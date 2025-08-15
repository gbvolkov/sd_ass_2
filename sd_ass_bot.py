
"""
sd_ass_aiogram_bot.py
---------------------
aiogram-based Telegram bot.

Key features carried over:
 • Per-user/per-chat throttling middleware
 • Robust chat actions: show UPLOAD_* during file work, then TYPING while streaming
 • Placeholder -> stream (no partials) -> final edit, with chunking
 • Voice transcription (.ogg) via recognise_text
 • Image pipeline: download -> base64 -> image_to_uri -> summarise_image (off-thread)
 • Periodic KB refresh with pause/resume around heavy work
 • Rating flow stored to Google Sheets via GoogleSheetsManager
"""

import asyncio
import contextlib
import base64
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")

import os
import time
from datetime import datetime
from collections import defaultdict
from typing import Dict, Tuple, Optional, List

from aiogram import Bot, Dispatcher, types
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.enums import ChatAction

import config

if config.NO_CUDA == "True":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

from thread_settings import ThreadSettings
from agents.utils import summarise_image, image_to_uri, ModelType  # noqa: F401 (ModelType kept for parity)
from langchain_core.messages import HumanMessage
from agents.retrievers.retriever import refresh_indexes
from store_managers.google_sheets_man import GoogleSheetsManager

# Voice recognition
from vrecog.vrecog import recognise_text

import telegramify_markdown

# Periodic KB updater (threaded)
from utils.periodic_task import PeriodicTask


# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------
async def collect_final_text_from_stream(assistant, payload_msg, cfg) -> str:
    """
    Run the blocking assistant.stream(...) in a background thread and
    return the final assistant text (concatenated).
    """
    def _run_sync():
        final_parts = []
        printed_ids = set()
        events = assistant.stream({"messages": [payload_msg]}, cfg, stream_mode="values")
        for event in events:
            msg = event.get("messages") if isinstance(event, dict) else None
            if not msg:
                continue
            if isinstance(msg, list):
                msg = msg[-1]
            mid = getattr(msg, "id", None)
            if mid in printed_ids:
                continue
            if getattr(msg, "type", "") == "ai":
                text = (getattr(msg, "content", "") or "").strip()
                if text:
                    final_parts.append(text)
                    printed_ids.add(mid)
        return "\n".join(final_parts).strip() or "—"

    return await asyncio.to_thread(_run_sync)

async def send_text_element(
    bot: Bot,
    chat_id: int,
    element_content: str,
    usr_msg: Optional[types.Message] = None,
) -> None:
    """
    Split large text, convert to MarkdownV2, send safely.
    """
    chunks = [element_content[i: i + 3800] for i in range(0, len(element_content), 3800)]
    for chunk in chunks:
        try:
            formatted = telegramify_markdown.markdownify(chunk)
            if usr_msg:
                await bot.send_message(
                    chat_id=usr_msg.chat.id,
                    text=formatted,
                    parse_mode="MarkdownV2",
                    reply_to_message_id=usr_msg.message_id,
                )
            else:
                await bot.send_message(
                    chat_id=chat_id,
                    text=formatted,
                    parse_mode="MarkdownV2",
                )
        except Exception:
            # Fallback to raw text without parse (avoid MarkdownV2 parse errors)
            if usr_msg:
                await bot.send_message(
                    chat_id=usr_msg.chat.id,
                    text=chunk,
                    reply_to_message_id=usr_msg.message_id,
                    parse_mode=None,
                )
            else:
                await bot.send_message(chat_id=chat_id, text=chunk, parse_mode=None)


async def show_typing(bot: Bot, chat_id: int, action: ChatAction = ChatAction.TYPING):
    """
    Pulse the given chat action every 4 seconds.
    """
    with contextlib.suppress(asyncio.CancelledError):
        while True:
            await bot.send_chat_action(chat_id, action)
            await asyncio.sleep(4)


async def start_show_typing(bot: Bot, chat_id: int, action: ChatAction = ChatAction.TYPING) -> asyncio.Task:
    """
    Send an immediate chat action frame, then pulse every 4 seconds.
    """
    await bot.send_chat_action(chat_id, action)
    return asyncio.create_task(show_typing(bot, chat_id, action))   


def create_rating_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Промазал:(", callback_data="rate_1")],
            # [InlineKeyboardButton(text="Похоже на правду.", callback_data="rate_2")],
            [
                InlineKeyboardButton(
                    text="В точку! Спасибо!", callback_data="rate_3"
                )
            ],
        ]
    )


# -------------------------------------------------------------
# Main bot
# -------------------------------------------------------------

async def main() -> None:
    logging.basicConfig(level=logging.INFO)

    # Aiogram ≥3.7.0: set default parse mode here
    bot = Bot(
        config.TELEGRAM_BOT_TOKEN,
        default=DefaultBotProperties(parse_mode="MarkdownV2"),
    )
    dp = Dispatcher()

    # Thread state (per chat_id, matching sd_ass_bot.py behavior)
    chats: Dict[int, ThreadSettings] = defaultdict(ThreadSettings)

    # Sheets manager (optional)
    try:
        sheets_manager = GoogleSheetsManager(config.GOOGLE_SHEETS_CRED, config.FEEDBACK_SHEET_ID)
    except Exception as e:
        logging.error(f"Error initializing Google Sheets manager: {e}")
        sheets_manager = None

    # Throttling middleware (simple, per user per chat)
    class ThrottlingMiddleware:
        def __init__(self, rate: float = 3.0):
            self.rate = rate
            self.last_called: Dict[Tuple[int, int], float] = {}

        async def __call__(self, handler, event, data):
            from_user = getattr(event, "from_user", None)
            chat = getattr(event, "chat", None)
            if from_user and chat:
                key = (from_user.id, chat.id)
                now = time.monotonic()
                last = self.last_called.get(key, 0.0)
                elapsed = now - last
                if elapsed < self.rate:
                    await asyncio.sleep(self.rate - elapsed)
                self.last_called[key] = time.monotonic()
            return await handler(event, data)

    dp.message.middleware(ThrottlingMiddleware(rate=3.0))

    # Global error handler
    @dp.errors()
    async def global_error_handler(event, exception, **kwargs):
        logging.exception(f"Unhandled exception occured: {exception}")
        return True

    # /start — reset memory, greet
    @dp.message(Command("start"))
    async def cmd_start(message: types.Message) -> None:
        chat_id = message.chat.id
        user_id = message.from_user.username
        chats[chat_id] = ThreadSettings(user_id=user_id, chat_id=chat_id)

        # Reset memory
        assistant = chats[chat_id].assistant
        assistant.invoke(
            {"messages": [HumanMessage(content=[{"type": "reset", "text": "RESET"}])]},
            chats[chat_id].get_config(),
            stream_mode="values",
        )

        # Greet via streaming final text only
        query = "Привет! Представься пожалуйста и расскажи о себе."
        payload = HumanMessage(content=[{"type": "text", "text": query}])

        typing_task = await start_show_typing(bot, chat_id, ChatAction.TYPING)
        try:
            final_text = await collect_final_text_from_stream(
                assistant, payload, chats[chat_id].get_config()
            )
            await send_text_element(bot, chat_id, final_text, usr_msg=message)
        finally:
            typing_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await typing_task

    # /reset — reset memory only
    @dp.message(Command("reset"))
    async def cmd_reset(message: types.Message) -> None:
        chat_id = message.chat.id
        user_id = message.from_user.username
        chats[chat_id] = ThreadSettings(user_id=user_id, chat_id=chat_id)

        assistant = chats[chat_id].assistant
        assistant.invoke(
            {"messages": [HumanMessage(content=[{"type": "reset", "text": "RESET"}])]},
            chats[chat_id].get_config(),
            stream_mode="values",
        )
        await bot.send_message(chat_id, "Память бота очищена.", parse_mode=None)

    # /reload — refresh indexes
    @dp.message(Command("reload"))
    async def cmd_reload(message: types.Message) -> None:
        chat_id = message.chat.id
        kb_update_thread.pause()
        refresh_indexes()
        kb_update_thread.resume()
        await bot.send_message(chat_id, "База знаний обновлена.", parse_mode=None)

    # Main message handler: text, voice, photo, document
    @dp.message(lambda m: m.content_type in {"text", "voice", "photo", "document"})
    async def handle_message(message: types.Message) -> None:
        kb_update_thread.pause()
        placeholder: Optional[types.Message] = None
        typing_task: Optional[asyncio.Task] = None

        try:
            chat_id = message.chat.id
            user_id = message.from_user.username
            image_uri_payload = []

            # Ensure thread settings exist
            if chat_id not in chats:
                chats[chat_id] = ThreadSettings(user_id=user_id, chat_id=chat_id)

            # Build query from text/caption/voice
            query = message.text or message.caption or ""

            # Voice handling
            if message.content_type == "voice":
                # keep upload voice while saving/transcribing
                typing_task = await start_show_typing(bot, chat_id, ChatAction.UPLOAD_VOICE)
                try:
                    file_id = message.voice.file_id
                    file_info = await bot.get_file(file_id)
                    voice_io = await bot.download_file(file_info.file_path)
                    raw = voice_io.getvalue() if hasattr(voice_io, "getvalue") else voice_io

                    # Save to temp ogg
                    tmp_path = f"voice_{user_id}_{int(time.time()*1000)}.ogg"
                    with open(tmp_path, "wb") as f:
                        f.write(raw)

                    # Transcribe
                    query = recognise_text(tmp_path)
                    with contextlib.suppress(Exception):
                        os.remove(tmp_path)

                    if not query:
                        await bot.send_message(
                            chat_id,
                            "Не удалось распознать голосовое сообщение. "
                            "Пожалуйста, отправьте текст вручную или попробуйте снова.",
                            parse_mode=None,
                        )
                        return
                finally:
                    if typing_task:
                        typing_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await typing_task
                    typing_task = None  # reset to reuse below

            # Photo or document handling (treated as image for summary)
            if message.content_type in {"photo", "document"} and not message.voice:
                # choose action based on type
                action = ChatAction.UPLOAD_PHOTO
                if message.document and (message.document.mime_type or "").lower().startswith("image/") is False:
                    action = ChatAction.UPLOAD_DOCUMENT

                typing_task = await start_show_typing(bot, chat_id, action)
                try:
                    if message.photo:
                        file_id = message.photo[-1].file_id
                    else:
                        file_id = message.document.file_id

                    file_info = await bot.get_file(file_id)
                    file_io = await bot.download_file(file_info.file_path)
                    raw_bytes = file_io.getvalue() if hasattr(file_io, "getvalue") else file_io
                    uri = image_to_uri(base64.b64encode(raw_bytes).decode())

                    # Summarise (off the loop)
                    summary = await asyncio.to_thread(summarise_image, uri)
                    if summary:
                        query = (query + "\n\n" + summary).strip()
                    image_uri_payload = [{"type": "image_url", "image_url": {"url": uri}}]
                finally:
                    if typing_task:
                        typing_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await typing_task
                    typing_task = None  # will be re-started for TYPING

            # Access the assistant from the thread state
            assistant = chats[chat_id].assistant

            # Reset memory if not replying in-thread
            if not message.reply_to_message:
                assistant.invoke(
                    {"messages": [HumanMessage(content=[{"type": "reset", "text": "RESET"}])]},
                    chats[chat_id].get_config(),
                    stream_mode="values",
                )

            # Post a placeholder and start TYPING while we stream
            placeholder = await message.reply("⌛ Обрабатываю запрос...", parse_mode=None)
            typing_task = await start_show_typing(bot, chat_id, ChatAction.TYPING)

            # Build the message for the assistant
            payload_msg = HumanMessage(content=[{"type": "text", "text": query}] + image_uri_payload)

            try:
                final_answer = await collect_final_text_from_stream(
                    assistant, payload_msg, chats[chat_id].get_config()
                )

                # Edit placeholder with the full answer (chunk if needed)
                def _chunks(s: str, n: int = 4000):
                    for i in range(0, len(s), n):
                        yield s[i:i+n]

                chunks = list(_chunks(final_answer))

                # First chunk: edit placeholder
                try:
                    await bot.edit_message_text(
                        chat_id=placeholder.chat.id,
                        message_id=placeholder.message_id,
                        text=chunks[0],
                    )
                except Exception:
                    # Fallback send if edit fails
                    await bot.send_message(chat_id, chunks[0], parse_mode=None)

                # Remaining chunks as follow-up messages
                for extra in chunks[1:]:
                    await send_text_element(bot, chat_id, extra)

                # Persist Q/A for rating flow
                chats[chat_id].question = query
                chats[chat_id].answer = final_answer

                # Ask for rating
                await bot.send_message(chat_id, "Пожалуйста, оцените ответ.", reply_markup=create_rating_keyboard(), parse_mode=None)

            except Exception as e:
                logging.exception("Error while streaming/answering")
                # Surface error into placeholder if possible
                with contextlib.suppress(Exception):
                    if placeholder:
                        await bot.edit_message_text(
                            chat_id=placeholder.chat.id,
                            message_id=placeholder.message_id,
                            text="❌ Ошибка при обработке ответа",
                        )
                    else:
                        await bot.send_message(chat_id, "❌ Ошибка при обработке ответа", parse_mode=None)
            finally:
                if typing_task:
                    typing_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await typing_task

        except Exception as e:
            logging.exception("Unexpected error in handle_message")
            if placeholder:
                with contextlib.suppress(Exception):
                    await bot.edit_message_text(
                        chat_id=placeholder.chat.id,
                        message_id=placeholder.message_id,
                        text="❌ Ошибка при обработке ответа",
                    )
            else:
                with contextlib.suppress(Exception):
                    await bot.send_message(message.chat.id, "❌ Ошибка при обработке ответа", parse_mode=None)
        finally:
            kb_update_thread.resume()

    # Rating callback handler
    @dp.callback_query(lambda c: c.data and c.data.startswith("rate_"))
    async def callback_rating(call: types.CallbackQuery):
        rating = call.data.split("_")[1]
        chat_id = call.message.chat.id
        user_id = call.from_user.id
        username = call.from_user.username or f"id:{user_id}"

        # Collect data from the thread state
        user_question = getattr(chats[chat_id], "question", "-")
        bot_response = getattr(chats[chat_id], "answer", "-")
        model = getattr(chats[chat_id].model, "value", "-") if getattr(chats[chat_id], "model", None) else "-"
        context = getattr(chats[chat_id], "context", "") or ""

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if sheets_manager:
            with contextlib.suppress(Exception):
                sheets_manager.append_row([
                    timestamp,
                    username,
                    user_question,
                    bot_response,
                    rating,
                    model,
                    context[:32000],
                ])

        # Acknowledge & remove keyboard
        with contextlib.suppress(Exception):
            await call.answer(f"Вы поставили рейтинг: {rating}")
        with contextlib.suppress(Exception):
            await bot.edit_message_reply_markup(chat_id=chat_id, message_id=call.message.message_id, reply_markup=None)

    # Periodic knowledge base update (threaded, paused during heavy work)
    def periodic_kb_update():
        refresh_indexes()

    kb_update_thread = PeriodicTask(periodic_kb_update)
    kb_update_thread.start()

    # Start polling
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
