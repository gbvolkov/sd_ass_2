import contextlib
from datetime import datetime
from http.client import HTTPException
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
#from palimpsest.logger_factory import setup_logging
#setup_logging("sd_assistant", project_console_level=logging.WARNING, other_console_level=logging.WARNING)


import config

import telebot
from telebot import types
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
from telebot.apihelper import ApiTelegramException

from langchain_core.messages import HumanMessage

import time, uuid, json, os, base64
from collections import defaultdict

if config.NO_CUDA == "True":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

from vrecog.vrecog import recognise_text

from thread_settings import ThreadSettings

from agents.utils import _send_response, summarise_image, image_to_uri, ModelType

from store_managers.google_sheets_man import GoogleSheetsManager

from agents.retrievers.retriever import refresh_indexes

from utils.periodic_task import PeriodicTask

from bot_helpers import TypingKeeper

def run_bot():

    bot = telebot.TeleBot(config.TELEGRAM_BOT_TOKEN)
    chats = defaultdict(ThreadSettings)

    try:
        sheets_manager = GoogleSheetsManager(config.GOOGLE_SHEETS_CRED, config.FEEDBACK_SHEET_ID)
    except Exception as e:
        logging.error(f"Error initializing Google Sheets manager: {e}")
        sheets_manager = None

    # Add this function to create the rating keyboard
    def create_rating_keyboard():
        keyboard = InlineKeyboardMarkup()
        keyboard.add(InlineKeyboardButton("Промазал:(", callback_data="rate_1"))
        #keyboard.add(InlineKeyboardButton("Похоже на правду.", callback_data="rate_2"))
        keyboard.add(InlineKeyboardButton("В точку! Спасибо!", callback_data="rate_3"))
        return keyboard

    @bot.message_handler(commands=['start'])
    def send_welcome(message):
        user_id = message.from_user.username
        chat_id = message.chat.id
        chats[chat_id] = ThreadSettings(user_id=user_id, chat_id=chat_id, model=ModelType.GPT)

        assistant = chats[chat_id].assistant
        #resetting memory
        assistant.invoke(
            {"messages": [HumanMessage(content=[{"type": "reset", "text": "RESET"}])]}, chats[chat_id].get_config(), stream_mode="values"
        )

        #Generating welcome message
        query = "Привет! Представься пожалуйста и расскажи о себе."
        messages = HumanMessage(
            content=[{"type": "text", "text": query}]
        )
        events = assistant.stream(
            {"messages": [messages]}, chats[chat_id].get_config(), stream_mode="values"
        )
        _printed = set()
        for event in events:
            bot.send_chat_action(chat_id=chat_id, action="typing", timeout=30)
            _send_response(event, _printed, thread=chats[chat_id], bot=bot)

    @bot.message_handler(commands=['reset'])
    def reset_memory(message):
        user_id = message.from_user.username
        chat_id = message.chat.id
        chats[chat_id] = ThreadSettings(user_id=user_id, chat_id=chat_id)

        #resetting memory
        assistant = chats[chat_id].assistant
        assistant.invoke(
            {"messages": [HumanMessage(content=[{"type": "reset", "text": "RESET"}])]}, chats[chat_id].get_config(), stream_mode="values"
        )
        bot.send_message(chat_id, "Память бота очищена.")

    @bot.message_handler(commands=['reload'])
    def reload_kb(message):
        kb_update_thread.pause()
        chat_id = message.chat.id
        refresh_indexes()
        kb_update_thread.resume()
        bot.send_message(chat_id, "База знаний обновлена.")

    @bot.message_handler(content_types=['text', 'voice', 'photo', 'document'])
    def handle_message(message):
        kb_update_thread.pause()

        placeholder = None
        _keeper = None

        try:
            chat_id = message.chat.id
            user_id = message.from_user.username
            image_uri = []

            # Ensure thread settings exist
            if chat_id not in chats:
                chats[chat_id] = ThreadSettings(user_id=user_id, chat_id=chat_id)

            # ----- Build query based on content type -----
            if message.content_type == 'voice':
                # Voice -> download, transcribe, build query
                try:
                    bot.send_chat_action(chat_id=chat_id, action="upload_voice", timeout=30)
                    file_info = bot.get_file(message.voice.file_id)
                    downloaded_file = bot.download_file(file_info.file_path)
                    ogg_file_path = f"voice_{user_id}_{uuid.uuid4()}.ogg"
                    with open(ogg_file_path, 'wb') as f:
                        f.write(downloaded_file)

                    query = recognise_text(ogg_file_path)
                    os.remove(ogg_file_path)

                    if not query:
                        bot.send_message(
                            chat_id,
                            "Не удалось распознать голосовое сообщение. "
                            "Пожалуйста, отправьте текст вручную или попробуйте снова."
                        )
                        return

                    logging.info(f"Распознанный текст:\n{query}")

                except Exception as e:
                    logging.error(f"Error processing voice message: {str(e)}")
                    bot.send_message(
                        chat_id,
                        "Не удалось распознать голосовое сообщение. "
                        "Пожалуйста, отправьте текст вручную или попробуйте снова."
                    )
                    return

            elif message.content_type in ('photo', 'document'):
                # Photo/Document -> download, summarise, build query + image_uri
                # Prefer any existing text (custom field) or caption as seed query
                seed_text = getattr(message, "any_text", None) or (getattr(message, "caption", None) or "")
                summary = ""
                query = seed_text
                try:
                    bot.send_chat_action(chat_id=chat_id, action="upload_photo", timeout=30)

                    if message.content_type == 'photo':
                        # Take the largest available photo
                        file_id = message.photo[-1].file_id
                    else:
                        file_id = message.document.file_id

                    file_info = bot.get_file(file_id)
                    img_bytes = bot.download_file(file_info.file_path)

                    # Convert to Base-64 data-URI and summarise
                    uri = image_to_uri(base64.b64encode(img_bytes).decode())
                    summary = summarise_image(uri)

                    # Avoid leading blank line if seed_text is empty
                    query = (query + "\n\n" if query else "") + summary
                    image_uri = [{"type": "image_url", "image_url": {"url": uri}}]

                except Exception as e:
                    logging.exception("Error processing image")
                    bot.send_message(
                        chat_id,
                        "Не удалось обработать изображение. "
                        "Попробуйте отправить другое или повторить позже."
                    )
                    return

            else:
                # Plain text
                query = message.text or ""

            assistant = chats[chat_id].assistant

            # Reset memory if this is not a threaded reply (preserves your behavior)
            if not message.reply_to_message:
                assistant.invoke(
                    {"messages": [HumanMessage(content=[{"type": "reset", "text": "RESET"}])]},
                    chats[chat_id].get_config(),
                    stream_mode="values"
                )

            # Message payload for the agent
            payload_msg = HumanMessage(content=[{"type": "text", "text": query}] + image_uri)

            # ----- Show persistent placeholder and keep 'typing' alive -----
            placeholder = bot.reply_to(
                message,
                "⌛ Обрабатываю запрос..."
            )
            _keeper = TypingKeeper(bot, chat_id)
            _keeper.start()

            final_answer = "—"
            # ----- Stream from agent, but DO NOT send partials; accumulate only -----
            try:
                events = assistant.stream(
                    {"messages": [payload_msg]},
                    chats[chat_id].get_config(),
                    stream_mode="values"
                )

                _printed = set()
                answer_parts = []

                for event in events:
                    # Keep ephemeral typing fresh as well
                    with contextlib.suppress(Exception):
                        bot.send_chat_action(chat_id=chat_id, action="typing", timeout=30)

                    # Extract the latest AI message text and collect it
                    if isinstance(event, dict) and event.get("messages"):
                        msg = event["messages"]
                        if isinstance(msg, list) and msg:
                            msg = msg[-1]

                        msg_id = getattr(msg, "id", None)
                        msg_type = getattr(msg, "type", "")
                        content = getattr(msg, "content", None)

                        if msg_id in _printed:
                            continue
                        if msg_type != "ai":
                            continue

                        if isinstance(content, str) and content.strip():
                            answer_parts.append(content.strip())
                        elif isinstance(content, list):
                            # Some SDKs return list of blocks; prefer textual ones
                            parts = []
                            for block in content:
                                t = getattr(block, "text", None) or getattr(block, "content", None)
                                if isinstance(t, str) and t.strip():
                                    parts.append(t.strip())
                            if parts:
                                answer_parts.append("\n".join(parts))

                        _printed.add(msg_id)

                final_answer = "\n".join(answer_parts).strip() or "—"

                # ----- Edit placeholder to the full answer (chunk if >4096) -----
                def _chunks(s, n=4000):
                    for i in range(0, len(s), n):
                        yield s[i:i+n]

                try:
                    chunks = list(_chunks(final_answer))
                    bot.edit_message_text(
                        chunks[0],
                        chat_id=placeholder.chat.id,
                        message_id=placeholder.message_id
                    )
                    for extra in chunks[1:]:
                        bot.send_message(chat_id, extra)
                except Exception:
                    # Fallback if edit fails (e.g. message deleted)
                    bot.send_message(chat_id, final_answer)

            except Exception as e:
                # Show the error on the placeholder so the user definitely sees it
                with contextlib.suppress(Exception):
                    bot.edit_message_text(
                        f"❌ Ошибка: {e}",
                        chat_id=placeholder.chat.id,
                        message_id=placeholder.message_id
                    )
                final_answer = f"Ошибка: {e}"
            finally:
                # Always stop typing keeper
                with contextlib.suppress(Exception):
                    _keeper.stop()

            # Save Q&A for the rating flow
            chats[chat_id].question = query
            chats[chat_id].answer = final_answer

            # Ask for rating (your existing keyboard)
            bot.send_message(chat_id, 'Пожалуйста, оцените ответ.', reply_markup=create_rating_keyboard())

        except Exception as e:
            logging.exception("Unexpected error in handle_message")
            # If we had a placeholder up, surface the error there
            with contextlib.suppress(Exception):
                if placeholder:
                    bot.edit_message_text(
                        f"❌ Ошибка: {e}",
                        chat_id=placeholder.chat.id,
                        message_id=placeholder.message_id
                    )
                else:
                    bot.send_message(message.chat.id, f"❌ Ошибка: {e}")
        finally:
            kb_update_thread.resume()


    @bot.callback_query_handler(func=lambda call: call.data.startswith("rate_"))
    def callback_rating(call):
        rating = call.data.split("_")[1]
        chat_id = call.message.chat.id
        user_id = call.from_user.id
        username = call.from_user.username

        # Retrieve the user's question and bot's response
        user_question = chats[chat_id].question
        bot_response = chats[chat_id].answer
        model = chats[chat_id].model.value
        context = chats[chat_id].context

        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Store the data in Google Sheets
        if sheets_manager:
            sheets_manager.append_row([
                timestamp,
                username,
                user_question,
                bot_response,
                rating,
                model,
                context[:32000]
            ])

        # Here you can add code to store the rating in a database or file
        # For now, we'll just log it
        logging.info(f"User {username} (ID: {user_id}) rated message {call.message.message_id} as {rating}")

        # Provide feedback to the user
        rating_text = {
            "1": "Unreliable",
            "2": "Somewhat Helpful",
            "3": "Very Helpful"
        }
        bot.answer_callback_query(call.id, f"Вы поставили рейтинг {rating_text[rating]}. Спасибо за оценку!")

        # Remove the rating keyboard after rating
        bot.edit_message_reply_markup(chat_id=chat_id, message_id=call.message.message_id, reply_markup=None)

    def periodic_kb_update():
        refresh_indexes()

    # Start the periodic update thread
    kb_update_thread = PeriodicTask(periodic_kb_update) #task.start() threading.Thread(target=periodic_kb_update, daemon=True)
    kb_update_thread.start()

    while True:
        try:
            bot.polling(none_stop=True, timeout=60)
        except ApiTelegramException as e:
            logging.error(f"Telegram API error: {e}")
            time.sleep(5)
            kb_update_thread.resume()
        except HTTPException as e:
            logging.error(f"HTTP error: {e}")
            time.sleep(5)
            kb_update_thread.resume()
        except Exception as e:
            logging.error(f"Unexpected error in bot polling: {e}")
            time.sleep(5)
            kb_update_thread.resume()


if __name__ == '__main__':
    pid = os.getpid()
    with open(".process", "w") as f:
        f.write(f"{pid}")
    run_bot()