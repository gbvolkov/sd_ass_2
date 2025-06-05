from http.client import HTTPException
import logging
logging.basicConfig(level=logging.INFO)

import config

import telebot
from telebot import types
from telebot.apihelper import ApiTelegramException
from langchain_core.messages import HumanMessage

import time, uuid, json, os, base64
from collections import defaultdict

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from vrecog.vrecog import recognise_text

from thread_settings import ThreadSettings

from utils import _send_response, summarise_image, image_to_uri, ModelType

from palimpsest.logger_factory import setup_logging


def run_bot():

    bot = telebot.TeleBot(config.TELEGRAM_BOT_TOKEN)
    chats = defaultdict(ThreadSettings)

    @bot.message_handler(commands=['start'])
    def send_welcome(message):
        user_id = message.from_user.id
        chat_id = message.chat.id
        chats[chat_id] = ThreadSettings(user_id=user_id, chat_id=chat_id, model=ModelType.YA)

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
        user_id = message.from_user.id
        chat_id = message.chat.id
        chats[chat_id] = ThreadSettings(user_id=user_id, chat_id=chat_id)

        #resetting memory
        assistant = chats[chat_id].assistant
        assistant.invoke(
            {"messages": [HumanMessage(content=[{"type": "reset", "text": "RESET"}])]}, chats[chat_id].get_config(), stream_mode="values"
        )
        bot.send_message(user_id, "Память бота очищена.")

    @bot.message_handler(content_types=['text', 'voice', 'photo', 'document'])
    def handle_message(message):
        chat_id = message.chat.id
        user_id = message.from_user.id
        image_uri = []
        if chat_id not in chats:
            chats[chat_id] = ThreadSettings(user_id=user_id, chat_id=chat_id)
        if message.content_type == 'voice':
            #bot.send_message(user_id, "Распознаю голосовое сообщение...")
            try:
                bot.send_chat_action(chat_id=chat_id, action="upload_voice", timeout=30)
                file_info = bot.get_file(message.voice.file_id)
                downloaded_file = bot.download_file(file_info.file_path)
                ogg_file_path = f"voice_{user_id}_{uuid.uuid4()}.ogg"
                with open(ogg_file_path, 'wb') as f:
                    f.write(downloaded_file)

                # Передаём путь к OGG-файлу в recognise_text
                query = recognise_text(ogg_file_path)
                os.remove(ogg_file_path)

                if not query:
                    bot.send_message(user_id, "Не удалось распознать голосовое сообщение. Пожалуйста, отправьте текст вручную или попробуйте снова.")
                    return
                logging.info(f"Распознанный текст:\n{query}")
            except Exception as e:
                logging.error(f"Error processing voice message: {str(e)}")
                bot.send_message(user_id, "Не удалось распознать голосовое сообщение. Пожалуйста, отправьте текст вручную или попробуйте снова.")
                return
        elif message.content_type in ('photo', 'document'):

            # Telegram compresses photos; if you need originals tell users to
            # send the image as a *document*.  We support both here.
            summary = ""
            query = message.any_text or ""
            try:
                bot.send_chat_action(chat_id=chat_id, action="upload_photo", timeout=30)
                if message.content_type == 'photo':
                    # photo array is sorted by size; take the last (largest) thumb
                    file_id = message.photo[-1].file_id
                else:
                    # document
                    file_id = message.document.file_id

                file_info = bot.get_file(file_id)
                img_bytes = bot.download_file(file_info.file_path)

                # Convert to Base‑64 data‑URI
                uri = image_to_uri(base64.b64encode(img_bytes).decode())
                #bot.send_message(user_id, "Обрабатываю изображение…")
                summary = summarise_image(uri)
                query = query + "\n\n" + summary
                image_uri = [{"type": "image_url", "image_url": {"url": uri}}]
                #bot.send_message(chat_id, f"🖼️  Вот краткое описание изображения:\n\n{summary}")
            except Exception as e:
                logging.exception("Error processing image")
                bot.send_message(user_id,
                                "Не удалось обработать изображение. "
                                "Попробуйте отправить другое или повторить позже.")
        else:
            query = message.text


        assistant = chats[chat_id].assistant
        if not message.reply_to_message:
            assistant.invoke(
                {"messages": [HumanMessage(content=[{"type": "reset", "text": "RESET"}])]}, chats[chat_id].get_config(), stream_mode="values"
            )

        messages = HumanMessage(
            content=[{"type": "text", "text": query}] + image_uri
        )

        events = assistant.stream(
            {"messages": [messages]}, chats[chat_id].get_config(), stream_mode="values"
        )
        _printed = set()
        for event in events:
            bot.send_chat_action(chat_id=chat_id, action="typing", timeout=30)
            _send_response(event, _printed, thread=chats[chat_id], bot=bot, usr_msg=message)

    while True:
        try:
            bot.polling(none_stop=True, timeout=60)
        except ApiTelegramException as e:
            logging.error(f"Telegram API error: {e}")
            time.sleep(5)
        except HTTPException as e:
            logging.error(f"HTTP error: {e}")
            time.sleep(5)
        #except Exception as e:
        #    logging.error(f"Unexpected error in bot polling: {e}")
        #    time.sleep(5)


if __name__ == '__main__':
    setup_logging("sd_assistant", project_console_level=logging.DEBUG, other_console_level=logging.WARNING)
    run_bot()