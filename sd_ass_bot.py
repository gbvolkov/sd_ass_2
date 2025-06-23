from datetime import datetime
from http.client import HTTPException
import logging
logging.basicConfig(level=logging.INFO)

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

from palimpsest.logger_factory import setup_logging
from store_managers.google_sheets_man import GoogleSheetsManager

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
        bot.send_message(user_id, "Память бота очищена.")
    
    """
    @bot.message_handler(commands=['role'])
    def set_role(message):
        user_id = message.from_user.username
        chat_id = message.chat.id
        role = message.any_text[len("/role"):].strip()
        #role = user_man.get_role(user_id)

        chats[chat_id] = ThreadSettings(user_id=user_id, chat_id=chat_id, role=role)

        #resetting memory
        assistant = chats[chat_id].assistant
        assistant.invoke(
            {"messages": [HumanMessage(content=[{"type": "reset", "text": "RESET"}])]}, chats[chat_id].get_config(), stream_mode="values"
        )
        
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
    """
    @bot.message_handler(content_types=['text', 'voice', 'photo', 'document'])
    def handle_message(message):
        chat_id = message.chat.id
        user_id = message.from_user.username
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
        
        bot.send_message(chat_id, 'Пожалуйста, оцените ответ.', reply_markup=create_rating_keyboard())

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
    setup_logging("sd_assistant", project_console_level=logging.WARNING, other_console_level=logging.WARNING)
    run_bot()