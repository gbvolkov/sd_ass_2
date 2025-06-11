import os
import config
from enum import Enum

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda

from langgraph.prebuilt import ToolNode

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

import telegramify_markdown
import telegramify_markdown.customize as customize

customize.strict_markdown = False

class ModelType(Enum):
    GPT = ("gpt", "GPT")
    YA = ("ya", "YandexGPT")
    SBER = ("sber", "Sber")
    LOCAL = ("local", "Local")
    MISTRAL = ("mistral", "MistralAI")
    GGUF = ("gguf", "GGUF")

    def __init__(self, value, display_name):
        self._value_ = value
        self.display_name = display_name

def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def _print_event(event: dict, _printed: set, max_length=1500):
    if current_state := event.get("dialog_state"):
        print("Currently in: ", current_state[-1])
    if message := event.get("messages"):
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = f"{msg_repr[:max_length]} ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)

def _print_response(event: dict, _printed: set, max_length=1500):
    if current_state := event.get("dialog_state"):
        print("Currently in: ", current_state[-1])
    if message := event.get("messages"):
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            if message.type == "ai" and message.content.strip() != "":
                msg_repr = message.content.strip()
                if len(msg_repr) > max_length:
                    msg_repr = f"{msg_repr[:max_length]} ... (truncated)"
                print(msg_repr)
            _printed.add(message.id)



def send_text_element(chat_id, element_content, bot, usr_msg = None):
    chunks = [element_content[i:i+3800] for i in range(0, len(element_content), 3800)]
    for chunk in chunks:
        try:
            formatted = telegramify_markdown.markdownify(chunk)
            if usr_msg:
                bot.reply_to(usr_msg, formatted, parse_mode='MarkdownV2')
            else:
                bot.send_message(chat_id, formatted, parse_mode='MarkdownV2')
        except Exception as e:
            bot.send_message(chat_id, chunk)


def _send_response(event: dict, _printed: set, thread, bot, usr_msg=None, max_length=0):
    if current_state := event.get("dialog_state"):
        print("Currently in: ", current_state[-1])
    if message := event.get("messages"):
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            if message.type == "ai" and message.content.strip() != "":
                msg_repr = message.content.strip()
                if max_length > 0 and len(msg_repr) > max_length:
                    msg_repr = f"{msg_repr[:max_length]} ... (truncated)"
                send_text_element(thread.chat_id, msg_repr, bot, usr_msg)
            _printed.add(message.id)

def _send_response_full(event: dict, _printed: set, thread, bot, usr_msg=None, max_length=0):
    if current_state := event.get("dialog_state"):
        print("Currently in: ", current_state[-1])
    if messages := event.get("messages"):
        if not isinstance(messages, list):
            messages = [messages]
        for message in messages:
            if message.id not in _printed:
                if message.type == "ai" and message.content.strip() != "":
                    msg_repr = message.content.strip()
                    if max_length > 0 and len(msg_repr) > max_length:
                        msg_repr = f"{msg_repr[:max_length]} ... (truncated)"
                    send_text_element(thread.chat_id, msg_repr, bot, usr_msg)
                _printed.add(message.id)

def show_graph(graph):
    try:
        png_data = graph.get_graph().draw_mermaid_png()
        # Write the PNG data to a file
        output_filename = "langgraph_visualization.png"
        with open(output_filename, "wb") as f:
            f.write(png_data)
        os.startfile(output_filename)
    except Exception as e:
        print(f"Error showing graph: {e}")


def image_to_uri(image_data: str) -> str:
    return f"data:image/jpeg;base64,{image_data}"

def summarise_image(image_uri: str):
    model = ChatOpenAI(model="gpt-4.1-nano")
    message = HumanMessage(
        content=[
            {"type": "text", "text": "generate up to four key words describing the image in Russian language"},
            {
                "type": "image_url",
                "image_url": {"url": f"{image_uri}"},
            },
        ],
    )
    response = model.invoke([message])
    return response.content


