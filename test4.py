import os

os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

import config

from langchain_core.messages import HumanMessage
from thread_settings import ThreadSettings
from agents.utils import ModelType

query = "Вопрос техподдержки: Как отформатировать диск в Ubuntu?"

chat = ThreadSettings(user_id="gv", chat_id=283983, model=ModelType.GPT)

assistant = chat.assistant

messages = HumanMessage(
    content=[{"type": "text", "text": query}]
)

response = assistant.invoke(
    {"messages": [messages]}, chat.get_config(), stream_mode="values"
)
print(response)