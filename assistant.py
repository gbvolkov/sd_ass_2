from datetime import date, datetime
from typing import List, Any
import config

from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
#from langchain_community.llms.yandex import YandexGPT
#from langchain_community.chat_models import ChatYandexGPT
from yandex_tools.yandex_tooling import ChatYandexGPTWithTools as ChatYandexGPT
from langchain_gigachat import GigaChat

from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace, HuggingFaceEndpoint

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_community.tools import DuckDuckGoSearchRun

from langchain.agents import initialize_agent, AgentType

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from retriever import get_search_tool
from tools import get_support_contact, get_discounts_and_actions, get_customer_manager_contact
from utils import ModelType
from state import State

from hf_tools.chat_local import ChatLocalTools

from palimpsest import Palimpsest
import logging
logger = logging.getLogger(__name__)

class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)
            if (
                result.tool_calls
                or result.content
                and (
                    not isinstance(result.content, list)
                    or result.content[0].get("text")
                )
            ):
                break
            messages = state["messages"] + [("user", "Respond with a real output.")]
            state = {**state, "messages": messages}
        return {"messages": result}


def assistant_factory(model: ModelType):

    # Haiku is faster and cheaper, but less accurate
    # llm = ChatAnthropic(model="claude-3-haiku-20240307")
    
    processor = Palimpsest(verbose=True)

    def anonymize(text):
        return processor.anonimize(text)

    def deanonymize(text):
        return processor.deanonimize(text)
    
    subject = "IT systems and business processes of Interleasing"
    
    with open("prompts/working_prompt.txt", encoding="utf-8") as f:
        prompt_txt = f.read()
    prompt = eval(f"f'''{prompt_txt}'''")

    if model == ModelType.MISTRAL:
        llm = ChatMistralAI(model="mistral-large-latest", temperature=1, frequency_penalty=0.3)
    elif model == ModelType.YA:
        #model_name=f'gpt://{config.YA_FOLDER_ID}/yandexgpt/rc'
        model_name=f'gpt://{config.YA_FOLDER_ID}/yandexgpt-32k/rc'
        
        llm = ChatYandexGPT(
            #iam_token = None,
            api_key = config.YA_API_KEY, 
            folder_id=config.YA_FOLDER_ID, 
            model_uri=model_name,
            temperature=1
            )
    elif model == ModelType.LOCAL:
        with open("prompts/working_prompt_ru_short.txt", encoding="utf-8") as f:
            prompt_txt = f.read()
        prompt = eval(f"f'''{prompt_txt}'''")
        
        prompt = "Ты бот, который отвечает на вопросы пользователей. Перед ответом извлеки информацию из базы знаний, при помощи инструментов."
        llm = ChatLocalTools(model_id="yandex/YandexGPT-5-Lite-8B-instruct")
    elif model == ModelType.SBER:
        llm = GigaChat(
            credentials=config.GIGA_CHAT_AUTH, 
            model="GigaChat-Pro",
            verify_ssl_certs=False,
            temperature=1,
            scope = config.GIGA_CHAT_SCOPE)
    else:
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=1, frequency_penalty=0.3)

    primary_assistant_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt),
            ("placeholder", "{messages}"),
        ]
    ).partial(time=datetime.now)

    web_search_tool = DuckDuckGoSearchRun()


    #assistant_chain = primary_assistant_prompt | llm.bind_tools(assistant_tools)
    from chat_model_wrapper import AnonimizedChatModelProxy, make_anonymized_tool

    #search_kb = get_search_tool(processor)
    search_kb = get_search_tool()
    assistant_tools = [
        search_kb
    ]

    #assistant_chain = {"messages": lambda txt: anonymize(txt, language="en")} | primary_assistant_prompt | llm.bind_tools(assistant_tools) | (lambda ai_message: deanonymize(ai_message))
    #anon_llm = ChatModelInterceptor(llm, anonymize, deanonymize)
    tooled_llm = llm.bind_tools(assistant_tools)
    anon_llm = AnonimizedChatModelProxy(tooled_llm, anonymize, deanonymize)

    assistant_chain = primary_assistant_prompt | anon_llm
    
    return assistant_chain, assistant_tools