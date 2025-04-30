from datetime import date, datetime
import config

from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
#from langchain_community.llms.yandex import YandexGPT
#from langchain_community.chat_models import ChatYandexGPT
from yandex_tools.yandex_tooling import ChatYandexGPTWithTools as ChatYandexGPT
from langchain_gigachat import GigaChat

from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace, HuggingFaceEndpoint

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_community.tools import DuckDuckGoSearchRun

from langchain.agents import initialize_agent, AgentType

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from retriever import search_kb
from tools import get_support_contact, get_discounts_and_actions, get_customer_manager_contact
from utils import ModelType
from state import State

class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


def assistant_factory(model: ModelType):

    # Haiku is faster and cheaper, but less accurate
    # llm = ChatAnthropic(model="claude-3-haiku-20240307")
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
    #elif model == ModelType.LOCAL:
    #    MODEL_NAME = "yandex/YandexGPT-5-Lite-8B-instruct"
    #    hf_llm = HuggingFaceEndpoint(repo_id=MODEL_NAME)
    #    llm = ChatHuggingFace(llm=hf_llm, verbose=True)
    elif model == ModelType.SBER:
        llm = GigaChat(
            credentials=config.GIGA_CHAT_AUTH, 
            model="GigaChat-Pro",
            verify_ssl_certs=False,
            temperature=1,
            scope = config.GIGA_CHAT_SCOPE)
    else:
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=1, frequency_penalty=0.3)

    subject = "IT systems and business processes of Interleasing"
    
    with open("prompts/working_prompt.txt", encoding="utf-8") as f:
        prompt_txt = f.read()
    
    prompt = eval(f"f'''{prompt_txt}'''")
    primary_assistant_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt),
            ("placeholder", "{messages}"),
        ]
    ).partial(time=datetime.now)

    web_search_tool = DuckDuckGoSearchRun()

    assistant_tools = [
        search_kb,
        web_search_tool
    ]
    assistant_chain = primary_assistant_prompt | llm.bind_tools(assistant_tools)
    
    return assistant_chain, assistant_tools