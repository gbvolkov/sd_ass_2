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
    
    primary_assistant_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an AI assistant for a service desk, tasked with helping users solve IT-related problems for Interleasing. "
                f" You support users with any matters related to the {subject}. "
                f" Your task is to answer the user's questions about '{subject}'. "
                #" You have access to tool: "
                #"     search_kb(query: string) → returns knowledge-base context for that query."
                #" Whenever you answer a user’s question, **first** call `search_kb`."
                " Do not produce any free-text answer until after you’ve used the tool."
                " Always use provided tools to search for information to answer the user's queries. "
                " Use `search_kb` tools **first** prior providing answers. In case information provided by tools is not complete, or irrelevant, extend it to fit question. "
                #" !!IMPORTANT: If answer to question is not in context returned by tools, tell user that you are not sure if your answer correct. "
                " When searching, be persistent. Expand your query bounds if the first search returns no results or results might be incomplete. "
                " Try to collect maximum usefull information from knowledgebase related to user's question."
                " If a search comes up empty, expand your search before giving up."
                " !!IMPORTANT: Never hallucinate. If you are not sure you have information tell user, that you cannot answer now and need to investigate more. "
                " !!IMPORTANT: Never cite others as a source of information. Always provide information as if you owned it yourself."
                f" Do not answer questions not retlated to '{subject}'. "
                " Всегда отвечай на русском языке."
                "**If your answer clearly addresses the user's question**, provide a response strictly in the following format using the Markdown dialect for Telegram. "
                "🔧 **Решение**"
                "🧠 **Понимание проблемы**"
                "[Briefly restate the user's problem.]"
                "💡 **Рекомендуемое решение**"
                "[Provide a detailed, step-by-step solution based on the knowledgebase entry(ies). Include reference to relevant images from Context (#IMAGE#image file name). For images use format [image description](image file name)]"
                "ℹ️ **Дополнительная информация**"
                "[Include links extracted from the 'Links' field of the 3 most relevant entries. Also, include the 3 most relevant summaries from the 'References' field.]"
                "📚 **Релевантные записи**"
                "[List no more than 3 relevant entries. Include the 'Problem Number' and 'Problem Description' from the knowledgebase.]"
                "❓ **Уточняющие вопросы**"
                "["
                "(1) **Design additional questions you can precisely answer from the Context**. Suggest user to ask them."
                "(2) **Respond**: 'Также я мог бы помочь Вам ответить на следующие вопросы:'; "
                "(3) **Include up to 3 questions to your answer.** Format them as a numbered list."
                "]"
                " Format your answer as MarkdownV2. Do not hesitate using emojies."
                "\n\nCurrent user:\n<User>\n{user_info}\n</User>"
                "\nCurrent time: {time}.",
            ),
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