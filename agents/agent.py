
import uuid
import os

os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

import config

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition

from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_gigachat import GigaChat
from agents.assistants.yandex_tools.yandex_tooling import ChatYandexGPTWithTools as ChatYandexGPTTolls

from langchain_core.messages.modifier import RemoveMessage

from langgraph_supervisor import create_supervisor
from langgraph.prebuilt.chat_agent_executor import create_react_agent


from classifier import classify_request
from agents.state.state import State, ConfigSchema
from agents.assistants.assistant import Assistant, assistant_factory
from agents.utils import create_tool_node_with_fallback, show_graph, _print_event, _print_response
from agents.user_info import user_info
from agents.utils import ModelType
from agents.tools.supervisor_tools import create_handoff_tool_no_history
from agents.retrievers.retriever import get_search_tool

import logging
logger = logging.getLogger(__name__)

def route_request(state: State) -> str:
    if state["messages"][-1].content[0].get("type") == "reset":
        return "reset_memory"
    agent_class = classify_request(state["messages"][-1].content[0]["text"])
    #return "assistant"
    return agent_class

def reset_memory(state: State) -> State:
    """
    Delete every message currently stored in the thread’s state.
    """
    all_msg_ids = [m.id for m in state["messages"]]
    # Returning RemoveMessage instances instructs the reducer to delete them
    return {
        "messages": [RemoveMessage(id=mid) for mid in all_msg_ids]
    }

def get_role_agent(model: ModelType = ModelType.GPT, role: str = "default"):
    assistant_node_name = f"assistant_{role}"

    llm, assistant_tools = assistant_factory(model, role)
    builder = StateGraph(State, config_schema=ConfigSchema)
    builder.add_node(assistant_node_name, Assistant(llm))
    builder.add_node("tools", create_tool_node_with_fallback(assistant_tools))
    builder.add_edge(START, assistant_node_name)
    builder.add_conditional_edges(
        assistant_node_name,
        tools_condition,
    )
    builder.add_edge("tools", assistant_node_name)

    # The checkpointer lets the graph persist its state
    # this is a complete memory for the entire graph.
    memory = MemorySaver()
    return builder.compile(name=assistant_node_name, checkpointer=memory), assistant_node_name


roles = {
    "service_desk": "Provides answers to questions related to resolving problems with issues in Interleasing's systems and business processes.",
    "sales_manager": "Provides answers to questions related to sales activities, products, sales conditions, discounts provided to our clients, leasing agreements and so on. Consults sales managers for all sales related processes, including activities of underwrighting, risks, operations and so on.",
    "default": "Provides answers to questions internal rules and features provided to Employees. Consults Interleasing Employees on any HR related questions."
}


def initialize_agent_supervisor(model: ModelType = ModelType.GPT, role: str = "default"):
    agent_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=1)
    team_llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.7)

    #llm_role, assistant_tools_role = assistant_factory(model, role)
    #llm_default, assistant_tools_default = assistant_factory(model, "default")
    search_kb = get_search_tool()
    search_tools = [
        search_kb
    ]
    with open("prompts/supervisor_prompt.txt", encoding="utf-8") as f:
        sv_prompt = f.read()
    with open("prompts/working_prompt_sales.txt", encoding="utf-8") as f:
        sm_prompt = f.read()
    with open("prompts/working_prompt.txt", encoding="utf-8") as f:
        sd_prompt = f.read()
    with open("prompts/working_prompt_employee.txt", encoding="utf-8") as f:
        default_prompt = f.read()

    sd_agent = create_react_agent(model=team_llm, tools=search_tools, prompt=sd_prompt, name="assistant_sd", debug=config.DEBUG_WORKFLOW)
    sm_agent = create_react_agent(model=team_llm, tools=search_tools, prompt=sm_prompt, name="assistant_sm", debug=config.DEBUG_WORKFLOW)
    default_agent = create_react_agent(model=team_llm, tools=search_tools, prompt=default_prompt, name="assistant_default", debug=config.DEBUG_WORKFLOW)
    
    #sd_agent, sd_agent_name = get_role_agent(model=ModelType.YA, role="service_desk")
    #sm_agent, sm_agent_name = get_role_agent(model=ModelType.YA, role="sales_manager")
    #default_agent, default_agent_name = get_role_agent(model=ModelType.YA, role="default")

    ho_sd = create_handoff_tool_no_history(
        agent_name = sd_agent.name, 
        agent_purpose=roles["service_desk"])
    ho_sm = create_handoff_tool_no_history(
        agent_name = sm_agent.name, 
        agent_purpose=roles["sales_manager"])
    ho_default = create_handoff_tool_no_history(
        agent_name = default_agent.name, 
        agent_purpose=roles["default"])
    
    ho_tools = [ho_sd, ho_sm, ho_default]
    team = [sd_agent, sm_agent, default_agent]

    supervisor_agent = create_supervisor(
        model=agent_llm, 
        agents=team,
        prompt=sv_prompt,
        tools=ho_tools,
        add_handoff_messages=False,
        add_handoff_back_messages=True,
        output_mode="last_message",
        parallel_tool_calls=True,
        supervisor_name="interleasing_qa"
    ).compile(name="interleasing_qa", debug = config.DEBUG_WORKFLOW)

    builder = StateGraph(State, config_schema=ConfigSchema)
    # Define nodes: these do the work
    builder.add_node("fetch_user_info", user_info)
    builder.add_node("reset_memory", reset_memory)

    #builder.add_node("assistant", Assistant(llm_role))
    builder.add_node("assistant", supervisor_agent)
    #builder.add_node("tools", create_tool_node_with_fallback(assistant_tools_role))
    # Define edges: these determine how the control flow moves

    builder.add_edge(START, "fetch_user_info")
    builder.add_conditional_edges(
        "fetch_user_info",
        route_request,
    )
    builder.add_edge("reset_memory", END)

    #builder.add_conditional_edges(
    #    "assistant",
    #    tools_condition,
    #)
    #builder.add_edge("tools", "assistant")

    # The checkpointer lets the graph persist its state
    # this is a complete memory for the entire graph.
    memory = MemorySaver()
    return builder.compile(name="interleasing_qa_agent", checkpointer=memory)

def initialize_agent(model: ModelType = ModelType.GPT, role: str = "default"):
    team_llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.7)

    search_kb = get_search_tool()
    search_tools = [
        search_kb
    ]
    with open("prompts/supervisor_prompt.txt", encoding="utf-8") as f:
        sv_prompt = f.read()
    with open("prompts/working_prompt_sales.txt", encoding="utf-8") as f:
        sm_prompt = f.read()
    with open("prompts/working_prompt.txt", encoding="utf-8") as f:
        sd_prompt = f.read()
    with open("prompts/working_prompt_employee.txt", encoding="utf-8") as f:
        default_prompt = f.read()

    sd_agent = create_react_agent(model=team_llm, tools=search_tools, prompt=sd_prompt, name="assistant_sd", debug=config.DEBUG_WORKFLOW)
    sm_agent = create_react_agent(model=team_llm, tools=search_tools, prompt=sm_prompt, name="assistant_sm", debug=config.DEBUG_WORKFLOW)
    default_agent = create_react_agent(model=team_llm, tools=search_tools, prompt=default_prompt, name="assistant_default", debug=config.DEBUG_WORKFLOW)
    

    builder = StateGraph(State, config_schema=ConfigSchema)
    # Define nodes
    builder.add_node("fetch_user_info", user_info)
    builder.add_node("reset_memory", reset_memory)

    builder.add_node("sm_agent", sm_agent)
    builder.add_node("sd_agent", sd_agent)
    builder.add_node("default_agent", default_agent)

    # Define edges
    builder.add_edge(START, "fetch_user_info")
    builder.add_conditional_edges(
        "fetch_user_info",
        route_request,
    )
    builder.add_edge("reset_memory", END)

    # The checkpointer lets the graph persist its state
    # this is a complete memory for the entire graph.
    memory = MemorySaver()
    return builder.compile(name="interleasing_qa_agent", checkpointer=memory)


if __name__ == "__main__":
    assistant_graph = initialize_agent(model=ModelType.GPT)

    #show_graph(assistant_graph)
    from langchain_core.messages import HumanMessage

    # Let's create an example conversation a user might have with the assistant
    tutorial_questions = [
        "Кто такие кей юзеры?",
        "Не работает МФУ",
        "Как отресетить график?"
    ]

    thread_id = str(uuid.uuid4())

    config = {
        "configurable": {
            # The passenger_id is used in our flight tools to
            # fetch the user's flight information
            "user_info": "3442 587242",
            # Checkpoints are accessed by thread_id
            "thread_id": thread_id,
        }
    }

    _printed = set()
    for question in tutorial_questions[:2]:
        events = assistant_graph.stream(
            {"messages": [HumanMessage(content=[{"type": "text", "text": question}])]}, config, stream_mode="values"
        )
        print("USER: ", question)
        print("-------------------")
        print("ASSISTANT:")
        for event in events:
            #_print_event(event, _printed)
            _print_response(event, _printed)
        print("===================")

    print("RESET")
    events = assistant_graph.invoke(
        {"messages": [HumanMessage(content=[{"type": "reset", "text": "RESET"}])]}, config, stream_mode="values"
    )
    #for event in events:
    #    _print_response(event, _printed)

    for question in tutorial_questions[2:]:
        events = assistant_graph.stream(
            {"messages": [HumanMessage(content=[{"type": "text", "text": question}])]}, config, stream_mode="values"
        )
        print("USER: ", question)
        print("-------------------")
        print("ASSISTANT:")
        for event in events:
            #_print_event(event, _printed)
            _print_response(event, _printed)
        print("===================")
