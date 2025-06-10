
import uuid
import os

os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

from agents.utils import ModelType

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition

from langchain_core.messages.modifier import RemoveMessage

from agents.state import State
from agents.assistants.assistant import Assistant, assistant_factory
from agents.utils import create_tool_node_with_fallback, show_graph, _print_event, _print_response
from agents.user_info import user_info



def reset_memory_condition(state: State) -> str:
    if state["messages"][-1].content[0].get("type") == "reset":
        return "reset_memory"
    return "assistant"


def reset_memory(state: State) -> State:
    """
    Delete every message currently stored in the thread’s state.
    """
    all_msg_ids = [m.id for m in state["messages"]]
    # Returning RemoveMessage instances instructs the reducer to delete them
    return {
        "messages": [RemoveMessage(id=mid) for mid in all_msg_ids]
    }

def initialize_agent(model: ModelType = ModelType.GPT, role: str = "default"):
    llm, assistant_tools = assistant_factory(model, role)


    builder = StateGraph(State)
    # Define nodes: these do the work
    builder.add_node("fetch_user_info", user_info)
    builder.add_node("reset_memory", reset_memory)
    builder.add_node("assistant", Assistant(llm))
    builder.add_node("tools", create_tool_node_with_fallback(assistant_tools))
    # Define edges: these determine how the control flow moves

    builder.add_edge(START, "fetch_user_info")
    builder.add_conditional_edges(
        "fetch_user_info",
        reset_memory_condition,
    )
    builder.add_edge("reset_memory", END)

    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    builder.add_edge("tools", "assistant")

    # The checkpointer lets the graph persist its state
    # this is a complete memory for the entire graph.
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


if __name__ == "__main__":
    assistant_graph = initialize_agent(model=ModelType.LOCAL)

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
