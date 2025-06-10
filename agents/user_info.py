from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from agents.state import State



@tool
def fetch_user_info(config: RunnableConfig)  -> dict:
    """Fetch all information of the user's profile along.

    Returns:
        Dictionary containing the user profile details.
    """
    configuration = config.get("configurable", {})
    user_id = configuration.get("user_info", None)
    if not user_id:
        raise ValueError("No user ID configured.")
    return {"user_info": user_id}

def user_info(state: State):
    # TODO: This should come from telegram chat bot info
    return {"user_info": fetch_user_info.invoke({})}