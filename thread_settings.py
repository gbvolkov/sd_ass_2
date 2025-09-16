from agents.utils import ModelType
from agents.agent import initialize_agent
from agents.state.state import ConfigSchema

from user_manager.utils import UserManager
from langchain_core.runnables import RunnableConfig
import config

class ThreadSettings():
    user_man = UserManager()

    def __init__(self, user_id, chat_id, model=ModelType.GPT):
        self.model=model
        self.question = ''
        self.answer = ''
        self.context = ''
        # Extend with a rating “workflow” state if needed
        self.stage = 'idle'         # will track: 'waiting_for_rephrased', 'waiting_for_correct_answer', etc.
        self.rephrased_question = ''
        self.correct_answer = ''
        self._assistant = None
        self.user_id = user_id
        self.chat_id = chat_id
        self.role = ThreadSettings.user_man.get_role(user_id)

    def is_allowed(self)-> bool:
        return config.CHECK_RIGHTS.strip().lower()!='true' or self.user_man.is_allowed(self.user_id)

    @property
    def assistant(self): 
        if self._assistant is None:
            self._assistant = initialize_agent(self.model, self.role)

        return self._assistant
    
    @assistant.setter 
    def assistant(self, assistant):
        self._assistant = assistant

    def get_config(self):
        return RunnableConfig(ConfigSchema({"user_id": self.user_id, "user_role": self.role, "model": ModelType.GPT, "thread_id": self.chat_id}))
    #   {
    #        "configurable": {
    #            # The user_id is used in our tools to
    #            # fetch the user's information
    #            "user_info": self.user_id,
    #            "user_role": self.role,
    #            # Checkpoints are accessed by thread_id
    #            "thread_id": self.chat_id,
    #        }
    #    }