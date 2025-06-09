from utils import ModelType
from agent import initialize_agent

class ThreadSettings():
    def __init__(self, user_id, chat_id, model=ModelType.GPT, role="default"):
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
        self.role = role

    @property
    def assistant(self): 
        if self._assistant is None:
            self._assistant = initialize_agent(self.model, self.role)

        return self._assistant
    
    @assistant.setter 
    def assistant(self, assistant):
        self._assistant = assistant

    def get_config(self):
        return {
            "configurable": {
                # The user_id is used in our tools to
                # fetch the user's information
                "user_info": self.user_id,
                "user_role": self.role,
                # Checkpoints are accessed by thread_id
                "thread_id": self.chat_id,
            }
        }