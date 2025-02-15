
from app.core.repository.models import ChatModelRepository
from langchain_core.messages import BaseMessage


class ChatModelService:
    def __init__(self, model_repository: ChatModelRepository):
        self.chat_model = model_repository.get_model()

    def invoke(self, messages: list[BaseMessage]):
        return self.chat_model.invoke(messages)
