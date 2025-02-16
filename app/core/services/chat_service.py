
from app.core.repository.models import ChatModelRepository
from langchain_core.messages import BaseMessage

from pydantic import BaseModel, Field
class ResponseFormatter(BaseModel):
    """Always use this tool to structure your response to the user."""
    answer: str = Field(description="The answer to the user's question")
    supporting_text: str = Field(description="A text from the context that supports the answer.")

class ChatModelService:
    def __init__(self, model_repository: ChatModelRepository):
        self.chat_model = model_repository.get_model()
        self.chat_model.bind_tools([ResponseFormatter])

    def invoke(self, messages: list[BaseMessage]):
        return self.chat_model.invoke(messages)
