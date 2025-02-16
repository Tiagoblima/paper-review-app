from enum import Enum
import os
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings

class ModelProvider(Enum):
    GROQ = "groq"
    OPENAI = "openai"
class EmbeddingsModelProvider(Enum):
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
class EmbeddingModel(Enum):
    BAAI_EMBEDDING = "BAAI/bge-base-en-v1.5"
    OPENAI_EMBEDDING = "text-embedding-3-small" 
    SENTENCE_TRANSFORMER = "sentence-transformers/all-mpnet-base-v2"


class BaseModelRepository:
    def __init__(self):
       pass

    def get_model(self):
        pass

class ChatModelRepository(BaseModelRepository):
    def __init__(self, model_name: str, api_key: str, provider: str):
       
        self.chat_model = init_chat_model(
            model=model_name,
            api_key=api_key,
            model_provider=provider
        )

    def get_model(self):
        return self.chat_model


class EmbeddingsModelRepository(BaseModelRepository):
    def __init__(self, model_name: str):
       self.embeddings_model = HuggingFaceEmbeddings(
            model_name=model_name,
        )

    def get_model(self):
        return self.embeddings_model
