
import os
from app.agents.review_agent import PaperReviewAgent
from app.core.repository.models import ChatModelRepository, EmbeddingModel, EmbeddingsModelRepository, ModelProvider
from app.core.repository.vectorstore import VectorStoreRepository
from app.core.services.chat_service import ChatModelService
from app.core.services.processing_service import ProcessingService
from app.core.services.vectorstore_service import VectorStoreService


class ReviewService:
    def __init__(self, paper_path: str):
        processing_service = ProcessingService()
        pages = processing_service.process_paper(paper_path)
        vector_store = VectorStoreRepository(EmbeddingsModelRepository(EmbeddingModel.SENTENCE_TRANSFORMER))
        vectorstore_service = VectorStoreService(vector_store)
        vectorstore_service.add_documents(pages)
        chat_service = ChatModelService(model_repository=ChatModelRepository(model_name=os.getenv("CHAT_MODEL"),
                                                                              api_key=os.getenv("GROQ_API_KEY"),
                                                                                provider=ModelProvider.GROQ.value))
        self.review_agent = PaperReviewAgent(chat_service, vectorstore_service)
        self.agent = self.review_agent.build_agent()

    def invoke(self, question: str):
        return self.agent.invoke({"question": question})
    
    def review_paper_from_url(self, url: str):
        pass
