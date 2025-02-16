
import os
from app.agents.basic_info_agent import BasicInfoAgent
from app.agents.review_agent import PaperReviewAgent
from app.core.repository.models import ChatModelRepository, EmbeddingModel, EmbeddingsModelRepository, ModelProvider
from app.core.repository.vectorstore import VectorStoreRepository
from app.core.services.chat_service import ChatModelService
from app.core.services.processing_service import ProcessingService
from app.core.services.vectorstore_service import VectorStoreService

basic_info_questions = ["What is the title of the paper?",
                        "What are the authors of the paper?",
                        "What is the year of the paper?",
                        "What is the abstract of the paper?",
                        "What are the keywords of the paper?",
                        "What is the doi of the paper?",
                        "What is the country of the paper?",
                        "What is the conference/journal of the paper?"]

class ReviewService:
    def __init__(self, paper_path: str, basic_info_keys: list[str]):
        
        vector_store = VectorStoreRepository(EmbeddingsModelRepository(EmbeddingModel.SENTENCE_TRANSFORMER))
        vectorstore_service = ProcessingService(vector_store, paper_path).get_vectorstore_service()
        

        chat_service = ChatModelService(model_repository=ChatModelRepository(model_name=os.getenv("CHAT_MODEL"),
                                                                              api_key=os.getenv("GROQ_API_KEY"),
                                                                                provider=ModelProvider.GROQ.value))
        self.review_agent = PaperReviewAgent(chat_service, vectorstore_service).build_agent()
        self.basic_info_agent = BasicInfoAgent(chat_service, vectorstore_service, basic_info_keys).build_agent()

    def invoke(self, question: str):
        return self.review_agent.invoke({"question": question})
    
    def get_basic_info(self):
        return self.basic_info_agent.invoke({"question": "\n".join(basic_info_questions)})




class BasicInfoService:
    def __init__(self, paper_path: str, basic_info_keys: list[str]):
        processing_service = ProcessingService()
        pages = processing_service.process_paper(paper_path)
        vector_store = VectorStoreRepository(EmbeddingsModelRepository(EmbeddingModel.SENTENCE_TRANSFORMER))
        vectorstore_service = VectorStoreService(vector_store)
        vectorstore_service.add_documents(pages)
        chat_service = ChatModelService(model_repository=ChatModelRepository(model_name=os.getenv("CHAT_MODEL"),
                                                                              api_key=os.getenv("GROQ_API_KEY"),
                                                                                provider=ModelProvider.GROQ.value))
        self.review_agent = PaperReviewAgent(chat_service, vectorstore_service).build_agent()
        self.basic_info_agent = BasicInfoAgent(chat_service, vectorstore_service, basic_info_keys).build_agent()

    def invoke(self, question: str):
        return self.review_agent.invoke({"question": question})
    
    def get_basic_info(self):
        return self.basic_info_agent.invoke({"question": "\n".join(basic_info_questions)})




