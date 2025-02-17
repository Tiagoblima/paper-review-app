
from langchain_community.document_loaders import PyPDFLoader
from langchain_unstructured import UnstructuredLoader
from app.core.repository.models import EmbeddingsModelRepository
from app.core.repository.vectorstore import VectorStoreRepository
from app.core.services.vectorstore_service import VectorStoreService
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_experimental.text_splitter import SemanticChunker


class ProcessingService:
    def __init__(self, vector_store: VectorStoreRepository,
                 embedding_model: EmbeddingsModelRepository,
                 paper_url: str):
        self.paper_url = paper_url
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.text_splitter = SemanticChunker(embedding_model.get_model())

    def process_paper(self, ):
        #loader = PyPDFLoader(self.paper_url)
        loader = UnstructuredLoader(
        file_path=self.paper_url,
        strategy="hi_res",)
        pages = []
        for page in loader.load():
            pages.append(page)
        return pages
    
    def get_vectorstore_service(self):

        pages = self.process_paper()
        vectorstore_service = VectorStoreService(self.vector_store)
       
        vectorstore_service.add_documents(self.text_splitter.split_documents(pages))
        return vectorstore_service