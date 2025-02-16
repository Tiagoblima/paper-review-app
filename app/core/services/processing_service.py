
from langchain_community.document_loaders import PyPDFLoader
from app.core.repository.vectorstore import VectorStoreRepository
from app.core.services.vectorstore_service import VectorStoreService
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)


class ProcessingService:
    def __init__(self, vector_store: VectorStoreRepository, paper_url: str):
        self.paper_url = paper_url
        self.vector_store = vector_store

    def process_paper(self, ):
        loader = PyPDFLoader(self.paper_url)
        pages = []
        for page in loader.load():
            pages.append(page)
        return pages
    
    def get_vectorstore_service(self):

        pages = self.process_paper()
        vectorstore_service = VectorStoreService(self.vector_store)
        vectorstore_service.add_documents(text_splitter.split_documents(pages))
        return vectorstore_service