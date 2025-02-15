
from langchain_community.document_loaders import PyPDFLoader
from app.core.services.vectorstore_service import VectorStoreService


class ProcessingService:
    def __init__(self):
       pass
    def process_paper(self, paper_url: str):
        loader = PyPDFLoader(paper_url)
        pages = []
        for page in loader.load():
            pages.append(page)
        return pages
    