

from app.core.repository.vectorstore import VectorStoreRepository
from langchain_core.documents import Document

class VectorStoreService:
    def __init__(self, vector_store: VectorStoreRepository):
        self.vector_store = vector_store

    def add_documents(self, documents: list[Document]):
        self.vector_store.add_documents(documents)

    def get_documents(self):
        return self.vector_store.get_documents()

    def get_document_count(self):
        return self.vector_store.get_document_count()

    def retrieve(self, query: str):
        return self.vector_store.retrieve(query)

    def delete(self):
        self.vector_store.delete()
