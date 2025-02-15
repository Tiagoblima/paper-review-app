from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from app.core.repository.models import EmbeddingsModelRepository




class VectorStoreRepository:
    def __init__(self, embeddings_model: EmbeddingsModelRepository):
        self.vector_store = InMemoryVectorStore(embeddings=embeddings_model.get_model())

    def add_documents(self, documents: list[Document]):
        self.vector_store.add_documents(documents)

    def get_vector_store(self):
        return self.vector_store
    
    def get_documents(self):
        return self.vector_store.get_documents()
    
    def get_document_count(self):
        return self.vector_store.count()
    
    def retrieve(self, query: str):
        return self.vector_store.similarity_search(query)
    
    def delete(self):
        self.vector_store.delete_all()
    
    