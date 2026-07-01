# vector_base.py
from abc import ABC, abstractmethod
from typing import List, Union
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
class BaseVectorDBHelper(ABC):
    """
    Abstract interface defining the contract for Vector DB helper implementations.
    """
    
    @abstractmethod
    def upsert_documents(self, collection_name: str, documents: List[Document], ids: List[str | int]) -> None:
        """Inserts or updates vector records with unique IDs."""
        pass

    @abstractmethod
    def similarity_search(self, collection_name: str, query_text: str, limit: int = 3) -> List[tuple[Document, float]]:
        """Runs a semantic search against a collection and returns matching Documents."""
        pass

    @abstractmethod
    def delete_collection(self, collection_name: str) -> None:
        """Completely deletes a vector collection index."""
        pass
    
    @abstractmethod
    def get_embeddings(self) -> Embeddings:
        pass