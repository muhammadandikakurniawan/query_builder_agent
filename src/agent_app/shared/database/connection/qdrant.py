from typing import List, Union

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import VectorParams, Distance
from qdrant_client import QdrantClient

from agent_app.shared.database.helper.vectordb import BaseVectorDBHelper


class QdrantDBHelper(BaseVectorDBHelper):
    """
    Concrete implementation of BaseVectorDBHelper using Qdrant.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        host: str = "localhost",
        port: int = 6333,
    ):
        self._client = QdrantClient(host=host, port=port)
        self._embeddings = embeddings

    def get_embeddings(self) -> Embeddings:
        return self._embeddings

    def _ensure_collection_exists(self, collection_name: str) -> None:
        """
        Creates the collection if it does not already exist.
        """
        collections = self._client.get_collections().collections
        existing_names = {c.name for c in collections}

        if collection_name not in existing_names:
            vector_size = len(
                self._embeddings.embed_query("test")
            )

            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def upsert_documents(
        self,
        collection_name: str,
        documents: List[Document],
        ids: List[str|int],
    ) -> None:
        """
        Inserts or updates documents using explicit IDs.
        """
        self._ensure_collection_exists(collection_name)

        vectorstore = QdrantVectorStore(
            client=self._client,
            collection_name=collection_name,
            embedding=self._embeddings,
        )

        vectorstore.add_documents(
            documents=documents,
            ids=ids,
        )

    def similarity_search(
        self,
        collection_name: str,
        query_text: str,
        limit: int = 3,
    ) -> List[tuple[Document, float]]:
        """
        Performs semantic similarity search.
        """
        vectorstore = QdrantVectorStore(
            client=self._client,
            collection_name=collection_name,
            embedding=self._embeddings,
        )

        return vectorstore.similarity_search_with_score(
            query=query_text,
            k=limit,
        )

    def delete_collection(self, collection_name: str) -> None:
        """
        Deletes an entire collection.
        """
        try:
            self._client.delete_collection(collection_name=collection_name)
        except Exception:
            pass