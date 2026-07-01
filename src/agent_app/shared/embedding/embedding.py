from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from typing import List

class LocalSentenceEmbeddings(Embeddings):
    def __init__(self, 
        model_name="BAAI/bge-large-en-v1.5", 
        hf_token: str = None,
        default_truncate_dim: int = 1024
    ):
        self._default_truncate_dim = default_truncate_dim
        hf_token = f"Bearer {hf_token}" if hf_token else None 
        self.model = SentenceTransformer(
            model_name,
            use_auth_token= hf_token
        )
       
    def embed_documents(self, texts: List[str], truncate_dim: int | None = None) -> List[List[float]]:
        truncate_dim = truncate_dim or self._default_truncate_dim
        return self.model.encode(texts, normalize_embeddings=True, batch_size=100, truncate_dim=truncate_dim).tolist()

    def embed_query(self, text: str, truncate_dim: int | None = None) -> List[float]:
        truncate_dim = truncate_dim or self._default_truncate_dim
        return self.model.encode(text, normalize_embeddings=True, truncate_dim=truncate_dim).tolist()
