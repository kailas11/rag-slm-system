"""Sentence-transformers based embedder (local, no API key needed)."""

import numpy as np

from rag_slm_system.embeddings.base import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    """Embedding using sentence-transformers models (runs locally).

    Default model: all-MiniLM-L6-v2 (384 dimensions, fast, good quality).
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32,
        normalize: bool = True,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required: pip install sentence-transformers"
            ) from e

        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.normalize = normalize
        self._dimension = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=len(texts) > 100,
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        embedding = self.model.encode(
            [query],
            normalize_embeddings=self.normalize,
        )
        return np.array(embedding[0], dtype=np.float32)

    @property
    def dimension(self) -> int:
        return self._dimension
