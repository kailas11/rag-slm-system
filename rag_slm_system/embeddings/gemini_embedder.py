"""Google Gemini embedding model."""

import numpy as np

from rag_slm_system.embeddings.base import BaseEmbedder


class GeminiEmbedder(BaseEmbedder):
    """Embedding using Google Gemini embedding API.

    Uses the text-embedding-004 model (768 dimensions).
    Requires GOOGLE_API_KEY environment variable or explicit api_key.
    """

    DEFAULT_MODEL = "text-embedding-004"
    EMBEDDING_DIM = 768

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        batch_size: int = 32,
    ):
        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "google-genai is required: pip install google-genai"
            ) from e

        self._model_name = model_name or self.DEFAULT_MODEL
        self.batch_size = batch_size
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()

    def embed(self, texts: list[str]) -> np.ndarray:
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            result = self._client.models.embed_content(
                model=self._model_name,
                contents=batch,
            )
            for embedding in result.embeddings:
                all_embeddings.append(embedding.values)

        return np.array(all_embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        result = self._client.models.embed_content(
            model=self._model_name,
            contents=[query],
        )
        return np.array(result.embeddings[0].values, dtype=np.float32)

    @property
    def dimension(self) -> int:
        return self.EMBEDDING_DIM
