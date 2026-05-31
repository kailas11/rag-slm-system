"""Reranking modules for hybrid retrieval.

Supports:
- CohereReranker: uses the Cohere Rerank API (requires COHERE_API_KEY)
- CrossEncoderReranker: local cross-encoder model (no API key needed)
"""

import logging
import os
from abc import ABC, abstractmethod

from rag_slm_system.vectorstore.base import SearchResult

logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    """Abstract base class for rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Rerank search results given a query.

        Args:
            query: The user query.
            results: Candidate results from initial retrieval.
            top_k: Number of top results to return after reranking.

        Returns:
            Reranked list of SearchResult, ordered by relevance.
        """


class CohereReranker(BaseReranker):
    """Reranker using the Cohere Rerank API.

    Requires ``cohere`` package and a valid COHERE_API_KEY.
    """

    DEFAULT_MODEL = "rerank-v3.5"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        try:
            import cohere
        except ImportError as e:
            raise ImportError(
                "cohere is required for CohereReranker: pip install cohere"
            ) from e

        resolved_key = api_key or os.environ.get("COHERE_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Cohere API key required. Set COHERE_API_KEY env var "
                "or pass api_key parameter."
            )

        self._client = cohere.Client(resolved_key)
        self.model = model or self.DEFAULT_MODEL

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        if not results:
            return []

        documents = [r.chunk.text for r in results]
        top_k = min(top_k, len(documents))

        response = self._client.rerank(
            query=query,
            documents=documents,
            model=self.model,
            top_n=top_k,
        )

        reranked: list[SearchResult] = []
        for rank, item in enumerate(response.results):
            original = results[item.index]
            reranked.append(
                SearchResult(
                    chunk=original.chunk,
                    score=float(item.relevance_score),
                    rank=rank,
                )
            )

        return reranked


class CrossEncoderReranker(BaseReranker):
    """Reranker using a local cross-encoder model (no API key needed).

    Default model: cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, ~80MB).
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        model_name: str | None = None,
        device: str = "cpu",
        batch_size: int = 32,
    ):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderReranker"
            ) from e

        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device
        self.batch_size = batch_size
        self._model = CrossEncoder(self.model_name, device=self.device)
        logger.info(f"CrossEncoderReranker loaded: {self.model_name}")

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        if not results:
            return []

        pairs = [(query, r.chunk.text) for r in results]
        scores = self._model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )

        scored = list(zip(results, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        top_k = min(top_k, len(scored))
        reranked: list[SearchResult] = []
        for rank, (result, score) in enumerate(scored[:top_k]):
            reranked.append(
                SearchResult(
                    chunk=result.chunk,
                    score=float(score),
                    rank=rank,
                )
            )

        return reranked
