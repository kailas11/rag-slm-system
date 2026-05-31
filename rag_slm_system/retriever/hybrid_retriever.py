"""Hybrid retriever combining dense (embedding) and sparse (BM25) retrieval
with reciprocal rank fusion and optional reranking."""

import logging
from dataclasses import dataclass

from rag_slm_system.chunking.base import Chunk
from rag_slm_system.embeddings.base import BaseEmbedder
from rag_slm_system.retriever.bm25_retriever import BM25Retriever
from rag_slm_system.retriever.rag_retriever import RetrievalResult
from rag_slm_system.retriever.reranker import BaseReranker
from rag_slm_system.vectorstore.base import BaseVectorStore, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class FusionResult:
    """Intermediate result from rank fusion before reranking."""

    chunk: Chunk
    dense_score: float
    sparse_score: float
    fused_score: float


class HybridRetriever:
    """Hybrid retriever: dense + BM25 → Reciprocal Rank Fusion → Reranker.

    Pipeline:
        1. Dense retrieval via embeddings + vector store
        2. Sparse retrieval via BM25 keyword matching
        3. Reciprocal Rank Fusion (RRF) to merge results
        4. Optional reranking (Cohere or cross-encoder) for final ordering

    Args:
        embedder: Embedding model for dense retrieval.
        vector_store: FAISS or other vector store for dense search.
        reranker: Optional reranker (CohereReranker or CrossEncoderReranker).
        top_k: Number of final results to return.
        dense_weight: Weight for dense retrieval in fusion (0-1).
        sparse_weight: Weight for sparse retrieval in fusion (0-1).
        rrf_k: RRF constant (higher = more uniform blending).
        fusion_top_k: Number of candidates to pass to reranker after fusion.
        score_threshold: Minimum score threshold for final results.
        bm25_k1: BM25 term frequency saturation parameter.
        bm25_b: BM25 document length normalization parameter.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        reranker: BaseReranker | None = None,
        top_k: int = 5,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        rrf_k: int = 60,
        fusion_top_k: int = 20,
        score_threshold: float = 0.0,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.reranker = reranker
        self.top_k = top_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k
        self.fusion_top_k = fusion_top_k
        self.score_threshold = score_threshold
        self._bm25 = BM25Retriever(k1=bm25_k1, b=bm25_b)

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Add chunks to the BM25 index (dense store is managed separately)."""
        self._bm25.add(chunks)

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        """Retrieve relevant chunks using hybrid dense + sparse + reranking.

        Args:
            query: Search query.
            top_k: Override default top_k for this query.

        Returns:
            RetrievalResult with ranked chunks and formatted context.
        """
        k = top_k or self.top_k
        fetch_k = max(self.fusion_top_k, k * 3)

        # 1. Dense retrieval
        query_embedding = self.embedder.embed_query(query)
        dense_results = self.vector_store.search(query_embedding, top_k=fetch_k)

        # 2. Sparse (BM25) retrieval
        sparse_results = self._bm25.search(query, top_k=fetch_k)

        # 3. Reciprocal Rank Fusion
        fused = self._reciprocal_rank_fusion(dense_results, sparse_results)

        # Take top candidates for reranking
        candidates = fused[: self.fusion_top_k]

        # Convert to SearchResult for reranker
        search_results = [
            SearchResult(chunk=f.chunk, score=f.fused_score, rank=i)
            for i, f in enumerate(candidates)
        ]

        # 4. Rerank if reranker is available
        if self.reranker and search_results:
            search_results = self.reranker.rerank(query, search_results, top_k=k)
        else:
            search_results = search_results[:k]

        # Apply score threshold
        if self.score_threshold > 0:
            search_results = [
                r for r in search_results if r.score >= self.score_threshold
            ]

        context = self._format_context(search_results)

        return RetrievalResult(
            results=search_results,
            context=context,
            query=query,
        )

    def retrieve_with_answer(
        self,
        query: str,
        answer_fn: callable,
        top_k: int | None = None,
    ) -> dict:
        """Retrieve context and generate an answer using the provided function."""
        retrieval = self.retrieve(query, top_k)
        answer = answer_fn(query, retrieval.context)

        return {
            "query": query,
            "answer": answer,
            "context": retrieval.context,
            "sources": [
                {
                    "text": r.chunk.text[:200],
                    "source": r.chunk.source,
                    "score": r.score,
                }
                for r in retrieval.results
            ],
        }

    def _reciprocal_rank_fusion(
        self,
        dense_results: list[SearchResult],
        sparse_results: list[SearchResult],
    ) -> list[FusionResult]:
        """Merge dense and sparse results using Reciprocal Rank Fusion.

        RRF score = Σ weight / (k + rank)
        where k is a constant (default 60) that controls how much
        lower-ranked results are penalized.
        """
        chunk_scores: dict[int, FusionResult] = {}

        for rank, result in enumerate(dense_results):
            key = id(result.chunk)
            rrf_score = self.dense_weight / (self.rrf_k + rank + 1)
            if key not in chunk_scores:
                chunk_scores[key] = FusionResult(
                    chunk=result.chunk,
                    dense_score=result.score,
                    sparse_score=0.0,
                    fused_score=rrf_score,
                )
            else:
                chunk_scores[key].dense_score = result.score
                chunk_scores[key].fused_score += rrf_score

        for rank, result in enumerate(sparse_results):
            key = self._chunk_identity_key(result.chunk, chunk_scores)
            rrf_score = self.sparse_weight / (self.rrf_k + rank + 1)
            if key not in chunk_scores:
                chunk_scores[key] = FusionResult(
                    chunk=result.chunk,
                    dense_score=0.0,
                    sparse_score=result.score,
                    fused_score=rrf_score,
                )
            else:
                chunk_scores[key].sparse_score = result.score
                chunk_scores[key].fused_score += rrf_score

        fused = sorted(
            chunk_scores.values(), key=lambda f: f.fused_score, reverse=True
        )
        return fused

    @staticmethod
    def _chunk_identity_key(
        chunk: Chunk,
        existing: dict[int, FusionResult],
    ) -> int:
        """Find matching chunk in existing results by content, or use id()."""
        for key, fusion in existing.items():
            if (
                fusion.chunk.source == chunk.source
                and fusion.chunk.chunk_id == chunk.chunk_id
            ):
                return key
        return id(chunk)

    @staticmethod
    def _format_context(results: list[SearchResult]) -> str:
        if not results:
            return ""

        parts: list[str] = []
        for i, result in enumerate(results, 1):
            source_info = (
                f" (source: {result.chunk.source})" if result.chunk.source else ""
            )
            parts.append(f"[Document {i}{source_info}]\n{result.chunk.text}")

        return "\n\n---\n\n".join(parts)
