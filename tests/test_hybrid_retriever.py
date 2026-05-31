"""Tests for hybrid retrieval, BM25, and reranking."""

import numpy as np

from rag_slm_system.chunking.base import Chunk
from rag_slm_system.retriever.bm25_retriever import BM25Retriever
from rag_slm_system.retriever.hybrid_retriever import HybridRetriever
from rag_slm_system.retriever.reranker import BaseReranker
from rag_slm_system.vectorstore.base import SearchResult


def _make_chunks(texts: list[str], source: str = "test.txt") -> list[Chunk]:
    return [
        Chunk(text=t, chunk_id=i, source=source, content_type="text")
        for i, t in enumerate(texts)
    ]


class FakeEmbedder:
    """Deterministic embedder for testing."""

    def __init__(self, dim: int = 8):
        self._dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        rng = np.random.RandomState(42)
        vecs = rng.randn(len(texts), self._dim).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1, norms)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]

    @property
    def dimension(self) -> int:
        return self._dim


class FakeReranker(BaseReranker):
    """Reranker that reverses the input order for testing."""

    def rerank(
        self, query: str, results: list[SearchResult], top_k: int = 5
    ) -> list[SearchResult]:
        reversed_results = list(reversed(results))[:top_k]
        return [
            SearchResult(chunk=r.chunk, score=1.0 / (i + 1), rank=i)
            for i, r in enumerate(reversed_results)
        ]


class TestBM25Retriever:
    def test_basic_search(self):
        bm25 = BM25Retriever()
        chunks = _make_chunks([
            "Python is a programming language",
            "Java is used for enterprise applications",
            "Python supports machine learning libraries",
        ])
        bm25.add(chunks)

        results = bm25.search("Python programming", top_k=2)
        assert len(results) > 0
        assert any("Python" in r.chunk.text for r in results)

    def test_empty_search(self):
        bm25 = BM25Retriever()
        results = bm25.search("anything", top_k=5)
        assert results == []

    def test_size(self):
        bm25 = BM25Retriever()
        assert bm25.size == 0
        chunks = _make_chunks(["hello world", "foo bar"])
        bm25.add(chunks)
        assert bm25.size == 2

    def test_clear(self):
        bm25 = BM25Retriever()
        bm25.add(_make_chunks(["some text"]))
        assert bm25.size == 1
        bm25.clear()
        assert bm25.size == 0

    def test_no_match_returns_empty(self):
        bm25 = BM25Retriever()
        bm25.add(_make_chunks(["apple banana cherry"]))
        results = bm25.search("xylophone", top_k=5)
        assert results == []

    def test_scores_are_positive(self):
        bm25 = BM25Retriever()
        bm25.add(_make_chunks([
            "machine learning with neural networks",
            "deep learning and transformers",
        ]))
        results = bm25.search("machine learning", top_k=2)
        for r in results:
            assert r.score > 0


class TestHybridRetriever:
    def _build_retriever(self, texts, reranker=None):
        embedder = FakeEmbedder(dim=8)
        from rag_slm_system.vectorstore.faiss_store import FAISSVectorStore

        store = FAISSVectorStore(dimension=8, similarity_metric="cosine")
        chunks = _make_chunks(texts)
        embeddings = embedder.embed([c.text for c in chunks])
        store.add(chunks, embeddings)

        retriever = HybridRetriever(
            embedder=embedder,
            vector_store=store,
            reranker=reranker,
            top_k=3,
            fusion_top_k=10,
        )
        retriever.add_chunks(chunks)
        return retriever

    def test_hybrid_returns_results(self):
        texts = [
            "Python is a dynamic programming language",
            "Java is a statically typed language",
            "Machine learning uses Python extensively",
            "Databases store structured data",
            "REST APIs enable web communication",
        ]
        retriever = self._build_retriever(texts)
        result = retriever.retrieve("Python programming", top_k=3)

        assert len(result.results) > 0
        assert result.query == "Python programming"
        assert result.context != ""

    def test_hybrid_with_reranker(self):
        texts = [
            "Transformers are neural network architectures",
            "BERT is a transformer model",
            "Cooking pasta requires boiling water",
        ]
        reranker = FakeReranker()
        retriever = self._build_retriever(texts, reranker=reranker)
        result = retriever.retrieve("transformer models", top_k=2)

        assert len(result.results) <= 2
        assert all(r.rank >= 0 for r in result.results)

    def test_hybrid_without_reranker(self):
        texts = [
            "Deep learning with PyTorch",
            "Natural language processing basics",
        ]
        retriever = self._build_retriever(texts, reranker=None)
        result = retriever.retrieve("deep learning", top_k=2)
        assert len(result.results) > 0

    def test_retrieve_with_answer(self):
        texts = ["RAG combines retrieval and generation"]
        retriever = self._build_retriever(texts)

        def answer_fn(q, ctx):
            return f"Answer: {ctx[:50]}"

        result = retriever.retrieve_with_answer("what is RAG", answer_fn)
        assert "query" in result
        assert "answer" in result
        assert "context" in result
        assert "sources" in result

    def test_score_threshold(self):
        texts = ["alpha beta gamma", "delta epsilon zeta"]
        embedder = FakeEmbedder(dim=8)
        from rag_slm_system.vectorstore.faiss_store import FAISSVectorStore

        store = FAISSVectorStore(dimension=8, similarity_metric="cosine")
        chunks = _make_chunks(texts)
        embeddings = embedder.embed([c.text for c in chunks])
        store.add(chunks, embeddings)

        retriever = HybridRetriever(
            embedder=embedder,
            vector_store=store,
            score_threshold=999.0,
            top_k=5,
        )
        retriever.add_chunks(chunks)

        result = retriever.retrieve("alpha")
        assert len(result.results) == 0


class TestFakeReranker:
    def test_reranker_reverses(self):
        reranker = FakeReranker()
        chunks = _make_chunks(["a", "b", "c"])
        results = [
            SearchResult(chunk=c, score=float(i), rank=i)
            for i, c in enumerate(chunks)
        ]
        reranked = reranker.rerank("query", results, top_k=3)
        assert reranked[0].chunk.text == "c"
        assert reranked[1].chunk.text == "b"
        assert reranked[2].chunk.text == "a"

    def test_reranker_top_k(self):
        reranker = FakeReranker()
        chunks = _make_chunks(["a", "b", "c", "d"])
        results = [
            SearchResult(chunk=c, score=float(i), rank=i)
            for i, c in enumerate(chunks)
        ]
        reranked = reranker.rerank("query", results, top_k=2)
        assert len(reranked) == 2

    def test_reranker_empty(self):
        reranker = FakeReranker()
        assert reranker.rerank("query", [], top_k=5) == []
