"""BM25-based sparse retriever for keyword matching."""

import re

from rag_slm_system.chunking.base import Chunk
from rag_slm_system.vectorstore.base import SearchResult


class BM25Retriever:
    """Sparse retriever using BM25 (Okapi BM25) for keyword-based matching.

    Complements dense (embedding) retrieval by capturing exact keyword matches
    that dense models may miss.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._bm25 = None
        self._chunks: list[Chunk] = []
        self._tokenized_corpus: list[list[str]] = []

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer with lowercasing."""
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        return tokens

    def add(self, chunks: list[Chunk]) -> None:
        """Index chunks for BM25 search."""
        self._chunks.extend(chunks)
        self._tokenized_corpus.extend(
            self._tokenize(c.text) for c in chunks
        )
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Rebuild the BM25 index from the current corpus."""
        from rank_bm25 import BM25Okapi

        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(
                self._tokenized_corpus, k1=self.k1, b=self.b
            )

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Search for chunks matching the query using BM25 scoring."""
        if not self._bm25 or not self._chunks:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        top_indices = scores.argsort()[::-1][:top_k]

        results: list[SearchResult] = []
        for rank, idx in enumerate(top_indices):
            score = float(scores[idx])
            if score <= 0:
                continue
            results.append(
                SearchResult(
                    chunk=self._chunks[idx],
                    score=score,
                    rank=rank,
                )
            )
        return results

    @property
    def size(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:
        """Reset the index."""
        self._bm25 = None
        self._chunks = []
        self._tokenized_corpus = []
