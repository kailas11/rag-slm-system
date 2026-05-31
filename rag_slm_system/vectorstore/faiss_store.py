"""FAISS-based vector store."""

import json
from pathlib import Path

import numpy as np

from rag_slm_system.chunking.base import Chunk
from rag_slm_system.vectorstore.base import BaseVectorStore, SearchResult


class FAISSVectorStore(BaseVectorStore):
    """Vector store backed by Facebook AI Similarity Search (FAISS).

    Supports cosine similarity (via normalized L2), L2 distance, and inner product.
    """

    def __init__(self, dimension: int, similarity_metric: str = "cosine"):
        try:
            import faiss
        except ImportError as e:
            raise ImportError("faiss-cpu is required: pip install faiss-cpu") from e

        self._faiss = faiss
        self.dimension = dimension
        self.similarity_metric = similarity_metric

        if similarity_metric == "cosine":
            self.index = faiss.IndexFlatIP(dimension)
        elif similarity_metric == "l2":
            self.index = faiss.IndexFlatL2(dimension)
        elif similarity_metric == "ip":
            self.index = faiss.IndexFlatIP(dimension)
        else:
            raise ValueError(f"Unknown metric: {similarity_metric}. Use 'cosine', 'l2', or 'ip'.")

        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
            )

        embeddings = embeddings.astype(np.float32)

        if self.similarity_metric == "cosine":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            embeddings = embeddings / norms

        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[SearchResult]:
        if self.size == 0:
            return []

        query = query_embedding.astype(np.float32).reshape(1, -1)

        if self.similarity_metric == "cosine":
            norm = np.linalg.norm(query)
            if norm > 0:
                query = query / norm

        top_k = min(top_k, self.size)
        scores, indices = self.index.search(query, top_k)

        results: list[SearchResult] = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < 0:
                continue
            results.append(
                SearchResult(
                    chunk=self.chunks[idx],
                    score=float(score),
                    rank=rank,
                )
            )

        return results

    def save(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        self._faiss.write_index(self.index, str(path / "index.faiss"))

        chunks_data = []
        for chunk in self.chunks:
            chunks_data.append(
                {
                    "text": chunk.text,
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "content_type": chunk.content_type,
                    "metadata": chunk.metadata,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                }
            )

        with open(path / "chunks.json", "w") as f:
            json.dump(
                {
                    "dimension": self.dimension,
                    "similarity_metric": self.similarity_metric,
                    "chunks": chunks_data,
                },
                f,
                indent=2,
            )

    def load(self, directory: str) -> None:
        path = Path(directory)

        if not (path / "index.faiss").exists():
            raise FileNotFoundError(f"No FAISS index found at {path / 'index.faiss'}")

        self.index = self._faiss.read_index(str(path / "index.faiss"))

        with open(path / "chunks.json") as f:
            data = json.load(f)

        self.dimension = data["dimension"]
        self.similarity_metric = data["similarity_metric"]
        self.chunks = [
            Chunk(
                text=c["text"],
                chunk_id=c["chunk_id"],
                source=c["source"],
                content_type=c["content_type"],
                metadata=c["metadata"],
                start_char=c.get("start_char"),
                end_char=c.get("end_char"),
            )
            for c in data["chunks"]
        ]

    @property
    def size(self) -> int:
        return self.index.ntotal

    @classmethod
    def load_from(cls, directory: str) -> "FAISSVectorStore":
        """Load a vector store from disk."""
        path = Path(directory)
        with open(path / "chunks.json") as f:
            data = json.load(f)

        store = cls(
            dimension=data["dimension"],
            similarity_metric=data["similarity_metric"],
        )
        store.load(directory)
        return store
