"""Base vector store interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from rag_slm_system.chunking.base import Chunk


@dataclass
class SearchResult:
    """A search result from the vector store."""

    chunk: Chunk
    score: float
    rank: int


class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Add chunks and their embeddings to the store."""

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[SearchResult]:
        """Search for similar chunks."""

    @abstractmethod
    def save(self, directory: str) -> None:
        """Persist the vector store to disk."""

    @abstractmethod
    def load(self, directory: str) -> None:
        """Load the vector store from disk."""

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of vectors in the store."""
