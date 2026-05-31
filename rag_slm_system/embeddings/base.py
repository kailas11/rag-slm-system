"""Base embedding interface."""

from abc import ABC, abstractmethod

import numpy as np


class BaseEmbedder(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts into vectors.

        Args:
            texts: List of text strings to embed.

        Returns:
            numpy array of shape (len(texts), embedding_dim).
        """

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query text.

        Args:
            query: Query text to embed.

        Returns:
            numpy array of shape (embedding_dim,).
        """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
