"""Base classes for document chunking."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Chunk:
    """A chunk of text with metadata."""

    text: str
    chunk_id: int
    source: str = ""
    content_type: str = "text"
    metadata: dict = field(default_factory=dict)
    start_char: Optional[int] = None
    end_char: Optional[int] = None

    @property
    def token_estimate(self) -> int:
        return len(self.text.split())


class BaseChunker(ABC):
    """Abstract base class for content-aware chunkers."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64, min_chunk_size: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    @abstractmethod
    def chunk(self, text: str, source: str = "") -> list[Chunk]:
        """Split text into chunks."""

    def _merge_small_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Merge chunks smaller than min_chunk_size with adjacent chunks."""
        if not chunks:
            return chunks

        merged: list[Chunk] = []
        buffer = chunks[0]

        for chunk in chunks[1:]:
            if len(buffer.text) < self.min_chunk_size:
                buffer = Chunk(
                    text=buffer.text + "\n\n" + chunk.text,
                    chunk_id=buffer.chunk_id,
                    source=buffer.source,
                    content_type=buffer.content_type,
                    metadata={**buffer.metadata, **chunk.metadata},
                    start_char=buffer.start_char,
                    end_char=chunk.end_char,
                )
            else:
                merged.append(buffer)
                buffer = chunk

        merged.append(buffer)
        return merged

    def _apply_overlap(self, chunks: list[Chunk]) -> list[Chunk]:
        """Add overlap between consecutive chunks for context continuity."""
        if self.chunk_overlap <= 0 or len(chunks) <= 1:
            return chunks

        result: list[Chunk] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_text = chunks[i - 1].text
            overlap_text = prev_text[-self.chunk_overlap :]

            separator_idx = overlap_text.find(" ")
            if separator_idx > 0:
                overlap_text = overlap_text[separator_idx + 1 :]

            chunk = Chunk(
                text=overlap_text + " " + chunks[i].text,
                chunk_id=chunks[i].chunk_id,
                source=chunks[i].source,
                content_type=chunks[i].content_type,
                metadata=chunks[i].metadata,
                start_char=chunks[i].start_char,
                end_char=chunks[i].end_char,
            )
            result.append(chunk)

        return result
