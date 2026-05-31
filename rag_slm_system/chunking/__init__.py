"""Content-aware document chunking module."""

from rag_slm_system.chunking.base import BaseChunker, Chunk
from rag_slm_system.chunking.code_chunker import CodeChunker
from rag_slm_system.chunking.factory import ChunkerFactory, detect_content_type
from rag_slm_system.chunking.html_chunker import HTMLChunker
from rag_slm_system.chunking.markdown_chunker import MarkdownChunker
from rag_slm_system.chunking.pdf_chunker import PDFChunker
from rag_slm_system.chunking.text_chunker import TextChunker

__all__ = [
    "BaseChunker",
    "Chunk",
    "ChunkerFactory",
    "detect_content_type",
    "TextChunker",
    "MarkdownChunker",
    "CodeChunker",
    "PDFChunker",
    "HTMLChunker",
]
