"""Factory for auto-detecting content type and returning the appropriate chunker."""

import mimetypes
import re
from pathlib import Path

from rag_slm_system.chunking.base import BaseChunker
from rag_slm_system.chunking.code_chunker import CodeChunker
from rag_slm_system.chunking.html_chunker import HTMLChunker
from rag_slm_system.chunking.markdown_chunker import MarkdownChunker
from rag_slm_system.chunking.pdf_chunker import PDFChunker
from rag_slm_system.chunking.text_chunker import TextChunker
from rag_slm_system.config import ChunkingConfig

CODE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs",
    ".c", ".cpp", ".h", ".hpp", ".rb", ".php", ".swift", ".kt",
    ".scala", ".sh", ".bash", ".zsh", ".ps1", ".r", ".m", ".lua",
}


def detect_content_type(source: str, text: str = "") -> str:
    """Detect content type from file extension, MIME type, or content heuristics.

    Returns one of: "pdf", "markdown", "html", "code", "text".
    """
    if source:
        path = Path(source)
        ext = path.suffix.lower()

        if ext == ".pdf":
            return "pdf"
        if ext in {".md", ".markdown", ".mdx"}:
            return "markdown"
        if ext in {".html", ".htm", ".xhtml"}:
            return "html"
        if ext in CODE_EXTENSIONS:
            return "code"

        mime_type, _ = mimetypes.guess_type(source)
        if mime_type:
            if "pdf" in mime_type:
                return "pdf"
            if "html" in mime_type:
                return "html"
            if "markdown" in mime_type:
                return "markdown"

    if text:
        if text.strip().startswith(("<!DOCTYPE", "<html", "<HTML")):
            return "html"
        if re.search(r"^#{1,6}\s+", text, re.MULTILINE):
            return "markdown"
        code_pat = r"^(def |class |function |import |from |const |let |var )"
        if re.search(code_pat, text, re.MULTILINE):
            return "code"

    return "text"


class ChunkerFactory:
    """Factory that returns the appropriate chunker based on content type."""

    def __init__(self, config: ChunkingConfig | None = None):
        self.config = config or ChunkingConfig()

    def get_chunker(self, content_type: str) -> BaseChunker:
        """Get chunker for the given content type."""
        base_kwargs = {
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "min_chunk_size": self.config.min_chunk_size,
        }

        if content_type == "pdf":
            return PDFChunker(**base_kwargs)
        elif content_type == "markdown":
            return MarkdownChunker(
                **base_kwargs,
                split_on_heading_level=self.config.split_on_heading_level,
            )
        elif content_type == "code":
            return CodeChunker(
                **base_kwargs,
                max_function_chunk_size=self.config.max_function_chunk_size,
            )
        elif content_type == "html":
            return HTMLChunker(**base_kwargs)
        else:
            return TextChunker(
                **base_kwargs,
                respect_sentence_boundaries=self.config.respect_sentence_boundaries,
            )

    def auto_chunk(self, text: str, source: str = "") -> list:
        """Auto-detect content type and chunk accordingly."""
        content_type = detect_content_type(source, text)
        chunker = self.get_chunker(content_type)
        return chunker.chunk(text, source)
