"""PDF document chunker using pdfplumber."""

from pathlib import Path

from rag_slm_system.chunking.base import BaseChunker, Chunk
from rag_slm_system.chunking.text_chunker import TextChunker


class PDFChunker(BaseChunker):
    """Chunks PDF documents with page-aware splitting.

    Extracts text per page, preserves page numbers as metadata,
    then delegates to TextChunker for within-page splitting.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_size: int = 50,
    ):
        super().__init__(chunk_size, chunk_overlap, min_chunk_size)
        self._text_chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=0,
            min_chunk_size=min_chunk_size,
        )

    def chunk(self, text: str, source: str = "") -> list[Chunk]:
        """Chunk from pre-extracted text. Use chunk_file() for PDFs."""
        chunks = self._text_chunker.chunk(text, source)
        for c in chunks:
            c.content_type = "pdf"
        return chunks

    def chunk_file(self, file_path: str | Path) -> list[Chunk]:
        """Extract text from a PDF file and chunk it."""
        try:
            import pdfplumber
        except ImportError as e:
            raise ImportError(
                "pdfplumber is required for PDF chunking: pip install pdfplumber"
            ) from e

        file_path = Path(file_path)
        chunks: list[Chunk] = []
        chunk_id = 0

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                page_text = page_text.strip()
                if not page_text:
                    continue

                page_chunks = self._text_chunker.chunk(page_text, source=str(file_path))

                for pc in page_chunks:
                    pc.chunk_id = chunk_id
                    pc.content_type = "pdf"
                    pc.metadata["page_number"] = page_num
                    pc.metadata["total_pages"] = len(pdf.pages)
                    chunks.append(pc)
                    chunk_id += 1

        chunks = self._merge_small_chunks(chunks)
        chunks = self._apply_overlap(chunks)
        for i, c in enumerate(chunks):
            c.chunk_id = i
        return chunks
