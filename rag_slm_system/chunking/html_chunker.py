"""HTML document chunker using BeautifulSoup."""

import re

from rag_slm_system.chunking.base import BaseChunker, Chunk
from rag_slm_system.chunking.text_chunker import TextChunker


class HTMLChunker(BaseChunker):
    """Chunks HTML documents by semantic elements.

    Strips tags, splits on structural HTML elements (headings, divs, sections,
    articles), preserves element context as metadata.
    """

    STRUCTURAL_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "div", "p"}

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
        try:
            from bs4 import BeautifulSoup
        except ImportError as e:
            raise ImportError(
                "beautifulsoup4 is required for HTML chunking: pip install beautifulsoup4"
            ) from e

        soup = BeautifulSoup(text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        sections = self._extract_sections(soup)

        chunks: list[Chunk] = []
        chunk_id = 0

        for section in sections:
            section_text = self._clean_text(section["text"])
            if not section_text:
                continue

            if len(section_text) <= self.chunk_size:
                chunks.append(
                    Chunk(
                        text=section_text,
                        chunk_id=chunk_id,
                        source=source,
                        content_type="html",
                        metadata={
                            "tag": section.get("tag", ""),
                            "heading": section.get("heading", ""),
                        },
                    )
                )
                chunk_id += 1
            else:
                sub_chunks = self._text_chunker.chunk(section_text, source)
                for sc in sub_chunks:
                    sc.chunk_id = chunk_id
                    sc.content_type = "html"
                    sc.metadata["tag"] = section.get("tag", "")
                    sc.metadata["heading"] = section.get("heading", "")
                    chunks.append(sc)
                    chunk_id += 1

        chunks = self._merge_small_chunks(chunks)
        chunks = self._apply_overlap(chunks)
        for i, c in enumerate(chunks):
            c.chunk_id = i
        return chunks

    def _extract_sections(self, soup) -> list[dict]:
        sections: list[dict] = []
        current_heading = ""

        for element in soup.find_all(self.STRUCTURAL_TAGS):
            tag_name = element.name
            text = element.get_text(separator="\n", strip=True)

            if tag_name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                current_heading = text
                continue

            if text:
                sections.append(
                    {
                        "text": text,
                        "tag": tag_name,
                        "heading": current_heading,
                    }
                )

        if not sections:
            full_text = soup.get_text(separator="\n", strip=True)
            if full_text:
                sections.append({"text": full_text, "tag": "body", "heading": ""})

        return sections

    @staticmethod
    def _clean_text(text: str) -> str:
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        return text.strip()
