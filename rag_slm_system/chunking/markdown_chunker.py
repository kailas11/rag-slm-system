"""Markdown-aware chunker that splits on headings and structural elements."""

import re

from rag_slm_system.chunking.base import BaseChunker, Chunk


class MarkdownChunker(BaseChunker):
    """Chunks Markdown documents by heading hierarchy.

    Preserves heading context as metadata so each chunk knows its section path.
    Falls back to paragraph/sentence splitting within sections.
    """

    HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_size: int = 50,
        split_on_heading_level: int = 2,
    ):
        super().__init__(chunk_size, chunk_overlap, min_chunk_size)
        self.split_on_heading_level = split_on_heading_level

    def chunk(self, text: str, source: str = "") -> list[Chunk]:
        text = text.strip()
        if not text:
            return []

        sections = self._split_by_headings(text)
        chunks: list[Chunk] = []
        chunk_id = 0

        for section in sections:
            heading = section.get("heading", "")
            heading_level = section.get("level", 0)
            heading_path = section.get("path", [])
            body = section.get("body", "").strip()

            if not body and not heading:
                continue

            full_text = f"{heading}\n\n{body}".strip() if heading else body

            if len(full_text) <= self.chunk_size:
                chunks.append(
                    Chunk(
                        text=full_text,
                        chunk_id=chunk_id,
                        source=source,
                        content_type="markdown",
                        metadata={
                            "heading": heading,
                            "heading_level": heading_level,
                            "heading_path": heading_path,
                        },
                    )
                )
                chunk_id += 1
            else:
                sub_chunks = self._split_section(full_text, source, chunk_id)
                for sc in sub_chunks:
                    sc.metadata.update(
                        {
                            "heading": heading,
                            "heading_level": heading_level,
                            "heading_path": heading_path,
                        }
                    )
                chunks.extend(sub_chunks)
                chunk_id += len(sub_chunks)

        chunks = self._merge_small_chunks(chunks)
        chunks = self._apply_overlap(chunks)
        for i, c in enumerate(chunks):
            c.chunk_id = i
        return chunks

    def _split_by_headings(self, text: str) -> list[dict]:
        code_blocks: list[tuple[int, int]] = []
        for m in self.CODE_BLOCK_RE.finditer(text):
            code_blocks.append((m.start(), m.end()))

        headings: list[tuple[int, int, str, int]] = []
        for m in self.HEADING_RE.finditer(text):
            in_code = any(s <= m.start() < e for s, e in code_blocks)
            if not in_code:
                level = len(m.group(1))
                if level <= self.split_on_heading_level:
                    headings.append((m.start(), m.end(), m.group(0), level))

        if not headings:
            return [{"heading": "", "level": 0, "path": [], "body": text}]

        sections: list[dict] = []
        heading_stack: list[str] = []

        if headings[0][0] > 0:
            preamble = text[: headings[0][0]].strip()
            if preamble:
                sections.append({"heading": "", "level": 0, "path": [], "body": preamble})

        for i, (start, end, heading_text, level) in enumerate(headings):
            while len(heading_stack) >= level:
                heading_stack.pop()
            heading_stack.append(heading_text)

            body_start = end
            body_end = headings[i + 1][0] if i + 1 < len(headings) else len(text)
            body = text[body_start:body_end].strip()

            sections.append(
                {
                    "heading": heading_text,
                    "level": level,
                    "path": list(heading_stack),
                    "body": body,
                }
            )

        return sections

    def _split_section(self, text: str, source: str, start_id: int) -> list[Chunk]:
        paragraphs = re.split(r"\n\s*\n", text)
        chunks: list[Chunk] = []
        buffer = ""
        chunk_id = start_id

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            candidate = (buffer + "\n\n" + para).strip() if buffer else para

            if len(candidate) <= self.chunk_size:
                buffer = candidate
            else:
                if buffer:
                    chunks.append(
                        Chunk(
                            text=buffer,
                            chunk_id=chunk_id,
                            source=source,
                            content_type="markdown",
                        )
                    )
                    chunk_id += 1
                buffer = para if len(para) <= self.chunk_size else ""
                if len(para) > self.chunk_size:
                    words = para.split()
                    word_buf: list[str] = []
                    for w in words:
                        test = " ".join(word_buf + [w])
                        if len(test) <= self.chunk_size:
                            word_buf.append(w)
                        else:
                            if word_buf:
                                chunks.append(
                                    Chunk(
                                        text=" ".join(word_buf),
                                        chunk_id=chunk_id,
                                        source=source,
                                        content_type="markdown",
                                    )
                                )
                                chunk_id += 1
                            word_buf = [w]
                    if word_buf:
                        buffer = " ".join(word_buf)

        if buffer:
            chunks.append(
                Chunk(
                    text=buffer,
                    chunk_id=chunk_id,
                    source=source,
                    content_type="markdown",
                )
            )

        return chunks
