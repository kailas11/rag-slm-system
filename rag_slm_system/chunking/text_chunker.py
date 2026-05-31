"""Plain text chunker with sentence-boundary awareness."""

import re

from rag_slm_system.chunking.base import BaseChunker, Chunk


class TextChunker(BaseChunker):
    """Recursive text chunker that respects sentence boundaries.

    Splitting hierarchy: paragraphs -> sentences -> words.
    """

    SENTENCE_ENDINGS = re.compile(r"(?<=[.!?])\s+")
    PARAGRAPH_BREAK = re.compile(r"\n\s*\n")

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_size: int = 50,
        respect_sentence_boundaries: bool = True,
    ):
        super().__init__(chunk_size, chunk_overlap, min_chunk_size)
        self.respect_sentence_boundaries = respect_sentence_boundaries

    def chunk(self, text: str, source: str = "") -> list[Chunk]:
        text = text.strip()
        if not text:
            return []

        paragraphs = self.PARAGRAPH_BREAK.split(text)
        raw_chunks: list[Chunk] = []
        chunk_id = 0
        char_offset = 0

        buffer = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                char_offset += 1
                continue

            candidate = (buffer + "\n\n" + para).strip() if buffer else para

            if len(candidate) <= self.chunk_size:
                buffer = candidate
            else:
                if buffer:
                    raw_chunks.append(
                        Chunk(
                            text=buffer,
                            chunk_id=chunk_id,
                            source=source,
                            content_type="text",
                            start_char=char_offset,
                            end_char=char_offset + len(buffer),
                        )
                    )
                    chunk_id += 1
                    char_offset += len(buffer)

                if len(para) <= self.chunk_size:
                    buffer = para
                else:
                    sentence_chunks = self._split_by_sentences(
                        para, source, chunk_id, char_offset
                    )
                    raw_chunks.extend(sentence_chunks)
                    chunk_id += len(sentence_chunks)
                    char_offset += len(para)
                    buffer = ""

        if buffer:
            raw_chunks.append(
                Chunk(
                    text=buffer,
                    chunk_id=chunk_id,
                    source=source,
                    content_type="text",
                    start_char=char_offset,
                    end_char=char_offset + len(buffer),
                )
            )

        chunks = self._merge_small_chunks(raw_chunks)
        chunks = self._apply_overlap(chunks)

        for i, c in enumerate(chunks):
            c.chunk_id = i
        return chunks

    def _split_by_sentences(
        self, text: str, source: str, start_id: int, char_offset: int
    ) -> list[Chunk]:
        if self.respect_sentence_boundaries:
            sentences = self.SENTENCE_ENDINGS.split(text)
        else:
            sentences = [text]

        chunks: list[Chunk] = []
        buffer = ""
        chunk_id = start_id
        local_offset = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            candidate = (buffer + " " + sentence).strip() if buffer else sentence

            if len(candidate) <= self.chunk_size:
                buffer = candidate
            else:
                if buffer:
                    chunks.append(
                        Chunk(
                            text=buffer,
                            chunk_id=chunk_id,
                            source=source,
                            content_type="text",
                            start_char=char_offset + local_offset,
                            end_char=char_offset + local_offset + len(buffer),
                        )
                    )
                    chunk_id += 1
                    local_offset += len(buffer)

                if len(sentence) > self.chunk_size:
                    word_chunks = self._split_by_words(
                        sentence, source, chunk_id, char_offset + local_offset
                    )
                    chunks.extend(word_chunks)
                    chunk_id += len(word_chunks)
                    local_offset += len(sentence)
                    buffer = ""
                else:
                    buffer = sentence

        if buffer:
            chunks.append(
                Chunk(
                    text=buffer,
                    chunk_id=chunk_id,
                    source=source,
                    content_type="text",
                    start_char=char_offset + local_offset,
                    end_char=char_offset + local_offset + len(buffer),
                )
            )

        return chunks

    def _split_by_words(
        self, text: str, source: str, start_id: int, char_offset: int
    ) -> list[Chunk]:
        words = text.split()
        chunks: list[Chunk] = []
        buffer: list[str] = []
        chunk_id = start_id
        current_len = 0

        for word in words:
            new_len = current_len + len(word) + (1 if buffer else 0)
            if new_len <= self.chunk_size:
                buffer.append(word)
                current_len = new_len
            else:
                if buffer:
                    chunk_text = " ".join(buffer)
                    chunks.append(
                        Chunk(
                            text=chunk_text,
                            chunk_id=chunk_id,
                            source=source,
                            content_type="text",
                            start_char=char_offset,
                            end_char=char_offset + len(chunk_text),
                        )
                    )
                    chunk_id += 1
                    char_offset += len(chunk_text) + 1
                buffer = [word]
                current_len = len(word)

        if buffer:
            chunk_text = " ".join(buffer)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    source=source,
                    content_type="text",
                    start_char=char_offset,
                    end_char=char_offset + len(chunk_text),
                )
            )

        return chunks
