"""Code-aware chunker that splits on function/class boundaries."""

import re

from rag_slm_system.chunking.base import BaseChunker, Chunk


class CodeChunker(BaseChunker):
    """Chunks source code by structural boundaries (functions, classes, blocks).

    Supports Python, JavaScript/TypeScript, Java, C/C++, Go, Rust.
    Falls back to line-based splitting for unrecognized languages.
    """

    LANGUAGE_PATTERNS: dict[str, dict[str, re.Pattern]] = {  # type: ignore[type-arg]
        "python": {
            "function": re.compile(
                r"^([ \t]*(?:async\s+)?def\s+\w+\s*\(.*?\).*?:)", re.MULTILINE
            ),
            "class": re.compile(r"^([ \t]*class\s+\w+.*?:)", re.MULTILINE),
            "decorator": re.compile(r"^([ \t]*@\w+.*?)$", re.MULTILINE),
        },
        "javascript": {
            "function": re.compile(
                r"^([ \t]*(?:export\s+)?(?:async\s+)?function\s+\w+\s*\()", re.MULTILINE
            ),
            "arrow": re.compile(
                r"^([ \t]*(?:export\s+)?(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?\()",
                re.MULTILINE,
            ),
            "class": re.compile(
                r"^([ \t]*(?:export\s+)?class\s+\w+)", re.MULTILINE
            ),
        },
        "java": {
            "method": re.compile(
                r"^([ \t]*(?:public|private|protected|static|\s)*[\w<>\[\]]+\s+\w+\s*\()",
                re.MULTILINE,
            ),
            "class": re.compile(
                r"^([ \t]*(?:public|private|protected|abstract|static|\s)*class\s+\w+)",
                re.MULTILINE,
            ),
        },
    }

    EXTENSION_TO_LANG: dict[str, str] = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "javascript",
        ".tsx": "javascript",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".c": "c",
        ".cpp": "c",
        ".h": "c",
        ".hpp": "c",
    }

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_size: int = 50,
        max_function_chunk_size: int = 1024,
        language: str | None = None,
    ):
        super().__init__(chunk_size, chunk_overlap, min_chunk_size)
        self.max_function_chunk_size = max_function_chunk_size
        self.language = language

    def chunk(self, text: str, source: str = "") -> list[Chunk]:
        text = text.strip()
        if not text:
            return []

        lang = self.language or self._detect_language(source)
        blocks = self._split_into_blocks(text, lang)

        chunks: list[Chunk] = []
        chunk_id = 0

        for block in blocks:
            block_text = block["text"].strip()
            if not block_text:
                continue

            if len(block_text) <= self.max_function_chunk_size:
                chunks.append(
                    Chunk(
                        text=block_text,
                        chunk_id=chunk_id,
                        source=source,
                        content_type="code",
                        metadata={
                            "language": lang,
                            "block_type": block.get("type", "unknown"),
                            "name": block.get("name", ""),
                        },
                    )
                )
                chunk_id += 1
            else:
                line_chunks = self._split_by_lines(block_text, source, chunk_id, lang)
                for lc in line_chunks:
                    lc.metadata["block_type"] = block.get("type", "unknown")
                    lc.metadata["name"] = block.get("name", "")
                chunks.extend(line_chunks)
                chunk_id += len(line_chunks)

        chunks = self._merge_small_chunks(chunks)
        for i, c in enumerate(chunks):
            c.chunk_id = i
        return chunks

    def _detect_language(self, source: str) -> str:
        for ext, lang in self.EXTENSION_TO_LANG.items():
            if source.endswith(ext):
                return lang
        return "unknown"

    def _split_into_blocks(self, text: str, lang: str) -> list[dict]:
        patterns = self.LANGUAGE_PATTERNS.get(lang)
        if not patterns:
            return self._split_by_blank_lines(text)

        boundary_positions: list[int] = []
        for pattern in patterns.values():
            for m in pattern.finditer(text):
                boundary_positions.append(m.start())

        if not boundary_positions:
            return self._split_by_blank_lines(text)

        boundary_positions = sorted(set(boundary_positions))

        blocks: list[dict] = []

        if boundary_positions[0] > 0:
            preamble = text[: boundary_positions[0]].strip()
            if preamble:
                blocks.append({"text": preamble, "type": "preamble", "name": ""})

        for i, pos in enumerate(boundary_positions):
            end = boundary_positions[i + 1] if i + 1 < len(boundary_positions) else len(text)
            block_text = text[pos:end]

            block_type = "block"
            name = ""
            name_match = re.search(r"(?:def|function|class|const|let|var)\s+(\w+)", block_text)
            if name_match:
                name = name_match.group(1)
                block_type = "class" if "class " in block_text[:50] else "function"

            blocks.append({"text": block_text, "type": block_type, "name": name})

        return blocks

    def _split_by_blank_lines(self, text: str) -> list[dict]:
        segments = re.split(r"\n\s*\n", text)
        return [
            {"text": seg.strip(), "type": "block", "name": ""}
            for seg in segments
            if seg.strip()
        ]

    def _split_by_lines(
        self, text: str, source: str, start_id: int, lang: str
    ) -> list[Chunk]:
        lines = text.split("\n")
        chunks: list[Chunk] = []
        buffer: list[str] = []
        current_len = 0
        chunk_id = start_id

        for line in lines:
            line_len = len(line) + 1
            if current_len + line_len > self.chunk_size and buffer:
                chunks.append(
                    Chunk(
                        text="\n".join(buffer),
                        chunk_id=chunk_id,
                        source=source,
                        content_type="code",
                        metadata={"language": lang},
                    )
                )
                chunk_id += 1
                buffer = []
                current_len = 0

            buffer.append(line)
            current_len += line_len

        if buffer:
            chunks.append(
                Chunk(
                    text="\n".join(buffer),
                    chunk_id=chunk_id,
                    source=source,
                    content_type="code",
                    metadata={"language": lang},
                )
            )

        return chunks
