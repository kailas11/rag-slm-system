"""Tests for the content-aware chunking module."""


from rag_slm_system.chunking.code_chunker import CodeChunker
from rag_slm_system.chunking.factory import ChunkerFactory, detect_content_type
from rag_slm_system.chunking.html_chunker import HTMLChunker
from rag_slm_system.chunking.markdown_chunker import MarkdownChunker
from rag_slm_system.chunking.text_chunker import TextChunker


class TestTextChunker:
    def test_basic_chunking(self):
        chunker = TextChunker(chunk_size=100, chunk_overlap=0)
        text = "This is a test sentence. " * 20
        chunks = chunker.chunk(text, source="test.txt")
        assert len(chunks) > 0
        assert all(c.content_type == "text" for c in chunks)

    def test_empty_text(self):
        chunker = TextChunker()
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []

    def test_short_text_single_chunk(self):
        chunker = TextChunker(chunk_size=500)
        text = "Short text."
        chunks = chunker.chunk(text)
        assert len(chunks) == 1
        assert chunks[0].text == "Short text."

    def test_sentence_boundary_respect(self):
        chunker = TextChunker(chunk_size=50, chunk_overlap=0)
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1

    def test_chunk_ids_sequential(self):
        chunker = TextChunker(chunk_size=100, chunk_overlap=0)
        text = "Hello world. " * 50
        chunks = chunker.chunk(text)
        for i, c in enumerate(chunks):
            assert c.chunk_id == i


class TestMarkdownChunker:
    def test_heading_split(self):
        chunker = MarkdownChunker(chunk_size=50, chunk_overlap=0, min_chunk_size=10)
        text = (
            "# Title\n\nIntro text here.\n\n"
            "## Section 1\n\nContent for section one.\n\n"
            "## Section 2\n\nContent for section two."
        )
        chunks = chunker.chunk(text, source="doc.md")
        assert len(chunks) >= 2
        assert all(c.content_type == "markdown" for c in chunks)

    def test_preserves_heading_metadata(self):
        chunker = MarkdownChunker(chunk_size=500, chunk_overlap=0)
        text = "# Main\n\nIntro.\n\n## Sub\n\nContent."
        chunks = chunker.chunk(text)
        has_heading = any("heading" in c.metadata for c in chunks)
        assert has_heading

    def test_code_blocks_not_split_on_headings(self):
        chunker = MarkdownChunker(chunk_size=1000, chunk_overlap=0)
        text = "# Title\n\n```\n## Not a heading\ncode here\n```\n\nRegular text."
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1


class TestCodeChunker:
    def test_python_function_split(self):
        chunker = CodeChunker(chunk_size=500, chunk_overlap=0, language="python")
        text = (
            "import os\n\n"
            "def func_a():\n    return 1\n\n"
            "def func_b():\n    return 2\n\n"
            "class MyClass:\n    pass\n"
        )
        chunks = chunker.chunk(text, source="test.py")
        assert len(chunks) >= 2
        assert all(c.content_type == "code" for c in chunks)

    def test_language_detection(self):
        chunker = CodeChunker()
        assert chunker._detect_language("file.py") == "python"
        assert chunker._detect_language("file.js") == "javascript"
        assert chunker._detect_language("file.java") == "java"
        assert chunker._detect_language("file.xyz") == "unknown"


class TestHTMLChunker:
    def test_basic_html(self):
        chunker = HTMLChunker(chunk_size=500, chunk_overlap=0)
        text = "<html><body><h1>Title</h1><p>Paragraph one.</p><p>Paragraph two.</p></body></html>"
        chunks = chunker.chunk(text, source="page.html")
        assert len(chunks) >= 1
        assert all(c.content_type == "html" for c in chunks)

    def test_strips_script_style(self):
        chunker = HTMLChunker(chunk_size=500, chunk_overlap=0)
        text = "<html><script>alert('x')</script><style>.a{}</style><p>Real content.</p></html>"
        chunks = chunker.chunk(text)
        full_text = " ".join(c.text for c in chunks)
        assert "alert" not in full_text
        assert "Real content" in full_text


class TestContentTypeDetection:
    def test_file_extensions(self):
        assert detect_content_type("doc.pdf") == "pdf"
        assert detect_content_type("doc.md") == "markdown"
        assert detect_content_type("page.html") == "html"
        assert detect_content_type("main.py") == "code"
        assert detect_content_type("data.txt") == "text"

    def test_content_heuristics(self):
        assert detect_content_type("", "<html><body>test</body></html>") == "html"
        assert detect_content_type("", "# Heading\n\nSome text") == "markdown"
        assert detect_content_type("", "def foo():\n    pass") == "code"

    def test_factory_returns_correct_chunker(self):
        factory = ChunkerFactory()
        assert isinstance(factory.get_chunker("text"), TextChunker)
        assert isinstance(factory.get_chunker("markdown"), MarkdownChunker)
        assert isinstance(factory.get_chunker("code"), CodeChunker)
        assert isinstance(factory.get_chunker("html"), HTMLChunker)
