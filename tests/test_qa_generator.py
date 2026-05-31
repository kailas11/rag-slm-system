"""Tests for the QA pair generation module."""

import tempfile
from pathlib import Path

from rag_slm_system.qa_generator.base import QAPair
from rag_slm_system.qa_generator.template_qa_generator import TemplateQAGenerator


class TestQAPair:
    def test_to_alpaca(self):
        pair = QAPair(question="What is X?", answer="X is Y.", context="Context here.")
        result = pair.to_alpaca()
        assert result["instruction"] == "What is X?"
        assert result["output"] == "X is Y."
        assert result["input"] == "Context here."

    def test_to_sharegpt(self):
        pair = QAPair(question="Q?", answer="A.", context="C")
        result = pair.to_sharegpt()
        assert "conversations" in result
        assert len(result["conversations"]) == 3  # system + user + assistant

    def test_to_sharegpt_no_context(self):
        pair = QAPair(question="Q?", answer="A.")
        result = pair.to_sharegpt()
        assert len(result["conversations"]) == 2  # user + assistant

    def test_to_chat_ml(self):
        pair = QAPair(question="Q?", answer="A.")
        result = pair.to_chat_ml()
        assert "<|im_start|>user" in result["text"]
        assert "<|im_start|>assistant" in result["text"]

    def test_save_and_load_pairs(self):
        pairs = [
            QAPair(question="Q1?", answer="A1.", context="C1"),
            QAPair(question="Q2?", answer="A2.", context="C2"),
        ]

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name

        try:
            QAPair.to_alpaca  # sanity check method exists
            from rag_slm_system.qa_generator.base import BaseQAGenerator

            BaseQAGenerator.save_pairs(pairs, path, "alpaca")
            loaded = BaseQAGenerator.load_pairs(path, "alpaca")
            assert len(loaded) == 2
            assert loaded[0].question == "Q1?"
            assert loaded[1].answer == "A2."
        finally:
            Path(path).unlink(missing_ok=True)


class TestTemplateQAGenerator:
    def test_generates_pairs(self):
        gen = TemplateQAGenerator()
        text = (
            "Python is a programming language. "
            "Machine learning provides powerful tools for data analysis. "
            "The system includes a web server and a database."
        )
        pairs = gen.generate(text, num_pairs=3)
        assert len(pairs) > 0
        assert all(isinstance(p, QAPair) for p in pairs)

    def test_empty_text(self):
        gen = TemplateQAGenerator()
        assert gen.generate("") == []

    def test_max_pairs_limit(self):
        gen = TemplateQAGenerator()
        text = (
            "A is B. C is D. E is F. G is H. I is J. "
            "K is L. M is N. O is P. Q is R. S is T."
        )
        pairs = gen.generate(text, num_pairs=2)
        assert len(pairs) <= 2
