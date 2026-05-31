"""QA pair generation using Google Gemini via ADK."""

import json
import logging
import re

from google import genai
from google.genai import types

from rag_slm_system.qa_generator.base import BaseQAGenerator, QAPair

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTION = """You are an expert at generating high-quality question-answer pairs
from documents for training language models. Your QA pairs should:

1. Cover key facts, concepts, and relationships in the text
2. Vary in complexity (factual recall, inference, analysis)
3. Have clear, complete answers grounded in the provided text
4. Be self-contained — the question should make sense without seeing the source text
5. Avoid trivial yes/no questions

Return ONLY a JSON array of objects with "question", "answer", and "difficulty" fields.
Difficulty should be one of: "easy", "medium", "hard".
"""


class GeminiQAGenerator(BaseQAGenerator):
    """Generate QA pairs using Google Gemini via the GenAI SDK / ADK.

    Uses Gemini to analyze text chunks and produce diverse, high-quality
    question-answer pairs suitable for SLM fine-tuning.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()

    def generate(self, text: str, source: str = "", num_pairs: int = 3) -> list[QAPair]:
        """Generate QA pairs from a chunk of text using Gemini."""
        if not text.strip():
            return []

        prompt = self._build_prompt(text, source, num_pairs)

        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                    response_mime_type="application/json",
                ),
            )

            return self._parse_response(response.text, text, source)

        except Exception as e:
            logger.warning(f"Gemini QA generation failed: {e}")
            return []

    def _build_prompt(self, text: str, source: str, num_pairs: int) -> str:
        source_note = f"\nSource: {source}" if source else ""
        return (
            f"Generate exactly {num_pairs} question-answer pairs from the following text."
            f"{source_note}\n\n"
            f"Text:\n\"\"\"\n{text}\n\"\"\"\n\n"
            f"Return a JSON array of {num_pairs} objects, each with "
            f'"question", "answer", and "difficulty" fields.'
        )

    def _parse_response(self, response_text: str, context: str, source: str) -> list[QAPair]:
        """Parse Gemini response into QAPair objects."""
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            json_match = re.search(r"\[.*\]", response_text, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    logger.warning("Failed to parse Gemini response as JSON")
                    return []
            else:
                return []

        if not isinstance(data, list):
            data = [data]

        pairs: list[QAPair] = []
        for item in data:
            if isinstance(item, dict) and "question" in item and "answer" in item:
                pairs.append(
                    QAPair(
                        question=item["question"],
                        answer=item["answer"],
                        context=context,
                        source=source,
                        difficulty=item.get("difficulty", "medium"),
                    )
                )

        return pairs
