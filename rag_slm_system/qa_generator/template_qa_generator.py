"""Template-based QA generator (no LLM required, rule-based fallback)."""

import re

from rag_slm_system.qa_generator.base import BaseQAGenerator, QAPair


class TemplateQAGenerator(BaseQAGenerator):
    """Generate QA pairs using rule-based templates.

    Useful as a fallback when no LLM API is available, or for generating
    simple factual QA pairs from structured text.
    """

    TEMPLATES = [
        {
            "pattern": r"(?:^|\n)(.+?)\s+is\s+(.+?)(?:\.|$)",
            "question": "What is {0}?",
            "answer": "{0} is {1}.",
            "difficulty": "easy",
        },
        {
            "pattern": r"(?:^|\n)(.+?)\s+(?:are|were)\s+(.+?)(?:\.|$)",
            "question": "What are {0}?",
            "answer": "{0} are {1}.",
            "difficulty": "easy",
        },
        {
            "pattern": r"(?:^|\n)(.+?)\s+(?:can|could)\s+(.+?)(?:\.|$)",
            "question": "What can {0} do?",
            "answer": "{0} can {1}.",
            "difficulty": "medium",
        },
        {
            "pattern": r"(?:^|\n)(.+?)\s+(?:provides?|offers?)\s+(.+?)(?:\.|$)",
            "question": "What does {0} provide?",
            "answer": "{0} provides {1}.",
            "difficulty": "medium",
        },
        {
            "pattern": r"(?:^|\n)(.+?)\s+(?:includes?|contains?)\s+(.+?)(?:\.|$)",
            "question": "What does {0} include?",
            "answer": "{0} includes {1}.",
            "difficulty": "easy",
        },
    ]

    def generate(self, text: str, source: str = "", num_pairs: int = 3) -> list[QAPair]:
        if not text.strip():
            return []

        pairs: list[QAPair] = []
        seen_questions: set[str] = set()

        for template in self.TEMPLATES:
            if len(pairs) >= num_pairs:
                break

            matches = re.finditer(template["pattern"], text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if len(pairs) >= num_pairs:
                    break

                groups = match.groups()
                if len(groups) < 2:
                    continue

                subject = groups[0].strip()
                obj = groups[1].strip()

                if len(subject) < 3 or len(obj) < 3:
                    continue
                if len(subject) > 100:
                    continue

                question = template["question"].format(subject, obj)
                if question in seen_questions:
                    continue

                answer = template["answer"].format(subject, obj)

                pairs.append(
                    QAPair(
                        question=question,
                        answer=answer,
                        context=text,
                        source=source,
                        difficulty=template["difficulty"],
                    )
                )
                seen_questions.add(question)

        if len(pairs) < num_pairs:
            summary_pairs = self._generate_summary_qa(text, source, num_pairs - len(pairs))
            pairs.extend(summary_pairs)

        return pairs[:num_pairs]

    def _generate_summary_qa(self, text: str, source: str, count: int) -> list[QAPair]:
        """Generate summary-based QA pairs from the text."""
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        pairs: list[QAPair] = []
        for i, sentence in enumerate(sentences[:count]):
            pairs.append(
                QAPair(
                    question=f"What information is provided about: {sentence[:80]}...?",
                    answer=sentence,
                    context=text,
                    source=source,
                    difficulty="easy",
                )
            )

        return pairs
