"""Base QA generator interface."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class QAPair:
    """A question-answer pair with metadata for fine-tuning."""

    question: str
    answer: str
    context: str = ""
    source: str = ""
    content_type: str = ""
    difficulty: str = "medium"
    metadata: dict = field(default_factory=dict)

    def to_alpaca(self) -> dict:
        """Convert to Alpaca format for fine-tuning."""
        return {
            "instruction": self.question,
            "input": self.context if self.context else "",
            "output": self.answer,
        }

    def to_sharegpt(self) -> dict:
        """Convert to ShareGPT/conversation format."""
        messages = []
        if self.context:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Use the following context to answer "
                        f"the question:\n\n{self.context}"
                    ),
                }
            )
        messages.append({"role": "user", "content": self.question})
        messages.append({"role": "assistant", "content": self.answer})
        return {"conversations": messages}

    def to_chat_ml(self) -> dict:
        """Convert to ChatML format."""
        parts = []
        if self.context:
            parts.append(
                f"<|im_start|>system\nUse the following context "
                f"to answer:\n\n{self.context}<|im_end|>"
            )
        parts.append(f"<|im_start|>user\n{self.question}<|im_end|>")
        parts.append(f"<|im_start|>assistant\n{self.answer}<|im_end|>")
        return {"text": "\n".join(parts)}


class BaseQAGenerator(ABC):
    """Abstract base class for QA pair generators."""

    @abstractmethod
    def generate(self, text: str, source: str = "", num_pairs: int = 3) -> list[QAPair]:
        """Generate QA pairs from a chunk of text."""

    def generate_from_chunks(self, chunks: list, num_pairs: int = 3) -> list[QAPair]:
        """Generate QA pairs from a list of Chunk objects."""
        all_pairs: list[QAPair] = []
        for chunk in chunks:
            pairs = self.generate(
                text=chunk.text,
                source=chunk.source,
                num_pairs=num_pairs,
            )
            for pair in pairs:
                pair.content_type = chunk.content_type
                pair.metadata.update(chunk.metadata)
            all_pairs.extend(pairs)
        return all_pairs

    @staticmethod
    def save_pairs(
        pairs: list[QAPair],
        output_path: str | Path,
        output_format: str = "alpaca",
    ) -> None:
        """Save QA pairs to a JSON file in the specified format."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if output_format == "alpaca":
            data = [p.to_alpaca() for p in pairs]
        elif output_format == "sharegpt":
            data = [p.to_sharegpt() for p in pairs]
        elif output_format == "chat_ml":
            data = [p.to_chat_ml() for p in pairs]
        else:
            raise ValueError(
                f"Unknown format: {output_format}. "
                "Use 'alpaca', 'sharegpt', or 'chat_ml'."
            )

        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def load_pairs(input_path: str | Path, input_format: str = "alpaca") -> list[QAPair]:
        """Load QA pairs from a JSON file."""
        with open(input_path) as f:
            data = json.load(f)

        pairs: list[QAPair] = []
        for item in data:
            if input_format == "alpaca":
                pairs.append(
                    QAPair(
                        question=item["instruction"],
                        answer=item["output"],
                        context=item.get("input", ""),
                    )
                )
            elif input_format == "sharegpt":
                convos = item["conversations"]
                question = ""
                answer = ""
                context = ""
                for msg in convos:
                    if msg["role"] == "user":
                        question = msg["content"]
                    elif msg["role"] == "assistant":
                        answer = msg["content"]
                    elif msg["role"] == "system":
                        context = msg["content"]
                pairs.append(QAPair(question=question, answer=answer, context=context))

        return pairs
