"""Format QA pairs into datasets for SLM fine-tuning."""

import json
import logging
from pathlib import Path

from rag_slm_system.qa_generator.base import QAPair

logger = logging.getLogger(__name__)


class DataFormatter:
    """Format QA pairs into training datasets for various SLM architectures.

    Supports Alpaca, ShareGPT, and ChatML formats with train/val splitting.
    """

    def __init__(self, output_format: str = "alpaca", val_split: float = 0.1):
        self.output_format = output_format
        self.val_split = val_split

    def format_pairs(self, pairs: list[QAPair]) -> list[dict]:
        """Convert QA pairs to the target format."""
        if self.output_format == "alpaca":
            return [p.to_alpaca() for p in pairs]
        elif self.output_format == "sharegpt":
            return [p.to_sharegpt() for p in pairs]
        elif self.output_format == "chat_ml":
            return [p.to_chat_ml() for p in pairs]
        else:
            raise ValueError(f"Unknown format: {self.output_format}")

    def prepare_dataset(
        self,
        pairs: list[QAPair],
        output_dir: str | Path,
    ) -> dict[str, Path]:
        """Prepare train/val datasets and save to disk.

        Args:
            pairs: List of QA pairs.
            output_dir: Directory to save the datasets.

        Returns:
            Dict with 'train' and 'val' paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        formatted = self.format_pairs(pairs)

        import random
        random.shuffle(formatted)

        split_idx = max(1, int(len(formatted) * (1 - self.val_split)))
        train_data = formatted[:split_idx]
        val_data = formatted[split_idx:]

        train_path = output_dir / "train.json"
        val_path = output_dir / "val.json"

        with open(train_path, "w") as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)

        with open(val_path, "w") as f:
            json.dump(val_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(train_data)} training and {len(val_data)} validation examples")

        stats = self._compute_stats(pairs)
        stats_path = output_dir / "dataset_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        return {"train": train_path, "val": val_path, "stats": stats_path}

    def _compute_stats(self, pairs: list[QAPair]) -> dict:
        """Compute dataset statistics."""
        q_lengths = [len(p.question.split()) for p in pairs]
        a_lengths = [len(p.answer.split()) for p in pairs]
        difficulties = {}
        sources = {}
        content_types = {}

        for p in pairs:
            difficulties[p.difficulty] = difficulties.get(p.difficulty, 0) + 1
            if p.source:
                sources[p.source] = sources.get(p.source, 0) + 1
            if p.content_type:
                content_types[p.content_type] = content_types.get(p.content_type, 0) + 1

        return {
            "total_pairs": len(pairs),
            "avg_question_length_words": sum(q_lengths) / max(len(q_lengths), 1),
            "avg_answer_length_words": sum(a_lengths) / max(len(a_lengths), 1),
            "difficulty_distribution": difficulties,
            "source_distribution": sources,
            "content_type_distribution": content_types,
        }

    def to_hf_dataset(self, pairs: list[QAPair]):
        """Convert to a HuggingFace Dataset object."""
        try:
            from datasets import Dataset
        except ImportError as e:
            raise ImportError("datasets is required: pip install datasets") from e

        formatted = self.format_pairs(pairs)

        if self.output_format == "alpaca":
            return Dataset.from_dict(
                {
                    "instruction": [d["instruction"] for d in formatted],
                    "input": [d["input"] for d in formatted],
                    "output": [d["output"] for d in formatted],
                }
            )
        elif self.output_format == "sharegpt":
            return Dataset.from_dict(
                {"conversations": [json.dumps(d["conversations"]) for d in formatted]}
            )
        elif self.output_format == "chat_ml":
            return Dataset.from_dict({"text": [d["text"] for d in formatted]})
        else:
            raise ValueError(f"Unknown format: {self.output_format}")
