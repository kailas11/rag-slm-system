"""SLM fine-tuning trainer using PEFT/LoRA."""

import json
import logging
from pathlib import Path

from rag_slm_system.config import FineTuningConfig
from rag_slm_system.fine_tuning.data_formatter import DataFormatter
from rag_slm_system.qa_generator.base import QAPair

logger = logging.getLogger(__name__)


class SLMTrainer:
    """Fine-tune Small Language Models using LoRA/PEFT.

    Supports models like Phi-2, Phi-3-mini, TinyLlama, Gemma-2B, etc.
    Uses QLoRA for memory-efficient training.
    """

    SUPPORTED_MODELS = {
        "microsoft/phi-2": {"type": "causal_lm", "max_seq_length": 2048},
        "microsoft/Phi-3-mini-4k-instruct": {"type": "causal_lm", "max_seq_length": 4096},
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {"type": "causal_lm", "max_seq_length": 2048},
        "google/gemma-2b": {"type": "causal_lm", "max_seq_length": 8192},
        "Qwen/Qwen2-1.5B-Instruct": {"type": "causal_lm", "max_seq_length": 4096},
    }

    def __init__(self, config: FineTuningConfig | None = None):
        self.config = config or FineTuningConfig()

    def prepare_and_train(
        self,
        pairs: list[QAPair],
        output_dir: str | None = None,
    ) -> dict:
        """End-to-end: format data, prepare dataset, and train.

        Args:
            pairs: QA pairs to train on.
            output_dir: Override config output directory.

        Returns:
            Dict with training results and model path.
        """
        out_dir = Path(output_dir or self.config.output_dir)
        data_dir = out_dir / "data"

        formatter = DataFormatter(
            output_format=self.config.output_format,
            val_split=self.config.val_split,
        )
        dataset_paths = formatter.prepare_dataset(pairs, data_dir)

        logger.info(f"Training data saved to {data_dir}")

        return self.train(
            train_path=str(dataset_paths["train"]),
            val_path=str(dataset_paths["val"]),
            output_dir=str(out_dir / "model"),
        )

    def train(
        self,
        train_path: str,
        val_path: str | None = None,
        output_dir: str | None = None,
    ) -> dict:
        """Train the SLM with LoRA/PEFT.

        Args:
            train_path: Path to training data JSON.
            val_path: Path to validation data JSON.
            output_dir: Directory for model output.

        Returns:
            Dict with training results.
        """
        try:
            import torch
            from peft import LoraConfig, TaskType, get_peft_model
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                Trainer,
                TrainingArguments,
            )
        except ImportError as e:
            raise ImportError(
                "Required packages: pip install torch transformers peft datasets accelerate"
            ) from e

        out_dir = output_dir or self.config.output_dir

        logger.info(f"Loading model: {self.config.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=self._get_target_modules(),
        )

        model = get_peft_model(model, lora_config)
        trainable_params, total_params = model.get_nb_trainable_parameters()
        logger.info(
            f"Trainable params: {trainable_params:,} / {total_params:,} "
            f"({100 * trainable_params / total_params:.2f}%)"
        )

        train_dataset = self._load_and_tokenize(train_path, tokenizer)
        val_dataset = self._load_and_tokenize(val_path, tokenizer) if val_path else None

        training_args = TrainingArguments(
            output_dir=out_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch" if val_dataset else "no",
            save_total_limit=2,
            load_best_model_at_end=bool(val_dataset),
            report_to="none",
            fp16=torch.cuda.is_available(),
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
        )

        logger.info("Starting training...")
        train_result = trainer.train()

        trainer.save_model(out_dir)
        tokenizer.save_pretrained(out_dir)

        metrics = {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples": len(train_dataset),
            "model_path": out_dir,
            "trainable_params": trainable_params,
            "total_params": total_params,
        }

        with open(Path(out_dir) / "training_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Model saved to {out_dir}")
        return metrics

    def _load_and_tokenize(self, data_path: str, tokenizer):
        from datasets import Dataset

        with open(data_path) as f:
            data = json.load(f)

        texts: list[str] = []
        for item in data:
            if self.config.output_format == "alpaca":
                text = self._format_alpaca_prompt(item)
            elif self.config.output_format == "chat_ml":
                text = item.get("text", "")
            else:
                text = json.dumps(item)
            texts.append(text)

        dataset = Dataset.from_dict({"text": texts})

        def tokenize_fn(examples):
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_seq_length,
                padding="max_length",
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        return dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    def _format_alpaca_prompt(self, item: dict) -> str:
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output = item.get("output", "")

        if input_text:
            return (
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}\n\n"
                f"### Response:\n{output}"
            )
        return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

    def _get_target_modules(self) -> list[str]:
        """Get LoRA target modules based on model architecture."""
        model_name = self.config.model_name.lower()
        if "phi" in model_name:
            return ["q_proj", "k_proj", "v_proj", "dense"]
        elif "llama" in model_name or "tinyllama" in model_name:
            return ["q_proj", "k_proj", "v_proj", "o_proj"]
        elif "gemma" in model_name:
            return ["q_proj", "k_proj", "v_proj", "o_proj"]
        elif "qwen" in model_name:
            return ["q_proj", "k_proj", "v_proj", "o_proj"]
        else:
            return ["q_proj", "v_proj"]

    def generate_training_config(self, output_path: str | Path) -> None:
        """Export training configuration to a YAML-like JSON file."""
        config_dict = {
            "model": {
                "name": self.config.model_name,
                "max_seq_length": self.config.max_seq_length,
            },
            "lora": {
                "r": self.config.lora_r,
                "alpha": self.config.lora_alpha,
                "dropout": self.config.lora_dropout,
                "target_modules": self._get_target_modules(),
            },
            "training": {
                "epochs": self.config.num_epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                "warmup_ratio": self.config.warmup_ratio,
                "weight_decay": self.config.weight_decay,
            },
            "data": {
                "format": self.config.output_format,
                "val_split": self.config.val_split,
            },
        }

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)
