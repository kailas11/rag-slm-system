"""Global configuration for the RAG + SLM system."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""

    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 50
    respect_sentence_boundaries: bool = True
    max_function_chunk_size: int = 1024
    split_on_heading_level: int = 2


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""

    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32
    normalize: bool = True


@dataclass
class VectorStoreConfig:
    """Configuration for vector store."""

    store_type: str = "faiss"
    persist_directory: str = "./vector_store"
    similarity_metric: str = "cosine"


@dataclass
class RerankerConfig:
    """Configuration for result reranking."""

    enabled: bool = True
    method: str = "cross_encoder"  # "cohere" or "cross_encoder"
    model: str = ""  # empty = use default per method
    cohere_api_key: Optional[str] = None
    top_k: int = 10  # candidates passed to reranker


@dataclass
class RetrieverConfig:
    """Configuration for RAG retrieval."""

    mode: str = "hybrid"  # "dense", "sparse", or "hybrid"
    top_k: int = 5
    score_threshold: float = 0.3
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    rrf_k: int = 60
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    reranker: RerankerConfig = field(default_factory=RerankerConfig)


@dataclass
class QAGeneratorConfig:
    """Configuration for QA pair generation."""

    method: str = "gemini"
    num_questions_per_chunk: int = 3
    include_context: bool = True
    gemini_model: str = "gemini-2.0-flash"
    gemini_api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1024


@dataclass
class FineTuningConfig:
    """Configuration for SLM fine-tuning."""

    model_name: str = "microsoft/phi-2"
    output_dir: str = "./fine_tuned_model"
    num_epochs: int = 3
    learning_rate: float = 2e-5
    batch_size: int = 4
    max_seq_length: int = 512
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    output_format: str = "alpaca"
    val_split: float = 0.1


@dataclass
class RAGConfig:
    """Top-level configuration combining all sub-configs."""

    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    qa_generator: QAGeneratorConfig = field(default_factory=QAGeneratorConfig)
    fine_tuning: FineTuningConfig = field(default_factory=FineTuningConfig)
