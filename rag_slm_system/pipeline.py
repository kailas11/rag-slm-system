"""End-to-end RAG + SLM fine-tuning pipeline."""

import logging
from pathlib import Path

from rag_slm_system.chunking.base import Chunk
from rag_slm_system.chunking.factory import ChunkerFactory, detect_content_type
from rag_slm_system.config import RAGConfig
from rag_slm_system.embeddings.base import BaseEmbedder
from rag_slm_system.qa_generator.base import BaseQAGenerator, QAPair
from rag_slm_system.retriever.hybrid_retriever import HybridRetriever
from rag_slm_system.retriever.rag_retriever import RAGRetriever
from rag_slm_system.vectorstore.base import BaseVectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Unified pipeline for document ingestion, RAG retrieval, and QA generation.

    Usage:
        config = RAGConfig()
        pipeline = RAGPipeline(config)
        pipeline.ingest_documents(["doc1.pdf", "doc2.md"])
        results = pipeline.query("What is X?")
        qa_pairs = pipeline.generate_qa_pairs()
        pipeline.export_for_fine_tuning("./output")
    """

    def __init__(self, config: RAGConfig | None = None):
        self.config = config or RAGConfig()
        self._chunker_factory = ChunkerFactory(self.config.chunking)
        self._embedder: BaseEmbedder | None = None
        self._vector_store: BaseVectorStore | None = None
        self._retriever: RAGRetriever | HybridRetriever | None = None
        self._qa_generator: BaseQAGenerator | None = None
        self._all_chunks: list[Chunk] = []
        self._all_qa_pairs: list[QAPair] = []

    @property
    def embedder(self) -> BaseEmbedder:
        if self._embedder is None:
            self._embedder = self._create_embedder()
        return self._embedder

    @property
    def vector_store(self) -> BaseVectorStore:
        if self._vector_store is None:
            self._vector_store = self._create_vector_store()
        return self._vector_store

    @property
    def retriever(self) -> RAGRetriever | HybridRetriever:
        if self._retriever is None:
            self._retriever = self._create_retriever()
        return self._retriever

    @property
    def qa_generator(self) -> BaseQAGenerator:
        if self._qa_generator is None:
            self._qa_generator = self._create_qa_generator()
        return self._qa_generator

    def ingest_text(self, text: str, source: str = "") -> list[Chunk]:
        """Ingest raw text, chunk it, and add to vector store."""
        content_type = detect_content_type(source, text)
        chunker = self._chunker_factory.get_chunker(content_type)
        chunks = chunker.chunk(text, source)

        if not chunks:
            return []

        logger.info(f"Created {len(chunks)} chunks from {source or 'text'} ({content_type})")

        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed(texts)
        self.vector_store.add(chunks, embeddings)

        if isinstance(self.retriever, HybridRetriever):
            self.retriever.add_chunks(chunks)

        self._all_chunks.extend(chunks)
        logger.info(f"Total chunks in store: {self.vector_store.size}")

        return chunks

    def ingest_file(self, file_path: str | Path) -> list[Chunk]:
        """Ingest a file, auto-detecting content type."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content_type = detect_content_type(str(file_path))

        if content_type == "pdf":
            from rag_slm_system.chunking.pdf_chunker import PDFChunker

            chunker = PDFChunker(
                chunk_size=self.config.chunking.chunk_size,
                chunk_overlap=self.config.chunking.chunk_overlap,
                min_chunk_size=self.config.chunking.min_chunk_size,
            )
            chunks = chunker.chunk_file(file_path)
        else:
            text = file_path.read_text(encoding="utf-8", errors="replace")
            chunker = self._chunker_factory.get_chunker(content_type)
            chunks = chunker.chunk(text, source=str(file_path))

        if not chunks:
            return []

        logger.info(f"Created {len(chunks)} chunks from {file_path} ({content_type})")

        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed(texts)
        self.vector_store.add(chunks, embeddings)

        if isinstance(self.retriever, HybridRetriever):
            self.retriever.add_chunks(chunks)

        self._all_chunks.extend(chunks)
        return chunks

    def ingest_documents(self, paths: list[str | Path]) -> int:
        """Ingest multiple documents. Returns total chunk count."""
        total = 0
        for path in paths:
            try:
                chunks = self.ingest_file(path)
                total += len(chunks)
            except Exception as e:
                logger.error(f"Failed to ingest {path}: {e}")
        return total

    def query(self, question: str, top_k: int | None = None) -> dict:
        """Query the RAG system."""
        return self.retriever.retrieve(question, top_k)

    def generate_qa_pairs(
        self,
        chunks: list[Chunk] | None = None,
        num_pairs_per_chunk: int | None = None,
    ) -> list[QAPair]:
        """Generate QA pairs from ingested chunks.

        Args:
            chunks: Specific chunks to use. Defaults to all ingested chunks.
            num_pairs_per_chunk: Override config for QA pairs per chunk.

        Returns:
            List of QAPair objects.
        """
        target_chunks = chunks or self._all_chunks
        n = num_pairs_per_chunk or self.config.qa_generator.num_questions_per_chunk

        logger.info(f"Generating {n} QA pairs per chunk for {len(target_chunks)} chunks...")
        pairs = self.qa_generator.generate_from_chunks(target_chunks, num_pairs=n)

        self._all_qa_pairs.extend(pairs)
        logger.info(f"Generated {len(pairs)} QA pairs total")
        return pairs

    def export_for_fine_tuning(
        self,
        output_dir: str | Path,
        pairs: list[QAPair] | None = None,
        output_format: str | None = None,
    ) -> dict[str, Path]:
        """Export QA pairs in a format ready for SLM fine-tuning.

        Args:
            output_dir: Directory to save formatted datasets.
            pairs: Specific QA pairs. Defaults to all generated pairs.
            output_format: Override config format ('alpaca', 'sharegpt', 'chat_ml').

        Returns:
            Dict with paths to train/val files.
        """
        from rag_slm_system.fine_tuning.data_formatter import DataFormatter

        target_pairs = pairs or self._all_qa_pairs
        fmt = output_format or self.config.fine_tuning.output_format

        formatter = DataFormatter(
            output_format=fmt,
            val_split=self.config.fine_tuning.val_split,
        )

        return formatter.prepare_dataset(target_pairs, output_dir)

    def save(self, directory: str | Path) -> None:
        """Save the pipeline state (vector store + QA pairs)."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        self.vector_store.save(str(directory / "vector_store"))

        if self._all_qa_pairs:
            BaseQAGenerator.save_pairs(
                self._all_qa_pairs,
                directory / "qa_pairs.json",
                self.config.fine_tuning.output_format,
            )

        logger.info(f"Pipeline state saved to {directory}")

    def _create_embedder(self) -> BaseEmbedder:
        cfg = self.config.embedding
        from rag_slm_system.embeddings.sentence_transformer import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(
            model_name=cfg.model_name,
            device=cfg.device,
            batch_size=cfg.batch_size,
            normalize=cfg.normalize,
        )

    def _create_vector_store(self) -> BaseVectorStore:
        from rag_slm_system.vectorstore.faiss_store import FAISSVectorStore

        return FAISSVectorStore(
            dimension=self.embedder.dimension,
            similarity_metric=self.config.vector_store.similarity_metric,
        )

    def _create_retriever(self) -> RAGRetriever | HybridRetriever:
        cfg = self.config.retriever

        if cfg.mode == "hybrid":
            reranker = self._create_reranker()
            retriever = HybridRetriever(
                embedder=self.embedder,
                vector_store=self.vector_store,
                reranker=reranker,
                top_k=cfg.top_k,
                dense_weight=cfg.dense_weight,
                sparse_weight=cfg.sparse_weight,
                rrf_k=cfg.rrf_k,
                fusion_top_k=cfg.reranker.top_k if cfg.reranker.enabled else cfg.top_k * 3,
                score_threshold=cfg.score_threshold,
                bm25_k1=cfg.bm25_k1,
                bm25_b=cfg.bm25_b,
            )
            if self._all_chunks:
                retriever.add_chunks(self._all_chunks)
            return retriever

        return RAGRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            top_k=cfg.top_k,
            score_threshold=cfg.score_threshold,
        )

    def _create_reranker(self):
        cfg = self.config.retriever.reranker
        if not cfg.enabled:
            return None

        if cfg.method == "cohere":
            from rag_slm_system.retriever.reranker import CohereReranker

            return CohereReranker(
                api_key=cfg.cohere_api_key,
                model=cfg.model or None,
            )

        from rag_slm_system.retriever.reranker import CrossEncoderReranker

        return CrossEncoderReranker(
            model_name=cfg.model or None,
        )

    def _create_qa_generator(self) -> BaseQAGenerator:
        cfg = self.config.qa_generator

        if cfg.method == "gemini":
            from rag_slm_system.qa_generator.gemini_qa_generator import GeminiQAGenerator

            return GeminiQAGenerator(
                api_key=cfg.gemini_api_key,
                model_name=cfg.gemini_model,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )
        else:
            from rag_slm_system.qa_generator.template_qa_generator import TemplateQAGenerator

            return TemplateQAGenerator()
