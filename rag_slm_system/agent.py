"""Gemini ADK-based RAG Agent.

An agentic RAG system using Google's Agent Development Kit (ADK).
The agent can ingest documents, answer questions, and generate QA pairs.
"""

import json
import logging

from google.adk import Agent

from rag_slm_system.config import RAGConfig
from rag_slm_system.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


def create_rag_agent(
    config: RAGConfig | None = None,
    api_key: str | None = None,
    model: str = "gemini-2.0-flash",
) -> Agent:
    """Create a Gemini ADK agent with RAG capabilities.

    The agent has tools to:
    - Ingest documents into the knowledge base
    - Answer questions using RAG retrieval
    - Generate QA pairs for fine-tuning
    - Export data for SLM training

    Args:
        config: RAG configuration. Uses defaults if None.
        api_key: Gemini API key. Falls back to GOOGLE_API_KEY env var.
        model: Gemini model to use for the agent.

    Returns:
        An ADK Agent instance.
    """
    pipeline = RAGPipeline(config)

    def search_knowledge_base(query: str, top_k: int = 5) -> str:
        """Search the knowledge base for relevant information.

        Args:
            query: The search query or question.
            top_k: Number of results to return.

        Returns:
            Retrieved context from the knowledge base.
        """
        result = pipeline.retriever.retrieve(query, top_k=top_k)
        if not result.results:
            return "No relevant information found in the knowledge base."
        return result.context

    def ingest_document(file_path: str) -> str:
        """Ingest a document into the knowledge base.

        Args:
            file_path: Path to the document file.

        Returns:
            Summary of ingestion results.
        """
        try:
            chunks = pipeline.ingest_file(file_path)
            return (
                f"Successfully ingested {file_path}: "
                f"{len(chunks)} chunks created, "
                f"total vectors in store: {pipeline.vector_store.size}"
            )
        except Exception as e:
            return f"Failed to ingest {file_path}: {e}"

    def generate_qa_pairs(num_pairs_per_chunk: int = 3) -> str:
        """Generate question-answer pairs from ingested documents.

        Args:
            num_pairs_per_chunk: Number of QA pairs to generate per chunk.

        Returns:
            Summary and sample QA pairs.
        """
        pairs = pipeline.generate_qa_pairs(num_pairs_per_chunk=num_pairs_per_chunk)
        if not pairs:
            return "No QA pairs generated. Make sure documents are ingested first."

        sample = pairs[:3]
        sample_text = "\n".join(
            f"Q: {p.question}\nA: {p.answer}\n" for p in sample
        )

        return (
            f"Generated {len(pairs)} QA pairs.\n\n"
            f"Sample pairs:\n{sample_text}\n"
            f"Use export_training_data to save all pairs."
        )

    def export_training_data(
        output_dir: str = "./training_data",
        output_format: str = "alpaca",
    ) -> str:
        """Export QA pairs as training data for SLM fine-tuning.

        Args:
            output_dir: Directory to save the training data.
            output_format: Format ('alpaca', 'sharegpt', or 'chat_ml').

        Returns:
            Paths to exported files.
        """
        paths = pipeline.export_for_fine_tuning(output_dir, output_format=output_format)
        return (
            f"Training data exported to {output_dir}:\n"
            f"  Train: {paths['train']}\n"
            f"  Val: {paths['val']}\n"
            f"  Stats: {paths['stats']}"
        )

    def get_knowledge_base_stats() -> str:
        """Get statistics about the current knowledge base.

        Returns:
            Summary of ingested documents and chunks.
        """
        total_chunks = len(pipeline._all_chunks)
        total_vectors = pipeline.vector_store.size if pipeline._vector_store else 0
        total_qa = len(pipeline._all_qa_pairs)

        content_types: dict[str, int] = {}
        sources: set[str] = set()
        for chunk in pipeline._all_chunks:
            ct = chunk.content_type
            content_types[ct] = content_types.get(ct, 0) + 1
            if chunk.source:
                sources.add(chunk.source)

        return (
            f"Knowledge Base Stats:\n"
            f"  Total chunks: {total_chunks}\n"
            f"  Total vectors: {total_vectors}\n"
            f"  Total QA pairs: {total_qa}\n"
            f"  Content types: {json.dumps(content_types)}\n"
            f"  Sources: {len(sources)} documents"
        )

    agent = Agent(
        model=model,
        name="rag_slm_agent",
        description=(
            "A RAG agent that can ingest documents, answer questions using "
            "retrieval-augmented generation, and generate QA pairs for "
            "fine-tuning small language models."
        ),
        instruction=(
            "You are a helpful RAG assistant. You can:\n"
            "1. Ingest documents into a knowledge base\n"
            "2. Answer questions by searching the knowledge base\n"
            "3. Generate QA pairs for training language models\n"
            "4. Export training data in various formats\n\n"
            "When answering questions, always search the knowledge base first "
            "and base your answers on the retrieved context. If the context "
            "doesn't contain relevant information, say so clearly.\n\n"
            "When asked to generate training data, first ingest documents, "
            "then generate QA pairs, and finally export them."
        ),
        tools=[
            search_knowledge_base,
            ingest_document,
            generate_qa_pairs,
            export_training_data,
            get_knowledge_base_stats,
        ],
    )

    return agent
