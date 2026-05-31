"""Quick start example for the RAG + SLM Fine-Tuning System.

This example demonstrates:
1. Content-aware document chunking
2. Embedding and vector store creation
3. RAG retrieval
4. QA pair generation
5. Exporting training data for SLM fine-tuning
"""

import os
import tempfile
from pathlib import Path

from rag_slm_system.config import RAGConfig, QAGeneratorConfig
from rag_slm_system.pipeline import RAGPipeline


# Sample documents for demonstration
SAMPLE_MARKDOWN = """
# Machine Learning Overview

## Supervised Learning

Supervised learning is a type of machine learning where the model is trained
on labeled data. The algorithm learns a mapping function from input variables
to output variables. Common algorithms include linear regression, decision
trees, and neural networks.

### Classification vs Regression

Classification predicts discrete labels (e.g., spam/not spam), while
regression predicts continuous values (e.g., house prices). Both are
fundamental tasks in supervised learning.

## Unsupervised Learning

Unsupervised learning works with unlabeled data. The algorithm tries to
find patterns and structure in the data without explicit guidance.
Clustering and dimensionality reduction are key techniques.

### K-Means Clustering

K-Means is a popular clustering algorithm that partitions data into K
clusters. Each data point is assigned to the nearest centroid, and
centroids are iteratively updated.
"""

SAMPLE_CODE = '''
def calculate_cosine_similarity(vec_a, vec_b):
    """Calculate cosine similarity between two vectors.

    Args:
        vec_a: First vector as a list of floats.
        vec_b: Second vector as a list of floats.

    Returns:
        Cosine similarity score between -1 and 1.
    """
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = sum(a ** 2 for a in vec_a) ** 0.5
    magnitude_b = sum(b ** 2 for b in vec_b) ** 0.5

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


class VectorDatabase:
    """Simple in-memory vector database for similarity search."""

    def __init__(self):
        self.vectors = []
        self.metadata = []

    def add(self, vector, meta=None):
        self.vectors.append(vector)
        self.metadata.append(meta or {})

    def search(self, query_vector, top_k=5):
        scores = [
            (i, calculate_cosine_similarity(query_vector, v))
            for i, v in enumerate(self.vectors)
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
'''

SAMPLE_TEXT = """
Retrieval-Augmented Generation (RAG) is a technique that enhances large language
models by providing them with relevant external knowledge at inference time. Instead
of relying solely on the model's parametric memory, RAG retrieves documents from a
knowledge base and includes them in the prompt context.

The RAG pipeline consists of three main stages: indexing, retrieval, and generation.
During indexing, documents are chunked, embedded, and stored in a vector database.
At query time, the user's question is embedded and used to retrieve the most relevant
chunks. These chunks are then passed to the language model along with the question
to generate a grounded answer.

RAG offers several advantages over pure parametric models. It reduces hallucination
by grounding responses in retrieved evidence. It allows easy knowledge updates without
retraining. And it enables attribution of answers to source documents.
"""


def main():
    # Use template-based QA if no Gemini key available
    qa_method = "gemini" if os.environ.get("GOOGLE_API_KEY") else "template"

    config = RAGConfig(
        qa_generator=QAGeneratorConfig(method=qa_method),
    )
    pipeline = RAGPipeline(config)

    print("=" * 60)
    print("RAG + SLM Fine-Tuning System - Quick Start")
    print("=" * 60)

    # --- Step 1: Ingest documents ---
    print("\n1. Ingesting documents with content-aware chunking...")

    # Write sample files to temp dir
    with tempfile.TemporaryDirectory() as tmpdir:
        md_path = Path(tmpdir) / "ml_overview.md"
        md_path.write_text(SAMPLE_MARKDOWN)

        code_path = Path(tmpdir) / "similarity.py"
        code_path.write_text(SAMPLE_CODE)

        pipeline.ingest_file(md_path)
        pipeline.ingest_file(code_path)
        pipeline.ingest_text(SAMPLE_TEXT, source="rag_overview.txt")

        print(f"   Total chunks in vector store: {pipeline.vector_store.size}")

        # --- Step 2: RAG Retrieval ---
        print("\n2. RAG Retrieval...")
        queries = [
            "What is supervised learning?",
            "How does cosine similarity work?",
            "What are the advantages of RAG?",
        ]

        for query in queries:
            result = pipeline.query(query, top_k=2)
            print(f"\n   Q: {query}")
            for r in result.results:
                preview = r.chunk.text[:120].replace("\n", " ")
                print(f"   -> [{r.score:.3f}] {preview}...")

        # --- Step 3: Generate QA pairs ---
        print("\n\n3. Generating QA pairs for SLM fine-tuning...")
        pairs = pipeline.generate_qa_pairs(num_pairs_per_chunk=2)
        print(f"   Generated {len(pairs)} QA pairs")

        if pairs:
            print("\n   Sample QA pairs:")
            for i, p in enumerate(pairs[:3], 1):
                print(f"\n   [{i}] Q: {p.question}")
                print(f"       A: {p.answer[:150]}...")
                print(f"       Difficulty: {p.difficulty}")

        # --- Step 4: Export for fine-tuning ---
        print("\n\n4. Exporting training data...")
        output_dir = Path(tmpdir) / "training_output"
        paths = pipeline.export_for_fine_tuning(output_dir, output_format="alpaca")
        for key, path in paths.items():
            print(f"   {key}: {path}")

        print("\n" + "=" * 60)
        print("Done! The system is ready for SLM fine-tuning.")
        print("=" * 60)


if __name__ == "__main__":
    main()
