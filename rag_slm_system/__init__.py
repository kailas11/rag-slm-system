"""
RAG + SLM Fine-Tuning System

A reusable system for:
1. Content-aware document chunking (PDF, Markdown, Code, HTML, plain text)
2. Embedding generation and vector storage (FAISS)
3. RAG-based retrieval
4. QA pair generation using Gemini ADK
5. SLM fine-tuning orchestration with LoRA/PEFT
"""

from rag_slm_system.config import RAGConfig

__all__ = ["RAGConfig"]
__version__ = "0.1.0"
