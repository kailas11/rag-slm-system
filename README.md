# RAG + SLM Fine-Tuning System

A modular, production-ready system for **Retrieval-Augmented Generation (RAG)** and **Small Language Model (SLM) fine-tuning**. Ingest documents with content-aware chunking, build a searchable knowledge base, generate QA pairs using **Gemini ADK**, and fine-tune SLMs with **LoRA/PEFT**.

## Architecture

```
Documents ──▶ Content-Aware Chunking ──▶ Embeddings ──▶ FAISS Vector Store
                    │                                         │
                    │                                         ▼
                    │                                   RAG Retrieval
                    │                                         │
                    ▼                                         ▼
              QA Pair Generation ◄─── Gemini ADK Agent ──▶ Answers
                    │
                    ▼
              SLM Fine-Tuning (LoRA/PEFT)
```

## Features

### Content-Aware Chunking
Automatically detects document type and applies the optimal chunking strategy:

| Content Type | Strategy | Preserves |
|---|---|---|
| **PDF** | Page-aware extraction + sentence splitting | Page numbers, document structure |
| **Markdown** | Heading-hierarchy splitting | Section paths, heading levels |
| **Code** | Function/class boundary detection | Language, function names, structure |
| **HTML** | Semantic element extraction | Tag context, headings |
| **Plain Text** | Recursive sentence-boundary splitting | Paragraph structure |

### Embeddings & Vector Store
- **Local embeddings**: sentence-transformers (all-MiniLM-L6-v2, no API key needed)
- **Gemini embeddings**: text-embedding-004 via Google GenAI API
- **FAISS vector store** with cosine/L2/inner-product similarity
- Persistence: save/load vector stores to disk

### Gemini ADK Agent
An agentic RAG system using Google's Agent Development Kit:
- Interactive document Q&A
- Tool-based knowledge base management
- Automated QA pair generation

### QA Pair Generation
Reusable module for generating training data:
- **Gemini-based**: High-quality, diverse QA pairs with difficulty levels
- **Template-based**: Rule-based fallback (no API needed)
- Multiple output formats: Alpaca, ShareGPT, ChatML

### SLM Fine-Tuning
- **LoRA/QLoRA** via PEFT for memory-efficient training
- Supports: Phi-2, Phi-3-mini, TinyLlama, Gemma-2B, Qwen2-1.5B
- Auto train/val splitting with dataset statistics

## Installation

```bash
pip install -e .

# For development
pip install -e ".[dev]"
```

## Quick Start

### Python API

```python
from rag_slm_system import RAGConfig
from rag_slm_system.pipeline import RAGPipeline

# Initialize
config = RAGConfig()
pipeline = RAGPipeline(config)

# Ingest documents
pipeline.ingest_file("docs/guide.md")
pipeline.ingest_file("paper.pdf")
pipeline.ingest_file("src/main.py")

# Query (RAG retrieval)
result = pipeline.query("How does the authentication system work?")
for r in result.results:
    print(f"[{r.score:.3f}] {r.chunk.text[:100]}...")

# Generate QA pairs for fine-tuning
qa_pairs = pipeline.generate_qa_pairs(num_pairs_per_chunk=3)

# Export for SLM training
pipeline.export_for_fine_tuning("./training_data", output_format="alpaca")
```

### Gemini ADK Agent

```python
from rag_slm_system.agent import create_rag_agent
from google.adk.runners import InMemoryRunner
from google.genai import types

agent = create_rag_agent(api_key="your-gemini-key")
runner = InMemoryRunner(agent=agent, app_name="rag_slm_agent")
session = runner.session_service.create_session(
    app_name="rag_slm_agent", user_id="user"
)

# Chat with your documents
content = types.Content(role="user", parts=[types.Part(text="Summarize the main findings")])
for event in runner.run(user_id="user", session_id=session.id, new_message=content):
    if event.content and event.content.parts:
        for part in event.content.parts:
            if part.text:
                print(part.text)
```

### CLI

```bash
# Ingest documents
rag-slm ingest docs/ papers/ --save-to ./knowledge_base

# Query
rag-slm query "What is the main algorithm?" --load-from ./knowledge_base

# Generate QA pairs
rag-slm generate-qa docs/ --num-pairs 5 --output qa_pairs.json --preview

# Export for fine-tuning
rag-slm export qa_pairs.json --output-dir ./training_data --format alpaca

# Fine-tune SLM
rag-slm train qa_pairs.json --model microsoft/phi-2 --output-dir ./my_model

# Interactive agent
rag-slm agent --model gemini-2.0-flash
```

## Project Structure

```
rag_slm_system/
├── chunking/           # Content-aware document chunking
│   ├── base.py         # Chunk dataclass + BaseChunker ABC
│   ├── text_chunker.py # Recursive sentence-boundary splitting
│   ├── markdown_chunker.py  # Heading-hierarchy splitting
│   ├── code_chunker.py # Function/class boundary detection
│   ├── pdf_chunker.py  # Page-aware PDF extraction
│   ├── html_chunker.py # Semantic HTML element extraction
│   └── factory.py      # Auto-detection + ChunkerFactory
├── embeddings/         # Embedding generation
│   ├── base.py         # BaseEmbedder ABC
│   ├── sentence_transformer.py  # Local sentence-transformers
│   └── gemini_embedder.py       # Google Gemini embeddings
├── vectorstore/        # Vector storage and search
│   ├── base.py         # BaseVectorStore ABC
│   └── faiss_store.py  # FAISS implementation
├── retriever/          # RAG retrieval pipeline
│   └── rag_retriever.py
├── qa_generator/       # QA pair generation
│   ├── base.py         # QAPair dataclass + base class
│   ├── gemini_qa_generator.py   # Gemini-powered generation
│   └── template_qa_generator.py # Rule-based fallback
├── fine_tuning/        # SLM fine-tuning
│   ├── data_formatter.py # Dataset preparation
│   └── trainer.py      # LoRA/PEFT training
├── agent.py            # Gemini ADK RAG agent
├── pipeline.py         # End-to-end pipeline
├── config.py           # Configuration dataclasses
└── cli.py              # CLI interface
```

## Configuration

```python
from rag_slm_system.config import RAGConfig, ChunkingConfig, QAGeneratorConfig

config = RAGConfig(
    chunking=ChunkingConfig(
        chunk_size=512,
        chunk_overlap=64,
        split_on_heading_level=2,
    ),
    qa_generator=QAGeneratorConfig(
        method="gemini",
        gemini_model="gemini-2.0-flash",
        num_questions_per_chunk=5,
    ),
)
```

## Environment Variables

| Variable | Description |
|---|---|
| `GOOGLE_API_KEY` | Gemini API key (for QA generation & embeddings) |

## License

MIT
