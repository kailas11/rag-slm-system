"""CLI interface for the RAG + SLM system."""

import argparse
import json
import logging
import sys
from pathlib import Path

from rag_slm_system.config import RAGConfig


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest documents into the knowledge base."""
    from rag_slm_system.pipeline import RAGPipeline

    config = _load_config(args)
    pipeline = RAGPipeline(config)

    files = []
    for path_str in args.files:
        p = Path(path_str)
        if p.is_dir():
            for ext in ("*.pdf", "*.md", "*.txt", "*.html", "*.py", "*.js", "*.ts"):
                files.extend(p.glob(f"**/{ext}"))
        else:
            files.append(p)

    total = pipeline.ingest_documents(files)
    print(f"Ingested {len(files)} files -> {total} chunks")

    if args.save_to:
        pipeline.save(args.save_to)
        print(f"Saved vector store to {args.save_to}")


def cmd_query(args: argparse.Namespace) -> None:
    """Query the RAG system."""
    from rag_slm_system.pipeline import RAGPipeline

    config = _load_config(args)
    pipeline = RAGPipeline(config)

    if args.load_from:
        pipeline.vector_store.load(str(Path(args.load_from) / "vector_store"))

    result = pipeline.query(args.question, top_k=args.top_k)
    print(f"\nQuery: {args.question}")
    print(f"\nRetrieved {len(result.results)} chunks:\n")
    for r in result.results:
        print(f"  [{r.rank + 1}] (score: {r.score:.4f}) {r.chunk.text[:200]}...")
        print()


def cmd_generate_qa(args: argparse.Namespace) -> None:
    """Generate QA pairs from documents."""
    from rag_slm_system.pipeline import RAGPipeline

    config = _load_config(args)
    pipeline = RAGPipeline(config)

    files = []
    for path_str in args.files:
        p = Path(path_str)
        if p.is_dir():
            for ext in ("*.pdf", "*.md", "*.txt", "*.html", "*.py", "*.js"):
                files.extend(p.glob(f"**/{ext}"))
        else:
            files.append(p)

    pipeline.ingest_documents(files)
    pairs = pipeline.generate_qa_pairs(num_pairs_per_chunk=args.num_pairs)

    print(f"\nGenerated {len(pairs)} QA pairs from {len(files)} files")

    if args.output:
        from rag_slm_system.qa_generator.base import BaseQAGenerator

        BaseQAGenerator.save_pairs(pairs, args.output, args.format)
        print(f"Saved to {args.output} ({args.format} format)")

    if args.preview:
        print("\nPreview (first 5 pairs):")
        for i, p in enumerate(pairs[:5], 1):
            print(f"\n  [{i}] Q: {p.question}")
            print(f"      A: {p.answer[:200]}...")


def cmd_export(args: argparse.Namespace) -> None:
    """Export QA pairs for fine-tuning."""
    from rag_slm_system.pipeline import RAGPipeline
    from rag_slm_system.qa_generator.base import BaseQAGenerator

    pairs = BaseQAGenerator.load_pairs(args.input, args.input_format)
    print(f"Loaded {len(pairs)} QA pairs from {args.input}")

    pipeline = RAGPipeline()
    paths = pipeline.export_for_fine_tuning(args.output_dir, pairs=pairs, output_format=args.format)

    print(f"\nExported to {args.output_dir}:")
    for key, path in paths.items():
        print(f"  {key}: {path}")


def cmd_train(args: argparse.Namespace) -> None:
    """Fine-tune an SLM on QA pairs."""
    from rag_slm_system.fine_tuning.trainer import SLMTrainer
    from rag_slm_system.qa_generator.base import BaseQAGenerator

    config = _load_config(args)
    if args.model:
        config.fine_tuning.model_name = args.model

    pairs = BaseQAGenerator.load_pairs(args.input, args.input_format)
    print(f"Loaded {len(pairs)} QA pairs")

    trainer = SLMTrainer(config.fine_tuning)
    metrics = trainer.prepare_and_train(pairs, output_dir=args.output_dir)

    print("\nTraining complete!")
    print(f"  Model saved to: {metrics['model_path']}")
    print(f"  Training loss: {metrics['train_loss']:.4f}")


def cmd_agent(args: argparse.Namespace) -> None:
    """Run the Gemini ADK RAG agent interactively."""
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    from rag_slm_system.agent import create_rag_agent

    config = _load_config(args)
    agent = create_rag_agent(config, model=args.model)

    runner = InMemoryRunner(agent=agent, app_name="rag_slm_agent")
    session = runner.session_service.create_session(
        app_name="rag_slm_agent", user_id="user"
    )

    print("RAG Agent ready. Type 'quit' to exit.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        content = types.Content(
            role="user", parts=[types.Part(text=user_input)]
        )

        response_text = ""
        for event in runner.run(
            user_id="user",
            session_id=session.id,
            new_message=content,
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        response_text += part.text

        print(f"\nAgent: {response_text}\n")


def _load_config(args: argparse.Namespace) -> RAGConfig:
    if hasattr(args, "config") and args.config:
        with open(args.config) as f:
            _ = json.load(f)
        return RAGConfig()
    return RAGConfig()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rag-slm",
        description="RAG + SLM Fine-Tuning System",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--config", help="Path to config JSON file")
    subparsers = parser.add_subparsers(dest="command")

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Ingest documents")
    p_ingest.add_argument("files", nargs="+", help="Files or directories to ingest")
    p_ingest.add_argument("--save-to", help="Directory to save vector store")

    # query
    p_query = subparsers.add_parser("query", help="Query the RAG system")
    p_query.add_argument("question", help="Question to ask")
    p_query.add_argument("--load-from", help="Directory to load vector store from")
    p_query.add_argument("--top-k", type=int, default=5)

    # generate-qa
    p_qa = subparsers.add_parser("generate-qa", help="Generate QA pairs")
    p_qa.add_argument("files", nargs="+", help="Files or directories")
    p_qa.add_argument("--num-pairs", type=int, default=3)
    p_qa.add_argument("--output", "-o", help="Output file path")
    p_qa.add_argument("--format", default="alpaca", choices=["alpaca", "sharegpt", "chat_ml"])
    p_qa.add_argument("--preview", action="store_true")

    # export
    p_export = subparsers.add_parser("export", help="Export QA pairs for fine-tuning")
    p_export.add_argument("input", help="Input QA pairs JSON file")
    p_export.add_argument("--output-dir", "-o", default="./training_data")
    p_export.add_argument("--format", default="alpaca", choices=["alpaca", "sharegpt", "chat_ml"])
    p_export.add_argument("--input-format", default="alpaca")

    # train
    p_train = subparsers.add_parser("train", help="Fine-tune an SLM")
    p_train.add_argument("input", help="Input QA pairs JSON file")
    p_train.add_argument("--output-dir", "-o", default="./fine_tuned_model")
    p_train.add_argument("--model", help="Model name to fine-tune")
    p_train.add_argument("--input-format", default="alpaca")

    # agent
    p_agent = subparsers.add_parser("agent", help="Run interactive RAG agent")
    p_agent.add_argument("--model", default="gemini-2.0-flash")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "ingest": cmd_ingest,
        "query": cmd_query,
        "generate-qa": cmd_generate_qa,
        "export": cmd_export,
        "train": cmd_train,
        "agent": cmd_agent,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
