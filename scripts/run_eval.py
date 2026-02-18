"""CLI entry point for running evaluations."""

import argparse
from pathlib import Path

from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation")
    parser.add_argument(
        "--mode",
        choices=["rag", "agentic", "combined"],
        default=None,
        help="Retrieval mode (defaults to RETRIEVAL_MODE from .env)",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("evaluation/dataset/questions.json"),
        help="Path to evaluation questions JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/reports"),
        help="Output directory for reports",
    )
    args = parser.parse_args()

    from agent.graph import create_agent_for_mode
    from config.settings import settings
    from data_pipeline.embedder.embedding_model import EmbeddingModel
    from data_pipeline.store.vector_store import VectorStoreClient
    from evaluation.ragas_eval import run_evaluation
    from retrieval.reranker import Reranker

    mode = args.mode or settings.retrieval_mode

    # Initialize resources based on mode
    store = None
    embedding_model = None
    reranker = None

    if mode in ("rag", "combined"):
        store = VectorStoreClient()
        store.connect()
        embedding_model = EmbeddingModel()
        reranker = Reranker()

    agent = create_agent_for_mode(
        mode=mode,
        store=store,
        embedding_model=embedding_model,
        reranker=reranker,
        hierarchy_dir=settings.hierarchy_dir,
    )

    logger.info(f"Running evaluation in '{mode}' mode")

    # Run evaluation
    if not args.questions.exists():
        logger.error(f"Questions file not found: {args.questions}")
        logger.info("Create evaluation/dataset/questions.json with test questions first.")
        return

    run_evaluation(agent, args.questions, args.output)

    if store:
        store.disconnect()


if __name__ == "__main__":
    main()
