"""CLI entry point for running evaluations."""

from pathlib import Path

from loguru import logger


def main():
    from agent.graph import create_agent
    from data_pipeline.embedder.embedding_model import EmbeddingModel
    from data_pipeline.store.vector_store import VectorStoreClient
    from evaluation.ragas_eval import run_evaluation
    from retrieval.reranker import Reranker

    # Initialize resources
    store = VectorStoreClient()
    store.connect()

    embedding_model = EmbeddingModel()
    reranker = Reranker()

    agent = create_agent(
        store=store,
        embedding_model=embedding_model,
        reranker=reranker,
    )

    # Run evaluation
    questions_path = Path("evaluation/dataset/questions.json")
    output_dir = Path("evaluation/reports")

    if not questions_path.exists():
        logger.error(f"Questions file not found: {questions_path}")
        logger.info("Create evaluation/dataset/questions.json with test questions first.")
        return

    run_evaluation(agent, questions_path, output_dir)

    store.disconnect()


if __name__ == "__main__":
    main()
