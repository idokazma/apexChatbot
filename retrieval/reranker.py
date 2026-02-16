"""Cross-encoder reranking for retrieved documents."""

from loguru import logger
from sentence_transformers import CrossEncoder


class Reranker:
    """Reranks retrieved documents using a cross-encoder model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        logger.info(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """Rerank documents by relevance to the query.

        Args:
            query: User query.
            documents: List of result dicts from initial retrieval.
            top_k: Number of top documents to return.

        Returns:
            Top-k documents sorted by cross-encoder score.
        """
        if not documents:
            return []

        if len(documents) <= top_k:
            return documents

        # Create query-document pairs for the cross-encoder
        pairs = [(query, doc["content"]) for doc in documents]
        scores = self.model.predict(pairs)

        # Attach scores and sort
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        documents.sort(key=lambda x: x["rerank_score"], reverse=True)

        logger.debug(
            f"Reranked {len(documents)} docs -> top {top_k}. "
            f"Score range: {scores.min():.3f} to {scores.max():.3f}"
        )
        return documents[:top_k]
