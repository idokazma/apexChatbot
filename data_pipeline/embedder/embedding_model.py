"""Embedding model wrapper for multilingual document embeddings."""

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from config.settings import settings


class EmbeddingModel:
    """Wraps a SentenceTransformer model with query/passage prefixing for E5 models."""

    def __init__(self, model_name: str = settings.embedding_model):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded. Dimension: {self.dim}")

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed document passages with 'passage:' prefix for E5 models."""
        prefixed = [f"passage: {t}" for t in texts]
        embeddings = self.model.encode(
            prefixed,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a search query with 'query:' prefix for E5 models."""
        embedding = self.model.encode(
            f"query: {query}",
            normalize_embeddings=True,
        )
        return embedding

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        """Embed multiple search queries."""
        prefixed = [f"query: {q}" for q in queries]
        return self.model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
