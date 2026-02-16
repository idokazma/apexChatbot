"""Batch embedding of document chunks with progress tracking."""

from loguru import logger
from tqdm import tqdm

from data_pipeline.chunker.chunk_models import Chunk
from data_pipeline.embedder.embedding_model import EmbeddingModel


def embed_chunks(
    chunks: list[Chunk],
    embedding_model: EmbeddingModel | None = None,
    batch_size: int = 32,
) -> list[dict]:
    """Embed all chunks and return dicts ready for Milvus insertion.

    Args:
        chunks: List of Chunk objects to embed.
        embedding_model: Pre-loaded model, or None to create one.
        batch_size: Number of chunks to embed at once.

    Returns:
        List of dicts with 'chunk' and 'embedding' keys.
    """
    if not chunks:
        return []

    if embedding_model is None:
        embedding_model = EmbeddingModel()

    # Use content_with_context for embedding (includes metadata header)
    texts = [chunk.content_with_context for chunk in chunks]

    logger.info(f"Embedding {len(texts)} chunks in batches of {batch_size}...")
    embeddings = embedding_model.embed_documents(texts, batch_size=batch_size)

    results = []
    for chunk, embedding in zip(chunks, embeddings):
        results.append({
            "chunk": chunk,
            "embedding": embedding.tolist(),
        })

    logger.info(f"Embedded {len(results)} chunks successfully")
    return results
