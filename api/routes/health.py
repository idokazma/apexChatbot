"""Health check endpoint."""

from fastapi import APIRouter

from api.dependencies import resources
from api.schemas import HealthResponse
from config.settings import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Check the health of all system components."""
    vectordb_ok = False
    ollama_ok = False
    embedding_ok = False
    count = 0

    try:
        if resources.store:
            count = resources.store.get_count()
            vectordb_ok = True
    except Exception:
        pass

    try:
        if settings.inference_llm == "claude":
            ollama_ok = resources.ollama_client is not None
        elif resources.ollama_client:
            ollama_ok = resources.ollama_client.is_available()
    except Exception:
        pass

    try:
        embedding_ok = resources.embedding_model is not None
    except Exception:
        pass

    all_ok = vectordb_ok and ollama_ok and embedding_ok

    return HealthResponse(
        status="healthy" if all_ok else "degraded",
        vector_db=vectordb_ok,
        ollama=ollama_ok,
        embedding_model=embedding_ok,
        collection_count=count,
        retrieval_mode=settings.retrieval_mode,
    )
