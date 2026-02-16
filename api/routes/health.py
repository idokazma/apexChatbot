"""Health check endpoint."""

from fastapi import APIRouter

from api.dependencies import resources
from api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Check the health of all system components."""
    milvus_ok = False
    ollama_ok = False
    embedding_ok = False
    count = 0

    try:
        if resources.milvus_client:
            count = resources.milvus_client.get_count()
            milvus_ok = True
    except Exception:
        pass

    try:
        if resources.ollama_client:
            ollama_ok = resources.ollama_client.is_available()
    except Exception:
        pass

    try:
        embedding_ok = resources.embedding_model is not None
    except Exception:
        pass

    all_ok = milvus_ok and ollama_ok and embedding_ok

    return HealthResponse(
        status="healthy" if all_ok else "degraded",
        milvus=milvus_ok,
        ollama=ollama_ok,
        embedding_model=embedding_ok,
        collection_count=count,
    )
