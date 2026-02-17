"""Request and response schemas for the API."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Chat endpoint request body."""

    message: str = Field(..., min_length=1, max_length=2000)
    conversation_id: str | None = None
    language: str | None = None  # "he" | "en" | auto-detect


class Citation(BaseModel):
    """A single citation reference."""

    source_url: str = ""
    document_title: str = ""
    section: str = ""
    relevant_text: str = ""
    page_number: int = 0
    source_file_path: str = ""


class ChatResponse(BaseModel):
    """Chat endpoint response body."""

    answer: str
    citations: list[Citation] = []
    domain: str | None = None
    confidence: float = 0.0
    conversation_id: str = ""
    language: str = "he"


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    vector_db: bool
    ollama: bool
    embedding_model: bool
    collection_count: int = 0
