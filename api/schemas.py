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
    page_number: int | None = 0
    source_file_path: str = ""


class ChatResponse(BaseModel):
    """Chat endpoint response body."""

    answer: str
    citations: list[Citation] = []
    domain: str | None = None
    confidence: float = 0.0
    conversation_id: str = ""
    language: str = "he"
    retrieval_mode: str = "rag"
    navigation_path: dict = {}


# --------------- OpenAI-Compatible Completions Schemas ---------------


class CompletionsChatMessage(BaseModel):
    """A single message in the completions request."""

    role: str
    content: str


class CompletionsRequest(BaseModel):
    """OpenAI-compatible chat completions request."""

    model: str = "harel-chatbot"
    messages: list[CompletionsChatMessage]
    max_tokens: int | None = 512
    temperature: float | None = 0.1
    stream: bool | None = False


class CompletionsSource(BaseModel):
    """A source reference in the completions response."""

    link: str
    page: int


class CompletionsChoice(BaseModel):
    """A single choice in the completions response."""

    index: int
    text: str
    sources: list[CompletionsSource] = []
    finish_reason: str | None = None


class CompletionsResponse(BaseModel):
    """OpenAI-compatible chat completions response."""

    id: str
    object: str = "chat.completion"
    created: float
    model: str
    choices: list[CompletionsChoice]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    vector_db: bool
    ollama: bool
    embedding_model: bool
    collection_count: int = 0
    retrieval_mode: str = "rag"


# --------------- Admin Dashboard Schemas ---------------


class ComponentStatus(BaseModel):
    """Status of a single system component."""

    name: str
    status: str  # "online" | "offline" | "degraded"
    detail: str = ""


class SystemStatusResponse(BaseModel):
    """Full system status for admin dashboard."""

    overall: str  # "healthy" | "degraded" | "down"
    components: list[ComponentStatus]
    uptime_seconds: float = 0.0


class DomainDocCount(BaseModel):
    """Document count for a single domain."""

    domain: str
    domain_he: str
    count: int


class DocumentStatsResponse(BaseModel):
    """Document/collection statistics."""

    total_chunks: int
    domains: list[DomainDocCount]


class QueryStatsResponse(BaseModel):
    """Aggregate query statistics."""

    total_queries: int
    total_errors: int
    total_fallbacks: int
    avg_duration_ms: float
