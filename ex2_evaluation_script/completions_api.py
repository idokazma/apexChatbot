import asyncio
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import List, Literal, Optional

from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel

# Add project root to path so we can import our modules
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from api.dependencies import AppResources

app = FastAPI(title="Completions-Compatible REST API")

# Shared resources (agent, vector store, etc.)
resources = AppResources()

# Thread pool for running agent queries without blocking the event loop
_agent_executor = ThreadPoolExecutor(max_workers=4)


@app.on_event("startup")
def startup():
    resources.initialize()
    logger.info("Agent initialized and ready for evaluation")


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "mock-model"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.1
    stream: Optional[bool] = False

class Choice(BaseModel):
    index: int
    text: str
    finish_reason: Optional[str] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"]
    created: float
    model: str
    choices: List[Choice]

# ----------------------
# Endpoints
# ----------------------

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def question_endpoint_v1(request: ChatCompletionRequest):
    response = await process_completions_request(request)
    return response

@app.post("/chat/completions", response_model=ChatCompletionResponse)
async def question_endpoint(request: ChatCompletionRequest):
    response = await process_completions_request(request)
    return response

async def process_completions_request(request: ChatCompletionRequest) -> ChatCompletionResponse:
    # Extract the user message (last message with role "user")
    user_message = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_message = msg.content
            break

    logger.info(f"Processing: {user_message[:80]}...")

    # Build agent input (same format as our chat endpoint)
    agent_input = {
        "query": user_message,
        "messages": [],
        "rewritten_query": "",
        "detected_domains": [],
        "detected_language": "he",
        "navigation_path": {},
        "retrieved_documents": [],
        "graded_documents": [],
        "generation": "",
        "citations": [],
        "is_grounded": False,
        "retry_count": 0,
        "should_fallback": False,
        "quality_action": "",
        "quality_reasoning": "",
        "reasoning_trace": [],
    }

    # Run agent in thread pool to avoid blocking
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _agent_executor, partial(resources.agent.invoke, agent_input)
        )
        answer_text = result.get("generation", "")
    except Exception as exc:
        logger.error(f"Agent error: {exc}")
        answer_text = f"Error processing query: {exc}"

    return ChatCompletionResponse(
        id=uuid.uuid4().hex[:12],
        object="chat.completion",
        created=time.time(),
        model=request.model,
        choices=[
            Choice(
                index=0,
                text=answer_text,
                finish_reason="stop"
            )
        ]
    )

# ----------------------
# Run with:
# uvicorn ex2_evaluation_script.completions_api:app --host 0.0.0.0 --port 8000
# ----------------------
