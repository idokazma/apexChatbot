# Harel Insurance Chatbot

## Project Overview
Production-grade RAG customer support chatbot for Harel Insurance covering 8 insurance domains:
Car, Life, Travel, Health, Dental, Mortgage, Business, Apartment.

## Tech Stack
- **LLM (inference):** Gemma via Ollama (local)
- **LLM (preprocessing):** Claude API (Anthropic)
- **Vector DB:** Milvus (docker-compose)
- **Agent Framework:** LangGraph
- **Document Processing:** Docling
- **Embeddings:** intfloat/multilingual-e5-large
- **API:** FastAPI
- **UI:** Custom HTML/CSS/JS chat interface
- **Evaluation:** RAGAS + custom citation scorer

## Development
- Use a virtual environment: `make setup`
- Start infrastructure: `make infra-up` (Milvus via Docker)
- Run data pipeline: `make pipeline`
- Start API server: `make serve`
- Run tests: `make test`
- Lint: `make lint`

## Project Structure
- `config/` - Settings, domain registry, prompt templates
- `data_pipeline/` - Scraping, parsing, chunking, embedding, Milvus storage
- `retrieval/` - Hybrid search, reranking, query processing
- `agent/` - LangGraph state graph with 7 nodes
- `llm/` - Ollama and Claude client wrappers
- `api/` - FastAPI endpoints
- `ui/` - Chat interface (RTL Hebrew/English)
- `evaluation/` - RAGAS harness, citation scorer, metrics
- `scripts/` - CLI entry points

## Conventions
- Follow PEP 8 / ruff linting
- Use type hints for function signatures
- Pydantic models for data schemas
- loguru for logging
- Keep functions focused and small
- All answers must include citations back to source documents
