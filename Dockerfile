FROM python:3.11-slim

WORKDIR /app

# System deps for chromadb (sqlite) and sentence-transformers
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install only runtime dependencies (skip pipeline/eval/voice packages)
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy application code and UI
COPY config/ config/
COPY agent/ agent/
COPY api/ api/
COPY llm/ llm/
COPY retrieval/ retrieval/
COPY ui/ ui/
COPY quizzer/ quizzer/
COPY scripts/ scripts/

# Data is expected to be mounted as a volume at /app/data
# containing chromadb/ and hierarchy/ subdirectories

ENV PORT=8000

EXPOSE ${PORT}

CMD python -m uvicorn api.main:app --host 0.0.0.0 --port ${PORT}
