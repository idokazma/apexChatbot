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

COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENV PORT=8000

EXPOSE ${PORT}

ENTRYPOINT ["./entrypoint.sh"]
