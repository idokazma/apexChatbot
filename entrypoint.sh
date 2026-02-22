#!/bin/bash
mkdir -p /app/data

if [ -d "/app/data/chromadb" ] && [ -d "/app/data/hierarchy" ]; then
    echo "Volume data found, starting server."
else
    echo "WARNING: Volume is missing data (chromadb/hierarchy). Use /api/upload-seed to seed."
fi

exec python -m uvicorn api.main:app --host 0.0.0.0 --port ${PORT}
