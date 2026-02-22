#!/bin/bash
# Wait briefly for volume mount
sleep 2

echo "Checking /app/data contents:"
ls -la /app/data/ 2>/dev/null || echo "/app/data does not exist"

if [ -d "/app/data/chromadb" ] && [ -d "/app/data/hierarchy" ]; then
    echo "Volume data found, starting server."
else
    echo "WARNING: Volume is missing data (chromadb/hierarchy). Use /api/upload-seed to seed."
fi

exec python -m uvicorn api.main:app --host 0.0.0.0 --port ${PORT}
