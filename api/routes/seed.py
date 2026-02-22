"""Temporary endpoint to upload seed data to the volume."""

import os
import tarfile
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException
from loguru import logger

router = APIRouter(prefix="/api", tags=["seed"])

SEED_TOKEN = os.getenv("SEED_TOKEN", "")
DATA_DIR = Path("/app/data")


@router.post("/upload-seed")
async def upload_seed(file: UploadFile = File(...), token: str = ""):
    """Upload a tar.gz containing chromadb/ and hierarchy/ to the volume."""
    if not SEED_TOKEN or token != SEED_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid seed token")

    if not file.filename or not file.filename.endswith(".tar.gz"):
        raise HTTPException(status_code=400, detail="File must be a .tar.gz")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = DATA_DIR / "seed-upload.tar.gz"

    try:
        # Save uploaded file
        content = await file.read()
        tmp_path.write_bytes(content)
        size_mb = len(content) / (1024 * 1024)
        logger.info(f"Received seed data: {size_mb:.1f} MB")

        # Extract
        with tarfile.open(tmp_path, "r:gz") as tar:
            tar.extractall(path=DATA_DIR)
        logger.info(f"Extracted seed data to {DATA_DIR}")

        # Verify
        has_chromadb = (DATA_DIR / "chromadb").exists()
        has_hierarchy = (DATA_DIR / "hierarchy").exists()

        return {
            "status": "ok",
            "size_mb": round(size_mb, 1),
            "chromadb": has_chromadb,
            "hierarchy": has_hierarchy,
        }
    finally:
        tmp_path.unlink(missing_ok=True)


@router.get("/seed-status")
async def seed_status():
    """Check if seed data is present on the volume."""
    chromadb_path = DATA_DIR / "chromadb"
    hierarchy_path = DATA_DIR / "hierarchy"
    return {
        "chromadb": chromadb_path.exists(),
        "hierarchy": hierarchy_path.exists(),
        "data_dir": str(DATA_DIR),
    }
