"""Orchestrator for the full data pipeline: scrape → parse → chunk → embed → store."""

import asyncio
import json
from pathlib import Path

from loguru import logger

from config.domains import DOMAINS
from config.settings import settings
from data_pipeline.chunker.semantic_chunker import chunk_parsed_documents
from data_pipeline.embedder.batch_embedder import embed_chunks
from data_pipeline.embedder.embedding_model import EmbeddingModel
from data_pipeline.parser.docling_parser import DoclingParser
from data_pipeline.parser.metadata_extractor import enrich_metadata
from data_pipeline.scraper.aspx_scraper import AspxScraper
from data_pipeline.scraper.pdf_downloader import PdfDownloader
from data_pipeline.scraper.sitemap_crawler import crawl_all
from data_pipeline.store.milvus_client import MilvusClient


async def run_scrape(raw_dir: Path) -> dict:
    """Step 1: Crawl and scrape all domains."""
    logger.info("=== Step 1: Scraping ===")

    # Discover URLs
    manifests = await crawl_all(raw_dir)

    # Scrape HTML pages
    scraper = AspxScraper(output_dir=raw_dir)
    pdf_downloader = PdfDownloader(output_dir=raw_dir)

    for domain_name, manifest in manifests.items():
        # Scrape pages
        await scraper.scrape_domain(domain_name, manifest["pages"])

        # Download PDFs
        if manifest["pdfs"]:
            await pdf_downloader.download_domain_pdfs(domain_name, manifest["pdfs"])

    return manifests


def run_parse(raw_dir: Path, parsed_dir: Path) -> list[dict]:
    """Step 2: Parse scraped content with Docling."""
    logger.info("=== Step 2: Parsing ===")

    parser = DoclingParser()
    all_parsed: list[dict] = []

    for domain_name in DOMAINS:
        domain_raw = raw_dir / domain_name

        # Load scrape results
        scrape_results_path = domain_raw / "scrape_results.json"
        if scrape_results_path.exists():
            scrape_results = json.loads(scrape_results_path.read_text())
        else:
            scrape_results = []

        # Add PDF files
        pdf_dir = domain_raw / "pdfs"
        if pdf_dir.exists():
            for pdf_file in pdf_dir.glob("*.pdf"):
                scrape_results.append({
                    "file_path": str(pdf_file),
                    "url": "",  # URL from manifest if available
                    "title": pdf_file.stem,
                })

        # Parse all files
        parsed_docs = parser.parse_domain(domain_name, domain_raw, parsed_dir, scrape_results)

        # Enrich metadata
        for doc in parsed_docs:
            enrich_metadata(doc)

        all_parsed.extend(parsed_docs)

    logger.info(f"Total parsed documents: {len(all_parsed)}")
    return all_parsed


def run_chunk(parsed_dir: Path, chunks_dir: Path) -> list:
    """Step 3: Chunk parsed documents."""
    logger.info("=== Step 3: Chunking ===")

    # Load all parsed documents
    all_parsed = []
    for json_file in parsed_dir.rglob("*.json"):
        doc = json.loads(json_file.read_text(encoding="utf-8"))
        all_parsed.append(doc)

    chunks = chunk_parsed_documents(all_parsed)

    # Save chunks to disk
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunks_data = [chunk.model_dump() for chunk in chunks]
    chunks_file = chunks_dir / "all_chunks.json"
    chunks_file.write_text(json.dumps(chunks_data, ensure_ascii=False, indent=2))
    logger.info(f"Saved {len(chunks)} chunks to {chunks_file}")

    return chunks


def run_embed_and_store(chunks_dir: Path) -> int:
    """Step 4: Embed chunks and store in Milvus."""
    logger.info("=== Step 4: Embedding & Storing ===")

    from data_pipeline.chunker.chunk_models import Chunk

    # Load chunks
    chunks_file = chunks_dir / "all_chunks.json"
    chunks_data = json.loads(chunks_file.read_text(encoding="utf-8"))
    chunks = [Chunk(**c) for c in chunks_data]

    # Embed
    embedding_model = EmbeddingModel()
    embedded = embed_chunks(chunks, embedding_model)

    # Store in Milvus
    client = MilvusClient()
    client.connect()
    client.create_collection(drop_existing=True)
    count = client.insert_chunks(embedded)
    client.load_collection()

    logger.info(f"Pipeline complete. {count} chunks stored in Milvus.")
    return count


async def run_full_pipeline():
    """Run the complete data pipeline end-to-end."""
    raw_dir = settings.raw_data_dir
    parsed_dir = settings.parsed_data_dir
    chunks_dir = settings.chunks_data_dir

    # Step 1: Scrape
    await run_scrape(raw_dir)

    # Step 2: Parse
    run_parse(raw_dir, parsed_dir)

    # Step 3: Chunk
    run_chunk(parsed_dir, chunks_dir)

    # Step 4: Embed & Store
    run_embed_and_store(chunks_dir)

    logger.info("=== Full pipeline complete ===")
