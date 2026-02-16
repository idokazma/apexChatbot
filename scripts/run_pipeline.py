"""CLI entry point for running data pipeline steps."""

import asyncio
import sys

from loguru import logger

from config.settings import settings


def main():
    step = sys.argv[1] if len(sys.argv) > 1 else "all"

    logger.info(f"Running pipeline step: {step}")

    if step in ("scrape", "all"):
        from data_pipeline.pipeline import run_scrape

        asyncio.run(run_scrape(settings.raw_data_dir))

    if step in ("parse", "all"):
        from data_pipeline.pipeline import run_parse

        run_parse(settings.raw_data_dir, settings.parsed_data_dir)

    if step in ("chunk", "all"):
        from data_pipeline.pipeline import run_chunk

        run_chunk(settings.parsed_data_dir, settings.chunks_data_dir)

    if step in ("embed", "all"):
        from data_pipeline.pipeline import run_embed_and_store

        run_embed_and_store(settings.chunks_data_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
