"""CLI entry point for running data pipeline steps."""

import argparse
import asyncio

from loguru import logger

from config.settings import settings


def main():
    parser = argparse.ArgumentParser(description="Run data pipeline steps")
    parser.add_argument(
        "step",
        nargs="?",
        default="all",
        choices=["scrape", "parse", "chunk", "enrich", "embed", "all"],
    )
    parser.add_argument(
        "--llm",
        default="auto",
        choices=["auto", "claude", "ollama", "gemini"],
        help="LLM mode for enrichment (default: auto)",
    )
    args = parser.parse_args()

    step = args.step
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

    if step in ("enrich", "all"):
        from data_pipeline.pipeline import run_enrich

        run_enrich(settings.chunks_data_dir, llm_mode=args.llm)

    if step in ("embed", "all"):
        from data_pipeline.pipeline import run_embed_and_store

        run_embed_and_store(settings.chunks_data_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
