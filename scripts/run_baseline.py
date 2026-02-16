"""CLI entry point for running GPT baseline evaluation."""

import argparse
from pathlib import Path

from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="Run GPT baseline evaluation")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model name")
    parser.add_argument(
        "--questions",
        default="evaluation/dataset/questions.json",
        help="Path to questions JSON",
    )
    parser.add_argument("--output", default="evaluation/reports", help="Output directory")
    args = parser.parse_args()

    from evaluation.baseline_eval import run_baseline

    questions_path = Path(args.questions)
    if not questions_path.exists():
        logger.error(f"Questions file not found: {questions_path}")
        return

    run_baseline(questions_path, Path(args.output), model=args.model)


if __name__ == "__main__":
    main()
