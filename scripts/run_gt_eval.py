"""CLI entry point for running ground truth evaluation."""

import argparse
import sys

from loguru import logger


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the chatbot against the curated ground truth dataset.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="evaluation/dataset/ground_truth_test.json",
        help="Path to the ground truth JSON file (default: evaluation/dataset/ground_truth_test.json).",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the chatbot API (default: http://localhost:8000).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation/reports",
        help="Directory to save reports (default: evaluation/reports).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="API request timeout in seconds (default: 60).",
    )

    args = parser.parse_args()

    from evaluation.gt_evaluator import run_gt_evaluation

    logger.info("Starting ground truth evaluation")
    logger.info(f"Dataset: {args.dataset} | API: {args.api_url}")

    result = run_gt_evaluation(
        dataset_path=args.dataset,
        api_base_url=args.api_url,
        api_timeout=args.timeout,
        output_dir=args.output_dir,
    )

    if not result.results:
        logger.error("No results collected. Check that the API is running.")
        sys.exit(1)


if __name__ == "__main__":
    main()
