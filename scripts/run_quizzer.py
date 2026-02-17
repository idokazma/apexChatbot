"""CLI entry point for running the quizzer module."""

import argparse
import sys

from loguru import logger


def main():
    parser = argparse.ArgumentParser(
        description="Run the automated quizzer against the chatbot API.",
    )
    parser.add_argument(
        "-n", "--num-questions",
        type=int,
        default=1000,
        help="Number of questions to generate and test (default: 1000).",
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
        default="quizzer/reports",
        help="Directory to save reports (default: quizzer/reports).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="API request timeout in seconds (default: 60).",
    )
    parser.add_argument(
        "--docs-per-question",
        type=int,
        default=2,
        help="Number of source documents per question (default: 2).",
    )
    parser.add_argument(
        "--no-intermediate",
        action="store_true",
        help="Disable saving intermediate results.",
    )

    args = parser.parse_args()

    from quizzer.runner import QuizRunConfig, run_quiz

    config = QuizRunConfig(
        num_questions=args.num_questions,
        docs_per_question=args.docs_per_question,
        api_base_url=args.api_url,
        api_timeout=args.timeout,
        output_dir=args.output_dir,
        save_intermediate=not args.no_intermediate,
    )

    logger.info(f"Starting quizzer with {config.num_questions} questions")
    logger.info(f"API: {config.api_base_url} | Output: {config.output_dir}")

    result = run_quiz(config)

    if result.scores:
        avg = sum(s.overall_score for s in result.scores) / len(result.scores)
        logger.info(f"Quiz complete: avg_score={avg:.3f}")
        logger.info(f"Report: {config.output_dir}/quiz_report.html")
    else:
        logger.error("No scores collected. Check that the API is running and documents are loaded.")
        sys.exit(1)


if __name__ == "__main__":
    main()
