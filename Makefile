.PHONY: setup infra scrape parse chunk embed pipeline serve telegram eval quiz quiz-small gt-eval test lint format

# Setup
setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -e ".[dev]"
	playwright install chromium

# Infrastructure
infra-up:
	docker compose up -d

infra-down:
	docker compose down

# Data pipeline steps
scrape:
	python -m scripts.run_pipeline scrape

parse:
	python -m scripts.run_pipeline parse

chunk:
	python -m scripts.run_pipeline chunk

embed:
	python -m scripts.run_pipeline embed

# Full pipeline
pipeline:
	python -m scripts.run_pipeline all

# Serve
serve:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Telegram bot
telegram:
	python -m scripts.run_telegram

# Evaluation
eval:
	python -m scripts.run_eval

# Quizzer (automated stress test)
quiz:
	python -m scripts.run_quizzer

quiz-small:
	python -m scripts.run_quizzer -n 50

# Ground truth evaluation
gt-eval:
	python -m scripts.run_gt_eval

# Development
test:
	pytest tests/ -v

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .
