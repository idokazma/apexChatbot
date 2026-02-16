.PHONY: setup infra scrape parse chunk embed pipeline serve eval test lint

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

# Evaluation
eval:
	python -m scripts.run_eval

# Development
test:
	pytest tests/ -v

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .
