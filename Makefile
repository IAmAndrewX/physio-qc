.PHONY: help install run clean lint format test update

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

install:  ## Install dependencies with uv
	uv sync

install-dev:  ## Install dependencies including dev tools
	uv sync --all-extras

run:  ## Run the Streamlit application
	uv run streamlit run app.py

clean:  ## Remove virtual environment and cache files
	rm -rf .venv
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

lint:  ## Run linter (ruff)
	uv run ruff check .

lint-fix:  ## Run linter and auto-fix issues
	uv run ruff check --fix .

format:  ## Format code with ruff
	uv run ruff format .

format-check:  ## Check code formatting without modifying
	uv run ruff format --check .

type-check:  ## Run type checker (mypy)
	uv run mypy metrics/ algorithms/ utils/ --ignore-missing-imports

test:  ## Run tests
	uv run pytest -v

test-cov:  ## Run tests with coverage report
	uv run pytest --cov=metrics --cov=algorithms --cov=utils --cov-report=html --cov-report=term

update:  ## Update all dependencies
	uv sync --upgrade

lock:  ## Update lock file without installing
	uv lock

export-requirements:  ## Export requirements.txt for compatibility
	uv pip freeze > requirements.txt

check:  ## Run all checks (lint, format, type-check)
	@uv run ruff check .
	@uv run ruff format --check .
	@uv run mypy metrics/ algorithms/ utils/ --ignore-missing-imports

setup:  ## First-time setup
	@command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
	@make install

dev:  ## Setup development environment
	@make install-dev
