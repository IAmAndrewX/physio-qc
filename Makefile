.PHONY: help install run clean lint format test update

VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
PIP ?= $(VENV)/bin/pip
STREAMLIT ?= $(VENV)/bin/streamlit
PORT ?= 8501
RUFF ?= $(VENV)/bin/ruff
MYPY ?= $(VENV)/bin/mypy
PYTEST ?= $(VENV)/bin/pytest

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

install:  ## Create .venv and install runtime dependencies
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -e .

install-dev:  ## Create .venv and install runtime + dev dependencies
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -e ".[dev]"

run:  ## Run the Streamlit application (PORT=XXXX to override)
	@# Remove broken symlinks in static/ left over from previous sessions
	@find static/ -type l ! -exec test -e {} \; -delete 2>/dev/null || true
	$(STREAMLIT) run app.py --server.port $(PORT)

clean:  ## Remove virtual environment and cache files
	rm -rf .venv
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find static/ -type l ! -exec test -e {} \; -delete 2>/dev/null || true

lint:  ## Run linter (ruff)
	$(RUFF) check .

lint-fix:  ## Run linter and auto-fix issues
	$(RUFF) check --fix .

format:  ## Format code with ruff
	$(RUFF) format .

format-check:  ## Check code formatting without modifying
	$(RUFF) format --check .

type-check:  ## Run type checker (mypy)
	$(MYPY) metrics/ algorithms/ utils/ --ignore-missing-imports

test:  ## Run tests
	$(PYTEST) -v

test-cov:  ## Run tests with coverage report
	$(PYTEST) --cov=metrics --cov=algorithms --cov=utils --cov-report=html --cov-report=term

update:  ## Update all dependencies
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install --upgrade -e ".[dev]"

lock:  ## Export pinned requirements
	$(PIP) freeze > requirements.txt

export-requirements:  ## Export requirements.txt for compatibility
	$(PIP) freeze > requirements.txt

check:  ## Run all checks (lint, format, type-check)
	@$(RUFF) check .
	@$(RUFF) format --check .
	@$(MYPY) metrics/ algorithms/ utils/ --ignore-missing-imports

setup:  ## First-time setup
	@make install

dev:  ## Setup development environment
	@make install-dev
