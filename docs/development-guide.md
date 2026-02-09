# Development Guide

This guide covers how to set up and work with the physio-qc project using modern Python tooling.

## First-Time Setup

### Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

### Setup Project

```bash
uv sync
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
```

## Daily Development Workflow

### Running the Application

```bash
source .venv/bin/activate
streamlit run app.py
```

### Managing Dependencies

```bash
uv add package-name
uv add --dev package-name
uv add "package-name>=1.0.0,<2.0.0"
uv sync --upgrade
uv remove package-name
```

## Code Quality

```bash
uv sync --all-extras
uv run ruff check .
uv run ruff check --fix .
uv run ruff format .
uv run mypy metrics/ algorithms/ utils/
uv run pytest
uv run pytest --cov=metrics --cov=algorithms --cov=utils
```

## Project Structure

```
physio-qc/
├── pyproject.toml          # Project metadata and dependencies (PEP 621)
├── uv.lock                 # Locked dependency versions
├── .python-version         # Python version specification
├── .venv/                  # Virtual environment (created by uv sync)
├── app.py                  # Main Streamlit application
├── config.py               # Configuration
├── metrics/                # Signal processing modules
│   ├── ecg.py
│   ├── rsp.py
│   ├── ppg.py
│   └── blood_pressure.py
├── algorithms/             # Specialized algorithms
│   ├── bp_delineator.py
│   └── quality_detection.py
└── utils/                  # Utility functions
    ├── file_io.py
    ├── peak_editing.py
    └── export.py
```

## Why uv?

- 10-100x faster than pip
- Built-in dependency resolver
- Uses pyproject.toml (PEP 621)
- Lock file for reproducibility
- Single command replaces multiple steps

## Common Tasks

### Common Commands

```bash
rm -rf .venv && uv sync
uv pip freeze > requirements.txt
uv run streamlit run app.py
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: uv sync
      - name: Run tests
        run: uv run pytest
```

## Troubleshooting

```bash
source ~/.bashrc
uv python install 3.11
uv venv --python 3.11
rm uv.lock && uv sync
```

