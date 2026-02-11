# CLAUDE.md

Operational notes for contributors and coding agents working in this repository.

## Scope

This project is a Streamlit app for physiological signal QC and export.

## Entry Points

- `app.py`: Streamlit application
- `config.py`: runtime paths/defaults
- `metrics/`: signal processing pipelines
- `utils/`: shared I/O/export/PMU helpers

## Local Commands

Install dependencies:

```bash
uv sync
```

Run app:

```bash
uv run streamlit run app.py
```

Run checks:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy metrics/ algorithms/ utils/ --ignore-missing-imports
```

Useful make targets:

```bash
make run
make lint
make format
make test
make check
```

## PMU Notes

- App PMU integration logic is in `utils/pmu_integration.py`.
- Session-B PMU diagnostics are in `scripts/diagnostics/` and `scripts/pmu/`.
- Use `scripts/diagnostics/diagnose_pmu_integration.py` first when PMU enrichment fails.

## Repository Hygiene

- Keep root folder limited to core app/config/build files.
- Put operational scripts under `scripts/` subfolders.
- Do not keep backup/temp files in the repo (`*.backup`, `*.working_backup`, `*.bak`, etc.).
- Update folder `README.md` files when moving/adding scripts.
