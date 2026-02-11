# Development Guide

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run

```bash
source .venv/bin/activate
streamlit run app.py
```

## Static checks

```bash
source .venv/bin/activate
ruff check .
ruff format --check .
mypy metrics/ algorithms/ utils/ --ignore-missing-imports
```

Auto-fix formatting/lint:

```bash
source .venv/bin/activate
ruff check --fix .
ruff format .
```

## Tests

```bash
source .venv/bin/activate
pytest -v
```

The repository currently has limited automated UI coverage; Streamlit behavior is still verified manually.

## Useful make targets

```bash
make run
make lint
make format
make test
make check
```

## Script entry points

Diagnostics:

```bash
./scripts/diagnostics/diagnose_pmu_integration.py --participant sub-1027 --session ses-2 --task rest
```

PMU utilities:

```bash
./scripts/pmu/audit_pmu_availability.py
./scripts/pmu/visualize_pmu_recording.py 1027
./scripts/pmu/extract_pmu_scan.py 1027 rest
```
