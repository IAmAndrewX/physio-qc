# PMU Scripts

Standalone PMU utilities used outside the Streamlit app.

## Scripts

- `scripts/pmu/audit_pmu_availability.py`
  - Audits all `sub-*` folders for PMU file availability and basic parseability.
- `scripts/pmu/visualize_pmu_recording.py`
  - Builds a multi-panel diagnostic figure for one subject's PMU recording.
- `scripts/pmu/extract_pmu_scan.py`
  - Extracts PMU signals for one scan (or all scans) and writes parquet + plots.

## Defaults

By default, these scripts read runtime paths/sessions from `config.py`:

- `BASE_DATA_PATH` for physio files
- `PMU_BIDS_BASE_PATH` for BIDS files
- `PMU_PHYSIO_SESSION` and `PMU_BIDS_SESSION`
- `PMU_SAMPLING_RATE`

Output defaults to repo-local directories under `outputs/pmu/`.

## Usage

Audit availability:

```bash
./scripts/pmu/audit_pmu_availability.py
```

With overrides:

```bash
./scripts/pmu/audit_pmu_availability.py \
  --physio-dir /export02/projects/LCS/01_physio \
  --session ses-2
```

Visualize one subject:

```bash
./scripts/pmu/visualize_pmu_recording.py 1027
```

Extract one scan:

```bash
./scripts/pmu/extract_pmu_scan.py 1027 rest
```

Extract all scans for a subject:

```bash
./scripts/pmu/extract_pmu_scan.py 1027
```

## Related

- PMU integration diagnostics used by the app: `scripts/diagnostics/diagnose_pmu_integration.py`
- App PMU matching code: `utils/pmu_integration.py`
