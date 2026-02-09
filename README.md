# Physio QC

Streamlit app for physiological signal quality control, manual editing, and export.

## What it supports

- ECG, RSP, PPG, Blood Pressure
- ETCO2, ETO2, SpO2
- Spirometer waveform (when present)
- Session B PMU integration (Siemens `.resp/.puls/.ext`) mapped to app `RSP/PPG` by selected task

## Quick start

```bash
uv sync
uv run streamlit run app.py
```

## Typical workflow

1. Select participant, session, and task.
2. Click **Load Data**.
3. Process available tabs.
4. Edit peaks/troughs where needed.
5. Export outputs.

## Data layout

ACQ input:

```text
BASE_DATA_PATH/
  sub-XXXX/
    ses-YY/
      sub-XXXX_ses-YY_task-<task>_physio.acq
```

PMU input used for Session B (if enabled in `config.py`):

```text
01_physio/
  sub-XXXX/
    ses-2/Scanner_physio|Scanner_Physio/
      *.resp *.puls *.ext
```

BIDS timing for PMU task segmentation:

```text
PMU_BIDS_BASE_PATH/sub-XXXX/ses-02/sub-XXXX_ses-02_scans.tsv
```

## Configuration

Edit `config.py`:

- `BASE_DATA_PATH`, `OUTPUT_BASE_PATH`
- signal processing defaults
- PMU integration settings (`PMU_*`)

## Docs

- `docs/README.md`
- `docs/quick-start.md`
- `docs/development-guide.md`
- `docs/installation-guide.md`
- `docs/etco2-eto2-integration-guide.md`

## License

MIT
