# Phase Completion, Y-Axis Labels, and Phase Toggle

**Date:** 2026-02-13
**Files modified:** `config.py`, `metrics/rsp.py`, `metrics/ecg.py`, `app.py`, `utils/export.py`

---

## Overview

This update adds three features to physio-qc:

1. **Respiratory phase & cycle completion** — binary inhalation/exhalation phase, within-phase completion, and full-cycle completion (0-1 sawtooth preserving I:E ratio)
2. **Cardiac phase & cycle completion** — ventricular systole/diastole phase, within-phase completion, and full-cycle completion (R-peak to R-peak)
3. **Y-axis titles** on all signal plots (previously missing on ECG, RSP, PPG, BP, Spirometer)
4. **Show Phase toggle** — checkbox in the RSP and ECG tabs to overlay cycle completion on the signal plot

---

## What Was Added

### 1. `config.py` — Y-Axis Label Constants

A `Y_AXIS_LABELS` dictionary (31 entries) was added before the UI THEME section. It maps every subplot row across all signal types to a descriptive y-axis label. This centralises label strings so they can be edited in one place.

Covers: ECG, RSP, Spirometer, PPG, BP, ETCO2, ETO2, SpO2.

### 2. `metrics/rsp.py` — `compute_rsp_phase()`

New function added before `process_rsp()`.

**What it does:**

- Calls `nk.rsp_phase()` with detected peaks and troughs to get:
  - `RSP_Phase` — binary: 1 = inhalation, 0 = exhalation
  - `RSP_Phase_Completion` — 0-1 within each phase half
- Computes **full cycle completion** (0→1 sawtooth) for each respiratory cycle (trough to trough):
  - Calculates the inhalation proportion of each cycle (`inhalation_samples / cycle_length`)
  - Maps inhalation completion to `[0, inhalation_proportion]`
  - Maps exhalation completion to `[inhalation_proportion, 1]`
  - This preserves the natural I:E ratio per breath
- Handles the last incomplete cycle using average cycle duration

**Returns:** `rsp_phase`, `rsp_phase_completion`, `rsp_cycle_completion`

**Integration:** Called automatically from `process_rsp()` after breathing rate computation. Always runs — no user parameter needed.

### 3. `metrics/ecg.py` — `compute_ecg_phase()`

New function added before `process_ecg()`.

**What it does:**

- Calls `nk.ecg_phase()` with R-peaks and sampling rate to get:
  - `ECG_Phase_Ventricular` — binary: 1 = systole, 0 = diastole
  - `ECG_Phase_Completion_Ventricular` — 0-1 within each phase
- Separates systole and diastole completions into distinct arrays
- Computes **full cardiac cycle completion** (0→1 linspace) from R-peak[i] to R-peak[i+1]
- Detects **diastole onsets** — sample indices where ventricular phase transitions from 1→0

**Returns:** `ecg_ventricular_phase`, `ecg_ventricular_completion`, `ecg_systole_completion`, `ecg_diastole_completion`, `ecg_cardiac_cycle_completion`, `ecg_diastole_onsets`

**Integration:** Called automatically from `process_ecg()` after HR calculation and quality metrics.

### 4. `app.py` — Plot Functions

#### `create_signal_plot()` (ECG / PPG)

- New `phase_data` parameter
- When provided, overlays a green dotted line (cycle completion, 0-1) on **row 2** (signal + peaks) using a **secondary y-axis** on the right side
- Added y-axis titles to all 3 rows using `config.Y_AXIS_LABELS`

#### `create_rsp_bp_plot()` (RSP / Spirometer / BP)

- New `phase_data` parameter
- When provided (RSP/Spirometer only, not BP), overlays cycle completion on **row 2** using a secondary y-axis (0-1 on right)
- Row 2 specs dynamically enable `secondary_y` when phase is active
- Added y-axis titles to all rows using `config.Y_AXIS_LABELS`, with prefix detection (`rsp_`, `spiro_`, `bp_`)

#### Design Decision: Overlay vs Separate Row

The phase completion is **overlaid on the signal+peaks subplot** (row 2) rather than added as a separate subplot row. This allows direct visual comparison between the phase algorithm output and the detected peaks/troughs for validation. The signal amplitude uses the left y-axis; the phase (0-1) uses the right y-axis.

### 5. `app.py` — UI Controls

#### RSP / Spirometer tabs (`render_rsp_like_tab`)

- Added **"Show Phase"** checkbox (default OFF) in the process column, below the Process button
- When toggled ON, passes `rsp_cycle_completion` from the result dict to `create_rsp_bp_plot()`

#### ECG tab

- Added **"Show Phase"** checkbox (default OFF) in col2, below "Calculate Quality"
- When toggled ON, passes `ecg_cardiac_cycle_completion` to `create_signal_plot()`

### 6. `app.py` — Gas Channel Y-Axis Labels

Replaced hardcoded y-axis title strings with `config.Y_AXIS_LABELS` lookups:

| Signal | Row 1 | Row 2 |
|--------|-------|-------|
| ETCO2 | `etco2_raw` → "CO2 (mmHg)" | `etco2_envelope` → "ETCO2 (mmHg)" |
| ETO2 | `eto2_raw` → "O2 (mmHg)" | `eto2_envelope` → "ETO2 (mmHg)" |
| SpO2 | `spo2_raw` → "SpO2 (%)" | `spo2_clean` → "SpO2 (%)" |

### 7. `app.py` — Export Tab Fix

The export tab was missing population of ECG, RSP, and PPG results into `results_dict`. Added:

```python
if st.session_state.ecg_result is not None:
    results_dict['ecg'] = st.session_state.ecg_result
    params_dict['ecg'] = st.session_state.ecg_params
# (same for rsp, ppg)
```

Without this, only BP data was being exported.

### 8. `utils/export.py` — Phase Columns in CSV and JSON

#### `create_combined_dataframe()`

New columns added when phase data is present:

| Column | Type | Source |
|--------|------|--------|
| `rsp_phase` | int8 | Binary inhalation (1) / exhalation (0) |
| `rsp_phase_completion` | float32 | Within-phase completion (0-1) |
| `rsp_cycle_completion` | float32 | Full cycle completion (0-1, preserving I:E) |
| `ecg_phase_ventricular` | int8 | Binary systole (1) / diastole (0) |
| `ecg_phase_completion_ventricular` | float32 | Within-phase completion (0-1) |
| `ecg_cardiac_cycle_completion` | float32 | Full cardiac cycle completion (0-1) |

#### `create_metadata_json()`

- Added `"PhaseComputed": true` to RSP and ECG metadata sections when phase data exists
- Appended new column names to the `Columns` list

---

## Verification Checklist

- [ ] Load a recording with ECG + RSP
- [ ] RSP tab: "Show Phase" checkbox appears, OFF by default
- [ ] Toggle ON → green dotted sawtooth (0-1) appears on signal plot (right y-axis)
- [ ] Toggle OFF → sawtooth disappears
- [ ] ECG tab: "Show Phase" checkbox appears, toggle shows cardiac cycle completion
- [ ] All plots: y-axis titles present on every subplot row
- [ ] ETCO2/ETO2/SpO2: y-axis titles still display correctly
- [ ] Export: CSV contains `rsp_phase`, `rsp_phase_completion`, `rsp_cycle_completion` columns
- [ ] Export: CSV contains `ecg_phase_ventricular`, `ecg_phase_completion_ventricular`, `ecg_cardiac_cycle_completion` columns
- [ ] Export: JSON metadata includes `PhaseComputed: true` for RSP and ECG
