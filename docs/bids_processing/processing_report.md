# Physio QC Processing Report

This report documents the current per-metric processing pipelines, method options, NeuroKit2 usage, and task protocol handling.

## Scope

- Repository: `physio-qc`
- Reviewed modules:
  - `metrics/ecg.py`
  - `metrics/rsp.py`
  - `metrics/ppg.py`
  - `metrics/blood_pressure.py`
  - `metrics/etco2.py`
  - `metrics/eto2.py`
  - `metrics/spo2.py`
  - `app.py`
  - `config.py`

## Consistency updates applied

1. Main entry-point naming consistency:
   - Added `process_etco2()` in `metrics/etco2.py` (wrapper over `extract_etco2_envelope()`).
   - Added `process_eto2()` in `metrics/eto2.py` (wrapper over `extract_eto2_envelope()`).
   - Updated `app.py` to call `process_etco2()` and `process_eto2()`.

2. Output key consistency (backward-compatible aliases):
   - ECG/RSP/PPG/BP now include `raw_signal` and `cleaned_signal` aliases.
   - ETCO2/ETO2 now include `raw` and `clean` aliases (`clean` maps to envelope).
   - SpO2 now includes `raw` and `clean` aliases.
   - Existing keys used by UI/export were retained.

3. Parameter key consistency:
   - ECG/RSP/PPG pipelines now accept `cleaning_method` as alias for `method`.

4. NeuroKit2 alignment where appropriate:
   - ECG custom cleaning now explicitly applies NeuroKit2 powerline filtering (`60 Hz` default, configurable 50/60).
   - SpO2 low-pass cleaning now uses `nk.signal_filter()`.

5. Functional bug fix in SpO2 events:
   - `desaturation_drop` (`min_drop`) is now enforced against a local pre-event baseline.

## Per-metric pipeline summary

## ECG

- Pipeline function: `process_ecg()`
- Cleaning: `clean_ecg()`
  - Default: `method='neurokit'`
  - NeuroKit2: `nk.ecg_clean(...)`
  - Custom mode: `nk.signal_filter(...)` + explicit NK powerline step
- Peaks: `detect_r_peaks()` via `nk.ecg_peaks(...)`
- Rate: `calculate_hr()` via `nk.ecg_rate(...)`
- Quality (optional): `calculate_ecg_quality()` via `nk.ecg_quality(...)`
- Powerline: explicitly handled (default 60 Hz)

## RSP

- Pipeline function: `process_rsp()`
- Cleaning: `clean_rsp()`
  - Default: `method='khodadad2018'`
  - NeuroKit2: `nk.rsp_clean(...)`
  - Custom mode: `nk.signal_filter(...)`
- Peaks/troughs: `detect_breath_peaks()` via `nk.rsp_peaks(...)`
- Rate: `calculate_breathing_rate()` via `nk.signal_rate(...)`
- Powerline: not explicitly applied (not typically needed for respiration-band processing)

## PPG

- Pipeline function: `process_ppg()`
- Cleaning: `clean_ppg()`
  - Default: `method='elgendi'`
  - NeuroKit2: `nk.ppg_clean(...)`
  - Custom mode: `nk.signal_filter(...)`
- Peaks: `detect_ppg_peaks()` via `nk.ppg_peaks(...)`
- Quality: `calculate_ppg_quality()` via `nk.ppg_quality(...)`
- Rate: `calculate_hr_from_ppg()` via `nk.signal_rate(...)`
- Powerline: no explicit dedicated stage (cleaning/band-limits typically remove high-frequency mains noise)

## Blood Pressure

- Pipeline function: `process_bp()`
- Filtering: `filter_bp()`
  - Default: Bessel low-pass path (`bessel_25hz`) for delineation compatibility
  - Alternative: Butterworth low-pass
  - Custom: `nk.signal_filter(...)`
- Peaks/troughs: `detect_bp_peaks()`
  - `delineator` (custom algorithm)
  - `prominence` (scipy)
- Metrics: `calculate_bp_metrics()` (4 Hz interpolation + MAP)
- Powerline: not explicitly applied (BP pipeline is low-pass dominant)

## ETCO2

- Pipeline function: `process_etco2()`
- Core extraction: `extract_etco2_envelope()`
- Detection methods:
  - `detect_peaks_diff()` (derivative/curvature)
  - `detect_peaks_prominence()` (scipy)
- Library basis: scipy/numpy (no direct NeuroKit2 ETCO2-specific equivalent used)
- Output: peak indices + interpolated ETCO2 envelope

## ETO2

- Pipeline function: `process_eto2()`
- Core extraction: `extract_eto2_envelope()`
- Detection methods:
  - `detect_troughs_diff()` (derivative/curvature)
  - `detect_troughs_prominence()` (scipy on inverted signal)
- Library basis: scipy/numpy (no direct NeuroKit2 ETO2-specific equivalent used)
- Output: trough indices + interpolated ETO2 envelope

## SpO2

- Pipeline function: `process_spo2()`
- Cleaning: `clean_spo2()`
  - `lowpass`: `nk.signal_filter(...)`
  - `savgol`: scipy Savitzky-Golay
  - `none`: passthrough
- Events: `detect_desaturation_events()`
  - Threshold + minimum duration
  - Minimum drop now enforced from local pre-event baseline
- Metrics: `calculate_spo2_metrics()`

## Task protocols and overlays

Task event overlays are driven by `config.py` and rendered in `app.py`.

- Default task events: `TASK_EVENTS`
- Alias mapping: `TASK_EVENT_ALIASES`
- Participant-specific overrides: `TASK_EVENTS_PARTICIPANT_OVERRIDES`

### sub-00 gas override

Implemented a gas-task override for participant IDs matching `sub-00*`.

- Default gas protocol is still used for non-`sub-00*` participants.
- `sub-00*` gas overlay now follows the operator table sequence:
  - Repeated cycle with simple labels: `Air / Hypercapnia / Air / Hypoxia / ...`.
  - Uses 1140 s sequence end (`Gas Off`) from the screenshot protocol.

## BIDS-style artifacts in this folder

- `processing_functions.tsv`
- `processing_functions.json`
- `task_protocols.tsv`
- `task_protocols.json`

These are intended as machine-readable companions to this narrative report.
