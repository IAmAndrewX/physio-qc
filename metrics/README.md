# Metrics

Signal-specific processing modules used by `app.py`.

## Files

- `metrics/ecg.py`: ECG cleaning, R-peak detection, heart-rate derivation.
- `metrics/rsp.py`: Respiration cleaning, breath peak/trough detection, respiratory-rate derivation.
- `metrics/ppg.py`: PPG peak detection and pulse-rate derivation.
- `metrics/blood_pressure.py`: BP signal processing and SBP/MAP/DBP derivation.
- `metrics/etco2.py`: End-tidal CO2 processing (`process_etco2` wrapper over envelope extraction).
- `metrics/eto2.py`: End-tidal O2 processing (`process_eto2` wrapper over envelope extraction).
- `metrics/spo2.py`: SpO2 cleaning and desaturation event detection.

Each module exposes a main `process_*` entry point plus helper functions.
