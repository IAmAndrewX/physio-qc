# Utils

Shared helpers used across app tabs and scripts.

## Files

- `utils/file_io.py`: Data discovery/loading and signal mapping.
- `utils/export.py`: Export CSV/JSON outputs and metadata.
- `utils/peak_editing.py`: Peak/trough edit bookkeeping and validation.
- `utils/pmu_integration.py`: PMU parsing/matching logic for session enrichment.
- `utils/conversions.py`: Gas channel unit conversions.

Utility modules should stay stateless and reusable from both app and scripts.
