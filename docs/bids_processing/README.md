# BIDS-Style Processing Documentation

This folder contains machine-readable processing and protocol documentation.

## Files

- `processing_report.md`: human-readable audit report.
- `processing_functions.tsv`: per-metric processing-function inventory.
- `processing_functions.json`: sidecar metadata for `processing_functions.tsv`.
- `task_protocols.tsv`: task procedure table, including participant-specific overrides.
- `task_protocols.json`: sidecar metadata for `task_protocols.tsv`.

## Notes

- Tables use tab-separated values and are intended to be easy to parse in Python/R.
- Participant-specific gas protocol override for `sub-00*` is documented here and implemented in `config.py` + `app.py`.
