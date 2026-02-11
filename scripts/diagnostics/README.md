# Diagnostics Scripts

## Files

- `scripts/diagnostics/diagnose_pmu_integration.py`
  - Replays the app PMU matching logic for one participant/session/task.
  - Prints scanner-folder checks, PMU file presence, scans.tsv task matches, and final matcher result.

## Example

```bash
./scripts/diagnostics/diagnose_pmu_integration.py \
  --participant sub-1027 --session ses-2 --task rest
```
