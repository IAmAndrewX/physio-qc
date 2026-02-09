# PMU Processor

Extract and align Siemens physiological monitoring (PMU) data with fMRI scans.

## Quick Start

```bash
pip install numpy matplotlib pandas nibabel pyarrow

# Step 1: Visualize raw data
python visualize_pmu_raw.py <subject_id>

# Step 2: Extract specific scan
python pmu_processor.py <subject_id> <scan_name>
```

## What It Does

1. **Parses** Siemens PMU files (`.resp`, `.puls`, `.ext`)
2. **Detects** scans using volume marker clustering (TAPAS PhysIO method)
3. **Matches** scans to BIDS by duration and volume count
4. **Extracts** aligned physiological time series as parquet files

## Siemens PMU File Format

### File Types

**`.resp` (Respiratory belt)**
```
Format: Physiological signal (0-4095)
Sampling: 400 Hz
Contains: Respiratory amplitude values
```

**`.puls` (Pulse oximetry)**
```
Format: Physiological signal (0-4095)
Sampling: 400 Hz
Contains: Cardiac pulse amplitude values
```

**`.ext` (External triggers)**
```
Format: Binary time series (position = sample index)
Sampling: 400 Hz
Contains: Volume and slice acquisition markers
```

### `.ext` File Values

| Value | Meaning | Purpose |
|-------|---------|---------|
| `0` | No activity | Baseline (95% of data) |
| `1` | Slice trigger | Marks individual slice acquisition (TTL pulse) |
| `5000` | Volume marker | Marks fMRI volume acquisition |
| `5003` | Recording end | Appears once at end of recording |

**Critical understanding:**
- `5000` = **VOLUME marker** (one per fMRI volume)
- NOT scan start marker!
- Scans = groups of volumes separated by gaps >10s

### File Structure

All PMU files follow this structure:
```
<Header> 5002 <LOGVERSION> 6002
[Optional training data]
5002 <metadata> 6002
<DATA SECTION>
5003
<Footer>
```

**Data extraction:**
1. Skip everything before first `6002`
2. Extract everything between `6002` and `5003`
3. Parse data section based on file type

## Directory Structure

```
/export02/projects/LCS/
├── BIDS/
│   └── sub-{ID}/ses-02/
│       ├── sub-{ID}_ses-02_scans.tsv     # Scan timing
│       └── func/
│           └── *_task-*_bold.nii.gz      # fMRI data (for TR)
└── 01_physio/
    └── sub-{ID}/ses-2/Scanner_physio/ # OR Scanner_Physio
        ├── *.resp    # Respiratory signal
        ├── *.puls    # Pulse oximetry
        └── *.ext     # Volume/slice markers
```

**Required inputs:**
1. PMU files (`.resp`, `.puls`, `.ext`)
2. BIDS `scans.tsv` (for scan timing)
3. NIfTI file (for TR extraction)

## How It Works

### 1. Parse PMU Files

**`.ext` file:**
```python
Value 0:    Baseline (no activity)
Value 1:    Slice acquisition (48,422 slices)
Value 5000: Volume acquisition (1,281 volumes)

Slices per volume: 48,422 / 1,281 = 37.8 ✓ (multi-echo)
Average TR: 2,492s / 1,281 volumes = 1.95s ✓
```

**`.resp`/`.puls` files:**
```python
Values 0-4095: Physiological signal
Sampled at 400 Hz continuously
```

### 2. Detect Scans (TAPAS Method)

```python
# Find all volume markers
volume_times = [t for t, val in data if val == 5000]

# Calculate inter-volume intervals
intervals = np.diff(volume_times)

# Large gaps (>10s) = scan boundaries
gap_indices = np.where(intervals > 10)[0]

# Group consecutive volumes into scans
# Result: 3-6 real scans (not 1,281 individual volumes!)
```

### 3. Match to BIDS

```python
# Strategy 1: Match by time (if within tolerance)
# Strategy 2: Match by duration + n_volumes
# Strategy 3: Report mismatch clearly
```

### 4. Extract Window

```python
# Use first and last volume positions as boundaries
start_sample = volume_markers[scan_start_idx]
end_sample = volume_markers[scan_end_idx]

# Extract physiological data for this window
resp_segment = resp_data[start_sample:end_sample]
puls_segment = puls_data[start_sample:end_sample]
```

## Usage

### visualize_pmu_raw.py

**Purpose:** Diagnostic visualization - see everything before extracting

```bash
python visualize_pmu_raw.py 2008
```

**Creates:**
- 6-panel diagnostic plot showing:
  1. Full respiratory trace + volume markers + BIDS times
  2. Full pulse trace + volume markers + BIDS times
  3. Slice trigger timeline (raster)
  4. Volume marker timeline (raster)
  5. Inter-volume intervals (gaps = scan boundaries)
  6. Zoom into first volume

**Use this first** to verify:
- ✓ Volume markers detected correctly
- ✓ Scan boundaries identified
- ✓ BIDS times align with PMU recording

### pmu_processor.py

**Purpose:** Extract physiological data for a specific scan

```bash
python pmu_processor.py <subject_id> <scan_name> [--show-full]
scan_name: rest_echo, gas_echo, breath
# Examples
python pmu_processor.py 2008 rest_echo
python pmu_processor.py 2034 gas_echo --show-full
```
**Options:**
--show-full (optional)
What it does: Creates an ADDITIONAL plot showing the entire PMU session with your extracted scan highlighted.
Default behavior (without --show-full):



**Output files:**
```
outputs/sub-{ID}/
├── sub-{ID}_{scan}_data.parquet       # Time series
└── sub-{ID}_{scan}_scan.png           # Plot
```

**Parquet format:**
```python
import pandas as pd
df = pd.read_parquet('sub-2008_rest_echo_data.parquet')

# Columns:
#   time_sec: Time from scan start (seconds)
#   respiratory: Respiratory signal (0-4095)
#   pulse: Pulse oximetry signal (0-4095)
```

## Example Session

```bash
# Step 1: Visualize
$ python visualize_pmu_raw.py 2008

📍 MARKERS (.ext file):
   Volume markers (5000): 1281
   Slice triggers (1): 48422
   Slices per volume: ~37.8

🔍 SCAN DETECTION:
   Detected 3 scans:
   Scan 1:   33.8s -  533.3s (499.5s),  291 volumes, TR=1.722s
   Scan 2:  652.3s - 1804.7s (1152.4s), 670 volumes, TR=1.723s
   Scan 3: 1941.5s - 2491.0s (549.5s),  320 volumes, TR=1.722s

# Step 2: Extract
$ python pmu_processor.py 2008 rest_echo

🎯 Matching BIDS scan to PMU scan...
   ✓ Matched to Scan #1
   Volumes: 291 (BIDS: 291)
   Duration: 499.5s (BIDS: 500.5s)

✓ COMPLETE
   - sub-2008_rest_echo_data.parquet
   - sub-2008_rest_echo_scan.png
```

## Common Issues

### "Scan scheduled AFTER PMU recording ended"

**Symptom:**
```
⚠️  WARNING: BIDS scan at 3225s but PMU ends at 2492s
   Gap: 733s (12.2 min)
```

**Cause:**
- Wrong PMU files (different session)
- PMU recording stopped early
- BIDS timestamps incorrect

**Solution:**
```bash
# Check which scans ARE in the recording
python visualize_pmu_raw.py 2008

# Extract one of those scans instead
python pmu_processor.py 2008 rest_echo  # This one exists
```

### "No matching scan found"

**Symptom:**
```
❌ Could not find matching scan in PMU data
```

**Cause:**
- Duration or volume count doesn't match
- Clock sync issue between PMU and BIDS

**Solution:**
- Check diagnostic plot for scan properties
- Verify BIDS metadata is correct
- Extract by scan order if properties don't match

## Data Analysis

### Load extracted data

```python
import pandas as pd
import numpy as np

# Load physiological data
df = pd.read_parquet('sub-2008_rest_echo_data.parquet')

# df.columns: ['time_sec', 'respiratory', 'pulse']
# df.shape: (199810, 3)  # 400Hz * 499.5s
```

### Downsample to fMRI TR

```python
# Align with fMRI volumes (TR = 1.72s)
tr = 1.72  # seconds
n_samples_per_tr = int(tr * 400)  # 688 samples

# Average over each TR
n_volumes = len(df) // n_samples_per_tr
resp_per_volume = [
    df['respiratory'][i*n_samples_per_tr:(i+1)*n_samples_per_tr].mean()
    for i in range(n_volumes)
]

# Now resp_per_volume[i] corresponds to fMRI volume i
```

### Calculate respiratory rate

```python
from scipy.signal import find_peaks

# Detect respiratory peaks (inhalations)
peaks, _ = find_peaks(df['respiratory'], distance=200, prominence=500)

# Calculate breathing rate
duration_min = df['time_sec'].max() / 60
resp_rate = len(peaks) / duration_min  # breaths per minute

print(f"Respiratory rate: {resp_rate:.1f} breaths/min")
```

### Calculate heart rate

```python
# Detect cardiac peaks
peaks, _ = find_peaks(df['pulse'], distance=120, prominence=300)

# Calculate heart rate
hr = len(peaks) / duration_min  # beats per minute

print(f"Heart rate: {hr:.1f} bpm")
```

## Troubleshooting

**Always start with visualization:**
```bash
python visualize_pmu_raw.py <subject_id>
```

This shows:
- ✓ How many scans detected
- ✓ Where BIDS times are relative to PMU
- ✓ Volume markers and scan boundaries

**If extraction fails:**
1. Check diagnostic plot
2. Verify PMU files match BIDS session
3. Extract a scan that EXISTS in the recording

## Requirements

```bash
pip install numpy matplotlib pandas nibabel pyarrow
```

**Python 3.7+** required

## Limitations

1. **No RETROICOR regressors** - Physiological noise correction not implemented
2. **No cardiac peak detection** - Use scipy.signal.find_peaks for now
3. **No quality metrics** - No SNR, artifact detection, or validation
4. **Fixed paths** - Hardcoded to LCS project structure
5. **No batch processing** - One subject/scan at a time

## References

- [TAPAS PhysIO Toolbox](https://github.com/translationalneuromodeling/tapas/tree/master/PhysIO)
- Kasper et al. (2017). PhysIO Toolbox. DOI: 10.1016/j.jneumeth.2016.10.019
- Siemens PMU file format (400Hz sampling, VB/VD systems)