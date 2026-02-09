# ETCO2/ETO2 Integration - Installation Complete

## ✅ What's Been Completed

### 1. Core Processing Modules (100%)
- **`metrics/etco2.py`** - Complete ETCO2 processing with derivative and prominence methods
- **`metrics/eto2.py`** - Complete ETO2 processing with derivative and prominence methods
- Both modules fully tested and documented

### 2. Configuration Updates (100%)
- **`config.py`** updated with:
  - ETCO2/ETO2 signal patterns for auto-detection
  - Method lists and descriptions
  - Default parameter dictionaries
  - All configurable options exposed

### 3. Session State Integration (100%)
- **`app.py`** updated with:
  - ETCO2/ETO2 imports added
  - Session state initialization for both signals
  - Zoom range tracking
  - Result storage
  - Tab detection in signal list

### 4. UI Tab Code (100%)
- **Complete tab implementation** in `ETCO2_ETO2_TAB_CODE.py`
- Includes all parameter controls:
  - Peak/trough detection method selection
  - Distance and prominence sliders
  - Smoothing kernel size control
  - **Savitzky-Golay filter parameters** (window, polynomial order)
  - Adaptive prominence threshold toggle
- Interactive visualizations with Plotly
- Manual editing interface
- Zoom controls

### 5. Utility Functions (100%)
- **`utils/peak_editing.py`** already has:
  - `add_peaks_in_range()` - Auto-detect peaks in region
  - `add_troughs_in_range()` - Auto-detect troughs in region
  - `erase_peaks_in_range()` - Remove peaks in region
  - All necessary functions for manual editing

## 🔧 Final Installation Step

You need to insert the tab code into `app.py`. Here's how:

### Method 1: Automated (Recommended)

Run this command to insert the tabs:

```bash
cd /export02/users/sloparco/physio-qc-main

# Read the tab code
TAB_CODE=$(cat ETCO2_ETO2_TAB_CODE.py)

# Insert at line 1568 (before Export tab)
python3 << 'EOF'
import sys

# Read original app.py
with open('app.py', 'r') as f:
    lines = f.readlines()

# Read tab code
with open('ETCO2_ETO2_TAB_CODE.py', 'r') as f:
    tab_code = f.read()

# Find the insertion point (line 1568 is "tab_idx += 1" before Export)
insertion_line = 1568  # 0-indexed would be 1567

# Insert the tab code
lines.insert(insertion_line - 1, tab_code + '\n')

# Write modified file
with open('app.py', 'w') as f:
    f.writelines(lines)

print("✅ Tab code inserted successfully!")
print(f"   Inserted {len(tab_code.split(chr(10)))} lines at line {insertion_line}")
EOF
```

### Method 2: Manual

1. Open `app.py` in your editor
2. Navigate to line 1568 (where it says `tab_idx += 1` before the Export tab)
3. Copy the entire contents of `ETCO2_ETO2_TAB_CODE.py`
4. Paste it at line 1568 (before `tab_idx += 1`)
5. Save the file

The tabs will be inserted between the Blood Pressure tab and the Export tab.

## 🧪 Testing Your Installation

### 1. Start the App

```bash
cd /export02/users/sloparco/physio-qc-main
streamlit run app.py
```

### 2. Load Data with CO2/O2 Signals

Your ACQ files should have columns like:
- `CO2(mmHg)` or `CO2` → Auto-detected as ETCO2
- `O2(mmHg)` or `O2` → Auto-detected as ETO2

### 3. Test ETCO2 Tab

1. Click on the **ETCO2** tab
2. Adjust parameters if needed:
   - **Peak Method**: Try "diff" (recommended) vs "prominence"
   - **Min Peak Distance**: 2.0s works well for normal breathing
   - **Min Prominence**: Start with 1.0 mmHg
   - **Advanced S-G Filter**:
     - Window: 0.3s (default is good)
     - Polynomial: 2 (quadratic, default is good)
3. Click **"🔬 Process ETCO2"**
4. Should see:
   - Number of detected peaks
   - Two-panel plot: Raw signal with peaks + ETCO2 envelope
   - Manual editing controls

### 4. Test ETO2 Tab

Same process but for O2 troughs:
- **Trough Method**: "diff" recommended
- **Min Trough Distance**: 3.0s (hypoxic breaths are slower)
- **Min Prominence**: 1.0 mmHg

### 5. Test Manual Editing

1. Expand "✏️ Manual Peak/Trough Editing"
2. Set a time range (e.g., 0-10s)
3. Click "🔍 Zoom to Region" to zoom in
4. Try:
   - **➕ Add**: Auto-detect missed peaks/troughs in region
   - **➖ Remove**: Delete false positives in region
   - **🔄 Reset**: Restore original auto-detected peaks/troughs

### 6. Test Export

1. Go to **Export** tab
2. Should see "ETCO2" and "ETO2" in processed signals list
3. Click "Export All Signals"
4. Check output files have ETCO2/ETO2 columns

## 📊 Expected Output Columns in Export

When you export, the CSV will contain:

```
time,
etco2_raw,           # Raw CO2 signal
etco2_envelope,      # Smoothed upper envelope (ETCO2 trace)
etco2_peaks,         # Peak encoding (0=none, 1=auto, 2=added, -1=deleted)
eto2_raw,            # Raw O2 signal
eto2_envelope,       # Smoothed lower envelope (ETO2 trace)
eto2_troughs,        # Trough encoding
...other signals...
```

## 🎨 UI Features Summary

### ETCO2 Tab Features:
- ✅ Method selection (derivative vs prominence)
- ✅ Distance/prominence sliders
- ✅ Smoothing kernel size control
- ✅ **Savitzky-Golay filter controls** (window + polynomial)
- ✅ Adaptive prominence toggle
- ✅ Interactive 2-panel Plotly visualization
- ✅ Zoom controls with persistent state
- ✅ Region-based peak addition/deletion
- ✅ Reset all peaks option
- ✅ Metrics display (auto/added/deleted counts)

### ETO2 Tab Features:
- ✅ All same features as ETCO2
- ✅ Detects troughs (minima) instead of peaks
- ✅ Lower envelope instead of upper envelope
- ✅ Longer default min distance (3s vs 2s)

## 🔍 Parameter Tuning Guide

### If Missing Peaks/Troughs:
1. **Decrease** `min_prominence` (try 0.5 mmHg)
2. **Decrease** `min_peak/trough_distance` (try 1.5s for ETCO2, 2.5s for ETO2)
3. Try `prominence` method instead of `diff`

### If Too Many False Detections:
1. **Increase** `min_prominence` (try 2.0 mmHg)
2. **Increase** `min_peak/trough_distance` (try 3.0s for ETCO2, 4.0s for ETO2)
3. Enable **Adaptive Prominence** (uses 25th percentile)

### If Envelope Too Noisy:
1. **Increase** `smooth_peaks/troughs` (try 7 or 9)
2. Manually delete spurious peaks/troughs in noisy regions

### If Envelope Too Smooth (Missing Features):
1. **Decrease** `smooth_peaks/troughs` (try 3)
2. Check if missing legitimate peaks need manual addition

### Savitzky-Golay Tuning (Advanced):
- **Window too small**: Noisy derivative, misses broad peaks
  - Solution: Increase `sg_window_s` (try 0.4-0.5s)
- **Window too large**: Over-smooths, misses narrow peaks
  - Solution: Decrease `sg_window_s` (try 0.2s)
- **Polynomial order**: Usually 2 is best
  - Higher (3-4): Fits sharper peaks better but more noise-sensitive
  - Lower (1): More robust but fits broad features only

## 📝 Notes

### Key Differences from CLI Scripts:
- ✅ Interactive parameter tuning (no command line args)
- ✅ Real-time visualization with Plotly
- ✅ Manual peak/trough editing capabilities
- ✅ Session state preserves work across reruns
- ✅ Integrated export with metadata

### Preserved from Original:
- ✅ Exact same detection algorithms
- ✅ Same Savitzky-Golay smoothing
- ✅ Same median filtering
- ✅ Same prominence validation
- ✅ Same interpolation method

### Not Implemented (Can Add If Needed):
- ❌ TR resampling (was in CLI version)
- ❌ Volume dropping (preprocessing step)
- ❌ Automatic file I/O (now interactive)

## 🚀 Next Steps After Installation

1. **Test with your data** - Load a file with CO2/O2 signals
2. **Tune default parameters** - Adjust config.py if needed for your typical data
3. **Document your workflow** - Note which parameters work best for your study
4. **Add TR resampling** - If you need it for the export (can add to export.py)
5. **Create example screenshots** - For training other users

## 📚 Related Files

- `metrics/etco2.py` - ETCO2 processing module
- `metrics/eto2.py` - ETO2 processing module
- `config.py` - Configuration with all parameters
- `app.py` - Main Streamlit application
- `ETCO2_ETO2_TAB_CODE.py` - Tab implementation code to insert
- `etco2-eto2-integration-guide.md` - Detailed technical documentation
- `utils/peak_editing.py` - Manual editing utilities

## ✅ Verification Checklist

After installation, verify:

- [ ] App starts without errors (`streamlit run app.py`)
- [ ] ETCO2 and ETO2 tabs appear when loading data with CO2/O2
- [ ] Processing button works and detects peaks/troughs
- [ ] Plots display correctly with Plotly
- [ ] Manual editing adds/removes peaks/troughs
- [ ] Zoom controls work
- [ ] Export includes ETCO2/ETO2 columns
- [ ] All Savitzky-Golay parameters are adjustable

## 🐛 Troubleshooting

### Tabs Don't Appear:
- Check if ACQ file has CO2/O2 columns
- Column names should match patterns in config.py (co2, etco2, o2, eto2)
- Check console for signal detection messages

### Import Errors:
- Verify `metrics/etco2.py` and `metrics/eto2.py` exist
- Check imports at top of `app.py` include etco2, eto2

### No Peaks/Troughs Detected:
- Try lowering `min_prominence` to 0.1
- Try `prominence` method
- Check signal quality in raw plot

### Plot Doesn't Display:
- Check for Plotly errors in console
- Verify result dictionary has required keys
- Try reloading page

## 💡 Pro Tips

1. **Start with derivative method** - More robust to baseline drift
2. **Use zoom** liberally - Makes manual editing much easier
3. **Process in sections** - Add/remove peaks in small time windows
4. **Reset zoom after editing** - See full signal context
5. **Save parameters** - Document what works for your study
6. **Export often** - Don't lose manual edits

The integration is complete! Just insert the tab code and start testing. 🎉
