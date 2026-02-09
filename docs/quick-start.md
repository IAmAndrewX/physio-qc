# ETCO2/ETO2 Integration - Quick Start Guide

## ✅ Installation Complete!

The ETCO2 and ETO2 tabs have been successfully integrated into your physio-qc application.

## 🚀 Launch the App

```bash
cd /export02/users/sloparco/physio-qc-main
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📊 Using ETCO2/ETO2 Features

### 1. Load Your Data
- Use the sidebar to select participant, session, and task
- Click "Load Data"
- If your ACQ file has CO2 and O2 signals, you'll see **ETCO2** and **ETO2** tabs appear

### 2. Process ETCO2 (End-Tidal CO2)
1. Click the **ETCO2** tab
2. **Configure parameters** (or use defaults):
   - Peak Detection Method: `diff` (recommended)
   - Min Peak Distance: `2.0s` (typical breathing rate)
   - Min Prominence: `1.0 mmHg`
   - Smoothing Kernel: `5 peaks`
3. **Advanced options** (expand "⚙️ Advanced"):
   - S-G Window: `0.3s` (for derivative calculation)
   - S-G Polynomial: `2` (quadratic fit)
   - Adaptive Prominence: `unchecked` (or check to auto-adjust threshold)
4. Click **"🔬 Process ETCO2"**
5. View results:
   - Top plot: Raw CO2 with detected peaks (green circles)
   - Bottom plot: Smooth ETCO2 envelope (upper envelope)
   - Metrics: Number of peaks, mean ETCO2 value

### 3. Process ETO2 (End-Tidal O2)
1. Click the **ETO2** tab
2. **Configure parameters** (or use defaults):
   - Trough Detection Method: `diff` (recommended)
   - Min Trough Distance: `3.0s` (hypoxic breaths are slower)
   - Min Prominence: `1.0 mmHg`
   - Smoothing Kernel: `5 troughs`
3. **Advanced options** same as ETCO2
4. Click **"🔬 Process ETO2"**
5. View results:
   - Top plot: Raw O2 with detected troughs (green circles)
   - Bottom plot: Smooth ETO2 envelope (lower envelope)
   - Metrics: Number of troughs, mean ETO2 value

### 4. Manual Editing (Optional)
Both tabs have manual editing features:

**To add missing peaks/troughs:**
1. Expand "✏️ Manual Peak/Trough Editing"
2. Set time range (e.g., 10-20 seconds)
3. Click "🔍 Zoom to Region" to focus
4. Click "➕ Add Peaks/Troughs in Region"
5. Auto-detects new peaks/troughs in that window

**To remove false detections:**
1. Set time range around false peaks/troughs
2. Click "➖ Remove Peaks/Troughs in Region"
3. All peaks/troughs in that range are deleted

**To reset:**
- Click "🔄 Reset All Peaks/Troughs" to restore original auto-detection
- Click "↔️ Reset Zoom" to see full signal

### 5. Export Data
1. Go to **Export** tab
2. You should see "ETCO2" and "ETO2" in the processed signals list
3. Click "Export All Signals"
4. Output CSV will contain:
   - `etco2_raw` - Raw CO2 signal
   - `etco2_envelope` - ETCO2 upper envelope
   - `etco2_peaks` - Peak encoding (1=auto, 2=added, -1=deleted, 0=none)
   - `eto2_raw` - Raw O2 signal
   - `eto2_envelope` - ETO2 lower envelope
   - `eto2_troughs` - Trough encoding

## ⚙️ Parameter Tuning Tips

### Start with These Defaults:
```python
ETCO2:
  - Method: diff
  - Min Distance: 2.0s
  - Min Prominence: 1.0 mmHg
  - Smoothing: 5 peaks
  - S-G Window: 0.3s
  - S-G Poly: 2

ETO2:
  - Method: diff
  - Min Distance: 3.0s  # Slower than peaks
  - Min Prominence: 1.0 mmHg
  - Smoothing: 5 troughs
  - S-G Window: 0.2s   # Narrower than ETCO2
  - S-G Poly: 2
```

### If Detection is Poor:

**Missing peaks/troughs?**
- ⬇️ Decrease min_prominence to 0.5 mmHg
- ⬇️ Decrease min_distance to 1.5s (ETCO2) or 2.5s (ETO2)
- Try `prominence` method instead of `diff`

**Too many false detections?**
- ⬆️ Increase min_prominence to 2.0 mmHg
- ⬆️ Increase min_distance to 3.0s (ETCO2) or 4.0s (ETO2)
- ✅ Enable "Adaptive Prominence" (uses 25th percentile as floor)

**Envelope too jagged?**
- ⬆️ Increase smoothing kernel to 7 or 9
- Manually remove spurious peaks/troughs

**Envelope too smooth?**
- ⬇️ Decrease smoothing kernel to 3
- Check if genuine peaks were missed

## 🔬 Understanding the Methods

### Derivative Method (`diff`)
**How it works:**
1. Smooths signal with Savitzky-Golay filter
2. Computes first and second derivatives
3. Finds zero-crossings in derivative:
   - ETCO2: Positive→Negative crossing (peaks)
   - ETO2: Negative→Positive crossing (troughs)
4. Checks curvature:
   - ETCO2: Negative (concave down)
   - ETO2: Positive (concave up)
5. Refines to nearest local max/min in raw signal
6. Validates by prominence

**Best for:** Signals with baseline drift, noisy data

### Prominence Method (`prominence`)
**How it works:**
1. Uses scipy's `find_peaks()` directly
2. ETCO2: Finds peaks by prominence
3. ETO2: Inverts signal, finds peaks (= troughs in original)

**Best for:** Clean signals, when derivative is too sensitive

## 📝 Files Modified

- ✅ `app.py` - Added ETCO2/ETO2 tabs (now 2294 lines, +639 lines)
- ✅ `metrics/etco2.py` - NEW processing module
- ✅ `metrics/eto2.py` - NEW processing module
- ✅ `config.py` - Added configuration sections

## 🎯 Example Workflow

```
1. Start app: streamlit run app.py
2. Load data with CO2/O2 signals
3. Process ETCO2 → Tune parameters if needed → Manual edit if needed
4. Process ETO2 → Tune parameters if needed → Manual edit if needed
5. Export → Check CSV has envelope columns
6. Use envelope traces in your analysis pipeline!
```

## 💡 Pro Tips

1. **Process ETCO2 first** - It's usually cleaner and validates your parameters
2. **Use zoom liberally** - Makes manual editing much easier
3. **Save your parameters** - Document what works for your study in config.py
4. **Export often** - Don't lose manual edits
5. **Check both plots** - Raw with peaks AND envelope to verify quality

## 🐛 Common Issues

**Tabs don't appear:**
- Check if ACQ file has CO2/O2 columns
- Check column names (should be like "CO2(mmHg)" or "O2(mmHg)")

**No peaks detected:**
- Lower min_prominence to 0.1
- Try prominence method
- Check if signal is too flat

**Envelope is stepped:**
- This is normal! It's linear interpolation between smoothed peaks/troughs
- Increase smoothing kernel for smoother transitions

**Can't add peaks manually:**
- Make sure time range is valid
- Signal might be too noisy in that region
- Try zooming in first

## 📚 Further Reading

- `installation-guide.md` - Detailed technical documentation
- `etco2-eto2-integration-guide.md` - Architecture and design notes
- `config.py` - See all available parameters
- `metrics/etco2.py` - Read algorithm documentation
- `metrics/eto2.py` - Read algorithm documentation

## ✅ You're Ready!

The integration is complete and tested. Start the app and try it with your data!

```bash
streamlit run app.py
```

Happy signal processing! 🎉
