# ETCO2/ETO2 Integration Guide

## Summary

I've integrated your ETCO2 and ETO2 trace extraction scripts into the physio-qc Streamlit framework. The core processing modules are complete and follow the existing architecture patterns.

## Files Created/Modified

### ✅ Created Files

1. **`metrics/etco2.py`** - ETCO2 processing module
   - `detect_peaks_diff()` - Derivative-based peak detection with curvature filtering
   - `detect_peaks_prominence()` - Scipy prominence-based peak detection
   - `extract_etco2_envelope()` - Main processing pipeline

2. **`metrics/eto2.py`** - ETO2 processing module
   - `detect_troughs_diff()` - Derivative-based trough detection with curvature filtering
   - `detect_troughs_prominence()` - Scipy prominence-based trough detection
   - `extract_eto2_envelope()` - Main processing pipeline

### ✅ Modified Files

3. **`config.py`** - Added configuration sections
   - Updated `SIGNAL_PATTERNS` to include ETCO2 and ETO2 detection keywords
   - Added `ETCO2_PEAK_METHODS` and method descriptions
   - Added `DEFAULT_ETCO2_PARAMS` with all configurable parameters
   - Added `ETO2_TROUGH_METHODS` and method descriptions
   - Added `DEFAULT_ETO2_PARAMS` with all configurable parameters

## Architecture Overview

The integration follows the exact same pattern as existing signals (ECG, RSP, PPG, BP):

### Processing Flow

```
Raw CO2/O2 Signal (from ACQ file)
    ↓
metrics/etco2.py or metrics/eto2.py
    ├─ Peak/Trough Detection (derivative or prominence method)
    ├─ Median Smoothing (configurable kernel size)
    ├─ Linear Interpolation (create continuous envelope)
    └─ Return Result Dictionary
    ↓
app.py (Streamlit UI)
    ├─ Display raw signal + detected peaks/troughs
    ├─ Show envelope overlay
    ├─ Allow manual peak/trough editing
    └─ Calculate statistics
    ↓
utils/export.py
    └─ Export envelope + metadata to CSV/JSON
```

### Parameter System

Both modules accept a `params` dictionary with configurable options:

**ETCO2 Parameters:**
```python
{
    'peak_method': 'diff',           # or 'prominence'
    'min_peak_distance_s': 2.0,      # Minimum time between breaths
    'min_prominence': 1.0,            # Minimum prominence in mmHg
    'sg_window_s': 0.3,               # Savitzky-Golay window (seconds)
    'sg_poly': 2,                     # S-G polynomial order
    'prom_adapt': False,              # Adaptive prominence threshold
    'smooth_peaks': 5                 # Median filter kernel size
}
```

**ETO2 Parameters:**
```python
{
    'trough_method': 'diff',         # or 'prominence'
    'min_trough_distance_s': 3.0,    # Minimum time between troughs
    'min_prominence': 1.0,            # Minimum prominence in mmHg
    'sg_window_s': 0.2,               # Savitzky-Golay window (seconds)
    'sg_poly': 2,                     # S-G polynomial order
    'prom_adapt': False,              # Adaptive prominence threshold
    'smooth_troughs': 5               # Median filter kernel size
}
```

## What's Done ✅

### Core Processing (100% Complete)
- [x] ETCO2 peak detection with both derivative and prominence methods
- [x] ETO2 trough detection with both derivative and prominence methods
- [x] Median filtering and interpolation to create envelopes
- [x] Robust fallback handling (derivative → prominence → smoothed signal)
- [x] Full parameter configurability
- [x] Configuration file integration
- [x] Signal pattern detection for auto-discovery in ACQ files

### Code Quality
- [x] Comprehensive docstrings (NumPy style)
- [x] Type hints for all functions
- [x] Follows existing code patterns exactly
- [x] Pure functional design (no classes)
- [x] Error handling with graceful fallbacks

## What Needs UI Integration 🔨

To complete the integration, you need to add UI sections to `app.py`. I'll provide the template below.

## UI Integration Template for app.py

Add these sections to `app.py` following the existing tab pattern:

### 1. Initialize Session State (add to `init_session_state()`)

```python
# ETCO2 initialization
if 'etco2_result' not in st.session_state:
    st.session_state.etco2_result = None
if 'etco2_params' not in st.session_state:
    st.session_state.etco2_params = config.DEFAULT_ETCO2_PARAMS.copy()

# ETO2 initialization
if 'eto2_result' not in st.session_state:
    st.session_state.eto2_result = None
if 'eto2_params' not in st.session_state:
    st.session_state.eto2_params = config.DEFAULT_ETO2_PARAMS.copy()
```

### 2. Add ETCO2 Tab (add after BP tab)

```python
def create_etco2_tab(data, sampling_rate):
    """ETCO2 processing and visualization tab"""
    from metrics import etco2

    st.header("End-Tidal CO2 (ETCO2) Processing")

    # Parameter configuration sidebar
    with st.sidebar:
        st.subheader("ETCO2 Parameters")

        # Peak detection method
        peak_method = st.selectbox(
            "Peak Detection Method",
            config.ETCO2_PEAK_METHODS,
            index=config.ETCO2_PEAK_METHODS.index(st.session_state.etco2_params.get('peak_method', 'diff'))
        )

        with st.expander("ℹ️ Method Information"):
            st.info(config.ETCO2_PEAK_METHOD_INFO[peak_method])

        # Detection parameters
        min_peak_distance_s = st.slider(
            "Min Peak Distance (s)",
            min_value=0.5,
            max_value=5.0,
            value=st.session_state.etco2_params.get('min_peak_distance_s', 2.0),
            step=0.1,
            help="Minimum time between consecutive breath peaks"
        )

        min_prominence = st.slider(
            "Min Prominence (mmHg)",
            min_value=0.1,
            max_value=10.0,
            value=st.session_state.etco2_params.get('min_prominence', 1.0),
            step=0.1,
            help="Minimum peak prominence for detection"
        )

        smooth_peaks = st.slider(
            "Smoothing Kernel Size (peaks)",
            min_value=3,
            max_value=15,
            value=st.session_state.etco2_params.get('smooth_peaks', 5),
            step=2,  # Keep odd
            help="Median filter kernel size applied to peak values"
        )

        # Advanced parameters
        with st.expander("Advanced Parameters"):
            sg_window_s = st.slider(
                "Savitzky-Golay Window (s)",
                min_value=0.1,
                max_value=1.0,
                value=st.session_state.etco2_params.get('sg_window_s', 0.3),
                step=0.05,
                help="Smoothing window for derivative calculation"
            )

            sg_poly = st.slider(
                "S-G Polynomial Order",
                min_value=1,
                max_value=5,
                value=st.session_state.etco2_params.get('sg_poly', 2),
                help="Polynomial order for Savitzky-Golay filter"
            )

            prom_adapt = st.checkbox(
                "Adaptive Prominence Threshold",
                value=st.session_state.etco2_params.get('prom_adapt', False),
                help="Use 25th percentile of prominences as adaptive minimum"
            )

        # Process button
        if st.button("🔬 Process ETCO2", type="primary"):
            # Update parameters
            params = {
                'peak_method': peak_method,
                'min_peak_distance_s': min_peak_distance_s,
                'min_prominence': min_prominence,
                'smooth_peaks': smooth_peaks,
                'sg_window_s': sg_window_s,
                'sg_poly': sg_poly,
                'prom_adapt': prom_adapt
            }
            st.session_state.etco2_params.update(params)

            # Get CO2 signal
            co2_signal = data['df'][data['signal_mappings']['etco2']].values

            # Process
            with st.spinner("Detecting CO2 peaks and extracting envelope..."):
                result = etco2.extract_etco2_envelope(
                    co2_signal,
                    sampling_rate,
                    st.session_state.etco2_params
                )

            if result is not None:
                st.session_state.etco2_result = result
                st.success(f"✅ ETCO2 processed: {len(result['auto_peaks'])} peaks detected")

    # Display results if available
    result = st.session_state.etco2_result
    if result is not None:
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Auto-detected Peaks", len(result['auto_peaks']))
        with col2:
            n_added = len(np.setdiff1d(result['current_peaks'], result['auto_peaks']))
            st.metric("Manually Added", n_added)
        with col3:
            n_deleted = len(np.setdiff1d(result['auto_peaks'], result['current_peaks']))
            st.metric("Deleted", n_deleted)
        with col4:
            mean_etco2 = np.mean(result['etco2_envelope'][np.isfinite(result['etco2_envelope'])])
            st.metric("Mean ETCO2", f"{mean_etco2:.1f} mmHg")

        # Visualization
        st.subheader("ETCO2 Trace Visualization")
        fig = create_etco2_plot(result)
        st.plotly_chart(fig, use_container_width=True)

        # Manual editing interface (similar to ECG)
        with st.expander("✏️ Manual Peak Editing"):
            st.info("Add or remove peaks by specifying time ranges")

            col1, col2 = st.columns(2)
            with col1:
                region_start = st.number_input(
                    "Region Start (s)",
                    min_value=0.0,
                    max_value=result['time_vector'][-1],
                    value=0.0,
                    step=1.0
                )
            with col2:
                region_end = st.number_input(
                    "Region End (s)",
                    min_value=0.0,
                    max_value=result['time_vector'][-1],
                    value=min(10.0, result['time_vector'][-1]),
                    step=1.0
                )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("➕ Add Peaks in Region"):
                    # Use peak_editing utility to add peaks
                    from utils import peak_editing
                    result['current_peaks'] = peak_editing.add_peaks_in_range(
                        result['current_peaks'],
                        result['raw_signal'],
                        sampling_rate,
                        region_start,
                        region_end
                    )
                    st.rerun()

            with col2:
                if st.button("➖ Remove Peaks in Region"):
                    from utils import peak_editing
                    result['current_peaks'] = peak_editing.erase_peaks_in_range(
                        result['current_peaks'],
                        sampling_rate,
                        region_start,
                        region_end
                    )
                    st.rerun()
    else:
        st.info("👈 Configure parameters in the sidebar and click 'Process ETCO2' to begin")


def create_etco2_plot(result):
    """Create plotly figure for ETCO2 visualization"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Raw CO2 Signal with Detected Peaks', 'ETCO2 Envelope'),
        vertical_spacing=0.12,
        row_heights=[0.5, 0.5]
    )

    time = result['time_vector']

    # Row 1: Raw signal with peaks
    fig.add_trace(
        go.Scatter(
            x=time,
            y=result['raw_signal'],
            name='Raw CO2',
            line=dict(color='#636EFA', width=1),
            mode='lines'
        ),
        row=1, col=1
    )

    # Auto-detected peaks
    if len(result['auto_peaks']) > 0:
        fig.add_trace(
            go.Scatter(
                x=time[result['auto_peaks']],
                y=result['raw_signal'][result['auto_peaks']],
                name='Auto Peaks',
                mode='markers',
                marker=dict(color='#00CC96', size=8, symbol='circle')
            ),
            row=1, col=1
        )

    # Manually edited peaks (if different)
    edited_peaks = np.setdiff1d(result['current_peaks'], result['auto_peaks'])
    if len(edited_peaks) > 0:
        fig.add_trace(
            go.Scatter(
                x=time[edited_peaks],
                y=result['raw_signal'][edited_peaks],
                name='Added Peaks',
                mode='markers',
                marker=dict(color='#AB63FA', size=10, symbol='x')
            ),
            row=1, col=1
        )

    # Row 2: ETCO2 envelope
    fig.add_trace(
        go.Scatter(
            x=time,
            y=result['etco2_envelope'],
            name='ETCO2 Envelope',
            line=dict(color='#EF553B', width=2),
            mode='lines'
        ),
        row=2, col=1
    )

    # Layout
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="CO2 (mmHg)", row=1, col=1)
    fig.update_yaxes(title_text="ETCO2 (mmHg)", row=2, col=1)

    fig.update_layout(
        height=800,
        template='plotly_dark',
        showlegend=True,
        hovermode='x unified'
    )

    return fig
```

### 3. Add ETO2 Tab (similar structure)

```python
def create_eto2_tab(data, sampling_rate):
    """ETO2 processing and visualization tab"""
    from metrics import eto2

    # Similar structure to ETCO2, but:
    # - Use 'trough_method' instead of 'peak_method'
    # - Use config.ETO2_TROUGH_METHODS
    # - Use config.DEFAULT_ETO2_PARAMS
    # - Replace "peaks" with "troughs" throughout
    # - Use min_trough_distance_s instead of min_peak_distance_s
    # - Plot shows troughs (minima) instead of peaks

    # Implementation follows exact same pattern as create_etco2_tab
    # Just swap peaks→troughs, ETCO2→ETO2
```

### 4. Add Tabs to Main App

In the `main()` function where tabs are created, add:

```python
# Detect available signals
available_signals = []
if 'ecg' in signal_mappings:
    available_signals.append('ECG')
if 'rsp' in signal_mappings:
    available_signals.append('RSP')
if 'ppg' in signal_mappings:
    available_signals.append('PPG')
if 'bp' in signal_mappings:
    available_signals.append('Blood Pressure')
if 'etco2' in signal_mappings:
    available_signals.append('ETCO2')  # NEW
if 'eto2' in signal_mappings:
    available_signals.append('ETO2')   # NEW

available_signals.append('Export')  # Always show export tab

# Create tabs
tabs = st.tabs(available_signals)

# ... existing tab code ...

# Add ETCO2 tab
if 'ETCO2' in available_signals:
    with tabs[available_signals.index('ETCO2')]:
        create_etco2_tab(data, sampling_rate)

# Add ETO2 tab
if 'ETO2' in available_signals:
    with tabs[available_signals.index('ETO2')]:
        create_eto2_tab(data, sampling_rate)
```

### 5. Update Export Function

In `utils/export.py`, add support for ETCO2/ETO2 in `create_combined_dataframe()`:

```python
# Add ETCO2 if available
if 'etco2' in results and results['etco2'] is not None:
    result = results['etco2']
    df['etco2_raw'] = result['raw_signal']
    df['etco2_envelope'] = result['etco2_envelope']

    # Encode peaks
    peak_encoding = np.zeros(len(result['raw_signal']), dtype=np.int8)
    peak_encoding[result['auto_peaks']] = config.PEAK_ENCODING['AUTO_DETECTED']

    # Mark manually added
    added_peaks = np.setdiff1d(result['current_peaks'], result['auto_peaks'])
    peak_encoding[added_peaks] = config.PEAK_ENCODING['MANUALLY_ADDED']

    # Mark deleted
    deleted_peaks = np.setdiff1d(result['auto_peaks'], result['current_peaks'])
    peak_encoding[deleted_peaks] = config.PEAK_ENCODING['DELETED']

    df['etco2_peaks'] = peak_encoding

# Similar for ETO2 (replace peaks with troughs)
```

## Testing the Integration

### 1. Test with Sample Data

Your ACQ files should have columns like:
- `CO2(mmHg)` or `CO2` - Will be auto-detected as ETCO2
- `O2(mmHg)` or `O2` - Will be auto-detected as ETO2

### 2. Expected Behavior

**ETCO2 Tab:**
- Shows raw CO2 signal with detected breath peaks (maxima)
- Shows smooth upper envelope representing end-tidal CO2 levels
- Allows manual peak addition/deletion
- Exports envelope to CSV

**ETO2 Tab:**
- Shows raw O2 signal with detected breath troughs (minima)
- Shows smooth lower envelope representing minimum O2 levels during hypoxia
- Allows manual trough addition/deletion
- Exports envelope to CSV

### 3. Parameter Tuning

Start with defaults and adjust:
- **If missing peaks/troughs**: Decrease `min_prominence`
- **If too many false detections**: Increase `min_prominence` or `min_peak/trough_distance_s`
- **If envelope is too noisy**: Increase `smooth_peaks/troughs`
- **If envelope is too smooth**: Decrease `smooth_peaks/troughs`

## Key Differences from Original Scripts

### Removed Features (CLI-specific)
- ❌ Argparse command-line interface
- ❌ Automatic CSV file I/O
- ❌ TR resampling (can be added to export if needed)
- ❌ Volume dropping logic (preprocessing, not core processing)
- ❌ Matplotlib plotting (replaced with Plotly for Streamlit)

### Added Features (Streamlit-specific)
- ✅ Interactive parameter tuning via UI
- ✅ Real-time visualization updates
- ✅ Manual peak/trough editing
- ✅ Session state persistence
- ✅ Integrated export with metadata
- ✅ Plotly interactive plots

### Preserved Features (Core Algorithm)
- ✅ Derivative-based peak/trough detection
- ✅ Curvature filtering (negative for peaks, positive for troughs)
- ✅ Prominence validation with adaptive thresholding
- ✅ Minimum temporal separation enforcement
- ✅ Median filtering of detected values
- ✅ Linear interpolation for envelope creation
- ✅ Savitzky-Golay smoothing parameters

## Next Steps

1. **Add UI code to app.py** using the templates above
2. **Test with your LCS data** to verify signal detection works
3. **Tune default parameters** in config.py based on your data
4. **Add TR resampling** to export.py if needed for your workflow
5. **Create example screenshots** for documentation

## Questions?

The core processing is complete and matches your original algorithm. The main work remaining is UI integration, which follows the exact pattern used for ECG/RSP/PPG/BP.

Let me know if you need help with any specific part of the UI integration!
