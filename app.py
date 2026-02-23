"""
Physiological Signal QC Application
Streamlit-based interface for quality control of physiological signals
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import streamlit.components.v1 as components

import config
from utils.file_io import scan_data_directory, find_file_path, load_acq_file
from utils.bids_scan import (
    scan_bids_subjects, get_structural_images, get_functional_images,
    get_available_tasks, get_cvr_methods, get_cvr_spaces, get_cvr_tasks,
    get_glm_images,
)
from neuro.niivue_component import build_niivue_html, colormap_css
from neuro.file_server import register_file, clear_cache as clear_nifti_cache
from neuro.masking import create_masked_volume
from metrics import ecg, rsp, ppg, blood_pressure, etco2, eto2, spo2
from utils import peak_editing, export


st.set_page_config(
    page_title="Physio QC",
    page_icon="📈",
    layout="wide"
)

# Clear stale NIfTI symlinks once per session (prevents >1GB static dir)
if '_nifti_cache_cleared' not in st.session_state:
    clear_nifti_cache()
    st.session_state['_nifti_cache_cleared'] = True


CSS = """
<style>
    .stApp {
        background-color: #0E1117;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #262730;
        border-radius: 4px;
        padding: 10px 20px;
        color: #FAFAFA;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF4B4B;
    }
    .metric-box {
        background-color: #262730;
        padding: 15px;
        border-radius: 8px;
        margin: 5px 0;
    }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables"""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False

    if 'loaded_data' not in st.session_state:
        st.session_state.loaded_data = None

    if 'ecg_result' not in st.session_state:
        st.session_state.ecg_result = None

    if 'rsp_result' not in st.session_state:
        st.session_state.rsp_result = None

    if 'ppg_result' not in st.session_state:
        st.session_state.ppg_result = None

    if 'bp_result' not in st.session_state:
        st.session_state.bp_result = None

    if 'ecg_params' not in st.session_state:
        st.session_state.ecg_params = config.DEFAULT_ECG_PARAMS.copy()

    if 'rsp_params' not in st.session_state:
        st.session_state.rsp_params = config.DEFAULT_RSP_PARAMS.copy()

    if 'ppg_params' not in st.session_state:
        st.session_state.ppg_params = config.DEFAULT_PPG_PARAMS.copy()

    if 'bp_params' not in st.session_state:
        st.session_state.bp_params = config.DEFAULT_BP_PARAMS.copy()

    if 'etco2_result' not in st.session_state:
        st.session_state.etco2_result = None

    if 'eto2_result' not in st.session_state:
        st.session_state.eto2_result = None

    if 'etco2_params' not in st.session_state:
        st.session_state.etco2_params = config.DEFAULT_ETCO2_PARAMS.copy()

    if 'eto2_params' not in st.session_state:
        st.session_state.eto2_params = config.DEFAULT_ETO2_PARAMS.copy()

    if 'spo2_result' not in st.session_state:
        st.session_state.spo2_result = None

    if 'spo2_params' not in st.session_state:
        st.session_state.spo2_params = config.DEFAULT_SPO2_PARAMS.copy()

    # Zoom ranges for each signal type
    if 'ecg_zoom_range' not in st.session_state:
        st.session_state.ecg_zoom_range = None

    if 'rsp_zoom_range' not in st.session_state:
        st.session_state.rsp_zoom_range = None

    if 'ppg_zoom_range' not in st.session_state:
        st.session_state.ppg_zoom_range = None

    if 'bp_zoom_range' not in st.session_state:
        st.session_state.bp_zoom_range = None

    if 'etco2_zoom_range' not in st.session_state:
        st.session_state.etco2_zoom_range = None

    if 'eto2_zoom_range' not in st.session_state:
        st.session_state.eto2_zoom_range = None

    if 'spo2_zoom_range' not in st.session_state:
        st.session_state.spo2_zoom_range = None

    if 'spirometer_result' not in st.session_state:
        st.session_state.spirometer_result = None

    if 'spirometer_params' not in st.session_state:
        st.session_state.spirometer_params = config.DEFAULT_RSP_PARAMS.copy()

    # Neuro mode state
    if 'neuro_data_loaded' not in st.session_state:
        st.session_state.neuro_data_loaded = False
    if 'neuro_participant' not in st.session_state:
        st.session_state.neuro_participant = None
    if 'neuro_session' not in st.session_state:
        st.session_state.neuro_session = None
    if 'neuro_structural_images' not in st.session_state:
        st.session_state.neuro_structural_images = {}
    if 'neuro_functional_tasks' not in st.session_state:
        st.session_state.neuro_functional_tasks = []
    if 'neuro_glm_available' not in st.session_state:
        st.session_state.neuro_glm_available = False


def create_signal_plot(time, raw, clean, current_peaks, auto_peaks, signal_name, sampling_rate,
                       hr_interpolated=None, hr_bpm=None, quality_continuous=None,
                       selected_quality_metrics=None, quality_data=None, ui_revision='constant',
                       zoom_range=None):
    """Create 3-panel plot for signal visualization with synchronized zooming"""
    signal_key = str(signal_name).strip().lower()
    if signal_key == 'ecg':
        labels = {
            'subplots': (
                'Raw ECG vs Filtered ECG',
                'Filtered ECG with R-Peak Markers',
                'Heart Rate (BPM)',
            ),
            'raw': 'Raw ECG',
            'clean': 'Filtered ECG',
            'signal': 'Filtered ECG',
            'peaks': 'R-Peaks',
            'rate': 'Heart Rate (Interpolated)',
        }
    elif signal_key == 'ppg':
        labels = {
            'subplots': (
                'Raw PPG vs Filtered PPG',
                'Filtered PPG with Systolic Peak Markers',
                'Pulse Rate (BPM)',
            ),
            'raw': 'Raw PPG',
            'clean': 'Filtered PPG',
            'signal': 'Filtered PPG',
            'peaks': 'Systolic Peaks',
            'rate': 'Pulse Rate (Interpolated)',
        }
    else:
        labels = {
            'subplots': (
                f'Raw {signal_name} vs Filtered {signal_name}',
                f'Filtered {signal_name} with Peak Markers',
                f'{signal_name} Rate',
            ),
            'raw': f'Raw {signal_name}',
            'clean': f'Filtered {signal_name}',
            'signal': f'Filtered {signal_name}',
            'peaks': 'Detected Peaks',
            'rate': f'{signal_name} Rate (Interpolated)',
        }

    deleted_peaks = np.setdiff1d(auto_peaks, current_peaks)
    added_peaks = np.setdiff1d(current_peaks, auto_peaks)

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=labels['subplots'],
        vertical_spacing=0.1,
        specs=[[{"secondary_y": False}], [{"secondary_y": True}], [{"secondary_y": False}]]
    )

    # Row 1: Raw vs Clean
    fig.add_trace(go.Scatter(x=time, y=raw, name=labels['raw'], line=dict(color='#808080', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=time, y=clean, name=labels['clean'], line=dict(color='#00D4FF', width=1)), row=1, col=1)

    # Row 2: Clean with Peaks
    fig.add_trace(
        go.Scatter(x=time, y=clean, name=labels['signal'], line=dict(color='#00D4FF', width=1)),
        row=2, col=1, secondary_y=False
    )

    if len(current_peaks) > 0:
        fig.add_trace(go.Scatter(
            x=time[current_peaks], y=clean[current_peaks],
            mode='markers', name=labels['peaks'],
            marker=dict(color='#FF4444', size=8, symbol='circle')
        ), row=2, col=1, secondary_y=False)

    # Quality metrics
    if selected_quality_metrics and quality_data:
        for i, metric in enumerate(selected_quality_metrics):
            if metric in quality_data and quality_data[metric] is not None:
                fig.add_trace(go.Scatter(
                    x=time, y=quality_data[metric],
                    name=metric,
                    line=dict(width=1.5, dash='dot'),
                    opacity=0.7
                ), row=2, col=1, secondary_y=True)

    # Row 3: Interpolated rate series
    if hr_interpolated is not None:
        fig.add_trace(go.Scatter(
            x=time, y=hr_interpolated,
            name=labels['rate'],
            line=dict(color='#FF6B6B', width=2)
        ), row=3, col=1)
        # Clamp y-axis to physiological range using percentiles
        finite_hr = hr_interpolated[np.isfinite(hr_interpolated)]
        if len(finite_hr) > 0:
            p1, p99 = np.percentile(finite_hr, [1, 99])
            pad = max(5, (p99 - p1) * 0.1)
            fig.update_yaxes(range=[max(0, p1 - pad), p99 + pad], row=3, col=1)

    fig.update_xaxes(matches='x', rangemode='nonnegative')
    if zoom_range is not None:
        fig.update_xaxes(range=[max(0, zoom_range[0]), zoom_range[1]])

    fig.update_layout(height=800, template='plotly_dark', showlegend=True, uirevision=ui_revision)

    return fig


def create_rsp_bp_plot(time, raw, clean, current_peaks, current_troughs, auto_peaks, auto_troughs, 
                       signal_name, rate_interpolated=None, rate_bpm=None, 
                       bp_data=None, hr_data=None, ui_revision='constant', 
                       zoom_range=None, calibration_regions=None):
    """Create 3 or 4-panel plot for RSP/BP with synchronized zooming"""
    signal_key = str(signal_name).strip().lower()
    is_bp = signal_key == 'bp'

    if is_bp:
        labels = {
            'subplots': [
                'Raw Arterial Pressure vs Filtered Pressure',
                'Filtered Pressure with Systolic/Diastolic Markers',
                'Blood Pressure Metrics (SBP / MAP / DBP)',
                'Heart Rate from BP Peaks (BPM)',
            ],
            'raw': 'Raw Arterial Pressure',
            'clean': 'Filtered Arterial Pressure',
            'signal': 'Filtered Arterial Pressure',
            'peaks': 'Systolic Peaks',
            'troughs': 'Diastolic Troughs',
            'rate': 'Respiratory Rate (Interpolated)',
            'hr_from_bp': 'Heart Rate from BP (Interpolated)',
        }
    else:
        if signal_key in {'spirometer', 'spiro', 'mask flow', 'maskflow'}:
            base_label = 'Spirometry Flow'
        elif signal_key in {'rsp', 'resp', 'respiration'}:
            base_label = 'Respiration'
        else:
            base_label = str(signal_name)
        labels = {
            'subplots': [
                f'Raw {base_label} vs Filtered {base_label}',
                f'Filtered {base_label} with Inhalation/Exhalation Markers',
                'Respiratory Rate (breaths/min)',
            ],
            'raw': f'Raw {base_label}',
            'clean': f'Filtered {base_label}',
            'signal': f'Filtered {base_label}',
            'peaks': 'Inhalation Peaks',
            'troughs': 'Exhalation Troughs',
            'rate': 'Respiratory Rate (Interpolated)',
            'hr_from_bp': 'Heart Rate from BP (Interpolated)',
        }

    # 1. Row Configuration
    n_rows = 4 if is_bp else 3
    height = 1000 if n_rows == 4 else 800

    fig = make_subplots(
        rows=n_rows, cols=1,
        subplot_titles=labels['subplots'],
        vertical_spacing=0.07,
        shared_xaxes=True
    )

    # --- Row 1: Raw vs Filtered ---
    fig.add_trace(go.Scatter(x=time, y=raw, name=labels['raw'], line=dict(color='#808080', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=time, y=clean, name=labels['clean'], line=dict(color='#00D4FF', width=1)), row=1, col=1)

    # --- Row 2: Signal with Peaks/Troughs ---
    fig.add_trace(
        go.Scatter(x=time, y=clean, name=labels['signal'], line=dict(color='#00D4FF', width=1), showlegend=False),
        row=2, col=1
    )

    if len(current_peaks) > 0:
        fig.add_trace(go.Scatter(
            x=time[current_peaks], y=clean[current_peaks], mode='markers',
            name=labels['peaks'], marker=dict(color='#FF4444', size=8)
        ), row=2, col=1)
    if len(current_troughs) > 0:
        fig.add_trace(go.Scatter(
            x=time[current_troughs], y=clean[current_troughs], mode='markers',
            name=labels['troughs'], marker=dict(color='#4444FF', size=8)
        ), row=2, col=1)

    # --- Row 3: BP Metrics (SBP/MAP/DBP) or RSP Rate ---
    if is_bp and bp_data is not None:
        t_4hz = bp_data['time_4hz']
        fig.add_trace(go.Scatter(x=t_4hz, y=bp_data['sbp_4hz'], name='SBP', line=dict(color='red', width=1.5)), row=3, col=1)
        fig.add_trace(go.Scatter(x=t_4hz, y=bp_data['map_4hz'], name='MAP', line=dict(color='green', width=2)), row=3, col=1)
        fig.add_trace(go.Scatter(x=t_4hz, y=bp_data['dbp_4hz'], name='DBP', line=dict(color='blue', width=1.5)), row=3, col=1)
    elif rate_interpolated is not None:
        fig.add_trace(
            go.Scatter(x=time, y=rate_interpolated, name=labels['rate'], line=dict(color='#FF6B6B', width=2)),
            row=3, col=1
        )
        # Clamp y-axis to physiological range using percentiles
        finite_rate = rate_interpolated[np.isfinite(rate_interpolated)]
        if len(finite_rate) > 0:
            p1, p99 = np.percentile(finite_rate, [1, 99])
            pad = max(2, (p99 - p1) * 0.1)
            fig.update_yaxes(range=[max(0, p1 - pad), p99 + pad], row=3, col=1)

    # --- Row 4: Heart Rate (BP only) ---
    if is_bp and hr_data is not None:
        fig.add_trace(
            go.Scatter(x=time, y=hr_data['hr_interpolated'], name=labels['hr_from_bp'], line=dict(color='#FF6B6B', width=2)),
            row=4, col=1
        )
        # Clamp y-axis for BP-derived HR
        finite_hr = hr_data['hr_interpolated'][np.isfinite(hr_data['hr_interpolated'])]
        if len(finite_hr) > 0:
            p1, p99 = np.percentile(finite_hr, [1, 99])
            pad = max(5, (p99 - p1) * 0.1)
            fig.update_yaxes(range=[max(0, p1 - pad), p99 + pad], row=4, col=1)

    # Formatting
    fig.update_xaxes(matches='x', rangemode='nonnegative')
    if zoom_range is not None:
        fig.update_xaxes(range=[max(0, zoom_range[0]), zoom_range[1]])

    fig.update_layout(height=height, template='plotly_dark', showlegend=True, hovermode='x unified', uirevision=ui_revision)
    fig.update_traces(connectgaps=False)
    return fig


def _resolve_task_key(task_name):
    """Map a task name from filename to a TASK_EVENTS key."""
    if not task_name:
        return None
    normalised = str(task_name).strip().lower().replace('-', '').replace('_', '')
    # Exact match first
    key = config.TASK_EVENT_ALIASES.get(normalised)
    if key:
        return key
    # Substring match: check if any alias is contained in the task name
    for alias, mapped_key in config.TASK_EVENT_ALIASES.items():
        if alias in normalised or normalised in alias:
            return mapped_key
    return None


def _format_event_time(seconds):
    """Format seconds as m:ss for compact event labels."""
    total = int(round(float(seconds)))
    mins = total // 60
    secs = total % 60
    return f"{mins}:{secs:02d}"


def _resolve_task_events(task_key, participant_label=None):
    """Resolve task events with participant-specific overrides when configured."""
    if participant_label:
        participant_norm = str(participant_label).strip().lower()
        for participant_prefix, per_task_overrides in config.TASK_EVENTS_PARTICIPANT_OVERRIDES.items():
            prefix_norm = str(participant_prefix).strip().lower()
            if participant_norm.startswith(prefix_norm):
                override_events = per_task_overrides.get(task_key)
                if override_events is not None:
                    return override_events
    return config.TASK_EVENTS.get(task_key, [])


def add_task_event_lines(fig, task_name, max_time, session_label=None, participant_label=None):
    """Add event boundary lines plus a readable top timeline legend."""
    task_key = _resolve_task_key(task_name)
    if task_key is None:
        return fig
    if task_key == 'sts' and session_label is not None:
        session_a_aliases = {str(alias).strip().lower() for alias in config.SPIROMETRY_SESSION_A_ALIASES}
        if str(session_label).strip().lower() not in session_a_aliases:
            return fig
    events = _resolve_task_events(task_key, participant_label=participant_label)
    if not events:
        return fig

    visible_events = []
    for t, label, color in events:
        if t > max_time:
            continue
        visible_events.append((t, label, color))
        # Vertical line across all subplot rows
        fig.add_vline(
            x=t, line_dash='dash', line_color=color,
            line_width=1, opacity=0.5,
        )

    # Add readable labels near the top of the first subplot (inside plotting area)
    # so they don't overlap subplot titles.
    if visible_events:
        sorted_events = sorted(visible_events, key=lambda e: e[0])
        min_sep_seconds = max(15.0, float(max_time) * 0.05) if max_time else 15.0
        label_rows = [0.96, 0.88]
        row_idx = 0
        prev_t = None

        edge_pad_seconds = max(20.0, float(max_time) * 0.03) if max_time else 20.0

        for t, label, color in sorted_events:
            if prev_t is None or (t - prev_t) >= min_sep_seconds:
                row_idx = 0
            else:
                row_idx = (row_idx + 1) % len(label_rows)
            prev_t = t

            # Keep edge labels fully visible (e.g., t=0 / last event).
            label_xanchor = 'center'
            if t <= edge_pad_seconds:
                label_xanchor = 'left'
            elif max_time and t >= (max_time - edge_pad_seconds):
                label_xanchor = 'right'

            fig.add_annotation(
                x=t,
                y=label_rows[row_idx],
                xref='x',
                yref='y domain',
                text=f"<b>{label}</b>",
                showarrow=False,
                xanchor=label_xanchor,
                yanchor='top',
                font=dict(size=13, color=color),
                bgcolor='rgba(14, 17, 23, 0.65)',
                bordercolor='rgba(120,120,120,0.35)',
                borderwidth=1,
                borderpad=2,
                row=1,
                col=1,
            )

        # Build a single-line horizontal legend strip with larger font.
        items = [
            f"<span style='color:{color}'>●</span> {_format_event_time(t)} {label}"
            for t, label, color in visible_events
        ]
        legend_text = " | ".join(items)

        # Increase top margin so the single-line timeline strip sits above subplot titles.
        current_margin_t = 0
        if fig.layout.margin and fig.layout.margin.t is not None:
            current_margin_t = int(fig.layout.margin.t)
        required_margin_t = 135
        fig.update_layout(margin=dict(t=max(current_margin_t, required_margin_t)))

        fig.add_annotation(
            x=0.01, y=1.14, xref='paper', yref='paper',
            text=legend_text,
            showarrow=False,
            align='left',
            xanchor='left', yanchor='top',
            font=dict(size=13, color='#E6E6E6'),
            bgcolor='rgba(14, 17, 23, 0.72)',
            bordercolor='rgba(120,120,120,0.45)',
            borderwidth=1,
            borderpad=6,
        )
    return fig


def is_session_a_selected(session_label):
    """Return True when current session should show the external spirometry placeholder."""
    aliases = {str(alias).strip().lower() for alias in config.SPIROMETRY_SESSION_A_ALIASES}
    return str(session_label).strip().lower() in aliases


def render_rsp_like_tab(data, sampling_rate, signal_key, state_prefix, header_title, plot_label):
    """Render an RSP-style processing tab (used for RSP belt and spirometer waveform)."""
    params_state_key = f"{state_prefix}_params"
    result_state_key = f"{state_prefix}_result"

    st.header(header_title)

    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

    with col1:
        method = st.selectbox("Cleaning Method", config.RSP_CLEANING_METHODS, key=f'{state_prefix}_method')
        with st.expander("ℹ️ Method Info"):
            st.info(config.RSP_CLEANING_INFO.get(method, "No info available"))

    with col2:
        peak_method = st.selectbox("Peak Detection", config.RSP_PEAK_METHODS, key=f'{state_prefix}_peak_method')
        with st.expander("ℹ️ Peak Method Info"):
            st.info(config.RSP_PEAK_INFO.get(peak_method, "No info available"))

    with col3:
        amplitude_method = st.selectbox("Amplitude Normalization", config.RSP_AMPLITUDE_METHODS, key=f'{state_prefix}_amplitude')
        with st.expander("ℹ️ Amplitude Info"):
            st.info(config.RSP_AMPLITUDE_INFO.get(amplitude_method, "No info available"))

    with col4:
        st.write("")
        st.write("")
        process_clicked = st.button(
            f"Process {plot_label}",
            type="primary",
            key=f'process_{state_prefix}',
            width='stretch'
        )

    if process_clicked:
        signal = data['df'][data['signal_mappings'][signal_key]].values

        params = {
            'method': method,
            'peak_method': peak_method,
            'amplitude_method': amplitude_method if amplitude_method != 'none' else None
        }
        st.session_state[params_state_key].update(params)

        result = rsp.process_rsp(signal, sampling_rate, st.session_state[params_state_key])

        if result is None:
            st.error("Processing failed: insufficient breaths detected")
        else:
            st.session_state[result_state_key] = result
            st.success(f"{plot_label} processed successfully")

    if st.session_state[result_state_key] is not None:
        result = st.session_state[result_state_key]

        st.subheader("Manual Breath Editing")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Auto Inhalations", len(result['auto_peaks']))
        with col2:
            st.metric("Auto Exhalations", len(result['auto_troughs']))
        with col3:
            n_added_peaks = len(np.setdiff1d(result['current_peaks'], result['auto_peaks']))
            st.metric("Added Inhalations", n_added_peaks)
        with col4:
            n_added_troughs = len(np.setdiff1d(result['current_troughs'], result['auto_troughs']))
            st.metric("Added Exhalations", n_added_troughs)

        time = np.arange(len(result['clean'])) / sampling_rate

        from metrics.rsp import calculate_breathing_rate
        if len(result['current_troughs']) > 1:
            br_data = calculate_breathing_rate(
                result['current_troughs'],
                sampling_rate,
                len(result['clean']),
                rate_method=st.session_state[params_state_key].get('rate_method', 'monotone_cubic')
            )
            result.update(br_data)
        else:
            result['br_bpm'] = np.array([])
            result['br_interpolated'] = np.zeros(len(result['clean']))
            result['mean_br'] = 0.0
            result['std_br'] = 0.0

        region_start_key = f'{state_prefix}_region_start'
        region_end_key = f'{state_prefix}_region_end'
        max_t = float(time[-1])
        if region_start_key not in st.session_state:
            st.session_state[region_start_key] = 0.0
        else:
            st.session_state[region_start_key] = min(st.session_state[region_start_key], max_t)
        if region_end_key not in st.session_state:
            st.session_state[region_end_key] = max_t
        else:
            st.session_state[region_end_key] = min(st.session_state[region_end_key], max_t)

        signal_zoom = (st.session_state[region_start_key], st.session_state[region_end_key])

        fig = create_rsp_bp_plot(
            time, result['raw'], result['clean'],
            result['current_peaks'], result['current_troughs'],
            result['auto_peaks'], result['auto_troughs'],
            plot_label,
            rate_interpolated=result.get('br_interpolated'),
            rate_bpm=result.get('br_bpm'),
            ui_revision=f'{state_prefix}_plot',
            zoom_range=signal_zoom
        )
        if hasattr(st.session_state, 'task'):
            add_task_event_lines(
                fig,
                st.session_state.task,
                float(time[-1]),
                st.session_state.get('session'),
                st.session_state.get('participant'),
            )

        st.plotly_chart(fig, width='stretch')

        st.subheader("Drag-Based Breath Editing")
        st.write("**Quick Range Selection:**")
        col_btn1, col_btn2, col_btn3, col_btn4, col_btn5 = st.columns(5)
        with col_btn1:
            if st.button("⏮️ First 10s", key=f'{state_prefix}_first_10s'):
                st.session_state[region_start_key] = 0.0
                st.session_state[region_end_key] = min(10.0, float(time[-1]))
                st.rerun()
        with col_btn2:
            if st.button("◀️ Previous 10s", key=f'{state_prefix}_prev_10s'):
                window = 10.0
                new_start = max(0.0, st.session_state[region_start_key] - window)
                new_end = max(window, st.session_state[region_end_key] - window)
                st.session_state[region_start_key] = new_start
                st.session_state[region_end_key] = min(new_end, float(time[-1]))
                st.rerun()
        with col_btn3:
            if st.button("▶️ Next 10s", key=f'{state_prefix}_next_10s'):
                window = 10.0
                new_start = min(float(time[-1]) - window, st.session_state[region_start_key] + window)
                new_end = min(float(time[-1]), st.session_state[region_end_key] + window)
                st.session_state[region_start_key] = new_start
                st.session_state[region_end_key] = new_end
                st.rerun()
        with col_btn4:
            if st.button("⏭️ Last 10s", key=f'{state_prefix}_last_10s'):
                st.session_state[region_start_key] = max(0.0, float(time[-1]) - 10.0)
                st.session_state[region_end_key] = float(time[-1])
                st.rerun()
        with col_btn5:
            if st.button("🔄 Reset Range", key=f'{state_prefix}_reset_range'):
                st.session_state[region_start_key] = 0.0
                st.session_state[region_end_key] = float(time[-1])
                st.rerun()

        st.write("**Manual Range Entry:** (Or look at zoomed plot X-axis and enter values)")
        col1, col2 = st.columns(2)
        with col1:
            region_start = st.number_input(
                "Region Start (s)",
                min_value=0.0,
                max_value=float(time[-1]),
                value=float(st.session_state[region_start_key]),
                step=1.0,
                format="%.2f",
                key=f'{state_prefix}_region_start_input',
                help="Enter the start time from the zoomed plot's X-axis, or use quick buttons above"
            )
            st.session_state[region_start_key] = region_start
        with col2:
            region_end = st.number_input(
                "Region End (s)",
                min_value=0.0,
                max_value=float(time[-1]),
                value=float(st.session_state[region_end_key]),
                step=1.0,
                format="%.2f",
                key=f'{state_prefix}_region_end_input',
                help="Enter the end time from the zoomed plot's X-axis, or use quick buttons above"
            )
            st.session_state[region_end_key] = region_end

        st.caption(f"Current range: {region_start:.2f}s to {region_end:.2f}s ({region_end - region_start:.2f}s window) | Full signal: {float(time[-1]):.2f}s")

        st.write("**Inhalation Peaks:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("➕ Add Inhalation Peaks", type="primary", key=f'{state_prefix}_add_peaks_btn', width='stretch'):
                st.session_state[result_state_key]['current_peaks'] = peak_editing.add_peaks_in_range(
                    result['clean'], result['current_peaks'], region_start, region_end, sampling_rate, min_distance_seconds=1.0
                )
                st.rerun()
        with col2:
            if st.button("➖ Remove Inhalation Peaks", type="secondary", key=f'{state_prefix}_remove_peaks_btn', width='stretch'):
                st.session_state[result_state_key]['current_peaks'] = peak_editing.erase_peaks_in_range(
                    result['current_peaks'], region_start, region_end, sampling_rate
                )
                st.rerun()
        with col3:
            if st.button("🔄 Reset Inhalations", key=f'{state_prefix}_reset_peaks_btn', width='stretch'):
                st.session_state[result_state_key]['current_peaks'] = result['auto_peaks'].copy()
                st.rerun()

        st.write("**Exhalation Troughs:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("➕ Add Exhalation Troughs", type="primary", key=f'{state_prefix}_add_troughs_btn', width='stretch'):
                st.session_state[result_state_key]['current_troughs'] = peak_editing.add_troughs_in_range(
                    result['clean'], result['current_troughs'], region_start, region_end, sampling_rate, min_distance_seconds=1.0
                )
                st.rerun()
        with col2:
            if st.button("➖ Remove Exhalation Troughs", type="secondary", key=f'{state_prefix}_remove_troughs_btn', width='stretch'):
                st.session_state[result_state_key]['current_troughs'] = peak_editing.erase_troughs_in_range(
                    result['current_troughs'], region_start, region_end, sampling_rate
                )
                st.rerun()
        with col3:
            if st.button("🔄 Reset Exhalations", key=f'{state_prefix}_reset_troughs_btn', width='stretch'):
                st.session_state[result_state_key]['current_troughs'] = result['auto_troughs'].copy()
                st.rerun()

        with st.expander("✏️ Single Peak/Trough Editing (Advanced)"):
            st.write("Add or remove individual peaks/troughs at specific times.")
            col1, _ = st.columns(2)
            with col1:
                single_time = st.number_input(
                    "Time (seconds)",
                    min_value=0.0,
                    max_value=float(time[-1]),
                    value=0.0,
                    step=0.1,
                    key=f'{state_prefix}_single_time'
                )

            st.write("**Inhalation Peaks:**")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Add Inhalation at Time", key=f'{state_prefix}_single_add_peak'):
                    st.session_state[result_state_key]['current_peaks'] = peak_editing.add_peak(
                        result['clean'], result['current_peaks'], single_time, sampling_rate
                    )
                    st.rerun()
            with col_b:
                if st.button("Delete Inhalation at Time", key=f'{state_prefix}_single_del_peak'):
                    st.session_state[result_state_key]['current_peaks'] = peak_editing.delete_peak(
                        result['current_peaks'], single_time, sampling_rate
                    )
                    st.rerun()

            st.write("**Exhalation Troughs:**")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Add Exhalation at Time", key=f'{state_prefix}_single_add_trough'):
                    st.session_state[result_state_key]['current_troughs'] = peak_editing.add_trough(
                        result['clean'], result['current_troughs'], single_time, sampling_rate
                    )
                    st.rerun()
            with col_b:
                if st.button("Delete Exhalation at Time", key=f'{state_prefix}_single_del_trough'):
                    st.session_state[result_state_key]['current_troughs'] = peak_editing.delete_trough(
                        result['current_troughs'], single_time, sampling_rate
                    )
                    st.rerun()

        st.subheader("Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Final Breath Count", len(result['current_troughs']))
        with col2:
            st.metric("Mean BR", f"{result['mean_br']:.1f} bpm")
        with col3:
            st.metric("BR Std Dev", f"{result['std_br']:.1f} bpm")


@st.fragment
def _build_neuro_tab(tab_key, image_paths, overlay_config):
    """
    Shared logic for the Structural and Functional neuro tabs.

    Decorated with @st.fragment so widget interactions inside a tab only
    re-execute this function, not the entire page (avoids reloading the
    other tab's NIfTI data).

    NIfTI files are served by a background HTTP server (file_server)
    and loaded by NiiVue via URL.  This keeps the HTML payload tiny
    and avoids Tornado's Content-Encoding decompression of .nii.gz.

    Parameters
    ----------
    tab_key : str
        Prefix for widget keys ('struct' or 'func').
    image_paths : dict
        {image_key: absolute_path} — must include 'T1w' for background.
    overlay_config : dict
        Overlay defaults from config (e.g. config.STRUCTURAL_OVERLAYS).
    """
    if not image_paths:
        st.warning("No images found for this subject")
        return

    # -- Build unified image config: T1w first, then overlays that exist --
    all_images = {}
    if 'T1w' in image_paths:
        all_images['T1w'] = {
            'colormap': 'gray', 'opacity': 1.0, 'label': 'T1w',
        }
    for key, defaults in overlay_config.items():
        if 'variants' in defaults:
            # Include if ANY variant file exists
            if any(v['key'] in image_paths for v in defaults['variants'].values()):
                all_images[key] = defaults
        elif key in image_paths:
            all_images[key] = defaults

    # -- Session state for layer ordering --
    layers_key = f'{tab_key}_layers'
    prev_sel_key = f'{tab_key}_prev_sel'

    if layers_key not in st.session_state:
        st.session_state[layers_key] = ['T1w'] if 'T1w' in all_images else []
    if prev_sel_key not in st.session_state:
        st.session_state[prev_sel_key] = set(st.session_state[layers_key])

    # Prune stale keys (e.g. after subject change)
    valid_keys = set(all_images.keys())
    st.session_state[layers_key] = [k for k in st.session_state[layers_key] if k in valid_keys]
    st.session_state[prev_sel_key] = st.session_state[prev_sel_key] & valid_keys

    # -- Label ↔ key mappings --
    label_to_key = {cfg['label']: key for key, cfg in all_images.items()}
    key_to_label = {key: cfg['label'] for key, cfg in all_images.items()}

    # -- Create containers in visual order --
    pills_container = st.container()
    viewer_container = st.container()
    stack_container = st.container()
    paths_container = st.container()

    # -- Pills: image selection --
    with pills_container:
        current_labels = [key_to_label[k] for k in st.session_state[layers_key]
                          if k in key_to_label]
        selected_labels = st.pills(
            "Images",
            list(key_to_label.values()),
            selection_mode="multi",
            default=current_labels,
            key=f'{tab_key}_pills',
        )

    # -- Diff to detect added/removed images --
    selected_keys = {label_to_key[lbl] for lbl in selected_labels} if selected_labels else set()
    prev_selected = st.session_state[prev_sel_key]
    added = selected_keys - prev_selected
    removed = prev_selected - selected_keys

    layers = [k for k in st.session_state[layers_key] if k not in removed]
    # Append newly added in config order (deterministic when multiple added at once)
    for key in all_images:
        if key in added:
            layers.append(key)
    st.session_state[layers_key] = layers
    st.session_state[prev_sel_key] = selected_keys

    # -- Auto-detect available masks from image_paths --
    available_masks = {k: v for k, v in image_paths.items() if 'mask' in k.lower()}

    # -- Layer stack UI (render before viewer so popover widgets populate state) --
    with stack_container:
        if layers:
            st.caption("Layer Stack")
            num_layers = len(layers)
            for display_idx, key in enumerate(reversed(layers)):
                stack_pos = num_layers - 1 - display_idx
                defaults = all_images[key]

                cols = st.columns([0.6, 2.5, 0.5, 0.5, 0.5, 0.5])

                # -- Visibility state --
                vis_key = f'{tab_key}_visible_{key}'
                if vis_key not in st.session_state:
                    st.session_state[vis_key] = True
                is_visible = st.session_state[vis_key]

                with cols[0]:
                    if stack_pos == num_layers - 1:
                        st.markdown("**TOP**")
                    elif stack_pos == 0:
                        st.markdown("**BTM**")
                    else:
                        st.markdown(f"**{stack_pos + 1}**")

                with cols[1]:
                    cur_cmap = st.session_state.get(
                        f'{tab_key}_cmap_{key}', defaults['colormap'])
                    cur_invert = st.session_state.get(
                        f'{tab_key}_invert_{key}', False)
                    gradient = colormap_css(cur_cmap, invert=cur_invert)
                    # Resolve effective cal_min/cal_max for display
                    disp_min = st.session_state.get(f'{tab_key}_cal_min_{key}')
                    disp_max = st.session_state.get(f'{tab_key}_cal_max_{key}')
                    if disp_min is None:
                        disp_min = defaults.get('cal_min')
                    if disp_max is None:
                        disp_max = defaults.get('cal_max')
                    # Build colorbar with tick marks when min/max are known
                    has_range = (disp_min is not None and disp_max is not None)
                    if is_visible:
                        if has_range:
                            n_ticks = 5
                            tick_vals = [
                                disp_min + i * (disp_max - disp_min) / (n_ticks - 1)
                                for i in range(n_ticks)
                            ]
                            tick_html = ''.join(
                                f'<span>{v:.2g}</span>' for v in tick_vals
                            )
                            st.markdown(
                                f'{defaults["label"]}'
                                f'<div style="height:10px;border-radius:2px;'
                                f'background:{gradient};margin-top:2px;"></div>'
                                f'<div style="display:flex;justify-content:space-between;'
                                f'font-size:14px;color:#888;margin-top:1px;">'
                                f'{tick_html}</div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f'{defaults["label"]}'
                                f'<div style="height:4px;border-radius:2px;'
                                f'background:{gradient};margin-top:2px;"></div>',
                                unsafe_allow_html=True,
                            )
                    else:
                        st.markdown(
                            f'~~{defaults["label"]}~~'
                            f'<div style="height:4px;border-radius:2px;'
                            f'background:#333;margin-top:2px;"></div>',
                            unsafe_allow_html=True,
                        )

                with cols[2]:
                    icon = "\U0001F441" if is_visible else "\u2014"
                    if st.button(icon, key=f'{tab_key}_vis_btn_{key}',
                                 help="Toggle visibility"):
                        st.session_state[vis_key] = not is_visible
                        st.rerun(scope="fragment")

                with cols[3]:
                    if stack_pos < num_layers - 1:
                        if st.button("\u2191", key=f'{tab_key}_up_{key}'):
                            lst = st.session_state[layers_key]
                            lst[stack_pos], lst[stack_pos + 1] = lst[stack_pos + 1], lst[stack_pos]
                            st.rerun(scope="fragment")

                with cols[4]:
                    if stack_pos > 0:
                        if st.button("\u2193", key=f'{tab_key}_down_{key}'):
                            lst = st.session_state[layers_key]
                            lst[stack_pos], lst[stack_pos - 1] = lst[stack_pos - 1], lst[stack_pos]
                            st.rerun(scope="fragment")

                with cols[5]:
                    with st.popover("⚙", use_container_width=True):
                        # Variant selector (e.g. tissue seg: All/GM/WM/CSF)
                        if 'variants' in defaults:
                            available = {name: v for name, v in defaults['variants'].items()
                                         if v['key'] in image_paths}
                            variant_name = st.selectbox(
                                "Type",
                                list(available.keys()),
                                key=f'{tab_key}_variant_{key}',
                            )
                            # Update default colormap to match the selected variant
                            variant_cmap = available[variant_name]['colormap']
                        else:
                            variant_cmap = defaults['colormap']

                        default_cmap = variant_cmap
                        st.selectbox(
                            "Colormap",
                            config.NIIVUE_COLORMAPS,
                            index=config.NIIVUE_COLORMAPS.index(default_cmap)
                                  if default_cmap in config.NIIVUE_COLORMAPS else 0,
                            key=f'{tab_key}_cmap_{key}',
                        )
                        st.checkbox(
                            "Invert colormap",
                            key=f'{tab_key}_invert_{key}',
                        )
                        st.slider(
                            "Opacity", 0.0, 1.0, defaults['opacity'],
                            key=f'{tab_key}_opacity_{key}',
                        )
                        st.number_input(
                            "Cal Min",
                            value=defaults.get('cal_min'),
                            format="%.1f",
                            help="Minimum display intensity (empty = auto)",
                            key=f'{tab_key}_cal_min_{key}',
                        )
                        st.number_input(
                            "Cal Max",
                            value=defaults.get('cal_max'),
                            format="%.1f",
                            help="Maximum display intensity (empty = auto)",
                            key=f'{tab_key}_cal_max_{key}',
                        )

                        # -- Mask controls --
                        # Don't show mask controls for mask images or the T1w background
                        is_background = key == 'T1w'
                        if available_masks and 'mask' not in key.lower() and not is_background:
                            st.divider()
                            mask_labels = {k: k.replace('_', ' ').title()
                                           for k in available_masks}
                            # Default mask ON for overlay images
                            mask_on_key = f'{tab_key}_mask_on_{key}'
                            if mask_on_key not in st.session_state:
                                st.session_state[mask_on_key] = True
                            st.checkbox(
                                "Apply mask",
                                key=mask_on_key,
                            )
                            if st.session_state.get(mask_on_key, True):
                                mask_keys = list(available_masks.keys())
                                st.selectbox(
                                    "Mask",
                                    mask_keys,
                                    format_func=lambda k: mask_labels[k],
                                    key=f'{tab_key}_mask_sel_{key}',
                                )
                                st.slider(
                                    "Mask strength", 0.0, 1.0, 1.0,
                                    help="1.0 = fully black outside mask, 0.0 = no masking",
                                    key=f'{tab_key}_mask_opacity_{key}',
                                )

    # -- Build volumes from session state and render viewer --
    with viewer_container:
        if not layers:
            st.info("Select at least one image above to view")
            return

        volumes = []
        for key in layers:
            # Skip hidden layers (visibility toggled off)
            if not st.session_state.get(f'{tab_key}_visible_{key}', True):
                continue

            defaults = all_images[key]
            cmap = st.session_state.get(f'{tab_key}_cmap_{key}', defaults['colormap'])
            opacity = st.session_state.get(f'{tab_key}_opacity_{key}', defaults['opacity'])

            # Resolve the actual file path key (may differ for variant overlays)
            file_key = key
            if 'variants' in defaults:
                variant_name = st.session_state.get(f'{tab_key}_variant_{key}')
                if variant_name and variant_name in defaults['variants']:
                    file_key = defaults['variants'][variant_name]['key']
                else:
                    # Fall back to first available variant
                    for v in defaults['variants'].values():
                        if v['key'] in image_paths:
                            file_key = v['key']
                            break

            if file_key not in image_paths:
                continue

            # Apply mask if enabled for this layer
            volume_filepath = image_paths[file_key]
            is_masked = False
            mask_result = None
            is_background = key == 'T1w'
            mask_on = st.session_state.get(f'{tab_key}_mask_on_{key}', False)
            if mask_on and 'mask' not in key.lower() and not is_background:
                mask_sel = st.session_state.get(f'{tab_key}_mask_sel_{key}')
                mask_strength = st.session_state.get(f'{tab_key}_mask_opacity_{key}', 1.0)
                if mask_sel and mask_sel in image_paths and mask_strength > 0:
                    mask_result = create_masked_volume(
                        volume_filepath, image_paths[mask_sel], mask_strength,
                    )
                    volume_filepath = mask_result['path']
                    is_masked = True

            invert = st.session_state.get(f'{tab_key}_invert_{key}', False)
            vol = {
                'path': register_file(volume_filepath),
                'name': f'{file_key}.nii.gz',
                'colormap': cmap,
                'opacity': opacity,
            }
            if invert:
                vol['colormap_invert'] = True

            cal_min = st.session_state.get(f'{tab_key}_cal_min_{key}')
            cal_max = st.session_state.get(f'{tab_key}_cal_max_{key}')
            # Masking uses a sentinel value (-1e10) for outside-brain voxels.
            # NiiVue must have explicit cal_min/cal_max so it doesn't auto-range
            # to include the sentinel. Use config defaults first, then fall
            # back to the robust range computed from inside-mask voxels.
            if is_masked and cal_min is None:
                cal_min = defaults.get('cal_min')
                if cal_min is None and mask_result:
                    cal_min = mask_result['cal_min']
            if is_masked and cal_max is None:
                cal_max = defaults.get('cal_max')
                if cal_max is None and mask_result:
                    cal_max = mask_result['cal_max']
            if cal_min is not None:
                vol['cal_min'] = cal_min
            if cal_max is not None:
                vol['cal_max'] = cal_max

            volumes.append(vol)

        html = build_niivue_html(volumes, height=700, viewer_id=tab_key)
        components.html(html, height=720, scrolling=False)

    # -- File paths --
    with paths_container:
        with st.expander("File paths"):
            for key in image_paths:
                st.text(f"{key}: {image_paths[key]}")


def run_neuro_mode():
    """Neuro mode: NIfTI image viewer with NiiVue (via Streamlit static serving)."""
    import os

    bids_path = config.BIDS_DATA_PATH
    deriv_path = config.FMRIPREP_DERIVATIVES_PATH

    # --- Sidebar: data selection ---
    with st.sidebar:
        if not os.path.isdir(bids_path):
            st.error(f"BIDS path not found:\n`{bids_path}`")
            st.stop()

        subjects = scan_bids_subjects(bids_path)
        if not subjects:
            st.error(f"No subjects found in {bids_path}")
            return

        participants = list(subjects.keys())
        participant = st.selectbox("Participant", participants, key='neuro_part_sel')

        sessions = subjects[participant]
        session = st.selectbox("Session", sessions, key='neuro_ses_sel')

        if st.button("Load Images", type="primary"):
            smri_path = getattr(config, 'SMRI_DERIVATIVES_PATH', None)
            struct_imgs = get_structural_images(bids_path, deriv_path, participant, session,
                                                smri_path=smri_path)
            func_tasks = get_available_tasks(deriv_path, participant, session)

            st.session_state.neuro_structural_images = struct_imgs
            st.session_state.neuro_functional_tasks = func_tasks
            st.session_state.neuro_participant = participant
            st.session_state.neuro_session = session
            st.session_state.neuro_data_loaded = True

            # Check for CVR/GLM derivatives
            cvr_path = config.CVR_DERIVATIVES_PATH
            st.session_state.neuro_glm_available = os.path.isdir(cvr_path)

            if struct_imgs:
                st.success(f"Found {len(struct_imgs)} structural images, {len(func_tasks)} tasks")
            else:
                st.warning("No fMRIPrep derivatives found for this subject")

        if st.session_state.neuro_data_loaded:
            st.divider()
            st.markdown(f"**Subject**: {st.session_state.neuro_participant}")
            st.markdown(f"**Session**: {st.session_state.neuro_session}")

    # --- Main area ---
    if not st.session_state.neuro_data_loaded:
        st.info("Select a participant and session from the sidebar to view NIfTI images")
        return

    struct_images = st.session_state.neuro_structural_images
    func_tasks = st.session_state.neuro_functional_tasks

    glm_available = st.session_state.get('neuro_glm_available', False)

    tab_names = ["Structural"]
    if func_tasks:
        tab_names.append("Functional")
    if glm_available:
        tab_names.append("CVR")

    tabs = st.tabs(tab_names)

    # ---- Structural tab ----
    with tabs[0]:
        st.header("Structural Images")
        _build_neuro_tab('struct', struct_images, config.STRUCTURAL_OVERLAYS)

    # ---- Functional tab ----
    if func_tasks and len(tabs) > 1:
        with tabs[tab_names.index("Functional")]:
            st.header("Functional Images")

            task = st.selectbox("Task", func_tasks, key='neuro_func_task')

            func_images = get_functional_images(
                config.FMRIPREP_DERIVATIVES_PATH,
                st.session_state.neuro_participant,
                st.session_state.neuro_session,
                task,
            )

            _build_neuro_tab('func', func_images, config.FUNCTIONAL_OVERLAYS)

    # ---- CVR tab ----
    if glm_available and "CVR" in tab_names:
        with tabs[tab_names.index("CVR")]:
            st.header("CVR Maps")

            participant = st.session_state.neuro_participant
            session = st.session_state.neuro_session

            methods = get_cvr_methods(config.CVR_DERIVATIVES_PATH)
            if not methods:
                st.warning("No CVR methods found in derivatives")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    method = st.selectbox("Method", methods, key='glm_method_sel')
                with col2:
                    spaces = get_cvr_spaces(config.CVR_DERIVATIVES_PATH, method)
                    if not spaces:
                        st.warning(f"No spaces found for method {method}")
                    else:
                        space = st.selectbox("Space", spaces, key='glm_space_sel')

                if methods and spaces:
                    # Task selector — auto-detect available tasks
                    tasks = get_cvr_tasks(
                        config.CVR_DERIVATIVES_PATH, method, space,
                        participant, session,
                    )
                    if len(tasks) > 1:
                        task = st.selectbox("Task", tasks, key='glm_task_sel')
                    elif tasks:
                        task = tasks[0]
                    else:
                        task = 'gas'

                    glm_images = get_glm_images(
                        config.CVR_DERIVATIVES_PATH,
                        config.FMRIPREP_DERIVATIVES_PATH,
                        participant,
                        session,
                        method,
                        space,
                        task=task,
                    )

                    if not glm_images:
                        st.warning(
                            f"No GLM images found for {participant}/{session} "
                            f"(method={method}, space={space}, task={task})"
                        )
                    else:
                        _build_neuro_tab('glm', glm_images, config.GLM_OVERLAYS)


def main():
    """Main application function"""
    init_session_state()

    with st.sidebar:
        mode = st.radio("Mode", ["Physio", "Neuro"], horizontal=True, key='app_mode')

    if mode == "Neuro":
        st.title("Neuro QC")
        run_neuro_mode()
        return

    st.title("Physiological Signal QC")

    with st.sidebar:
        st.header("Data Selection")

        # Check if data path exists and provide override option
        import os
        data_path_exists = os.path.isdir(config.BASE_DATA_PATH)

        if not data_path_exists:
            st.warning(f"⚠️ Default data path doesn't exist:\n`{config.BASE_DATA_PATH}`")

            with st.expander("🔧 Configure Data Paths", expanded=True):
                st.info("Update the paths below to point to your data directories, then click Apply.")

                # Allow user to override paths
                custom_data_path = st.text_input(
                    "Data Path",
                    value=config.BASE_DATA_PATH,
                    help="Path to raw physiological data files"
                )

                custom_output_path = st.text_input(
                    "Output Path",
                    value=config.OUTPUT_BASE_PATH,
                    help="Path where processed data will be saved"
                )

                if st.button("Apply Paths", type="primary"):
                    if os.path.isdir(custom_data_path):
                        config.BASE_DATA_PATH = custom_data_path
                        config.OUTPUT_BASE_PATH = custom_output_path
                        st.success("✅ Paths updated! Scanning for data...")
                        st.rerun()
                    else:
                        st.error(f"❌ Path doesn't exist: {custom_data_path}")

                st.stop()

        participants_data = scan_data_directory(config.BASE_DATA_PATH)

        if not participants_data:
            st.error(f"No data found in {config.BASE_DATA_PATH}")
            st.info("Make sure your data follows the expected structure:\n"
                    "`sub-{id}/ses-{id}/sub-{id}_ses-{id}_task-{task}_physio.{acq,csv}`")
            return

        participants = list(participants_data.keys())
        participant = st.selectbox("Participant", participants)

        sessions = list(participants_data[participant].keys())
        session = st.selectbox("Session", sessions)

        tasks = participants_data[participant][session]
        task = st.selectbox("Task", tasks)

        if st.button("Load Data", type="primary"):
            file_path = find_file_path(config.BASE_DATA_PATH, participant, session, task)

            if file_path is None:
                st.error("File not found")
                return

            data = load_acq_file(
                file_path,
                participant=participant,
                session=session,
                task=task
            )

            if data is None:
                st.error("Failed to load file")
                return

            st.session_state.loaded_data = data
            st.session_state.data_loaded = True
            st.session_state.participant = participant
            st.session_state.session = session
            st.session_state.task = task

            st.session_state.ecg_result = None
            st.session_state.rsp_result = None
            st.session_state.ppg_result = None
            st.session_state.bp_result = None
            st.session_state.etco2_result = None
            st.session_state.eto2_result = None
            st.session_state.spo2_result = None
            st.session_state.spirometer_result = None

            # Clear stale zoom/region states so they reinitialise for new data
            for key in list(st.session_state.keys()):
                if key.endswith('_region_start') or key.endswith('_region_end') or key.endswith('_zoom_range'):
                    del st.session_state[key]

            st.success(f"Loaded {file_path}")

        if st.session_state.data_loaded:
            data = st.session_state.loaded_data
            st.info(f"""
            **Samples**: {data['n_samples']:,}
            **Duration**: {data['duration']:.1f}s
            **Sampling Rate**: {data['sampling_rate']} Hz
            **Signals**: {', '.join(data['signal_mappings'].keys())}
            """)

            pmu_status = data.get('pmu_status', {})
            if pmu_status.get('attempted'):
                if pmu_status.get('success'):
                    scan_idx = pmu_status.get('scan_index')
                    strategy = pmu_status.get('match_strategy', 'unknown')
                    st.success(
                        f"PMU Session B enrichment active: scan #{scan_idx} "
                        f"(match: {strategy})"
                    )
                else:
                    st.warning(f"PMU enrichment not applied: {pmu_status.get('message', 'unknown reason')}")

    if not st.session_state.data_loaded:
        st.info("👈 Select data from the sidebar to begin")
        return

    data = st.session_state.loaded_data
    sampling_rate = data['sampling_rate']
    detected_signals = list(data['signal_mappings'].keys())
    session_a_selected = is_session_a_selected(st.session_state.session)

    tabs = []
    if 'ecg' in detected_signals:
        tabs.append("ECG")
    if 'rsp' in detected_signals:
        tabs.append("RSP")
    if 'spirometer' in detected_signals:
        tabs.append("Spirometer")
    if 'ppg' in detected_signals:
        tabs.append("PPG")
    if 'bp' in detected_signals:
        tabs.append("Blood Pressure")
    if 'etco2' in detected_signals:
        tabs.append("ETCO2")
    if 'eto2' in detected_signals:
        tabs.append("ETO2")
    if 'spo2' in detected_signals:
        tabs.append("SpO2")
    if session_a_selected:
        tabs.append("Spirometry")
    tabs.append("Export")

    tab_objects = st.tabs(tabs)
    tab_idx = 0

    if 'ecg' in detected_signals:
        with tab_objects[tab_idx]:
            st.header("ECG Processing")

            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                method = st.selectbox("Cleaning Method", config.ECG_CLEANING_METHODS, key='ecg_method')
                with st.expander("ℹ️ Method Info"):
                    st.info(config.ECG_CLEANING_INFO.get(method, "No info available"))

                if method == 'custom':
                    st.subheader("Custom Filter Options")
                    filter_type = st.selectbox("Filter Type", config.FILTER_TYPES, key='ecg_filter_type')
                    filter_mode = st.radio("Filter Mode", ["Bandpass", "Lowpass", "Highpass"], horizontal=True, key='ecg_filter_mode')

                    col_f1, col_f2 = st.columns(2)
                    with col_f1:
                        if filter_mode in ["Highpass", "Bandpass"]:
                            lowcut = st.number_input("High-pass (Hz)", min_value=0.01, max_value=100.0, value=0.5, step=0.1, key='ecg_lowcut')
                        else:
                            lowcut = None
                    with col_f2:
                        if filter_mode in ["Lowpass", "Bandpass"]:
                            highcut = st.number_input("Low-pass (Hz)", min_value=0.1, max_value=200.0, value=45.0, step=0.5, key='ecg_highcut')
                        else:
                            highcut = None

                    filter_order = st.slider("Filter Order", min_value=1, max_value=10, value=5, key='ecg_filter_order')
                else:
                    filter_type = 'butterworth'
                    filter_mode = "Bandpass"
                    lowcut = 0.5
                    highcut = 45.0
                    filter_order = 5

                peak_method = st.selectbox("Peak Detection", config.ECG_PEAK_METHODS, key='ecg_peak')
                with st.expander("ℹ️ Peak Method Info"):
                    st.info(config.ECG_PEAK_INFO.get(peak_method, "No info available"))

            with col2:
                powerline = st.selectbox("Powerline Frequency", config.POWERLINE_FREQUENCIES, key='ecg_powerline')
                correct_artifacts = st.checkbox("Artifact Correction", key='ecg_correct')
                calculate_quality = st.checkbox("Calculate Quality", value=True, key='ecg_quality')

            with col3:
                st.write("")
                st.write("")
                process_ecg_clicked = st.button("Process ECG", type="primary", key='process_ecg', width='stretch')

            if process_ecg_clicked:
                signal = data['df'][data['signal_mappings']['ecg']].values

                params = {
                    'method': method,
                    'powerline': powerline,
                    'peak_method': peak_method,
                    'correct_artifacts': correct_artifacts,
                    'calculate_quality': calculate_quality,
                    'filter_type': filter_type,
                    'filter_order': filter_order,
                    'lowcut': lowcut,
                    'highcut': highcut,
                    'apply_lowcut': lowcut is not None,
                    'apply_highcut': highcut is not None
                }
                st.session_state.ecg_params.update(params)

                result = ecg.process_ecg(signal, sampling_rate, st.session_state.ecg_params)

                if result is None:
                    st.error("Processing failed: insufficient peaks detected")
                else:
                    st.session_state.ecg_result = result
                    st.success("ECG processed successfully")

            if st.session_state.ecg_result is not None:
                result = st.session_state.ecg_result

                st.subheader("Manual Peak Editing")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Auto-detected Peaks", len(result['auto_r_peaks']))
                with col2:
                    n_added = len(np.setdiff1d(result['current_r_peaks'], result['auto_r_peaks']))
                    st.metric("Manually Added", n_added)
                with col3:
                    n_deleted = len(np.setdiff1d(result['auto_r_peaks'], result['current_r_peaks']))
                    st.metric("Deleted", n_deleted)

                # Quality metrics selection
                quality_available = []
                if st.session_state.ecg_params.get('calculate_quality', False):
                    st.subheader("Quality Metrics Display")

                    # Continuous metrics
                    continuous_metrics = []
                    if result.get('quality_templatematch') is not None:
                        continuous_metrics.append('quality_templatematch')
                    if result.get('quality_averageqrs') is not None:
                        continuous_metrics.append('quality_averageqrs')

                    # Overall metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if result.get('quality_templatematch_mean') is not None:
                            st.metric("Template Match Quality", f"{result['quality_templatematch_mean']:.3f}")
                    with col2:
                        if result.get('quality_averageqrs_mean') is not None:
                            st.metric("Average QRS Quality", f"{result['quality_averageqrs_mean']:.3f}")
                    with col3:
                        if result.get('quality_mean') is not None:
                            st.metric("Overall Quality", f"{result['quality_mean']:.3f}")

                    # Multiselect for continuous metrics
                    selected_quality_metrics = st.multiselect(
                        "Select continuous quality metrics to plot:",
                        options=continuous_metrics,
                        default=continuous_metrics,
                        key='ecg_quality_select',
                        format_func=lambda x: 'Template Match' if x == 'quality_templatematch' else 'Average QRS'
                    )
                else:
                    selected_quality_metrics = []

                time = np.arange(len(result['clean'])) / sampling_rate

                # Recalculate HR based on current peaks
                from metrics.ecg import calculate_hr
                if len(result['current_r_peaks']) > 1:
                    hr_data = calculate_hr(
                        result['current_r_peaks'],
                        sampling_rate,
                        len(result['clean']),
                        rate_method=st.session_state.ecg_params.get('rate_method', 'monotone_cubic')
                    )
                    result.update(hr_data)
                else:
                    result['hr_bpm'] = np.array([])
                    result['hr_interpolated'] = np.zeros(len(result['clean']))
                    result['mean_hr'] = 0.0
                    result['std_hr'] = 0.0

                # Create quality data dict for plotting
                quality_data = {
                    'quality_templatematch': result.get('quality_templatematch'),
                    'quality_averageqrs': result.get('quality_averageqrs')
                }

                # Initialize region range in session state if not exists (needed before plotting)
                ecg_max_t = float(time[-1])
                if 'ecg_region_start' not in st.session_state:
                    st.session_state.ecg_region_start = 0.0
                else:
                    st.session_state.ecg_region_start = min(st.session_state.ecg_region_start, ecg_max_t)
                if 'ecg_region_end' not in st.session_state:
                    st.session_state.ecg_region_end = ecg_max_t
                else:
                    st.session_state.ecg_region_end = min(st.session_state.ecg_region_end, ecg_max_t)

                # Get zoom range from session state
                ecg_zoom = (st.session_state.ecg_region_start, st.session_state.ecg_region_end)

                fig = create_signal_plot(
                    time, result['raw'], result['clean'],
                    result['current_r_peaks'], result['auto_r_peaks'],
                    'ECG', sampling_rate,
                    hr_interpolated=result.get('hr_interpolated'),
                    hr_bpm=result.get('hr_bpm'),
                    selected_quality_metrics=selected_quality_metrics,
                    quality_data=quality_data,
                    ui_revision='ecg_plot',  # Preserve zoom state
                    zoom_range=ecg_zoom  # Apply zoom from region inputs
                )
                add_task_event_lines(
                    fig,
                    st.session_state.task,
                    float(time[-1]),
                    st.session_state.get('session'),
                    st.session_state.get('participant'),
                )

                st.plotly_chart(fig, width='stretch')

                # Drag-based editing interface
                st.subheader("Drag-Based Peak Editing")

                # Quick navigation buttons
                st.write("**Quick Range Selection:**")
                col_btn1, col_btn2, col_btn3, col_btn4, col_btn5 = st.columns(5)
                with col_btn1:
                    if st.button("⏮️ First 10s", key='ecg_first_10s'):
                        st.session_state.ecg_region_start = 0.0
                        st.session_state.ecg_region_end = min(10.0, float(time[-1]))
                        st.rerun()
                with col_btn2:
                    if st.button("◀️ Previous 10s", key='ecg_prev_10s'):
                        window = 10.0
                        new_start = max(0.0, st.session_state.ecg_region_start - window)
                        new_end = max(window, st.session_state.ecg_region_end - window)
                        st.session_state.ecg_region_start = new_start
                        st.session_state.ecg_region_end = min(new_end, float(time[-1]))
                        st.rerun()
                with col_btn3:
                    if st.button("▶️ Next 10s", key='ecg_next_10s'):
                        window = 10.0
                        new_start = min(float(time[-1]) - window, st.session_state.ecg_region_start + window)
                        new_end = min(float(time[-1]), st.session_state.ecg_region_end + window)
                        st.session_state.ecg_region_start = new_start
                        st.session_state.ecg_region_end = new_end
                        st.rerun()
                with col_btn4:
                    if st.button("⏭️ Last 10s", key='ecg_last_10s'):
                        st.session_state.ecg_region_start = max(0.0, float(time[-1]) - 10.0)
                        st.session_state.ecg_region_end = float(time[-1])
                        st.rerun()
                with col_btn5:
                    if st.button("🔄 Reset Range", key='ecg_reset_range'):
                        st.session_state.ecg_region_start = 0.0
                        st.session_state.ecg_region_end = float(time[-1])
                        st.rerun()

                st.write("**Manual Range Entry:** (Or look at zoomed plot X-axis and enter values)")
                col1, col2 = st.columns(2)
                with col1:
                    region_start = st.number_input(
                        "Region Start (s)",
                        min_value=0.0,
                        max_value=float(time[-1]),
                        value=float(st.session_state.ecg_region_start),
                        step=1.0,
                        format="%.2f",
                        key='ecg_region_start_input',
                        help="Enter the start time from the zoomed plot's X-axis, or use quick buttons above"
                    )
                    # Update session state
                    st.session_state.ecg_region_start = region_start
                with col2:
                    region_end = st.number_input(
                        "Region End (s)",
                        min_value=0.0,
                        max_value=float(time[-1]),
                        value=float(st.session_state.ecg_region_end),
                        step=1.0,
                        format="%.2f",
                        key='ecg_region_end_input',
                        help="Enter the end time from the zoomed plot's X-axis, or use quick buttons above"
                    )
                    # Update session state
                    st.session_state.ecg_region_end = region_end

                # Show current range info
                st.caption(f"Current range: {region_start:.2f}s to {region_end:.2f}s ({region_end - region_start:.2f}s window) | Full signal: {float(time[-1]):.2f}s")

                st.write("**Choose Action:**")
                col1, col2, col3 = st.columns(3)

                with col1:
                    if st.button("➕ Add R-Peaks in Region", type="primary", key='ecg_add_region_btn', width='stretch'):
                        from utils import peak_editing
                        st.session_state.ecg_result['current_r_peaks'] = peak_editing.add_peaks_in_range(
                            result['clean'],
                            result['current_r_peaks'],
                            region_start,
                            region_end,
                            sampling_rate
                        )
                        st.rerun()

                with col2:
                    if st.button("➖ Remove R-Peaks in Region", type="secondary", key='ecg_remove_region_btn', width='stretch'):
                        from utils import peak_editing
                        st.session_state.ecg_result['current_r_peaks'] = peak_editing.erase_peaks_in_range(
                            result['current_r_peaks'],
                            region_start,
                            region_end,
                            sampling_rate
                        )
                        st.rerun()

                with col3:
                    if st.button("🔄 Reset to Auto-Detected", key='ecg_reset_btn', width='stretch'):
                        st.session_state.ecg_result['current_r_peaks'] = result['auto_r_peaks'].copy()
                        st.rerun()

                # Single peak editing
                with st.expander("✏️ Single Peak Editing (Advanced)"):
                    st.write("Add or remove individual peaks at specific times.")
                    col1, col2 = st.columns(2)
                    with col1:
                        single_peak_time = st.number_input(
                            "Time (seconds)",
                            min_value=0.0,
                            max_value=float(time[-1]),
                            value=0.0,
                            step=0.1,
                            key='ecg_single_peak_time'
                        )
                    with col2:
                        col_a, col_b = st.columns(2)
                        with col_a:
                            if st.button("Add at Time", key='ecg_single_add_btn'):
                                from utils import peak_editing
                                st.session_state.ecg_result['current_r_peaks'] = peak_editing.add_peak(
                                    result['clean'],
                                    result['current_r_peaks'],
                                    single_peak_time,
                                    sampling_rate
                                )
                                st.rerun()
                        with col_b:
                            if st.button("Delete at Time", key='ecg_single_del_btn'):
                                from utils import peak_editing
                                st.session_state.ecg_result['current_r_peaks'] = peak_editing.delete_peak(
                                    result['current_r_peaks'],
                                    single_peak_time,
                                    sampling_rate
                                )
                                st.rerun()

                st.subheader("Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Final Peak Count", len(result['current_r_peaks']))
                with col2:
                    st.metric("Mean HR", f"{result['mean_hr']:.1f} bpm")
                with col3:
                    st.metric("HR Std Dev", f"{result['std_hr']:.1f} bpm")

        tab_idx += 1

    if 'rsp' in detected_signals:
        with tab_objects[tab_idx]:
            render_rsp_like_tab(
                data=data,
                sampling_rate=sampling_rate,
                signal_key='rsp',
                state_prefix='rsp',
                header_title='RSP Processing',
                plot_label='RSP'
            )
        tab_idx += 1

    if 'spirometer' in detected_signals:
        with tab_objects[tab_idx]:
            render_rsp_like_tab(
                data=data,
                sampling_rate=sampling_rate,
                signal_key='spirometer',
                state_prefix='spirometer',
                header_title='Spirometer Processing (Mask Flow)',
                plot_label='SPIROMETER'
            )
        tab_idx += 1

    if 'ppg' in detected_signals:
        with tab_objects[tab_idx]:
            st.header("PPG Processing")

            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                method = st.selectbox("Cleaning Method", config.PPG_CLEANING_METHODS, key='ppg_method')
                with st.expander("ℹ️ Method Info"):
                    st.info(config.PPG_CLEANING_INFO.get(method, "No info available"))

                peak_method = st.selectbox("Peak Detection", config.PPG_PEAK_METHODS, key='ppg_peak')
                with st.expander("ℹ️ Peak Method Info"):
                    st.info(config.PPG_PEAK_INFO.get(peak_method, "No info available"))

            with col2:
                correct_artifacts = st.checkbox("Artifact Correction", key='ppg_correct')

            with col3:
                st.write("")
                st.write("")
                process_ppg_clicked = st.button("Process PPG", type="primary", key='process_ppg', width='stretch')

            if process_ppg_clicked:
                signal = data['df'][data['signal_mappings']['ppg']].values

                params = {
                    'method': method,
                    'peak_method': peak_method,
                    'correct_artifacts': correct_artifacts
                }
                st.session_state.ppg_params.update(params)

                result = ppg.process_ppg(signal, sampling_rate, st.session_state.ppg_params)

                if result is None:
                    st.error("Processing failed: insufficient peaks detected")
                else:
                    st.session_state.ppg_result = result
                    st.success("PPG processed successfully")

            if st.session_state.ppg_result is not None:
                result = st.session_state.ppg_result

                st.subheader("Manual Peak Editing")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Auto-detected Peaks", len(result['auto_peaks']))
                with col2:
                    n_added = len(np.setdiff1d(result['current_peaks'], result['auto_peaks']))
                    st.metric("Manually Added", n_added)
                with col3:
                    n_deleted = len(np.setdiff1d(result['auto_peaks'], result['current_peaks']))
                    st.metric("Deleted", n_deleted)

                time = np.arange(len(result['clean'])) / sampling_rate

                # Recalculate HR based on current peaks
                from metrics.ppg import calculate_hr_from_ppg
                if len(result['current_peaks']) > 1:
                    hr_data = calculate_hr_from_ppg(
                        result['current_peaks'],
                        sampling_rate,
                        len(result['clean']),
                        rate_method=st.session_state.ppg_params.get('rate_method', 'monotone_cubic')
                    )
                    result.update(hr_data)
                else:
                    result['hr_bpm'] = np.array([])
                    result['hr_interpolated'] = np.zeros(len(result['clean']))
                    result['mean_hr'] = 0.0
                    result['std_hr'] = 0.0

                # Initialize region range in session state if not exists (needed before plotting)
                ppg_max_t = float(time[-1])
                if 'ppg_region_start' not in st.session_state:
                    st.session_state.ppg_region_start = 0.0
                else:
                    st.session_state.ppg_region_start = min(st.session_state.ppg_region_start, ppg_max_t)
                if 'ppg_region_end' not in st.session_state:
                    st.session_state.ppg_region_end = ppg_max_t
                else:
                    st.session_state.ppg_region_end = min(st.session_state.ppg_region_end, ppg_max_t)

                # Get zoom range from session state
                ppg_zoom = (st.session_state.ppg_region_start, st.session_state.ppg_region_end)

                fig = create_signal_plot(
                    time, result['raw'], result['clean'],
                    result['current_peaks'], result['auto_peaks'],
                    'PPG', sampling_rate,
                    hr_interpolated=result.get('hr_interpolated'),
                    hr_bpm=result.get('hr_bpm'),
                    ui_revision='ppg_plot',  # Preserve zoom state
                    zoom_range=ppg_zoom  # Apply zoom from region inputs
                )
                add_task_event_lines(
                    fig,
                    st.session_state.task,
                    float(time[-1]),
                    st.session_state.get('session'),
                    st.session_state.get('participant'),
                )

                st.plotly_chart(fig, width='stretch')

                # Drag-based editing interface
                st.subheader("Drag-Based Peak Editing")

                # Quick navigation buttons
                st.write("**Quick Range Selection:**")
                col_btn1, col_btn2, col_btn3, col_btn4, col_btn5 = st.columns(5)
                with col_btn1:
                    if st.button("⏮️ First 10s", key='ppg_first_10s'):
                        st.session_state.ppg_region_start = 0.0
                        st.session_state.ppg_region_end = min(10.0, float(time[-1]))
                        st.rerun()
                with col_btn2:
                    if st.button("◀️ Previous 10s", key='ppg_prev_10s'):
                        window = 10.0
                        new_start = max(0.0, st.session_state.ppg_region_start - window)
                        new_end = max(window, st.session_state.ppg_region_end - window)
                        st.session_state.ppg_region_start = new_start
                        st.session_state.ppg_region_end = min(new_end, float(time[-1]))
                        st.rerun()
                with col_btn3:
                    if st.button("▶️ Next 10s", key='ppg_next_10s'):
                        window = 10.0
                        new_start = min(float(time[-1]) - window, st.session_state.ppg_region_start + window)
                        new_end = min(float(time[-1]), st.session_state.ppg_region_end + window)
                        st.session_state.ppg_region_start = new_start
                        st.session_state.ppg_region_end = new_end
                        st.rerun()
                with col_btn4:
                    if st.button("⏭️ Last 10s", key='ppg_last_10s'):
                        st.session_state.ppg_region_start = max(0.0, float(time[-1]) - 10.0)
                        st.session_state.ppg_region_end = float(time[-1])
                        st.rerun()
                with col_btn5:
                    if st.button("🔄 Reset Range", key='ppg_reset_range'):
                        st.session_state.ppg_region_start = 0.0
                        st.session_state.ppg_region_end = float(time[-1])
                        st.rerun()

                st.write("**Manual Range Entry:** (Or look at zoomed plot X-axis and enter values)")
                col1, col2 = st.columns(2)
                with col1:
                    region_start = st.number_input(
                        "Region Start (s)",
                        min_value=0.0,
                        max_value=float(time[-1]),
                        value=float(st.session_state.ppg_region_start),
                        step=1.0,
                        format="%.2f",
                        key='ppg_region_start_input',
                        help="Enter the start time from the zoomed plot's X-axis, or use quick buttons above"
                    )
                    # Update session state
                    st.session_state.ppg_region_start = region_start
                with col2:
                    region_end = st.number_input(
                        "Region End (s)",
                        min_value=0.0,
                        max_value=float(time[-1]),
                        value=float(st.session_state.ppg_region_end),
                        step=1.0,
                        format="%.2f",
                        key='ppg_region_end_input',
                        help="Enter the end time from the zoomed plot's X-axis, or use quick buttons above"
                    )
                    # Update session state
                    st.session_state.ppg_region_end = region_end

                # Show current range info
                st.caption(f"Current range: {region_start:.2f}s to {region_end:.2f}s ({region_end - region_start:.2f}s window) | Full signal: {float(time[-1]):.2f}s")

                st.write("**Choose Action:**")
                col1, col2, col3 = st.columns(3)

                with col1:
                    if st.button("➕ Add Systolic Peaks in Region", type="primary", key='ppg_add_region_btn', width='stretch'):
                        from utils import peak_editing
                        st.session_state.ppg_result['current_peaks'] = peak_editing.add_peaks_in_range(
                            result['clean'],
                            result['current_peaks'],
                            region_start,
                            region_end,
                            sampling_rate
                        )
                        st.rerun()

                with col2:
                    if st.button("➖ Remove Systolic Peaks in Region", type="secondary", key='ppg_remove_region_btn', width='stretch'):
                        from utils import peak_editing
                        st.session_state.ppg_result['current_peaks'] = peak_editing.erase_peaks_in_range(
                            result['current_peaks'],
                            region_start,
                            region_end,
                            sampling_rate
                        )
                        st.rerun()

                with col3:
                    if st.button("🔄 Reset to Auto-Detected", key='ppg_reset_btn', width='stretch'):
                        st.session_state.ppg_result['current_peaks'] = result['auto_peaks'].copy()
                        st.rerun()

                # Single peak editing
                with st.expander("✏️ Single Peak Editing (Advanced)"):
                    st.write("Add or remove individual peaks at specific times.")
                    col1, col2 = st.columns(2)
                    with col1:
                        single_peak_time = st.number_input(
                            "Time (seconds)",
                            min_value=0.0,
                            max_value=float(time[-1]),
                            value=0.0,
                            step=0.1,
                            key='ppg_single_peak_time'
                        )
                    with col2:
                        col_a, col_b = st.columns(2)
                        with col_a:
                            if st.button("Add at Time", key='ppg_single_add_btn'):
                                from utils import peak_editing
                                st.session_state.ppg_result['current_peaks'] = peak_editing.add_peak(
                                    result['clean'],
                                    result['current_peaks'],
                                    single_peak_time,
                                    sampling_rate
                                )
                                st.rerun()
                        with col_b:
                            if st.button("Delete at Time", key='ppg_single_del_btn'):
                                from utils import peak_editing
                                st.session_state.ppg_result['current_peaks'] = peak_editing.delete_peak(
                                    result['current_peaks'],
                                    single_peak_time,
                                    sampling_rate
                                )
                                st.rerun()

                st.subheader("Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Final Peak Count", len(result['current_peaks']))
                with col2:
                    st.metric("Mean HR", f"{result['mean_hr']:.1f} bpm")
                with col3:
                    st.metric("HR Std Dev", f"{result['std_hr']:.1f} bpm")
                with col4:
                    st.metric("Mean Quality", f"{result['quality_mean']:.2f}")

        tab_idx += 1

    if 'bp' in detected_signals:
        with tab_objects[tab_idx]:
            st.header("Blood Pressure Processing")

            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                filter_method = st.selectbox("Filter Method", config.BP_FILTER_METHODS, key='bp_filter')
                with st.expander("ℹ️ Filter Info"):
                    st.info(config.BP_FILTER_INFO.get(filter_method, "No info available"))

                peak_method = st.selectbox("Peak Detection", config.BP_PEAK_METHODS, key='bp_peak')
                with st.expander("ℹ️ Peak Method Info"):
                    st.info(config.BP_PEAK_INFO.get(peak_method, "No info available"))

            with col2:
                detect_calib = st.checkbox("Detect Calibration Artifacts", value=True, key='bp_calib')
                
                if peak_method == 'prominence':
                    prominence = st.number_input("Prominence", min_value=1, max_value=100, value=10, key='bp_prom')
                else:
                    prominence = 10

            with col3:
                st.write("")
                st.write("")
                process_bp_clicked = st.button(
                    "Process Blood Pressure",
                    type="primary",
                    key='process_bp',
                    width='stretch'
                )

            if process_bp_clicked:
                signal = data['df'][data['signal_mappings']['bp']].values

                params = {
                    'filter_method': filter_method,
                    'peak_method': peak_method,
                    'prominence': prominence,
                    'detect_calibration': detect_calib
                }
                st.session_state.bp_params.update(params)

                result = blood_pressure.process_bp(signal, sampling_rate, st.session_state.bp_params)

                if result is None:
                    st.error("Processing failed: insufficient peaks detected")
                else:
                    st.session_state.bp_result = result
                    st.success("Blood pressure processed successfully")

            if st.session_state.bp_result is not None:
                result = st.session_state.bp_result

                st.subheader("Manual Blood Pressure Editing")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Auto Systolic Peaks", len(result['auto_peaks']))
                with col2:
                    st.metric("Auto Diastolic Troughs", len(result['auto_troughs']))
                with col3:
                    n_added_peaks = len(np.setdiff1d(result['current_peaks'], result['auto_peaks']))
                    st.metric("Added Systolic", n_added_peaks)
                with col4:
                    n_added_troughs = len(np.setdiff1d(result['current_troughs'], result['auto_troughs']))
                    st.metric("Added Diastolic", n_added_troughs)

                time = np.arange(len(result['filtered'])) / sampling_rate

                # --- 1. Calculate Aligned 4Hz BP Metrics & Derived HR ---
                from metrics.blood_pressure import calculate_bp_metrics
                from metrics.ecg import calculate_hr
                
                bp_data_4hz = calculate_bp_metrics(
                    result['filtered'],
                    result['current_peaks'],
                    result['current_troughs'],
                    sampling_rate 
                )

                hr_from_bp = calculate_hr(
                    result['current_peaks'],
                    sampling_rate,
                    len(result['filtered']),
                    rate_method=st.session_state.bp_params.get('rate_method', 'monotone_cubic')
                )

                # Zoom initialization
                bp_max_t = float(time[-1])
                if 'bp_region_start' not in st.session_state:
                    st.session_state.bp_region_start = 0.0
                else:
                    st.session_state.bp_region_start = min(st.session_state.bp_region_start, bp_max_t)
                if 'bp_region_end' not in st.session_state:
                    st.session_state.bp_region_end = bp_max_t
                else:
                    st.session_state.bp_region_end = min(st.session_state.bp_region_end, bp_max_t)

                bp_zoom = (st.session_state.bp_region_start, st.session_state.bp_region_end)

                # --- 2. Generate Figure ---
                fig = create_rsp_bp_plot(
                    time, result['raw'], result['filtered'],
                    result['current_peaks'], result['current_troughs'],
                    result['auto_peaks'], result['auto_troughs'],
                    'BP',
                    bp_data=bp_data_4hz,
                    hr_data=hr_from_bp,
                    ui_revision='bp_plot',
                    zoom_range=bp_zoom
                )
                add_task_event_lines(
                    fig,
                    st.session_state.task,
                    float(time[-1]),
                    st.session_state.get('session'),
                    st.session_state.get('participant'),
                )

                # Calibration Artifact Sidebar Info
                calib = st.session_state.bp_result.get('calibration_artifacts')
                if calib:
                    st.sidebar.write(f"Artifacts found: {len(calib.get('starts', []))}")
                
                st.plotly_chart(fig, width='stretch')

                # --- 3. Statistics Section ---
                st.subheader("Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Final Cardiac Cycles", min(len(result['current_peaks']), len(result['current_troughs'])))
                with col2:
                    st.metric("Mean SBP", f"{bp_data_4hz['mean_sbp']:.1f} mmHg")
                with col3:
                    st.metric("Mean DBP", f"{bp_data_4hz['mean_dbp']:.1f} mmHg")
                with col4:
                    st.metric("Mean MAP", f"{bp_data_4hz['mean_mbp']:.1f} mmHg")

        tab_idx += 1

# ============================================================================
# ETCO2 TAB
# ============================================================================

    if 'etco2' in detected_signals:
        with tab_objects[tab_idx]:
            st.header("End-Tidal CO2 (ETCO2) Processing")
    
            col1, col2, col3 = st.columns([2, 1, 1])
    
            with col1:
                # Peak detection method selection
                peak_method = st.selectbox(
                    "Peak Detection Method",
                    config.ETCO2_PEAK_METHODS,
                    key='etco2_peak_method'
                )
    
                with st.expander("ℹ️ Method Info"):
                    st.info(config.ETCO2_PEAK_METHOD_INFO.get(peak_method, "No info available"))
    
                # Main parameters
                st.subheader("Detection Parameters")
    
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    min_peak_distance_s = st.slider(
                        "Min Peak Distance (s)",
                        min_value=0.5,
                        max_value=5.0,
                        value=st.session_state.etco2_params.get('min_peak_distance_s', 2.0),
                        step=0.1,
                        key='etco2_min_peak_distance',
                        help="Minimum time between consecutive breath peaks (prevents double-detection)"
                    )
    
                with col_p2:
                    min_prominence = st.slider(
                        "Min Prominence (mmHg)",
                        min_value=0.1,
                        max_value=10.0,
                        value=st.session_state.etco2_params.get('min_prominence', 1.0),
                        step=0.1,
                        key='etco2_min_prominence',
                        help="Minimum peak prominence for valid detection"
                    )
    
                smooth_peaks = st.slider(
                    "Smoothing Kernel Size",
                    min_value=3,
                    max_value=15,
                    value=st.session_state.etco2_params.get('smooth_peaks', 5),
                    step=2,
                    key='etco2_smooth_peaks',
                    help="Median filter kernel size (number of peaks). Must be odd."
                )
    
                # Advanced parameters (Savitzky-Golay filter)
                with st.expander("⚙️ Advanced: Savitzky-Golay Filter"):
                    st.markdown("**For derivative-based peak detection**")
    
                    col_sg1, col_sg2 = st.columns(2)
                    with col_sg1:
                        sg_window_s = st.slider(
                            "Window Duration (s)",
                            min_value=0.1,
                            max_value=1.0,
                            value=st.session_state.etco2_params.get('sg_window_s', 0.3),
                            step=0.05,
                            key='etco2_sg_window',
                            help="Smoothing window for computing derivatives"
                        )
    
                    with col_sg2:
                        sg_poly = st.slider(
                            "Polynomial Order",
                            min_value=1,
                            max_value=5,
                            value=st.session_state.etco2_params.get('sg_poly', 2),
                            key='etco2_sg_poly',
                            help="Polynomial order for S-G filter (2=quadratic)"
                        )
    
                    prom_adapt = st.checkbox(
                        "Adaptive Prominence Threshold",
                        value=st.session_state.etco2_params.get('prom_adapt', False),
                        key='etco2_prom_adapt',
                        help="Use 25th percentile of detected prominences as adaptive minimum"
                    )
    
            with col3:
                st.write("")
                st.write("")
                process_etco2_clicked = st.button(
                    "Process ETCO2",
                    type="primary",
                    key='process_etco2',
                    width='stretch'
                )

            if process_etco2_clicked:
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
                    result = etco2.process_etco2(
                        co2_signal,
                        sampling_rate,
                        st.session_state.etco2_params
                    )

                if result is not None:
                    st.session_state.etco2_result = result
                    st.success(f"✅ ETCO2 processed: {len(result['auto_peaks'])} peaks detected")
                    st.rerun()
    
            # Display results if available
            result = st.session_state.etco2_result
            if result is not None:
                st.divider()
    
                # Metrics row
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Auto-detected Peaks", len(result['auto_peaks']))
                with col2:
                    n_added = len(np.setdiff1d(result['current_peaks'], result['auto_peaks']))
                    st.metric("Manually Added", n_added, delta=f"+{n_added}" if n_added > 0 else None)
                with col3:
                    n_deleted = len(np.setdiff1d(result['auto_peaks'], result['current_peaks']))
                    st.metric("Deleted", n_deleted, delta=f"-{n_deleted}" if n_deleted > 0 else None)
                with col4:
                    if len(result['etco2_envelope']) > 0:
                        mean_etco2 = np.mean(result['etco2_envelope'][np.isfinite(result['etco2_envelope'])])
                        st.metric("Mean ETCO2", f"{mean_etco2:.1f} mmHg")
    
                # Visualization
                st.subheader("📊 ETCO2 Trace Visualization")
    
                # Create plotly figure
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=(
                        'Raw End-Tidal CO2 Waveform with Peak Markers',
                        'ETCO2 Envelope (Breath-Wise Maxima)'
                    ),
                    vertical_spacing=0.12,
                    row_heights=[0.55, 0.45]
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
    
                # Manually added peaks
                added_peaks = np.setdiff1d(result['current_peaks'], result['auto_peaks'])
                if len(added_peaks) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=time[added_peaks],
                            y=result['raw_signal'][added_peaks],
                            name='Added Peaks',
                            mode='markers',
                            marker=dict(color='#AB63FA', size=10, symbol='x', line=dict(width=2))
                        ),
                        row=1, col=1
                    )
    
                # Deleted peaks
                deleted_peaks = np.setdiff1d(result['auto_peaks'], result['current_peaks'])
                if len(deleted_peaks) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=time[deleted_peaks],
                            y=result['raw_signal'][deleted_peaks],
                            name='Deleted Peaks',
                            mode='markers',
                            marker=dict(color='#FF4444', size=8, symbol='x')
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
                        fill='tozeroy',
                        fillcolor='rgba(239, 85, 59, 0.2)',
                        mode='lines'
                    ),
                    row=2, col=1
                )
    
                # Layout
                fig.update_xaxes(title_text="Time (s)", row=2, col=1, rangemode='nonnegative')
                fig.update_xaxes(rangemode='nonnegative', row=1, col=1)
                fig.update_yaxes(title_text="CO2 (mmHg)", row=1, col=1)
                fig.update_yaxes(title_text="ETCO2 (mmHg)", row=2, col=1)

                fig.update_layout(
                    height=800,
                    template='plotly_dark',
                    showlegend=True,
                    hovermode='x unified'
                )

                add_task_event_lines(
                    fig,
                    st.session_state.task,
                    float(time[-1]),
                    st.session_state.get('session'),
                    st.session_state.get('participant'),
                )

                # Apply zoom if set
                if st.session_state.etco2_zoom_range is not None:
                    fig.update_xaxes(range=[max(0, st.session_state.etco2_zoom_range[0]), st.session_state.etco2_zoom_range[1]])

                st.plotly_chart(fig, width='stretch', key='etco2_plot')
    
                # Manual editing interface
                with st.expander("✏️ Manual Peak Editing"):
                    st.info("Add or remove peaks by specifying time ranges. Click 'Reset' to restore auto-detected peaks.")
    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        region_start = st.number_input(
                            "Region Start (s)",
                            min_value=0.0,
                            max_value=float(result['time_vector'][-1]),
                            value=0.0,
                            step=1.0,
                            key='etco2_region_start'
                        )
                    with col2:
                        region_end = st.number_input(
                            "Region End (s)",
                            min_value=0.0,
                            max_value=float(result['time_vector'][-1]),
                            value=min(10.0, float(result['time_vector'][-1])),
                            step=1.0,
                            key='etco2_region_end'
                        )
                    with col3:
                        st.write("")  # Spacer
                        st.write("")  # Spacer
                        if st.button("🔍 Zoom to Region", key='etco2_zoom'):
                            st.session_state.etco2_zoom_range = [region_start, region_end]
                            st.rerun()
    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("➕ Add Peaks in Region", key='etco2_add'):
                            result['current_peaks'] = peak_editing.add_peaks_in_range(
                                result['raw_signal'],
                                result['current_peaks'],
                                region_start,
                                region_end,
                                sampling_rate
                            )
                            st.rerun()
    
                    with col2:
                        if st.button("➖ Remove Peaks in Region", key='etco2_remove'):
                            result['current_peaks'] = peak_editing.erase_peaks_in_range(
                                result['current_peaks'],
                                sampling_rate,
                                region_start,
                                region_end
                            )
                            st.rerun()
    
                    with col3:
                        if st.button("🔄 Reset All Peaks", key='etco2_reset'):
                            result['current_peaks'] = result['auto_peaks'].copy()
                            st.success("Peaks reset to auto-detected")
                            st.rerun()
    
                    # Reset zoom button
                    if st.session_state.etco2_zoom_range is not None:
                        if st.button("↔️ Reset Zoom", key='etco2_reset_zoom'):
                            st.session_state.etco2_zoom_range = None
                            st.rerun()
    
            else:
                st.info("👆 Configure parameters above and click 'Process ETCO2' to begin")
    
        tab_idx += 1
    
    
    # ============================================================================
    # ETO2 TAB
    # ============================================================================
    
    if 'eto2' in detected_signals:
        with tab_objects[tab_idx]:
            st.header("End-Tidal O2 (ETO2) Processing")
    
            col1, col2, col3 = st.columns([2, 1, 1])
    
            with col1:
                # Trough detection method selection
                trough_method = st.selectbox(
                    "Trough Detection Method",
                    config.ETO2_TROUGH_METHODS,
                    key='eto2_trough_method'
                )
    
                with st.expander("ℹ️ Method Info"):
                    st.info(config.ETO2_TROUGH_METHOD_INFO.get(trough_method, "No info available"))
    
                # Main parameters
                st.subheader("Detection Parameters")
    
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    min_trough_distance_s = st.slider(
                        "Min Trough Distance (s)",
                        min_value=0.5,
                        max_value=6.0,
                        value=st.session_state.eto2_params.get('min_trough_distance_s', 3.0),
                        step=0.1,
                        key='eto2_min_trough_distance',
                        help="Minimum time between consecutive breath troughs (prevents double-detection)"
                    )
    
                with col_p2:
                    min_prominence = st.slider(
                        "Min Prominence (mmHg)",
                        min_value=0.1,
                        max_value=10.0,
                        value=st.session_state.eto2_params.get('min_prominence', 1.0),
                        step=0.1,
                        key='eto2_min_prominence',
                        help="Minimum trough prominence for valid detection (on inverted signal)"
                    )
    
                smooth_troughs = st.slider(
                    "Smoothing Kernel Size",
                    min_value=3,
                    max_value=15,
                    value=st.session_state.eto2_params.get('smooth_troughs', 5),
                    step=2,
                    key='eto2_smooth_troughs',
                    help="Median filter kernel size (number of troughs). Must be odd."
                )
    
                # Advanced parameters (Savitzky-Golay filter)
                with st.expander("⚙️ Advanced: Savitzky-Golay Filter"):
                    st.markdown("**For derivative-based trough detection**")
    
                    col_sg1, col_sg2 = st.columns(2)
                    with col_sg1:
                        sg_window_s = st.slider(
                            "Window Duration (s)",
                            min_value=0.1,
                            max_value=1.0,
                            value=st.session_state.eto2_params.get('sg_window_s', 0.2),
                            step=0.05,
                            key='eto2_sg_window',
                            help="Smoothing window for computing derivatives"
                        )
    
                    with col_sg2:
                        sg_poly = st.slider(
                            "Polynomial Order",
                            min_value=1,
                            max_value=5,
                            value=st.session_state.eto2_params.get('sg_poly', 2),
                            key='eto2_sg_poly',
                            help="Polynomial order for S-G filter (2=quadratic)"
                        )
    
                    prom_adapt = st.checkbox(
                        "Adaptive Prominence Threshold",
                        value=st.session_state.eto2_params.get('prom_adapt', False),
                        key='eto2_prom_adapt',
                        help="Use 25th percentile of detected prominences as adaptive minimum"
                    )
    
            with col3:
                st.write("")
                st.write("")
                process_eto2_clicked = st.button(
                    "Process ETO2",
                    type="primary",
                    key='process_eto2',
                    width='stretch'
                )

            if process_eto2_clicked:
                # Update parameters
                params = {
                    'trough_method': trough_method,
                    'min_trough_distance_s': min_trough_distance_s,
                    'min_prominence': min_prominence,
                    'smooth_troughs': smooth_troughs,
                    'sg_window_s': sg_window_s,
                    'sg_poly': sg_poly,
                    'prom_adapt': prom_adapt
                }
                st.session_state.eto2_params.update(params)

                # Get O2 signal
                o2_signal = data['df'][data['signal_mappings']['eto2']].values

                # Process
                with st.spinner("Detecting O2 troughs and extracting envelope..."):
                    result = eto2.process_eto2(
                        o2_signal,
                        sampling_rate,
                        st.session_state.eto2_params
                    )

                if result is not None:
                    st.session_state.eto2_result = result
                    st.success(f"✅ ETO2 processed: {len(result['auto_troughs'])} troughs detected")
                    st.rerun()
    
            # Display results if available
            result = st.session_state.eto2_result
            if result is not None:
                st.divider()
    
                # Metrics row
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Auto-detected Troughs", len(result['auto_troughs']))
                with col2:
                    n_added = len(np.setdiff1d(result['current_troughs'], result['auto_troughs']))
                    st.metric("Manually Added", n_added, delta=f"+{n_added}" if n_added > 0 else None)
                with col3:
                    n_deleted = len(np.setdiff1d(result['auto_troughs'], result['current_troughs']))
                    st.metric("Deleted", n_deleted, delta=f"-{n_deleted}" if n_deleted > 0 else None)
                with col4:
                    if len(result['eto2_envelope']) > 0:
                        mean_eto2 = np.mean(result['eto2_envelope'][np.isfinite(result['eto2_envelope'])])
                        st.metric("Mean ETO2", f"{mean_eto2:.1f} mmHg")
    
                # Visualization
                st.subheader("📊 ETO2 Trace Visualization")
    
                # Create plotly figure
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=(
                        'Raw End-Tidal O2 Waveform with Trough Markers',
                        'ETO2 Envelope (Breath-Wise Minima)'
                    ),
                    vertical_spacing=0.12,
                    row_heights=[0.55, 0.45]
                )
    
                time = result['time_vector']
    
                # Row 1: Raw signal with troughs
                fig.add_trace(
                    go.Scatter(
                        x=time,
                        y=result['raw_signal'],
                        name='Raw O2',
                        line=dict(color='#FFA15A', width=1),
                        mode='lines'
                    ),
                    row=1, col=1
                )
    
                # Auto-detected troughs
                if len(result['auto_troughs']) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=time[result['auto_troughs']],
                            y=result['raw_signal'][result['auto_troughs']],
                            name='Auto Troughs',
                            mode='markers',
                            marker=dict(color='#00CC96', size=8, symbol='circle')
                        ),
                        row=1, col=1
                    )
    
                # Manually added troughs
                added_troughs = np.setdiff1d(result['current_troughs'], result['auto_troughs'])
                if len(added_troughs) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=time[added_troughs],
                            y=result['raw_signal'][added_troughs],
                            name='Added Troughs',
                            mode='markers',
                            marker=dict(color='#AB63FA', size=10, symbol='x', line=dict(width=2))
                        ),
                        row=1, col=1
                    )
    
                # Deleted troughs
                deleted_troughs = np.setdiff1d(result['auto_troughs'], result['current_troughs'])
                if len(deleted_troughs) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=time[deleted_troughs],
                            y=result['raw_signal'][deleted_troughs],
                            name='Deleted Troughs',
                            mode='markers',
                            marker=dict(color='#FF4444', size=8, symbol='x')
                        ),
                        row=1, col=1
                    )
    
                # Row 2: ETO2 envelope
                fig.add_trace(
                    go.Scatter(
                        x=time,
                        y=result['eto2_envelope'],
                        name='ETO2 Envelope',
                        line=dict(color='#19D3F3', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(25, 211, 243, 0.2)',
                        mode='lines'
                    ),
                    row=2, col=1
                )
    
                # Layout
                fig.update_xaxes(title_text="Time (s)", row=2, col=1, rangemode='nonnegative')
                fig.update_xaxes(rangemode='nonnegative', row=1, col=1)
                fig.update_yaxes(title_text="O2 (mmHg)", row=1, col=1)
                fig.update_yaxes(title_text="ETO2 (mmHg)", row=2, col=1)

                fig.update_layout(
                    height=800,
                    template='plotly_dark',
                    showlegend=True,
                    hovermode='x unified'
                )

                add_task_event_lines(
                    fig,
                    st.session_state.task,
                    float(time[-1]),
                    st.session_state.get('session'),
                    st.session_state.get('participant'),
                )

                # Apply zoom if set
                if st.session_state.eto2_zoom_range is not None:
                    fig.update_xaxes(range=[max(0, st.session_state.eto2_zoom_range[0]), st.session_state.eto2_zoom_range[1]])

                st.plotly_chart(fig, width='stretch', key='eto2_plot')
    
                # Manual editing interface
                with st.expander("✏️ Manual Trough Editing"):
                    st.info("Add or remove troughs by specifying time ranges. Click 'Reset' to restore auto-detected troughs.")
    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        region_start = st.number_input(
                            "Region Start (s)",
                            min_value=0.0,
                            max_value=float(result['time_vector'][-1]),
                            value=0.0,
                            step=1.0,
                            key='eto2_region_start'
                        )
                    with col2:
                        region_end = st.number_input(
                            "Region End (s)",
                            min_value=0.0,
                            max_value=float(result['time_vector'][-1]),
                            value=min(10.0, float(result['time_vector'][-1])),
                            step=1.0,
                            key='eto2_region_end'
                        )
                    with col3:
                        st.write("")  # Spacer
                        st.write("")  # Spacer
                        if st.button("🔍 Zoom to Region", key='eto2_zoom'):
                            st.session_state.eto2_zoom_range = [region_start, region_end]
                            st.rerun()
    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("➕ Add Troughs in Region", key='eto2_add'):
                            # For troughs, we want minima not maxima
                            result['current_troughs'] = peak_editing.add_troughs_in_range(
                                result['raw_signal'],
                                result['current_troughs'],
                                region_start,
                                region_end,
                                sampling_rate
                            )
                            st.rerun()
    
                    with col2:
                        if st.button("➖ Remove Troughs in Region", key='eto2_remove'):
                            result['current_troughs'] = peak_editing.erase_peaks_in_range(
                                result['current_troughs'],
                                sampling_rate,
                                region_start,
                                region_end
                            )
                            st.rerun()
    
                    with col3:
                        if st.button("🔄 Reset All Troughs", key='eto2_reset'):
                            result['current_troughs'] = result['auto_troughs'].copy()
                            st.success("Troughs reset to auto-detected")
                            st.rerun()
    
                    # Reset zoom button
                    if st.session_state.eto2_zoom_range is not None:
                        if st.button("↔️ Reset Zoom", key='eto2_reset_zoom'):
                            st.session_state.eto2_zoom_range = None
                            st.rerun()
    
            else:
                st.info("👆 Configure parameters above and click 'Process ETO2' to begin")
    
        tab_idx += 1

    # --- SPO2 TAB ---
    if 'spo2' in detected_signals:
        with tab_objects[tab_idx]:
            st.header("SpO2 (Oxygen Saturation) Processing")

            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                # Cleaning method selection
                cleaning_method = st.selectbox(
                    "Cleaning Method",
                    config.SPO2_CLEANING_METHODS,
                    key='spo2_cleaning_method'
                )

                with st.expander("Method Info"):
                    st.info(config.SPO2_CLEANING_INFO.get(cleaning_method, "No info available"))

                # Parameters based on method
                st.subheader("Processing Parameters")

                if cleaning_method == 'lowpass':
                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        lowpass_cutoff = st.slider(
                            "Lowpass Cutoff (Hz)",
                            min_value=0.1,
                            max_value=2.0,
                            value=st.session_state.spo2_params.get('lowpass_cutoff', 0.5),
                            step=0.1,
                            key='spo2_lowpass_cutoff',
                            help="Cutoff frequency for lowpass filter"
                        )
                    with col_p2:
                        filter_order = st.slider(
                            "Filter Order",
                            min_value=1,
                            max_value=5,
                            value=st.session_state.spo2_params.get('filter_order', 2),
                            key='spo2_filter_order',
                            help="Butterworth filter order"
                        )
                    sg_window_s = 1.0
                    sg_poly = 2
                elif cleaning_method == 'savgol':
                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        sg_window_s = st.slider(
                            "Window Duration (s)",
                            min_value=0.5,
                            max_value=5.0,
                            value=st.session_state.spo2_params.get('sg_window_s', 1.0),
                            step=0.5,
                            key='spo2_sg_window',
                            help="Smoothing window for Savitzky-Golay filter"
                        )
                    with col_p2:
                        sg_poly = st.slider(
                            "Polynomial Order",
                            min_value=1,
                            max_value=5,
                            value=st.session_state.spo2_params.get('sg_poly', 2),
                            key='spo2_sg_poly',
                            help="Polynomial order for S-G filter"
                        )
                    lowpass_cutoff = 0.5
                    filter_order = 2
                else:
                    lowpass_cutoff = 0.5
                    filter_order = 2
                    sg_window_s = 1.0
                    sg_poly = 2

                # Desaturation detection parameters
                with st.expander("Desaturation Detection"):
                    desaturation_threshold = st.slider(
                        "Desaturation Threshold (%)",
                        min_value=80.0,
                        max_value=95.0,
                        value=st.session_state.spo2_params.get('desaturation_threshold', 90.0),
                        step=1.0,
                        key='spo2_desat_threshold',
                        help="SpO2 level below which is considered desaturation"
                    )

                    min_event_duration_s = st.slider(
                        "Min Event Duration (s)",
                        min_value=1.0,
                        max_value=30.0,
                        value=st.session_state.spo2_params.get('min_event_duration_s', 10.0),
                        step=1.0,
                        key='spo2_min_event_duration',
                        help="Minimum duration for a desaturation event"
                    )

            with col3:
                st.write("")
                st.write("")
                process_spo2_clicked = st.button("Process SpO2", type="primary", key='process_spo2', width='stretch')

            if process_spo2_clicked:
                # Update parameters
                params = {
                    'cleaning_method': cleaning_method,
                    'lowpass_cutoff': lowpass_cutoff,
                    'filter_order': filter_order,
                    'sg_window_s': sg_window_s,
                    'sg_poly': sg_poly,
                    'desaturation_threshold': desaturation_threshold,
                    'min_event_duration_s': min_event_duration_s
                }
                st.session_state.spo2_params.update(params)

                # Get SpO2 signal
                spo2_signal = data['df'][data['signal_mappings']['spo2']].values

                # Process
                with st.spinner("Processing SpO2 signal..."):
                    result = spo2.process_spo2(
                        spo2_signal,
                        sampling_rate,
                        st.session_state.spo2_params
                    )

                if result is not None:
                    st.session_state.spo2_result = result
                    metrics = result['metrics']
                    st.success(f"SpO2 processed: Mean {metrics['mean_spo2']:.1f}%")
                    st.rerun()

            # Display results if available
            result = st.session_state.spo2_result
            if result is not None:
                st.divider()

                # Metrics row
                metrics = result['metrics']
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean SpO2", f"{metrics['mean_spo2']:.1f}%")
                with col2:
                    st.metric("Min SpO2", f"{metrics['min_spo2']:.1f}%")
                with col3:
                    st.metric("Time < 90%", f"{metrics['time_below_90']:.1f}s ({metrics['time_below_90_pct']:.1f}%)")
                with col4:
                    st.metric("Desaturation Events", f"{metrics['n_desaturation_events']}")

                # Second metrics row
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Max SpO2", f"{metrics['max_spo2']:.1f}%")
                with col2:
                    st.metric("Std Dev", f"{metrics['std_spo2']:.2f}%")
                with col3:
                    st.metric("Time < 95%", f"{metrics['time_below_95']:.1f}s ({metrics['time_below_95_pct']:.1f}%)")
                with col4:
                    st.metric("Desat Index", f"{metrics['desaturation_index']:.1f}/hr")

                # Visualization
                st.subheader("SpO2 Trace Visualization")

                # Create plotly figure
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=(
                        'Raw vs Cleaned SpO2 Waveform',
                        'Cleaned SpO2 with Clinical Thresholds and Desaturation Events'
                    ),
                    vertical_spacing=0.12,
                    row_heights=[0.45, 0.55]
                )

                time = result['time_vector']

                # Row 1: Raw vs Cleaned
                fig.add_trace(
                    go.Scatter(
                        x=time,
                        y=result['raw_signal'],
                        name='Raw SpO2',
                        line=dict(color='#808080', width=1),
                        mode='lines'
                    ),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=time,
                        y=result['cleaned_signal'],
                        name='Cleaned SpO2',
                        line=dict(color='#00D4FF', width=1.5),
                        mode='lines'
                    ),
                    row=1, col=1
                )

                # Row 2: Cleaned with thresholds and events
                fig.add_trace(
                    go.Scatter(
                        x=time,
                        y=result['cleaned_signal'],
                        name='SpO2',
                        line=dict(color='#00D4FF', width=1.5),
                        mode='lines'
                    ),
                    row=2, col=1
                )

                # Add threshold lines
                desat_line = float(st.session_state.spo2_params.get('desaturation_threshold', desaturation_threshold))
                reference_line = desat_line + 5.0
                fig.add_hline(
                    y=desat_line, line_dash="dash", line_color="red",
                    annotation_text=f"{desat_line:.0f}%", row=2, col=1
                )
                fig.add_hline(
                    y=reference_line, line_dash="dash", line_color="yellow",
                    annotation_text=f"{reference_line:.0f}%", row=2, col=1
                )

                # Highlight desaturation events
                for event_start, event_end, min_val in result['desaturation_events']:
                    fig.add_vrect(
                        x0=time[event_start],
                        x1=time[min(event_end, len(time)-1)],
                        fillcolor="rgba(255, 0, 0, 0.2)",
                        layer="below",
                        line_width=0,
                        row=2, col=1
                    )

                # Layout
                fig.update_xaxes(title_text="Time (s)", row=2, col=1, rangemode='nonnegative')
                fig.update_xaxes(rangemode='nonnegative', row=1, col=1)
                fig.update_yaxes(title_text="SpO2 (%)", row=1, col=1)
                # Dynamic SpO2 axis limits to avoid hard-coded lower bounds.
                spo2_clean = np.asarray(result['cleaned_signal'], dtype=float)
                finite_spo2 = spo2_clean[np.isfinite(spo2_clean)]
                if len(finite_spo2) > 0:
                    p1, p99 = np.percentile(finite_spo2, [1, 99])
                    desat_thr = float(st.session_state.spo2_params.get('desaturation_threshold', 90.0))
                    ref_thr = desat_thr + 5.0
                    lower_anchor = min(p1, desat_thr, ref_thr)
                    upper_anchor = max(p99, desat_thr, ref_thr)
                    pad = max(0.5, (upper_anchor - lower_anchor) * 0.08)
                    y_min = max(0.0, lower_anchor - pad)
                    y_max = upper_anchor + pad
                    if y_max - y_min < 2.0:
                        y_max = y_min + 2.0
                    fig.update_yaxes(title_text="SpO2 (%)", row=2, col=1, range=[y_min, y_max])
                else:
                    fig.update_yaxes(title_text="SpO2 (%)", row=2, col=1)

                fig.update_layout(
                    height=700,
                    template='plotly_dark',
                    showlegend=True,
                    hovermode='x unified'
                )

                add_task_event_lines(
                    fig,
                    st.session_state.task,
                    float(time[-1]),
                    st.session_state.get('session'),
                    st.session_state.get('participant'),
                )

                # Apply zoom if set
                if st.session_state.spo2_zoom_range is not None:
                    fig.update_xaxes(range=[max(0, st.session_state.spo2_zoom_range[0]), st.session_state.spo2_zoom_range[1]])

                st.plotly_chart(fig, width='stretch', key='spo2_plot')

                # Zoom controls
                with st.expander("Zoom Controls"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        region_start = st.number_input(
                            "Region Start (s)",
                            min_value=0.0,
                            max_value=float(result['time_vector'][-1]),
                            value=0.0,
                            step=1.0,
                            key='spo2_region_start'
                        )
                    with col2:
                        region_end = st.number_input(
                            "Region End (s)",
                            min_value=0.0,
                            max_value=float(result['time_vector'][-1]),
                            value=min(60.0, float(result['time_vector'][-1])),
                            step=1.0,
                            key='spo2_region_end'
                        )
                    with col3:
                        st.write("")
                        st.write("")
                        if st.button("Zoom to Region", key='spo2_zoom'):
                            st.session_state.spo2_zoom_range = [region_start, region_end]
                            st.rerun()

                    if st.session_state.spo2_zoom_range is not None:
                        if st.button("Reset Zoom", key='spo2_reset_zoom'):
                            st.session_state.spo2_zoom_range = None
                            st.rerun()

                # Desaturation events table
                if result['desaturation_events']:
                    with st.expander("Desaturation Events"):
                        events_data = []
                        for i, (start_idx, end_idx, min_val) in enumerate(result['desaturation_events']):
                            events_data.append({
                                'Event': i + 1,
                                'Start (s)': f"{time[start_idx]:.1f}",
                                'End (s)': f"{time[min(end_idx, len(time)-1)]:.1f}",
                                'Duration (s)': f"{(end_idx - start_idx) / sampling_rate:.1f}",
                                'Min SpO2 (%)': f"{min_val:.1f}"
                            })
                        st.dataframe(events_data, width='stretch')

            else:
                st.info("Configure parameters above and click 'Process SpO2' to begin")

        tab_idx += 1

    # --- SPIROMETRY TAB (Session A) ---
    if session_a_selected:
        with tab_objects[tab_idx]:
            st.header("Spirometry (Session A)")
            st.warning("External spirometry export is currently disabled (vendor data issue).")
            st.info(
                "This tab is a placeholder for the external spirometry report. "
                "Waveform-level respiratory analysis is available in the new `Spirometer` tab when channel 12 is present."
            )

        tab_idx += 1

    # --- EXPORT TAB ---
    with tab_objects[-1]:
        st.header("Export Data")

        output_path = st.text_input(
            "Output Path",
            value=config.OUTPUT_BASE_PATH,
            help="Base output directory."
        )

        processed_signals = [s for s in ["ecg", "rsp", "ppg", "bp"] if st.session_state.get(f"{s}_result")]

        if not processed_signals:
            st.warning("No signals have been processed yet.")
        else:
            st.success(f"Signals ready: {', '.join(processed_signals)}")

            if st.button("Export All Signals", type="primary"):
                results_dict, params_dict = {}, {}

                if st.session_state.bp_result is not None:
                    from metrics.blood_pressure import calculate_bp_metrics
                    from metrics.ecg import calculate_hr
                    bp_result = st.session_state.bp_result.copy()
                    
                    metrics = calculate_bp_metrics(bp_result['filtered'], bp_result['current_peaks'], bp_result['current_troughs'], sampling_rate)
                    hr = calculate_hr(bp_result['current_peaks'], sampling_rate, len(bp_result['filtered']))
                    
                    bp_result.update(metrics)
                    bp_result['hr_from_bp'] = hr['hr_interpolated']
                    results_dict['bp'] = bp_result
                    params_dict['bp'] = st.session_state.bp_params

                # Create files
                df = export.create_combined_dataframe(results_dict, sampling_rate)
                metadata = export.create_metadata_json(results_dict, params_dict, sampling_rate)
                paths = export.export_physio_data(output_path, st.session_state.participant, st.session_state.session, st.session_state.task, df, metadata)

                st.success("Export complete!")
                st.info(f"**CSV**: `{paths['csv_path']}`")

if __name__ == "__main__":
    main()
