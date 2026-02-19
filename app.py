"""
Physiological Signal QC Application
Streamlit-based interface for quality control of physiological signals
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config
from utils.file_io import scan_data_directory, find_file_path, load_acq_file
from metrics import ecg, rsp, ppg, blood_pressure, etco2, eto2, spo2, doppler
from utils import peak_editing, export


st.set_page_config(
    page_title="Physio QC",
    page_icon="📈",
    layout="wide"
)


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

    if 'doppler_result' not in st.session_state:
        st.session_state.doppler_result = None

    if 'doppler_params' not in st.session_state:
        # simplest: start from BP defaults (or make DEFAULT_DOPPLER_PARAMS in config)
        st.session_state.doppler_params = config.DEFAULT_BP_PARAMS.copy()
    if 'doppler_zoom_range' not in st.session_state:
        st.session_state.doppler_zoom_range = None




def create_signal_plot(time, raw, clean, current_peaks, auto_peaks, signal_name, sampling_rate,
                       hr_interpolated=None, hr_bpm=None, quality_continuous=None,
                       selected_quality_metrics=None, quality_data=None, ui_revision='constant',
                       zoom_range=None):
    """Create 3-panel plot for signal visualization with synchronized zooming"""
    deleted_peaks = np.setdiff1d(auto_peaks, current_peaks)
    added_peaks = np.setdiff1d(current_peaks, auto_peaks)

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Raw vs Clean', 'Clean with Peaks', f'{signal_name} Rate'),
        vertical_spacing=0.1,
        specs=[[{"secondary_y": False}], [{"secondary_y": True}], [{"secondary_y": False}]]
    )

    # Row 1: Raw vs Clean
    fig.add_trace(go.Scatter(x=time, y=raw, name='Raw', line=dict(color='#808080', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=time, y=clean, name='Clean', line=dict(color='#00D4FF', width=1)), row=1, col=1)

    # Row 2: Clean with Peaks
    fig.add_trace(go.Scatter(x=time, y=clean, name='Signal', line=dict(color='#00D4FF', width=1)), row=2, col=1, secondary_y=False)

    if len(current_peaks) > 0:
        fig.add_trace(go.Scatter(
            x=time[current_peaks], y=clean[current_peaks],
            mode='markers', name='Valid Peaks',
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

    # Row 3: Heart Rate
    if hr_interpolated is not None:
        fig.add_trace(go.Scatter(
            x=time, y=hr_interpolated,
            name='HR Interpolated',
            line=dict(color='#FF6B6B', width=2)
        ), row=3, col=1)

    fig.update_xaxes(matches='x')
    if zoom_range is not None:
        fig.update_xaxes(range=[zoom_range[0], zoom_range[1]])

    fig.update_layout(height=800, template='plotly_dark', showlegend=True, uirevision=ui_revision)

    return fig


def create_rsp_bp_plot(time, raw, clean, current_peaks, current_troughs, auto_peaks, auto_troughs, 
                       signal_name, rate_interpolated=None, rate_bpm=None, 
                       bp_data=None, hr_data=None, ui_revision='constant', 
                       zoom_range=None, calibration_regions=None):
    """Create 3 or 4-panel plot for RSP/BP with synchronized zooming"""
    
    # 1. Row Configuration
    is_bp_like = signal_name in ('BP', 'DOPPLER')  # <-- add Doppler here

    n_rows = 4 if is_bp_like else 3
    height = 1000 if n_rows == 4 else 800

    titles = ['Raw vs Filtered', 'Signal with Peaks/Troughs']
    if is_bp_like:
        # keep layout identical; you can rename titles later if you want
        titles.extend(['Metrics (Peak/Mean/Trough)', 'Heart Rate (from peaks)'])
    else:
        titles.append(f'{signal_name} Rate')


    fig = make_subplots(
        rows=n_rows, cols=1,
        subplot_titles=titles,
        vertical_spacing=0.07,
        shared_xaxes=True
    )

    # --- Row 1: Raw vs Filtered ---
    fig.add_trace(go.Scatter(x=time, y=raw, name='Raw', line=dict(color='#808080', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=time, y=clean, name='Filtered', line=dict(color='#00D4FF', width=1)), row=1, col=1)

    # --- Row 2: Signal with Peaks/Troughs ---
    fig.add_trace(go.Scatter(x=time, y=clean, name='Signal', line=dict(color='#00D4FF', width=1), showlegend=False), row=2, col=1)

    if len(current_peaks) > 0:
        fig.add_trace(go.Scatter(x=time[current_peaks], y=clean[current_peaks], mode='markers', 
                                 name='Systolic/Inhale', marker=dict(color='#FF4444', size=8)), row=2, col=1)
    if len(current_troughs) > 0:
        fig.add_trace(go.Scatter(x=time[current_troughs], y=clean[current_troughs], mode='markers', 
                                 name='Diastolic/Exhale', marker=dict(color='#4444FF', size=8)), row=2, col=1)

    # --- Row 3: BP Metrics (SBP/MAP/DBP) or RSP Rate ---
    if is_bp_like and bp_data is not None:
        t_4hz = bp_data['time_4hz']
        fig.add_trace(go.Scatter(x=t_4hz, y=bp_data['sbp_4hz'], name='SBP', line=dict(color='red', width=1.5)), row=3, col=1)
        fig.add_trace(go.Scatter(x=t_4hz, y=bp_data['map_4hz'], name='MAP', line=dict(color='green', width=2)), row=3, col=1)
        fig.add_trace(go.Scatter(x=t_4hz, y=bp_data['dbp_4hz'], name='DBP', line=dict(color='blue', width=1.5)), row=3, col=1)
    elif rate_interpolated is not None:
        fig.add_trace(go.Scatter(x=time, y=rate_interpolated, name='Rate Interpolated', line=dict(color='#FF6B6B', width=2)), row=3, col=1)

    # --- Row 4: Heart Rate (BP only) ---
    if is_bp_like and hr_data is not None:
        fig.add_trace(go.Scatter(x=time, y=hr_data['hr_interpolated'], name='HR from BP', line=dict(color='#FF6B6B', width=2)), row=4, col=1)

    # Formatting
    fig.update_xaxes(matches='x')
    if zoom_range is not None:
        fig.update_xaxes(range=[zoom_range[0], zoom_range[1]])
    
    fig.update_layout(height=height, template='plotly_dark', showlegend=True, hovermode='x unified', uirevision=ui_revision)
    fig.update_traces(connectgaps=False)
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

    col1, col2 = st.columns([2, 1])

    with col1:
        method = st.selectbox("Cleaning Method", config.RSP_CLEANING_METHODS, key=f'{state_prefix}_method')
        with st.expander("ℹ️ Method Info"):
            st.info(config.RSP_CLEANING_INFO.get(method, "No info available"))

    with col2:
        amplitude_method = st.selectbox("Amplitude Normalization", config.RSP_AMPLITUDE_METHODS, key=f'{state_prefix}_amplitude')
        with st.expander("ℹ️ Amplitude Info"):
            st.info(config.RSP_AMPLITUDE_INFO.get(amplitude_method, "No info available"))

    if st.button(f"Process {plot_label}", type="primary", key=f'process_{state_prefix}'):
        signal = data['df'][data['signal_mappings'][signal_key]].values

        params = {
            'method': method,
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
        if region_start_key not in st.session_state:
            st.session_state[region_start_key] = 0.0
        if region_end_key not in st.session_state:
            st.session_state[region_end_key] = min(10.0, float(time[-1]))

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

        st.plotly_chart(fig, use_container_width=True)

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
                st.session_state[region_end_key] = min(10.0, float(time[-1]))
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
            if st.button("➕ Add Inhalation Peaks", type="primary", key=f'{state_prefix}_add_peaks_btn', use_container_width=True):
                st.session_state[result_state_key]['current_peaks'] = peak_editing.add_peaks_in_range(
                    result['clean'], result['current_peaks'], region_start, region_end, sampling_rate, min_distance_seconds=1.0
                )
                st.rerun()
        with col2:
            if st.button("➖ Remove Inhalation Peaks", type="secondary", key=f'{state_prefix}_remove_peaks_btn', use_container_width=True):
                st.session_state[result_state_key]['current_peaks'] = peak_editing.erase_peaks_in_range(
                    result['current_peaks'], region_start, region_end, sampling_rate
                )
                st.rerun()
        with col3:
            if st.button("🔄 Reset Inhalations", key=f'{state_prefix}_reset_peaks_btn', use_container_width=True):
                st.session_state[result_state_key]['current_peaks'] = result['auto_peaks'].copy()
                st.rerun()

        st.write("**Exhalation Troughs:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("➕ Add Exhalation Troughs", type="primary", key=f'{state_prefix}_add_troughs_btn', use_container_width=True):
                st.session_state[result_state_key]['current_troughs'] = peak_editing.add_troughs_in_range(
                    result['clean'], result['current_troughs'], region_start, region_end, sampling_rate, min_distance_seconds=1.0
                )
                st.rerun()
        with col2:
            if st.button("➖ Remove Exhalation Troughs", type="secondary", key=f'{state_prefix}_remove_troughs_btn', use_container_width=True):
                st.session_state[result_state_key]['current_troughs'] = peak_editing.erase_troughs_in_range(
                    result['current_troughs'], region_start, region_end, sampling_rate
                )
                st.rerun()
        with col3:
            if st.button("🔄 Reset Exhalations", key=f'{state_prefix}_reset_troughs_btn', use_container_width=True):
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


def main():
    """Main application function"""
    init_session_state()

    st.title("📈 Physiological Signal QC")

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
    if 'doppler' in detected_signals:
        tabs.append("Doppler")
    if session_a_selected:
        tabs.append("Spirometry")
    tabs.append("Export")

    tab_objects = st.tabs(tabs)
    tab_idx = 0

    if 'ecg' in detected_signals:
        with tab_objects[tab_idx]:
            st.header("ECG Processing")

            col1, col2 = st.columns([2, 1])

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

            if st.button("Process ECG", type="primary"):
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
                if 'ecg_region_start' not in st.session_state:
                    st.session_state.ecg_region_start = 0.0
                if 'ecg_region_end' not in st.session_state:
                    st.session_state.ecg_region_end = min(10.0, float(time[-1]))

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

                st.plotly_chart(fig, use_container_width=True)

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
                        st.session_state.ecg_region_end = min(10.0, float(time[-1]))
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
                    if st.button("➕ Add R-Peaks in Region", type="primary", key='ecg_add_region_btn', use_container_width=True):
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
                    if st.button("➖ Remove R-Peaks in Region", type="secondary", key='ecg_remove_region_btn', use_container_width=True):
                        from utils import peak_editing
                        st.session_state.ecg_result['current_r_peaks'] = peak_editing.erase_peaks_in_range(
                            result['current_r_peaks'],
                            region_start,
                            region_end,
                            sampling_rate
                        )
                        st.rerun()

                with col3:
                    if st.button("🔄 Reset to Auto-Detected", key='ecg_reset_btn', use_container_width=True):
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

            col1, col2 = st.columns([2, 1])

            with col1:
                method = st.selectbox("Cleaning Method", config.PPG_CLEANING_METHODS, key='ppg_method')
                with st.expander("ℹ️ Method Info"):
                    st.info(config.PPG_CLEANING_INFO.get(method, "No info available"))

                peak_method = st.selectbox("Peak Detection", config.PPG_PEAK_METHODS, key='ppg_peak')
                with st.expander("ℹ️ Peak Method Info"):
                    st.info(config.PPG_PEAK_INFO.get(peak_method, "No info available"))

            with col2:
                correct_artifacts = st.checkbox("Artifact Correction", key='ppg_correct')

            if st.button("Process PPG", type="primary"):
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
                if 'ppg_region_start' not in st.session_state:
                    st.session_state.ppg_region_start = 0.0
                if 'ppg_region_end' not in st.session_state:
                    st.session_state.ppg_region_end = min(10.0, float(time[-1]))

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

                st.plotly_chart(fig, use_container_width=True)

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
                        st.session_state.ppg_region_end = min(10.0, float(time[-1]))
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
                    if st.button("➕ Add Systolic Peaks in Region", type="primary", key='ppg_add_region_btn', use_container_width=True):
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
                    if st.button("➖ Remove Systolic Peaks in Region", type="secondary", key='ppg_remove_region_btn', use_container_width=True):
                        from utils import peak_editing
                        st.session_state.ppg_result['current_peaks'] = peak_editing.erase_peaks_in_range(
                            result['current_peaks'],
                            region_start,
                            region_end,
                            sampling_rate
                        )
                        st.rerun()

                with col3:
                    if st.button("🔄 Reset to Auto-Detected", key='ppg_reset_btn', use_container_width=True):
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

            col1, col2 = st.columns([2, 1])

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

            if st.button("Process Blood Pressure", type="primary"):
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
                if 'bp_region_start' not in st.session_state:
                    st.session_state.bp_region_start = 0.0
                if 'bp_region_end' not in st.session_state:
                    st.session_state.bp_region_end = min(10.0, float(time[-1]))

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
                
                # Calibration Artifact Sidebar Info
                calib = st.session_state.bp_result.get('calibration_artifacts')
                if calib:
                    st.sidebar.write(f"Artifacts found: {len(calib.get('starts', []))}")
                
                st.plotly_chart(fig, use_container_width=True)

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
    
            col1, col2 = st.columns([2, 1])
    
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
    
            with col2:
                if st.button("🔬 Process ETCO2", type="primary", key='process_etco2'):
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
                    subplot_titles=('Raw CO2 Signal with Detected Peaks', 'ETCO2 Upper Envelope'),
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
                fig.update_xaxes(title_text="Time (s)", row=2, col=1)
                fig.update_yaxes(title_text="CO2 (mmHg)", row=1, col=1)
                fig.update_yaxes(title_text="ETCO2 (mmHg)", row=2, col=1)
    
                fig.update_layout(
                    height=800,
                    template='plotly_dark',
                    showlegend=True,
                    hovermode='x unified'
                )
    
                # Apply zoom if set
                if st.session_state.etco2_zoom_range is not None:
                    fig.update_xaxes(range=st.session_state.etco2_zoom_range)
    
                st.plotly_chart(fig, use_container_width=True, key='etco2_plot')
    
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
    
            col1, col2 = st.columns([2, 1])
    
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
    
            with col2:
                if st.button("🔬 Process ETO2", type="primary", key='process_eto2'):
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
                        result = eto2.extract_eto2_envelope(
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
                    subplot_titles=('Raw O2 Signal with Detected Troughs', 'ETO2 Lower Envelope'),
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
                fig.update_xaxes(title_text="Time (s)", row=2, col=1)
                fig.update_yaxes(title_text="O2 (mmHg)", row=1, col=1)
                fig.update_yaxes(title_text="ETO2 (mmHg)", row=2, col=1)
    
                fig.update_layout(
                    height=800,
                    template='plotly_dark',
                    showlegend=True,
                    hovermode='x unified'
                )
    
                # Apply zoom if set
                if st.session_state.eto2_zoom_range is not None:
                    fig.update_xaxes(range=st.session_state.eto2_zoom_range)
    
                st.plotly_chart(fig, use_container_width=True, key='eto2_plot')
    
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

            col1, col2 = st.columns([2, 1])

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

            with col2:
                if st.button("Process SpO2", type="primary", key='process_spo2'):
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
                    subplot_titles=('Raw vs Cleaned SpO2', 'SpO2 with Thresholds'),
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
                fig.add_hline(
                    y=90, line_dash="dash", line_color="red",
                    annotation_text="90%", row=2, col=1
                )
                fig.add_hline(
                    y=95, line_dash="dash", line_color="yellow",
                    annotation_text="95%", row=2, col=1
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
                fig.update_xaxes(title_text="Time (s)", row=2, col=1)
                fig.update_yaxes(title_text="SpO2 (%)", row=1, col=1)
                fig.update_yaxes(title_text="SpO2 (%)", row=2, col=1, range=[80, 105])

                fig.update_layout(
                    height=700,
                    template='plotly_dark',
                    showlegend=True,
                    hovermode='x unified'
                )

                # Apply zoom if set
                if st.session_state.spo2_zoom_range is not None:
                    fig.update_xaxes(range=st.session_state.spo2_zoom_range)

                st.plotly_chart(fig, use_container_width=True, key='spo2_plot')

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
                        st.dataframe(events_data, use_container_width=True)

            else:
                st.info("Configure parameters above and click 'Process SpO2' to begin")

        tab_idx += 1



    if 'doppler' in detected_signals:
            with tab_objects[tab_idx]:
                st.header("Doppler Processing")

                # --- 1. UI FOR FILTER PARAMS ---
                col1, col2, col3 = st.columns(3)

                with col1:
                    filter_method = st.selectbox(
                        "Filter Method", 
                        ['sg_wavelet', 'bessel_25hz', 'butterworth', 'custom'], 
                        index=0, 
                        key='doppler_filter'
                    )   
                    # Hardcoding Peak Detection to Delineator
                    peak_method = 'delineator' 
                    st.info("Peak Detection locked to 'Delineator'.")

                with col2:
                    if filter_method == 'sg_wavelet':
                        sg_win = st.number_input("SG Window (s)", value=0.1, step=0.05, key='dopp_sg')
                        wavelet = st.selectbox("Wavelet Type", ['db6', 'sym8', 'db4'], index=0, key='dopp_wav')
                        level = st.number_input("Decomp Level", min_value=1, max_value=12, value=10, key='dopp_lvl')
                    else:
                        st.info("Standard method parameters applied.")

                with col3:
                    if filter_method == 'sg_wavelet':
                        alpha = st.number_input("IQR Alpha (Spikes)", value=4.0, step=0.5, key='dopp_alpha')
                        drop_levels = st.number_input("Drop Fine Levels", min_value=0, max_value=4, value=1, key='dopp_drop')

                # --- 2. EXECUTE PROCESSING ---
                if st.button("Process Doppler", type="primary"):
                    signal = data['df'][data['signal_mappings']['doppler']].values

                    # Build params dictionary dynamically
                    params = {
                        'filter_method': filter_method,
                        'peak_method': peak_method,
                    }
                    if filter_method == 'sg_wavelet':
                        params.update({
                            'sg_win': sg_win,
                            'wavelet': wavelet,
                            'level': level,
                            'alpha': alpha,
                            'drop_levels': drop_levels
                        })

                    st.session_state.doppler_params.update(params)

                    # Execute pipeline
                    result = doppler.process_doppler(signal, sampling_rate, st.session_state.doppler_params)

                    if result is None:
                        st.error("Processing failed: insufficient peaks detected.")
                    else:
                        st.session_state.doppler_result = result
                        st.success("Doppler processed successfully.")

                # --- 3. DISPLAY RESULTS ---
                if st.session_state.doppler_result is not None:
                    result = st.session_state.doppler_result

                    st.subheader("Manual Doppler Editing")

                    # Peak/Trough Editing Stats
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Auto Peaks", len(result['auto_peaks']))
                    with col2:
                        st.metric("Auto Troughs", len(result['auto_troughs']))
                    with col3:
                        n_added_peaks = len(np.setdiff1d(result['current_peaks'], result['auto_peaks']))
                        st.metric("Added Peaks", n_added_peaks)
                    with col4:
                        n_added_troughs = len(np.setdiff1d(result['current_troughs'], result['auto_troughs']))
                        st.metric("Added Troughs", n_added_troughs)

                    # Time Array
                    time = np.arange(len(result['filtered'])) / sampling_rate

                    # Calculate Aligned Metrics & HR
                    from metrics.doppler import calculate_doppler_metrics
                    from metrics.ecg import calculate_hr

                    doppler_data_4hz = calculate_doppler_metrics(
                        result['filtered'],
                        result['current_peaks'],
                        result['current_troughs'],
                        sampling_rate
                    )

                    hr_from_doppler = calculate_hr(
                        result['current_peaks'],
                        sampling_rate,
                        len(result['filtered']),
                        rate_method=st.session_state.doppler_params.get('rate_method', 'monotone_cubic')
                    )

                    # Zoom constraints
                    if 'doppler_region_start' not in st.session_state:
                        st.session_state.doppler_region_start = 0.0
                    if 'doppler_region_end' not in st.session_state:
                        st.session_state.doppler_region_end = min(10.0, float(time[-1]))
                    doppler_zoom = (st.session_state.doppler_region_start, st.session_state.doppler_region_end)

                    # Generate Plotly Chart
                    fig = create_rsp_bp_plot(
                        time, result['raw'], result['filtered'],
                        result['current_peaks'], result['current_troughs'],
                        result['auto_peaks'], result['auto_troughs'],
                        'DOPPLER',
                        bp_data=doppler_data_4hz,
                        hr_data=hr_from_doppler,
                        ui_revision='doppler_plot',
                        zoom_range=doppler_zoom
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # --- 4. STATISTICS SECTION ---
                    st.subheader("Statistics")
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Final Cycles", min(len(result['current_peaks']), len(result['current_troughs'])))
                    with col2:
                        st.metric("Mean Peak", f"{doppler_data_4hz['mean_peak']:.3f}")
                    with col3:
                        st.metric("Mean Trough", f"{doppler_data_4hz['mean_trough']:.3f}")
                    with col4:
                        st.metric("Mean Amplitude", f"{doppler_data_4hz['mean_amp']:.3f}")
                    with col5:
                        q_score = result.get('mean_quality', np.nan)
                        if not np.isnan(q_score):
                            # Streamlit delta styling: Green "Good" if score >= 0.8, Red "Review" if lower
                            delta_color = "normal" if q_score >= 0.8 else "inverse"
                            delta_str = "Good" if q_score >= 0.8 else "Review suggested"
                            st.metric("Avg Beat Quality", f"{q_score:.2f}", delta=delta_str, delta_color=delta_color)
                        else:
                            st.metric("Avg Beat Quality", "N/A")

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
