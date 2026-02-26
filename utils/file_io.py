"""
File I/O functions for loading and scanning physiological data files
Pure functions with no classes
"""

import re
from pathlib import Path

import bioread
import pandas as pd

import config
from utils.conversions import convert_doppler_channel, convert_gas_channels
from utils.pmu_integration import (
    extract_pmu_task_signals,
    resample_signal_to_length,
    session_matches_alias,
)


def scan_data_directory(base_path):
    """
    Scan the directory structure for available participants, sessions, and tasks

    Parameters
    ----------
    base_path : str or Path
        Base directory containing sub-*/ses-* structure

    Returns
    -------
    dict
        Nested dict: {participant: {session: [tasks]}}
    """
    base_path = Path(base_path)

    if not base_path.exists():
        return {}

    participants_data = {}

    for sub_dir in sorted(base_path.glob('sub-*')):
        if not sub_dir.is_dir():
            continue

        sub_id = sub_dir.name
        participants_data[sub_id] = {}

        for ses_dir in sorted(sub_dir.glob('ses-*')):
            if not ses_dir.is_dir():
                continue

            ses_id = ses_dir.name
            acq_files = list(ses_dir.glob('*.acq'))

            tasks = []
            for acq_file in acq_files:
                match = re.search(r'task-([^_]+)', acq_file.name)
                if match:
                    tasks.append(match.group(1))

            if tasks:
                participants_data[sub_id][ses_id] = sorted(set(tasks))

    return participants_data


def find_file_path(base_path, participant, session, task):
    """
    Find the ACQ file path for given participant, session, and task

    Parameters
    ----------
    base_path : str or Path
        Base directory containing data
    participant : str
        Participant ID (e.g., 'sub-2034')
    session : str
        Session ID (e.g., 'ses-01')
    task : str
        Task name (e.g., 'rest')

    Returns
    -------
    str or None
        Path to ACQ file or None if not found
    """
    base_path = Path(base_path)

    standard_pattern = base_path / participant / session / f"{participant}_{session}_task-{task}_physio.acq"
    if standard_pattern.exists():
        return str(standard_pattern)

    search_dir = base_path / participant / session
    if not search_dir.exists():
        return None

    matches = list(search_dir.glob(f"*task-{task}*.acq"))
    if matches:
        return str(matches[0])

    return None


def detect_signal_type(column_name):
    """
    Detect what type of signal a column contains based on its name

    Parameters
    ----------
    column_name : str
        Name of the column to check

    Returns
    -------
    str or None
        Signal type ('ecg', 'rsp', 'ppg', 'bp') or None if not detected
    """
    col_lower = column_name.lower()

    for signal_type, patterns in config.SIGNAL_PATTERNS.items():
        for pattern in patterns:
            if pattern in col_lower:
                return signal_type

    return None


def _attach_pmu_session_b_signals(df_raw, signal_mappings, participant, session, task):
    """Attach PMU respiration/pulse for Session B when available."""
    status = {
        'attempted': False,
        'success': False,
        'message': 'PMU integration not attempted for this session.',
    }

    if not participant or not session or not task:
        status['message'] = 'PMU integration skipped (missing participant/session/task).'
        return df_raw, signal_mappings, status

    aliases = getattr(config, 'PMU_SESSION_B_ALIASES', [])
    if not aliases or not session_matches_alias(session, aliases):
        return df_raw, signal_mappings, status

    status['attempted'] = True
    pmu_result = extract_pmu_task_signals(
        base_physio_path=config.BASE_DATA_PATH,
        bids_base_path=getattr(config, 'PMU_BIDS_BASE_PATH', config.BASE_DATA_PATH),
        participant=participant,
        session=session,
        task=task,
        pmu_session=getattr(config, 'PMU_PHYSIO_SESSION', None),
        bids_session=getattr(config, 'PMU_BIDS_SESSION', None),
        sampling_rate=getattr(config, 'PMU_SAMPLING_RATE', 400),
        scan_gap_seconds=getattr(config, 'PMU_SCAN_GAP_SECONDS', 10.0),
        time_tolerance_seconds=getattr(config, 'PMU_TIME_MATCH_TOLERANCE_SECONDS', 30.0),
    )

    status['message'] = pmu_result.get('message', 'PMU integration failed.')
    if not pmu_result.get('success'):
        return df_raw, signal_mappings, status

    n_samples = len(df_raw)
    rsp_resampled = resample_signal_to_length(pmu_result['rsp'], n_samples)
    ppg_resampled = resample_signal_to_length(pmu_result['ppg'], n_samples)

    rsp_column = 'PMU_RESP'
    ppg_column = 'PMU_PULS'
    df_raw[rsp_column] = rsp_resampled
    df_raw[ppg_column] = ppg_resampled

    prefer_pmu = bool(getattr(config, 'PMU_PREFER_SCANNER_SIGNALS', True))
    if prefer_pmu or 'rsp' not in signal_mappings:
        signal_mappings['rsp'] = rsp_column
    if prefer_pmu or 'ppg' not in signal_mappings:
        signal_mappings['ppg'] = ppg_column

    status.update({
        'success': True,
        'match_strategy': pmu_result.get('match_strategy'),
        'scan_index': pmu_result.get('scan_index'),
        'scan_duration_sec': pmu_result.get('scan_duration_sec'),
        'resolved_pmu_session': pmu_result.get('resolved_pmu_session'),
        'resolved_pmu_folder': pmu_result.get('resolved_pmu_folder'),
        'resolved_bids_session': pmu_result.get('resolved_bids_session'),
        'rsp_column': rsp_column,
        'ppg_column': ppg_column,
        'resampled_to_samples': n_samples,
    })
    return df_raw, signal_mappings, status


def load_acq_file(file_path, participant=None, session=None, task=None):
    """
    Load an ACQ file and return data with metadata

    Parameters
    ----------
    file_path : str or Path
        Path to ACQ file
    participant : str, optional
        Participant ID (e.g., 'sub-2034') for session-aware enrichment.
    session : str, optional
        Session ID (e.g., 'ses-02') for session-aware enrichment.
    task : str, optional
        Task name (e.g., 'breath') for PMU scan matching.

    Returns
    -------
    dict or None
        Dictionary containing:
        - df: DataFrame with all channels
        - sampling_rate: Sampling rate in Hz
        - channels: List of channel names
        - signal_mappings: Dict mapping signal types to column names
        - n_samples: Number of samples
        - duration: Duration in seconds
        Returns None if file doesn't exist
    """
    if not Path(file_path).exists():
        return None

    data = bioread.read(file_path)
    sampling_rate = int(data.samples_per_second)

    channels = {}
    for ch in data.channels:
        channels[ch.name] = ch.data

    df_raw = pd.DataFrame(channels)

    # Apply gas channel conversions (voltage to mmHg for CO2/O2)
    df_raw, gas_conversions = convert_gas_channels(df_raw)
    df_raw, doppler_conversions = convert_doppler_channel(df_raw)

    signal_mappings = {}
    for col in df_raw.columns:
        signal_type = detect_signal_type(col)
        if signal_type:
            if signal_type not in signal_mappings:
                signal_mappings[signal_type] = col

    # Fallback: map spirometer from a fixed Biopac channel index when naming is generic
    if 'spirometer' not in signal_mappings:
        channel_idx = getattr(config, 'SPIROMETER_CHANNEL_INDEX', None)
        if isinstance(channel_idx, int) and 1 <= channel_idx <= len(df_raw.columns):
            signal_mappings['spirometer'] = df_raw.columns[channel_idx - 1]

    # Prefer converted mmHg columns over raw voltage columns
    if 'co2' in gas_conversions:
        signal_mappings['etco2'] = gas_conversions['co2']
    if 'o2' in gas_conversions:
        signal_mappings['eto2'] = gas_conversions['o2']
    if 'doppler' in doppler_conversions:
        signal_mappings['doppler'] = doppler_conversions['doppler']

    # Session B: inject Siemens PMU pulse/respiration as PPG/RSP channels.
    df_raw, signal_mappings, pmu_status = _attach_pmu_session_b_signals(
        df_raw=df_raw,
        signal_mappings=signal_mappings,
        participant=participant,
        session=session,
        task=task,
    )

    return {
        'df': df_raw,
        'sampling_rate': sampling_rate,
        'channels': list(df_raw.columns),
        'signal_mappings': signal_mappings,
        'n_samples': len(df_raw),
        'duration': len(df_raw) / sampling_rate,
        'gas_conversions': gas_conversions,
        'pmu_status': pmu_status,
    }
