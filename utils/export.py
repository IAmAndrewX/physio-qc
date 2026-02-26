"""
Export functions for physiological data
CSV and JSON export with BIDS-inspired metadata
Pure functions with no classes
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import config
from utils.peak_editing import calculate_peak_delta, get_edited_peaks_info


def create_combined_dataframe(results_dict, sampling_rate):
    """
    Create combined DataFrame with all signals and peak columns

    Parameters
    ----------
    results_dict : dict
        Dictionary with keys: 'ecg', 'rsp', 'ppg', 'bp' (if available)
        Each value is the result dict from processing
    sampling_rate : int
        Sampling rate in Hz

    Returns
    -------
    DataFrame
        Combined dataframe with time and all signal columns
    """
    max_length = 0
    for signal_type in ['ecg', 'rsp', 'ppg', 'bp']:
        if signal_type in results_dict and results_dict[signal_type] is not None:
            max_length = max(max_length, len(results_dict[signal_type]['raw']))

    time = np.arange(max_length) / sampling_rate
    data = {'time': time}

    if 'ecg' in results_dict and results_dict['ecg'] is not None:
        ecg_result = results_dict['ecg']
        signal_length = len(ecg_result['raw'])

        data['ecg_raw'] = ecg_result['raw'].astype(config.EXPORT_DTYPE_SIGNALS)
        data['ecg_clean'] = ecg_result['clean'].astype(config.EXPORT_DTYPE_SIGNALS)
        data['ecg_r_peaks'] = calculate_peak_delta(
            ecg_result['auto_r_peaks'],
            ecg_result['current_r_peaks'],
            signal_length
        )
        data['ecg_hr_interpolated'] = ecg_result['hr_interpolated'].astype(config.EXPORT_DTYPE_SIGNALS)

        if 'ecg_ventricular_phase' in ecg_result:
            data['ecg_phase_ventricular'] = ecg_result['ecg_ventricular_phase'].astype(config.EXPORT_DTYPE_ONSETS)
            data['ecg_phase_completion_ventricular'] = ecg_result['ecg_ventricular_completion'].astype(config.EXPORT_DTYPE_SIGNALS)
            data['ecg_cardiac_cycle_completion'] = ecg_result['ecg_cardiac_cycle_completion'].astype(config.EXPORT_DTYPE_SIGNALS)

    if 'rsp' in results_dict and results_dict['rsp'] is not None:
        rsp_result = results_dict['rsp']
        signal_length = len(rsp_result['raw'])

        data['rsp_raw'] = rsp_result['raw'].astype(config.EXPORT_DTYPE_SIGNALS)
        data['rsp_clean'] = rsp_result['clean'].astype(config.EXPORT_DTYPE_SIGNALS)
        data['rsp_inhalation_onsets'] = calculate_peak_delta(
            rsp_result['auto_peaks'],
            rsp_result['current_peaks'],
            signal_length
        )
        data['rsp_exhalation_onsets'] = calculate_peak_delta(
            rsp_result['auto_troughs'],
            rsp_result['current_troughs'],
            signal_length
        )
        data['rsp_br_interpolated'] = rsp_result['br_interpolated'].astype(config.EXPORT_DTYPE_SIGNALS)

        if 'rvt' in rsp_result:
            data['rsp_rvt'] = rsp_result['rvt'].astype(config.EXPORT_DTYPE_SIGNALS)

        if 'rsp_phase' in rsp_result:
            data['rsp_phase'] = rsp_result['rsp_phase'].astype(config.EXPORT_DTYPE_ONSETS)
            data['rsp_phase_completion'] = rsp_result['rsp_phase_completion'].astype(config.EXPORT_DTYPE_SIGNALS)
            data['rsp_cycle_completion'] = rsp_result['rsp_cycle_completion'].astype(config.EXPORT_DTYPE_SIGNALS)

    if 'ppg' in results_dict and results_dict['ppg'] is not None:
        ppg_result = results_dict['ppg']
        signal_length = len(ppg_result['raw'])

        data['ppg_raw'] = ppg_result['raw'].astype(config.EXPORT_DTYPE_SIGNALS)
        data['ppg_clean'] = ppg_result['clean'].astype(config.EXPORT_DTYPE_SIGNALS)
        data['ppg_peaks'] = calculate_peak_delta(
            ppg_result['auto_peaks'],
            ppg_result['current_peaks'],
            signal_length
        )
        data['ppg_hr_interpolated'] = ppg_result['hr_interpolated'].astype(config.EXPORT_DTYPE_SIGNALS)

    if 'bp' in results_dict and results_dict['bp'] is not None:
        bp_result = results_dict['bp']
        signal_length = len(bp_result['raw'])

        data['bp_raw'] = bp_result['raw'].astype(config.EXPORT_DTYPE_SIGNALS)
        data['bp_filtered'] = bp_result['filtered'].astype(config.EXPORT_DTYPE_SIGNALS)
        data['bp_systolic_onsets'] = calculate_peak_delta(
            bp_result['auto_peaks'],
            bp_result['current_peaks'],
            signal_length
        )
        data['bp_diastolic_onsets'] = calculate_peak_delta(
            bp_result['auto_troughs'],
            bp_result['current_troughs'],
            signal_length
        )
        data['bp_sbp'] = bp_result['sbp_signal'].astype(config.EXPORT_DTYPE_SIGNALS)
        data['bp_dbp'] = bp_result['dbp_signal'].astype(config.EXPORT_DTYPE_SIGNALS)
        data['bp_mbp'] = bp_result['mbp_signal'].astype(config.EXPORT_DTYPE_SIGNALS)

    df = pd.DataFrame(data)
    return df


def create_metadata_json(results_dict, params_dict, sampling_rate, subject_metadata=None):
    """
    Create BIDS-inspired JSON metadata

    Parameters
    ----------
    results_dict : dict
        Dictionary with processing results
    params_dict : dict
        Dictionary with processing parameters used
    sampling_rate : int
        Sampling rate in Hz
    subject_metadata : dict, optional
        Participant/session/task metadata and notes payload.

    Returns
    -------
    dict
        Metadata dictionary ready for JSON export
    """
    metadata = {
        "SamplingFrequency": int(sampling_rate),
        "StartTime": 0,
        "ProcessingDate": datetime.now().isoformat(),
        "ProcessingSoftware": "physio-qc",
        "PeakEncoding": {
            "Description": "Peak/onset encoding in binary columns",
            "Values": {
                "1": "Auto-detected peak",
                "2": "Manually added peak",
                "0": "No peak",
                "-1": "Deleted peak"
            }
        }
    }

    columns = ["time"]

    if 'ecg' in results_dict and results_dict['ecg'] is not None:
        ecg_result = results_dict['ecg']
        ecg_params = params_dict.get('ecg', {})
        ecg_info = get_edited_peaks_info(
            ecg_result['auto_r_peaks'],
            ecg_result['current_r_peaks']
        )

        metadata['ECG'] = {
            "CleaningMethod": ecg_params.get('method', 'neurokit'),
            "CleaningParameters": {
                "powerline": ecg_params.get('powerline', 60),
                "lowcut": ecg_params.get('lowcut', 0.5),
                "highcut": ecg_params.get('highcut', 45.0)
            },
            "PeakDetectionMethod": ecg_params.get('peak_method', 'neurokit'),
            "ArtifactCorrection": ecg_params.get('correct_artifacts', False),
            "AutoDetectedPeaks": int(ecg_info['auto_count']),
            "ManuallyAddedPeaks": int(ecg_info['added_count']),
            "DeletedPeaks": int(ecg_info['deleted_count']),
            "FinalPeakCount": int(ecg_info['final_count']),
            "DeletedPeakIndices": ecg_info['deleted_indices'].tolist(),
            "AddedPeakIndices": ecg_info['added_indices'].tolist()
        }
        columns.extend(["ecg_raw", "ecg_clean", "ecg_r_peaks", "ecg_hr_interpolated"])
        if 'ecg_ventricular_phase' in ecg_result:
            metadata['ECG']['PhaseComputed'] = True
            columns.extend(["ecg_phase_ventricular", "ecg_phase_completion_ventricular", "ecg_cardiac_cycle_completion"])

    if 'rsp' in results_dict and results_dict['rsp'] is not None:
        rsp_result = results_dict['rsp']
        rsp_params = params_dict.get('rsp', {})
        peaks_info = get_edited_peaks_info(
            rsp_result['auto_peaks'],
            rsp_result['current_peaks']
        )
        troughs_info = get_edited_peaks_info(
            rsp_result['auto_troughs'],
            rsp_result['current_troughs']
        )

        metadata['RSP'] = {
            "CleaningMethod": rsp_params.get('method', 'khodadad2018'),
            "CleaningParameters": {
                "lowcut": rsp_params.get('lowcut', 0.05),
                "highcut": rsp_params.get('highcut', 3.0)
            },
            "AutoDetectedBreaths": int(troughs_info['auto_count']),
            "ManuallyAddedInhalations": int(peaks_info['added_count']),
            "DeletedInhalations": int(peaks_info['deleted_count']),
            "ManuallyAddedExhalations": int(troughs_info['added_count']),
            "DeletedExhalations": int(troughs_info['deleted_count']),
            "FinalBreathCount": int(troughs_info['final_count'])
        }
        if 'rvt' in rsp_result:
            metadata['RSP']['RVTMethod'] = rsp_params.get('rvt_method', 'unknown')
        if 'rsp_phase' in rsp_result:
            metadata['RSP']['PhaseComputed'] = True
        columns.extend(["rsp_raw", "rsp_clean", "rsp_inhalation_onsets", "rsp_exhalation_onsets", "rsp_br_interpolated"])
        if 'rvt' in rsp_result:
            columns.append("rsp_rvt")
        if 'rsp_phase' in rsp_result:
            columns.extend(["rsp_phase", "rsp_phase_completion", "rsp_cycle_completion"])

    if 'ppg' in results_dict and results_dict['ppg'] is not None:
        ppg_result = results_dict['ppg']
        ppg_params = params_dict.get('ppg', {})
        ppg_info = get_edited_peaks_info(
            ppg_result['auto_peaks'],
            ppg_result['current_peaks']
        )

        metadata['PPG'] = {
            "CleaningMethod": ppg_params.get('method', 'elgendi'),
            "PeakDetectionMethod": ppg_params.get('peak_method', 'elgendi'),
            "AutoDetectedPeaks": int(ppg_info['auto_count']),
            "ManuallyAddedPeaks": int(ppg_info['added_count']),
            "DeletedPeaks": int(ppg_info['deleted_count']),
            "FinalPeakCount": int(ppg_info['final_count'])
        }
        columns.extend(["ppg_raw", "ppg_clean", "ppg_peaks", "ppg_hr_interpolated"])

    if 'bp' in results_dict and results_dict['bp'] is not None:
        bp_result = results_dict['bp']
        bp_params = params_dict.get('bp', {})
        peaks_info = get_edited_peaks_info(
            bp_result['auto_peaks'],
            bp_result['current_peaks']
        )
        troughs_info = get_edited_peaks_info(
            bp_result['auto_troughs'],
            bp_result['current_troughs']
        )

        calib_regions = []
        if bp_result.get('calibration_artifacts') is not None:
            calib_art = bp_result['calibration_artifacts']
            if len(calib_art['starts']) > 0:
                calib_regions = [[int(s), int(e)] for s, e in zip(calib_art['starts'], calib_art['ends'])]

        metadata['BloodPressure'] = {
            "FilteringMethod": bp_params.get('filter_method', 'bessel_25hz'),
            "FilteringParameters": {
                "cutoff_freq": bp_params.get('cutoff_freq', 25)
            },
            "DelineationMethod": bp_params.get('peak_method', 'delineator'),
            "CalibrationArtifactsDetected": len(calib_regions),
            "CalibrationRegions": calib_regions,
            "AutoDetectedSystolicPeaks": int(peaks_info['auto_count']),
            "AutoDetectedDiastolicTroughs": int(troughs_info['auto_count']),
            "ManuallyAddedSystolicPeaks": int(peaks_info['added_count']),
            "DeletedSystolicPeaks": int(peaks_info['deleted_count']),
            "ManuallyAddedDiastolicTroughs": int(troughs_info['added_count']),
            "DeletedDiastolicTroughs": int(troughs_info['deleted_count']),
            "FinalCardiacCycles": int(min(peaks_info['final_count'], troughs_info['final_count']))
        }
        columns.extend(["bp_raw", "bp_filtered", "bp_systolic_onsets", "bp_diastolic_onsets", "bp_sbp", "bp_dbp", "bp_mbp"])

    metadata['Columns'] = columns

    total_edits = 0
    for signal_key in ['ECG', 'RSP', 'PPG', 'BloodPressure']:
        if signal_key in metadata:
            if signal_key == 'RSP':
                total_edits += metadata[signal_key].get('ManuallyAddedInhalations', 0)
                total_edits += metadata[signal_key].get('DeletedInhalations', 0)
                total_edits += metadata[signal_key].get('ManuallyAddedExhalations', 0)
                total_edits += metadata[signal_key].get('DeletedExhalations', 0)
            elif signal_key == 'BloodPressure':
                total_edits += metadata[signal_key].get('ManuallyAddedSystolicPeaks', 0)
                total_edits += metadata[signal_key].get('DeletedSystolicPeaks', 0)
                total_edits += metadata[signal_key].get('ManuallyAddedDiastolicTroughs', 0)
                total_edits += metadata[signal_key].get('DeletedDiastolicTroughs', 0)
            else:
                total_edits += metadata[signal_key].get('ManuallyAddedPeaks', 0)
                total_edits += metadata[signal_key].get('DeletedPeaks', 0)

    metadata['QualityControl'] = {
        "QCDate": datetime.now().isoformat(),
        "QCOperator": "automatic",
        "TotalManualEdits": int(total_edits)
    }

    if isinstance(subject_metadata, dict) and subject_metadata:
        metadata['SubjectMetadata'] = subject_metadata

    return metadata


def export_physio_data(output_path, participant, session, task, df, metadata):
    """
    Export physiological data to CSV and JSON files

    Parameters
    ----------
    output_path : str or Path
        Base output directory
    participant : str
        Participant ID (e.g., 'sub-2034')
    session : str
        Session ID (e.g., 'ses-01')
    task : str
        Task name (e.g., 'rest')
    df : DataFrame
        Combined dataframe with all signals
    metadata : dict
        Metadata dictionary

    Returns
    -------
    dict
        Dictionary with:
        - csv_path: Path to exported CSV file
        - json_path: Path to exported JSON file
    """
    output_path = Path(output_path)
    sub_ses_dir = output_path / participant / session
    sub_ses_dir.mkdir(parents=True, exist_ok=True)

    base_filename = f"{participant}_{session}_task-{task}_physio"

    csv_path = sub_ses_dir / f"{base_filename}.csv"
    json_path = sub_ses_dir / f"{base_filename}.json"

    df.to_csv(csv_path, index=False)

    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return {
        'csv_path': str(csv_path),
        'json_path': str(json_path)
    }
