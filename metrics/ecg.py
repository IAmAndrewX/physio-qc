"""
ECG signal processing functions
Pure functions with no classes
"""

import neurokit2 as nk
import numpy as np


def clean_ecg(
    signal,
    sampling_rate,
    method="neurokit",
    powerline=60,
    lowcut=0.5,
    highcut=45.0,
    filter_type="butterworth",
    filter_order=5,
    apply_lowcut=True,
    apply_highcut=True,
):
    """
    Clean ECG signal using NeuroKit2 methods or custom filtering

    Parameters
    ----------
    signal : array
        Raw ECG signal
    sampling_rate : int
        Sampling rate in Hz
    method : str
        Cleaning method ('neurokit', 'biosppy', 'custom', etc.)
    powerline : int
        Powerline frequency for filtering (50 or 60 Hz)
    lowcut : float
        High-pass cutoff frequency (Hz)
    highcut : float
        Low-pass cutoff frequency (Hz)
    filter_type : str
        Filter type for custom filtering ('butterworth', 'fir', etc.)
    filter_order : int
        Filter order
    apply_lowcut : bool
        Apply high-pass filter
    apply_highcut : bool
        Apply low-pass filter

    Returns
    -------
    array
        Cleaned ECG signal
    """
    if method == "custom":
        ecg_clean = signal.copy()
        low = lowcut if apply_lowcut else None
        high = highcut if apply_highcut else None
        if (low is not None and low > 0) or (high is not None and high < sampling_rate / 2):
            ecg_clean = nk.signal_filter(
                ecg_clean, sampling_rate=sampling_rate, lowcut=low, highcut=high, method=filter_type, order=filter_order
            )
        # Keep powerline handling consistent with NeuroKit ECG cleaning.
        if powerline in (50, 60):
            ecg_clean = nk.signal_filter(
                ecg_clean, sampling_rate=sampling_rate, method="powerline", powerline=powerline
            )
    else:
        ecg_clean = nk.ecg_clean(signal, sampling_rate=sampling_rate, method=method, powerline=powerline)

    return ecg_clean


def detect_r_peaks(signal, sampling_rate, method="neurokit", correct_artifacts=False):
    """
    Detect R-peaks in ECG signal

    Parameters
    ----------
    signal : array
        Cleaned ECG signal
    sampling_rate : int
        Sampling rate in Hz
    method : str
        Peak detection method
    correct_artifacts : bool
        Apply artifact correction to peaks

    Returns
    -------
    array
        R-peak sample indices
    """
    _, peaks_info = nk.ecg_peaks(
        signal, sampling_rate=sampling_rate, method=method, correct_artifacts=correct_artifacts
    )
    r_peaks = peaks_info["ECG_R_Peaks"]

    return r_peaks


def calculate_ecg_quality(signal, r_peaks, sampling_rate):
    """
    Calculate ECG quality metrics

    Parameters
    ----------
    signal : array
        Cleaned ECG signal
    r_peaks : array
        R-peak sample indices
    sampling_rate : int
        Sampling rate in Hz

    Returns
    -------
    dict
        Dictionary containing:
        - quality_zhao: Zhao2018 categorical classification
        - quality_templatematch: Template correlation (0-1, higher = better shape match)
        - quality_averageqrs: Distance from average (0-1, higher = closer to average)
        - quality_continuous: Alias for templatematch (backward compatibility)
        - quality_mean: Mean templatematch score
    """
    quality_zhao = nk.ecg_quality(
        signal, rpeaks=r_peaks, sampling_rate=sampling_rate, method="zhao2018", approach="simple"
    )

    quality_templatematch = nk.ecg_quality(signal, rpeaks=r_peaks, sampling_rate=sampling_rate, method="templatematch")

    quality_averageqrs = nk.ecg_quality(signal, rpeaks=r_peaks, sampling_rate=sampling_rate, method="averageQRS")

    return {
        "quality_zhao": quality_zhao,
        "quality_templatematch": quality_templatematch,
        "quality_templatematch_mean": np.nanmean(quality_templatematch),
        "quality_averageqrs": quality_averageqrs,
        "quality_averageqrs_mean": np.nanmean(quality_averageqrs),
        "quality_continuous": quality_templatematch,
        "quality_mean": np.nanmean(quality_templatematch),
    }


def calculate_hr(r_peaks, sampling_rate, signal_length, rate_method="monotone_cubic"):
    """
    Calculate heart rate from R-peaks

    Parameters
    ----------
    r_peaks : array
        R-peak sample indices
    sampling_rate : int
        Sampling rate in Hz
    signal_length : int
        Length of signal for interpolation
    rate_method : str
        Interpolation method

    Returns
    -------
    dict
        Dictionary containing:
        - hr_bpm: Beat-to-beat HR in bpm
        - hr_interpolated: Interpolated HR signal
        - mean_hr: Mean HR
        - std_hr: Standard deviation of HR
    """
    rr_intervals = np.diff(r_peaks) / sampling_rate
    hr_bpm = 60 / rr_intervals

    hr_interpolated = nk.ecg_rate(
        r_peaks, sampling_rate=sampling_rate, desired_length=signal_length, interpolation_method=rate_method
    )

    return {
        "hr_bpm": hr_bpm,
        "hr_interpolated": hr_interpolated,
        "mean_hr": np.nanmean(hr_bpm),
        "std_hr": np.nanstd(hr_bpm),
    }


def process_ecg(signal, sampling_rate, params):
    """
    Complete ECG processing pipeline

    Parameters
    ----------
    signal : array
        Raw ECG signal
    sampling_rate : int
        Sampling rate in Hz
    params : dict
        Processing parameters

    Returns
    -------
    dict or None
        Dictionary containing all processing results, or None if insufficient peaks
        Includes:
        - raw: Raw signal
        - clean: Cleaned signal
        - auto_r_peaks: Auto-detected R-peak indices (immutable)
        - current_r_peaks: Current R-peak indices (mutable for editing)
        - r_peaks_times: Peak times in seconds
        - hr_bpm: Beat-to-beat heart rate
        - hr_interpolated: Continuous heart rate signal
        - quality_*: Quality metrics if calculated
        - n_peaks: Number of peaks
        - mean_hr, std_hr: Statistics
    """
    signal = np.asarray(signal, dtype=float)

    cleaning_method = params.get("cleaning_method", params.get("method", "neurokit"))

    ecg_clean = clean_ecg(
        signal,
        sampling_rate,
        method=cleaning_method,
        powerline=params.get("powerline", 60),
        lowcut=params.get("lowcut", 0.5),
        highcut=params.get("highcut", 45.0),
        filter_type=params.get("filter_type", "butterworth"),
        filter_order=params.get("filter_order", 5),
        apply_lowcut=params.get("apply_lowcut", True),
        apply_highcut=params.get("apply_highcut", True),
    )

    r_peaks = detect_r_peaks(
        ecg_clean,
        sampling_rate,
        method=params.get("peak_method", "neurokit"),
        correct_artifacts=params.get("correct_artifacts", False),
    )

    if len(r_peaks) < 2:
        return None

    hr_result = calculate_hr(
        r_peaks, sampling_rate, len(ecg_clean), rate_method=params.get("rate_method", "monotone_cubic")
    )

    result = {
        "raw": signal,
        "clean": ecg_clean,
        "raw_signal": signal,
        "cleaned_signal": ecg_clean,
        "auto_r_peaks": r_peaks.copy(),
        "current_r_peaks": r_peaks.copy(),
        "auto_peaks": r_peaks.copy(),
        "current_peaks": r_peaks.copy(),
        "r_peaks_times": r_peaks / sampling_rate,
        "sampling_rate": sampling_rate,
        "params": params.copy(),
        "n_peaks": len(r_peaks),
    }

    result.update(hr_result)

    if params.get("calculate_quality", False):
        quality_result = calculate_ecg_quality(ecg_clean, r_peaks, sampling_rate)
        result.update(quality_result)
    else:
        result["quality_zhao"] = "Not calculated"
        result["quality_continuous"] = np.array([])
        result["quality_mean"] = 0.0

    return result
