"""
PPG (Photoplethysmography) signal processing functions
Pure functions with no classes
"""

import neurokit2 as nk
import numpy as np


def clean_ppg(
    signal,
    sampling_rate,
    method="elgendi",
    lowcut=0.5,
    highcut=8.0,
    filter_type="butterworth",
    filter_order=5,
    apply_lowcut=True,
    apply_highcut=True,
):
    """
    Clean PPG signal using NeuroKit2 methods or custom filtering

    Parameters
    ----------
    signal : array
        Raw PPG signal
    sampling_rate : int
        Sampling rate in Hz
    method : str
        Cleaning method ('elgendi', 'nabian2018', 'none', 'custom')
    lowcut : float
        High-pass cutoff frequency (Hz)
    highcut : float
        Low-pass cutoff frequency (Hz)
    filter_type : str
        Filter type for custom filtering
    filter_order : int
        Filter order
    apply_lowcut : bool
        Apply high-pass filter
    apply_highcut : bool
        Apply low-pass filter

    Returns
    -------
    array
        Cleaned PPG signal
    """
    if method == "custom":
        ppg_clean = signal.copy()
        low = lowcut if apply_lowcut else None
        high = highcut if apply_highcut else None
        if (low is not None and low > 0) or (high is not None and high < sampling_rate / 2):
            ppg_clean = nk.signal_filter(
                ppg_clean, sampling_rate=sampling_rate, lowcut=low, highcut=high, method=filter_type, order=filter_order
            )
    elif method == "none":
        ppg_clean = signal.copy()
    else:
        ppg_clean = nk.ppg_clean(signal, sampling_rate=sampling_rate, method=method)

    return ppg_clean


def detect_ppg_peaks(signal, sampling_rate, method="elgendi", correct_artifacts=False):
    """
    Detect systolic peaks in PPG signal

    Parameters
    ----------
    signal : array
        Cleaned PPG signal
    sampling_rate : int
        Sampling rate in Hz
    method : str
        Peak detection method ('elgendi', 'bishop', 'charlton')
    correct_artifacts : bool
        Apply artifact correction to peaks

    Returns
    -------
    array
        PPG peak sample indices
    """
    _, peaks_info = nk.ppg_peaks(
        signal, sampling_rate=sampling_rate, method=method, correct_artifacts=correct_artifacts
    )
    ppg_peaks = peaks_info["PPG_Peaks"]

    return ppg_peaks


def calculate_ppg_quality(signal, sampling_rate):
    """
    Calculate PPG quality metrics

    Parameters
    ----------
    signal : array
        Cleaned PPG signal
    sampling_rate : int
        Sampling rate in Hz

    Returns
    -------
    dict
        Dictionary containing:
        - quality: Quality metric per sample
        - quality_mean: Mean quality score
    """
    quality = nk.ppg_quality(signal, sampling_rate=sampling_rate)

    return {"quality": quality, "quality_mean": np.nanmean(quality)}


def calculate_hr_from_ppg(ppg_peaks, sampling_rate, signal_length, rate_method="monotone_cubic"):
    """
    Calculate heart rate from PPG peaks

    Parameters
    ----------
    ppg_peaks : array
        PPG peak sample indices
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
    peak_intervals = np.diff(ppg_peaks) / sampling_rate
    hr_bpm = 60 / peak_intervals

    hr_interpolated = nk.signal_rate(
        ppg_peaks, sampling_rate=sampling_rate, desired_length=signal_length, interpolation_method=rate_method
    )

    return {
        "hr_bpm": hr_bpm,
        "hr_interpolated": hr_interpolated,
        "mean_hr": np.nanmean(hr_bpm),
        "std_hr": np.nanstd(hr_bpm),
    }


def process_ppg(signal, sampling_rate, params):
    """
    Complete PPG processing pipeline

    Parameters
    ----------
    signal : array
        Raw PPG signal
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
        - auto_peaks: Auto-detected peak indices (immutable)
        - current_peaks: Current peak indices (mutable for editing)
        - peaks_times: Peak times in seconds
        - hr_bpm: Beat-to-beat heart rate
        - hr_interpolated: Continuous heart rate signal
        - quality: Quality metric per sample
        - quality_mean: Mean quality score
        - n_peaks: Number of peaks
        - mean_hr, std_hr: Statistics
    """
    signal = np.asarray(signal, dtype=float)

    cleaning_method = params.get("cleaning_method", params.get("method", "elgendi"))

    ppg_clean = clean_ppg(
        signal,
        sampling_rate,
        method=cleaning_method,
        lowcut=params.get("lowcut", 0.5),
        highcut=params.get("highcut", 8.0),
        filter_type=params.get("filter_type", "butterworth"),
        filter_order=params.get("filter_order", 5),
        apply_lowcut=params.get("apply_lowcut", True),
        apply_highcut=params.get("apply_highcut", True),
    )

    ppg_peaks = detect_ppg_peaks(
        ppg_clean,
        sampling_rate,
        method=params.get("peak_method", "elgendi"),
        correct_artifacts=params.get("correct_artifacts", False),
    )

    if len(ppg_peaks) < 2:
        return None

    hr_result = calculate_hr_from_ppg(
        ppg_peaks, sampling_rate, len(ppg_clean), rate_method=params.get("rate_method", "monotone_cubic")
    )

    quality_result = calculate_ppg_quality(ppg_clean, sampling_rate)

    result = {
        "raw": signal,
        "clean": ppg_clean,
        "raw_signal": signal,
        "cleaned_signal": ppg_clean,
        "auto_peaks": ppg_peaks.copy(),
        "current_peaks": ppg_peaks.copy(),
        "peaks_times": ppg_peaks / sampling_rate,
        "sampling_rate": sampling_rate,
        "params": params.copy(),
        "n_peaks": len(ppg_peaks),
    }

    result.update(hr_result)
    result.update(quality_result)

    return result
