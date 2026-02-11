"""
SpO2 (Oxygen Saturation) Processing Module

This module provides functions for processing SpO2 signals from pulse oximetry.
SpO2 values are percentages (0-100%) representing arterial oxygen saturation.

Functions:
    clean_spo2: Apply filtering/smoothing to SpO2 signal
    detect_desaturation_events: Find periods where SpO2 drops below threshold
    calculate_spo2_metrics: Compute summary statistics
    process_spo2: Main processing function
"""

from typing import Dict, List, Optional, Tuple

import neurokit2 as nk
import numpy as np
from scipy.signal import savgol_filter


def _nearest_odd(n: int) -> int:
    """Ensure window size is odd for Savitzky-Golay filter."""
    return n if n % 2 == 1 else n + 1


def clean_spo2(
    signal: np.ndarray,
    sampling_rate: float,
    method: str = "lowpass",
    lowpass_cutoff: float = 0.5,
    filter_order: int = 2,
    sg_window_s: float = 1.0,
    sg_poly: int = 2,
) -> np.ndarray:
    """
    Clean SpO2 signal by applying filtering/smoothing.

    Parameters
    ----------
    signal : np.ndarray
        Raw SpO2 signal (percentage values 0-100)
    sampling_rate : float
        Sampling frequency in Hz
    method : str
        Cleaning method: 'lowpass', 'savgol', or 'none'
    lowpass_cutoff : float
        Cutoff frequency for lowpass filter in Hz
    filter_order : int
        Filter order for Butterworth filter
    sg_window_s : float
        Window duration for Savitzky-Golay filter in seconds
    sg_poly : int
        Polynomial order for Savitzky-Golay filter

    Returns
    -------
    cleaned : np.ndarray
        Cleaned SpO2 signal
    """
    if method == "none":
        return signal.copy()

    if method == "lowpass":
        # NeuroKit2-backed low-pass filtering for consistency across modules.
        return nk.signal_filter(
            signal, sampling_rate=sampling_rate, highcut=lowpass_cutoff, method="butterworth", order=filter_order
        )

    if method == "savgol":
        # Savitzky-Golay smoothing
        win_pts = max(5, int(round(sg_window_s * sampling_rate)))
        win_pts = _nearest_odd(win_pts)
        try:
            cleaned = savgol_filter(signal, window_length=win_pts, polyorder=min(sg_poly, win_pts - 1))
        except ValueError:
            cleaned = signal.copy()
        return cleaned

    # Default fallback
    return signal.copy()


def detect_desaturation_events(
    signal: np.ndarray,
    sampling_rate: float,
    threshold: float = 90.0,
    min_drop: float = 3.0,
    min_duration_s: float = 10.0,
) -> List[Tuple[int, int, float]]:
    """
    Detect desaturation events in SpO2 signal.

    A desaturation event is defined as a period where SpO2 stays below
    `threshold` for at least `min_duration_s` seconds and reaches at least
    `min_drop` percentage points below a local pre-event baseline.

    Parameters
    ----------
    signal : np.ndarray
        Cleaned SpO2 signal (percentage values)
    sampling_rate : float
        Sampling frequency in Hz
    threshold : float
        SpO2 threshold in % (default: 90%)
    min_drop : float
        Minimum drop from baseline to count as event (default: 3%)
    min_duration_s : float
        Minimum event duration in seconds (default: 10s)

    Returns
    -------
    events : List[Tuple[int, int, float]]
        List of (start_idx, end_idx, min_spo2) tuples for each event
    """
    min_samples = int(min_duration_s * sampling_rate)
    baseline_window_samples = max(1, int(30 * sampling_rate))
    below_threshold = signal < threshold

    events = []
    in_event = False
    event_start = 0

    for i in range(len(signal)):
        if below_threshold[i] and not in_event:
            # Start of potential event
            in_event = True
            event_start = i
        elif not below_threshold[i] and in_event:
            # End of event
            event_duration = i - event_start
            if event_duration >= min_samples:
                min_spo2 = np.min(signal[event_start:i])
                baseline_start = max(0, event_start - baseline_window_samples)
                baseline_slice = signal[baseline_start:event_start]
                baseline = np.median(baseline_slice) if baseline_slice.size > 0 else np.median(signal)
                if (baseline - min_spo2) >= min_drop:
                    events.append((event_start, i, min_spo2))
            in_event = False

    # Handle event that extends to end of signal
    if in_event:
        event_duration = len(signal) - event_start
        if event_duration >= min_samples:
            min_spo2 = np.min(signal[event_start:])
            baseline_start = max(0, event_start - baseline_window_samples)
            baseline_slice = signal[baseline_start:event_start]
            baseline = np.median(baseline_slice) if baseline_slice.size > 0 else np.median(signal)
            if (baseline - min_spo2) >= min_drop:
                events.append((event_start, len(signal), min_spo2))

    return events


def calculate_spo2_metrics(signal: np.ndarray, sampling_rate: float, events: List[Tuple[int, int, float]]) -> Dict:
    """
    Calculate summary metrics for SpO2 signal.

    Parameters
    ----------
    signal : np.ndarray
        Cleaned SpO2 signal
    sampling_rate : float
        Sampling frequency in Hz
    events : List[Tuple[int, int, float]]
        Desaturation events from detect_desaturation_events()

    Returns
    -------
    metrics : Dict
        Dictionary containing:
        - mean_spo2: Mean SpO2 (%)
        - min_spo2: Minimum SpO2 (%)
        - max_spo2: Maximum SpO2 (%)
        - std_spo2: Standard deviation (%)
        - time_below_90: Time with SpO2 < 90% (seconds)
        - time_below_90_pct: Percentage of recording with SpO2 < 90%
        - time_below_95: Time with SpO2 < 95% (seconds)
        - time_below_95_pct: Percentage of recording with SpO2 < 95%
        - n_desaturation_events: Number of desaturation events
        - desaturation_index: Events per hour
        - total_duration: Total recording duration (seconds)
    """
    duration_s = len(signal) / sampling_rate
    duration_hours = duration_s / 3600

    # Basic statistics
    mean_spo2 = np.mean(signal)
    min_spo2 = np.min(signal)
    max_spo2 = np.max(signal)
    std_spo2 = np.std(signal)

    # Time below thresholds
    time_below_90 = np.sum(signal < 90) / sampling_rate
    time_below_90_pct = (time_below_90 / duration_s) * 100 if duration_s > 0 else 0

    time_below_95 = np.sum(signal < 95) / sampling_rate
    time_below_95_pct = (time_below_95 / duration_s) * 100 if duration_s > 0 else 0

    # Desaturation metrics
    n_events = len(events)
    desaturation_index = n_events / duration_hours if duration_hours > 0 else 0

    return {
        "mean_spo2": mean_spo2,
        "min_spo2": min_spo2,
        "max_spo2": max_spo2,
        "std_spo2": std_spo2,
        "time_below_90": time_below_90,
        "time_below_90_pct": time_below_90_pct,
        "time_below_95": time_below_95,
        "time_below_95_pct": time_below_95_pct,
        "n_desaturation_events": n_events,
        "desaturation_index": desaturation_index,
        "total_duration": duration_s,
    }


def process_spo2(signal: np.ndarray, sampling_rate: float, params: Optional[Dict] = None) -> Dict:
    """
    Process SpO2 signal: clean, detect events, calculate metrics.

    Parameters
    ----------
    signal : np.ndarray
        Raw SpO2 signal (percentage values 0-100)
    sampling_rate : float
        Sampling frequency in Hz
    params : dict, optional
        Processing parameters:
        - cleaning_method: 'lowpass', 'savgol', or 'none' (default: 'lowpass')
        - lowpass_cutoff: Cutoff frequency in Hz (default: 0.5)
        - filter_order: Butterworth filter order (default: 2)
        - sg_window_s: Savitzky-Golay window in seconds (default: 1.0)
        - sg_poly: Savitzky-Golay polynomial order (default: 2)
        - desaturation_threshold: Threshold for events in % (default: 90.0)
        - desaturation_drop: Minimum drop for event in % (default: 3.0)
        - min_event_duration_s: Minimum event duration in seconds (default: 10.0)

    Returns
    -------
    result : Dict
        Dictionary containing:
        - raw_signal: Original SpO2 signal
        - cleaned_signal: Filtered/smoothed signal
        - time_vector: Time axis in seconds
        - desaturation_events: List of (start, end, min) tuples
        - metrics: Summary statistics dictionary
        - sampling_rate: Sampling frequency
        - params: Processing parameters used
    """
    if params is None:
        params = {}

    signal = np.asarray(signal, dtype=float)

    # Extract parameters with defaults
    cleaning_method = params.get("cleaning_method", "lowpass")
    lowpass_cutoff = params.get("lowpass_cutoff", 0.5)
    filter_order = params.get("filter_order", 2)
    sg_window_s = params.get("sg_window_s", 1.0)
    sg_poly = params.get("sg_poly", 2)
    desaturation_threshold = params.get("desaturation_threshold", 90.0)
    desaturation_drop = params.get("desaturation_drop", 3.0)
    min_event_duration_s = params.get("min_event_duration_s", 10.0)

    # Clean signal
    cleaned = clean_spo2(
        signal,
        sampling_rate,
        method=cleaning_method,
        lowpass_cutoff=lowpass_cutoff,
        filter_order=filter_order,
        sg_window_s=sg_window_s,
        sg_poly=sg_poly,
    )

    # Clip to valid range (0-100%)
    cleaned = np.clip(cleaned, 0, 100)

    # Create time vector
    time_vector = np.arange(len(signal)) / sampling_rate

    # Detect desaturation events
    events = detect_desaturation_events(
        cleaned,
        sampling_rate,
        threshold=desaturation_threshold,
        min_drop=desaturation_drop,
        min_duration_s=min_event_duration_s,
    )

    # Calculate metrics
    metrics = calculate_spo2_metrics(cleaned, sampling_rate, events)

    result = {
        "raw_signal": signal,
        "cleaned_signal": cleaned,
        "raw": signal,
        "clean": cleaned,
        "time_vector": time_vector,
        "desaturation_events": events,
        "metrics": metrics,
        "sampling_rate": sampling_rate,
        "params": params.copy(),
    }
    return result
