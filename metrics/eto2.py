"""
End-Tidal O2 (ETO2) Processing Module

This module provides functions for extracting end-tidal O2 traces from continuous
O2 recordings. It detects breath-by-breath troughs (minimum O2 levels) and creates
a lower envelope representing minimum O2 levels across the respiratory cycle.

Functions:
    detect_troughs_diff: Derivative-based trough detection with curvature filtering
    detect_troughs_prominence: Prominence-based trough detection using scipy
    extract_eto2_envelope: Main processing function to create ETO2 trace
"""

from typing import Dict, Optional

import numpy as np
from scipy.signal import find_peaks, medfilt, peak_prominences, savgol_filter


def _nearest_odd(n: int) -> int:
    """Ensure window size is odd for Savitzky-Golay filter."""
    return n if n % 2 == 1 else n + 1


def detect_troughs_diff(
    signal: np.ndarray,
    sampling_rate: float,
    min_trough_distance_s: float = 3.0,
    min_prominence: float = 1.0,
    sg_window_s: float = 0.2,
    sg_poly: int = 2,
    prom_adapt: bool = False,
) -> np.ndarray:
    """
    Detect O2 troughs using derivative zero-crossings with curvature filtering.

    This method:
    1. Smooths signal with Savitzky-Golay filter for differentiability
    2. Finds zero-crossings where derivative goes from negative to non-negative
    3. Filters for positive curvature (concave up at troughs/minima)
    4. Refines to nearest local minimum in raw signal (±150ms window)
    5. Enforces minimum temporal separation via greedy amplitude ordering (lower first)
    6. Validates by prominence on inverted signal with optional adaptive thresholding

    Parameters
    ----------
    signal : np.ndarray
        Raw O2 signal in mmHg
    sampling_rate : float
        Sampling frequency in Hz
    min_trough_distance_s : float, optional
        Minimum time between troughs in seconds (default: 3.0)
    min_prominence : float, optional
        Minimum trough prominence in mmHg (on inverted signal) (default: 1.0)
    sg_window_s : float, optional
        Savitzky-Golay window duration in seconds (default: 0.2)
    sg_poly : int, optional
        Savitzky-Golay polynomial order (default: 2)
    prom_adapt : bool, optional
        Enable adaptive prominence threshold using 25th percentile (default: False)

    Returns
    -------
    troughs_idx : np.ndarray
        Array of trough indices in the signal
    """
    # Smooth with Savitzky-Golay
    win_pts = max(5, int(round(sg_window_s * sampling_rate)))
    win_pts = _nearest_odd(win_pts)

    try:
        x_s = savgol_filter(signal, window_length=win_pts, polyorder=max(1, sg_poly))
    except ValueError:
        x_s = signal.copy()

    # Compute first and second derivatives
    d1 = np.gradient(x_s)
    d2 = np.gradient(d1)

    # Zero-crossings from negative to non-negative (trough candidates)
    zc = np.where((d1[:-1] < 0) & (d1[1:] >= 0))[0] + 1
    if zc.size == 0:
        return zc

    # Enforce positive curvature (concave up at minima)
    cand = zc[d2[zc] > 0]
    if cand.size == 0:
        cand = zc

    # Refine to nearest local minimum in raw signal within ±150ms
    r = max(1, int(round(0.15 * sampling_rate)))
    refined = []
    n = len(signal)

    for idx in cand:
        lo = max(0, idx - r)
        hi = min(n, idx + r + 1)
        if hi - lo <= 0:
            continue
        local = signal[lo:hi]
        ridx = lo + int(np.argmin(local))  # Local minimum
        refined.append(ridx)

    if not refined:
        return np.array([], dtype=int)

    refined = np.array(refined, dtype=int)

    # Enforce minimum distance using greedy selection by ascending amplitude (lowest first)
    min_dist = int(round(min_trough_distance_s * sampling_rate))
    order = np.argsort(signal[refined])  # Ascending (lowest values first)
    selected = []
    taken = np.zeros(n, dtype=bool)

    for i in order:
        idx = refined[i]
        lo = max(0, idx - min_dist)
        hi = min(n, idx + min_dist + 1)
        if not taken[lo:hi].any():
            selected.append(idx)
            taken[lo:hi] = True

    troughs_idx = np.array(sorted(selected), dtype=int)

    if troughs_idx.size == 0:
        return troughs_idx

    # Validate by prominence on inverted signal
    try:
        prom, _, _ = peak_prominences(-signal, troughs_idx)
        thr = float(min_prominence)

        if prom_adapt and prom.size:
            # Use 25th percentile as adaptive floor
            thr = max(thr, float(np.quantile(prom, 0.25)))

        keep = prom >= thr
        troughs_idx = troughs_idx[keep]
    except Exception:
        pass

    return troughs_idx


def detect_troughs_prominence(
    signal: np.ndarray, sampling_rate: float, min_trough_distance_s: float = 3.0, min_prominence: float = 1.0
) -> np.ndarray:
    """
    Detect O2 troughs using scipy's prominence-based peak detection on inverted signal.

    Simpler alternative to derivative method, detects peaks in inverted signal
    (which correspond to troughs in original signal).

    Parameters
    ----------
    signal : np.ndarray
        Raw O2 signal in mmHg
    sampling_rate : float
        Sampling frequency in Hz
    min_trough_distance_s : float, optional
        Minimum time between troughs in seconds (default: 3.0)
    min_prominence : float, optional
        Minimum trough prominence in mmHg (default: 1.0)

    Returns
    -------
    troughs_idx : np.ndarray
        Array of trough indices in the signal
    """
    min_dist_samples = int(min_trough_distance_s * sampling_rate)
    troughs_idx, _ = find_peaks(
        -signal,  # Invert signal to find troughs
        prominence=min_prominence,
        distance=min_dist_samples,
    )
    return troughs_idx


def extract_eto2_envelope(signal: np.ndarray, sampling_rate: float, params: Optional[Dict] = None) -> Dict:
    """
    Extract end-tidal O2 envelope from continuous O2 recording.

    Detects breath-by-breath O2 troughs (minima), applies median smoothing, and
    interpolates to create a continuous lower envelope trace representing
    minimum O2 levels during hypoxic manipulations.

    Parameters
    ----------
    signal : np.ndarray
        Raw O2 signal in mmHg
    sampling_rate : float
        Sampling frequency in Hz
    params : dict, optional
        Processing parameters:
        - trough_method: 'diff' or 'prominence' (default: 'diff')
        - min_trough_distance_s: Minimum time between troughs in seconds (default: 3.0)
        - min_prominence: Minimum trough prominence in mmHg (default: 1.0)
        - sg_window_s: Savitzky-Golay window in seconds (default: 0.2)
        - sg_poly: Savitzky-Golay polynomial order (default: 2)
        - prom_adapt: Enable adaptive prominence threshold (default: False)
        - smooth_troughs: Median filter kernel size in troughs (default: 5)

    Returns
    -------
    result : dict
        Dictionary containing:
        - raw_signal: Original O2 signal
        - auto_troughs: Automatically detected trough indices
        - current_troughs: Current troughs (same as auto_troughs, can be edited)
        - eto2_envelope: Lower envelope trace (same length as signal)
        - trough_values: O2 values at trough locations
        - smoothed_trough_values: Median-filtered trough values
        - time_vector: Time axis in seconds
        - sampling_rate: Sampling frequency
        - params: Processing parameters used
    """
    if params is None:
        params = {}

    # Default parameters
    trough_method = params.get("trough_method", "diff")
    min_trough_distance_s = params.get("min_trough_distance_s", 3.0)
    min_prominence = params.get("min_prominence", 1.0)
    sg_window_s = params.get("sg_window_s", 0.2)
    sg_poly = params.get("sg_poly", 2)
    prom_adapt = params.get("prom_adapt", False)
    smooth_troughs = params.get("smooth_troughs", 5)

    # Ensure smooth_troughs is odd
    if smooth_troughs % 2 == 0:
        smooth_troughs += 1

    # Detect troughs
    if trough_method == "diff":
        troughs_idx = detect_troughs_diff(
            signal,
            sampling_rate,
            min_trough_distance_s=min_trough_distance_s,
            min_prominence=min_prominence,
            sg_window_s=sg_window_s,
            sg_poly=sg_poly,
            prom_adapt=prom_adapt,
        )

        # Fallback to prominence if derivative method fails
        if troughs_idx.size == 0:
            troughs_idx = detect_troughs_prominence(
                signal, sampling_rate, min_trough_distance_s=min_trough_distance_s, min_prominence=min_prominence
            )
    else:
        troughs_idx = detect_troughs_prominence(
            signal, sampling_rate, min_trough_distance_s=min_trough_distance_s, min_prominence=min_prominence
        )

    # Create time vector
    time_vector = np.arange(len(signal)) / sampling_rate

    # Extract trough values and apply median smoothing
    if troughs_idx.size >= smooth_troughs:
        trough_values = signal[troughs_idx]
        smoothed_trough_values = medfilt(trough_values, kernel_size=smooth_troughs)
    elif troughs_idx.size > 0:
        trough_values = signal[troughs_idx]
        smoothed_trough_values = trough_values.copy()
    else:
        trough_values = np.array([])
        smoothed_trough_values = np.array([])

    # Create envelope by interpolation
    if troughs_idx.size >= 2:
        envelope = np.interp(time_vector, time_vector[troughs_idx], smoothed_trough_values)
    elif troughs_idx.size == 1:
        # Single trough: flat line at trough value
        envelope = np.full_like(time_vector, smoothed_trough_values[0], dtype=float)
    else:
        # No troughs: fallback to smoothed signal
        try:
            win_pts = _nearest_odd(max(5, int(round(sg_window_s * sampling_rate))))
            envelope = savgol_filter(signal, window_length=win_pts, polyorder=max(1, sg_poly))
        except Exception:
            envelope = signal.astype(float)

    result = {
        "raw_signal": signal,
        "auto_troughs": troughs_idx.copy(),
        "current_troughs": troughs_idx.copy(),
        "eto2_envelope": envelope,
        "trough_values": trough_values,
        "smoothed_trough_values": smoothed_trough_values,
        "time_vector": time_vector,
        "sampling_rate": sampling_rate,
        "params": params.copy(),
    }
    # Backward/forward compatibility aliases for cross-metric consistency
    result["raw"] = result["raw_signal"]
    result["clean"] = result["eto2_envelope"]
    result["cleaned_signal"] = result["eto2_envelope"]
    return result


def process_eto2(signal: np.ndarray, sampling_rate: float, params: Optional[Dict] = None) -> Dict:
    """Wrapper to standardize per-metric entry-point naming."""
    return extract_eto2_envelope(signal=signal, sampling_rate=sampling_rate, params=params)
