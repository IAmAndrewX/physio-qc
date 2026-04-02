"""
Spirometry signal processing functions using BreathMetrics
"""

import traceback
import numpy as np
from scipy.interpolate import interp1d


def _add_volume_stats(bm_attr, key_prefix, result, bm, exclude_outliers, volume_outlier_sd=3.0):
    """
    Extract volume stats from a BreathMetrics attribute and store in result.

    BreathMetrics stores volume arrays as shape (1, n) — we index [0] to get
    the flat 1-D array of per-cycle values.
    """
    if not (hasattr(bm, bm_attr) and getattr(bm, bm_attr) is not None
            and len(getattr(bm, bm_attr)) > 0):
        return

    vols = getattr(bm, bm_attr)
    cycle_vols = vols[0]  # BreathMetrics stores volumes as (1, n); grab the inner array
    result[f'{key_prefix}_volumes'] = cycle_vols.flatten()

    valid_inds = np.where(~np.isnan(cycle_vols))[0]
    if len(valid_inds) == 0:
        return

    if exclude_outliers:
        mean_val = np.nanmean(cycle_vols[valid_inds])
        std_val = np.nanstd(cycle_vols[valid_inds])
        outlier_mask = (
            (cycle_vols < mean_val - volume_outlier_sd * std_val) |
            (cycle_vols > mean_val + volume_outlier_sd * std_val)
        )
        result[f'{key_prefix}_outliers'] = outlier_mask
        valid_non_outlier_inds = np.where(~np.isnan(cycle_vols) & ~outlier_mask)[0]
        result[f'mean_{key_prefix}_volume'] = float(
            np.mean(cycle_vols[valid_non_outlier_inds]) if len(valid_non_outlier_inds) > 0
            else np.mean(cycle_vols[valid_inds])
        )
    else:
        result[f'{key_prefix}_outliers'] = np.zeros(len(cycle_vols), dtype=bool)
        result[f'mean_{key_prefix}_volume'] = float(np.mean(cycle_vols[valid_inds]))


def process_breathmetrics(signal, sampling_rate, params):
    """
    Wrapper function to integrate breathMetricsClass with Streamlit.

    Parameters
    ----------
    signal : array
        Raw respiratory signal
    sampling_rate : int
        Sampling rate in Hz
    params : dict
        Processing parameters: verbose, zscore, baseline_method, simplify,
        exclude_outliers, exclude_duration_outliers

    Returns
    -------
    dict
        Results in format expected by Streamlit app
    """
    from metrics.breathmetricsClass import bmObject

    # Extract parameters
    data_type = params.get('data_type', 'humanAirflow')
    zscore = params.get('zscore', 0)
    baseline_method = params.get('baseline_method', 'sliding')
    simplify = params.get('simplify', 1)
    verbose = params.get('verbose', 0)
    exclude_outliers = params.get('exclude_outliers', 0)
    volume_outlier_sd = float(params.get('volume_outlier_sd', 2.0))
    exclude_duration_outliers = params.get('exclude_duration_outliers', 0)
    duration_outlier_sd = float(params.get('duration_outlier_sd', 2.0))

    try:
        if signal is None or len(signal) == 0:
            raise ValueError("Signal is empty or None")

        if not isinstance(sampling_rate, (int, float)) or sampling_rate < 20 or sampling_rate > 5000:
            raise ValueError(f"Invalid sampling rate: {sampling_rate}. Must be between 20-5000 Hz")

        signal = np.asarray(signal, dtype=float)
        if signal.ndim != 1:
            signal = signal.flatten()

        if np.any(np.isnan(signal)):
            print(f"WARNING: Signal contains {np.sum(np.isnan(signal))} NaN values, filling with 0")
            signal = np.nan_to_num(signal, nan=0.0)

        if np.any(np.isinf(signal)):
            print(f"WARNING: Signal contains {np.sum(np.isinf(signal))} infinite values, replacing with 0")
            signal = np.where(np.isinf(signal), 0.0, signal)

        valid_data_types = ['humanAirflow', 'rodentAirflow', 'humanBB', 'rodentThermocouple']
        if data_type not in valid_data_types:
            raise ValueError(f"Invalid data type '{data_type}'. Must be one of: {valid_data_types}")

        bm = bmObject(signal, sampling_rate, data_type)
        bm.estimateAllFeatures(
            zScore=zscore,
            baselineCorrectionMethod=baseline_method,
            simplify=simplify,
            verbose=verbose
        )

        # For airflow data, peaks are inhale peaks, troughs are exhale troughs.
        # For belt data, peaks are exhale onsets, troughs are inhale onsets.
        if data_type in ['humanAirflow', 'rodentAirflow']:
            current_peaks = bm.inhalePeaks
            current_troughs = bm.exhaleTroughs
            if hasattr(bm, 'peakInspiratoryFlows') and len(bm.peakInspiratoryFlows) > 0:
                peaks_values = bm.peakInspiratoryFlows
            elif len(bm.inhalePeaks) > 0:
                peaks_values = bm.smoothedRespiration[bm.inhalePeaks]
            else:
                peaks_values = np.array([])
            if hasattr(bm, 'troughExpiratoryFlows') and len(bm.troughExpiratoryFlows) > 0:
                troughs_values = bm.troughExpiratoryFlows
            elif len(bm.exhaleTroughs) > 0:
                troughs_values = bm.smoothedRespiration[bm.exhaleTroughs]
            else:
                troughs_values = np.array([])
        else:  # In case we want to try using breathmetrics on belt data, though this isn't supported in the app
            current_peaks = bm.exhaleOnsets if hasattr(bm, 'exhaleOnsets') else bm.inhaleOnsets
            current_troughs = bm.inhaleOnsets if hasattr(bm, 'inhaleOnsets') else bm.exhaleOnsets
            peaks_values = bm.smoothedRespiration[current_peaks] if len(current_peaks) > 0 else np.array([])
            troughs_values = bm.smoothedRespiration[current_troughs] if len(current_troughs) > 0 else np.array([])

        current_peaks = np.array(current_peaks) if len(current_peaks) > 0 else np.array([])
        current_troughs = np.array(current_troughs) if len(current_troughs) > 0 else np.array([])

        result = {
            'raw': signal,
            'clean': bm.smoothedRespiration,
            'baseline_corrected': bm.baselineCorrectedRespiration,
            'time': bm.time,
            'current_peaks': current_peaks, # Can be used for peak editing
            'current_troughs': current_troughs, # Can be used for peak editing
            'auto_peaks': current_peaks.copy(),
            'auto_troughs': current_troughs.copy(),
            'peaks_values': peaks_values,
            'troughs_values': troughs_values,
            'data_type': data_type,
            'params': params.copy(),
        }

        _add_volume_stats('inhaleVolumes', 'inhale', result, bm, exclude_outliers, volume_outlier_sd)
        _add_volume_stats('exhaleVolumes', 'exhale', result, bm, exclude_outliers, volume_outlier_sd)

        # Tidal volumes (inhale + exhale)
        # Calculated in BreathMetrics by adding abs(values) for all values within one breath, then dividing by the sampling rate 
        if 'inhale_volumes' in result and 'exhale_volumes' in result:
            inhale_vols = np.array(result['inhale_volumes']).flatten()
            exhale_vols = np.array(result['exhale_volumes']).flatten()
            min_len = min(len(inhale_vols), len(exhale_vols))
            tidal_volumes = inhale_vols[:min_len] + exhale_vols[:min_len]
            result['tidal_volumes'] = tidal_volumes

            if 'inhale_outliers' in result and 'exhale_outliers' in result:
                tidal_outliers = result['inhale_outliers'][:min_len] | result['exhale_outliers'][:min_len]
            else:
                tidal_outliers = np.zeros(min_len, dtype=bool)
            result['tidal_outliers'] = tidal_outliers

            # valid_tidal_inds already excludes NaNs and outlier cycles (via tidal_outliers,
            # which is the union of inhale/exhale outlier flags set by _add_volume_stats).
            # No second filtering pass needed — just mean the remaining cycles directly.
            valid_tidal_inds = np.where(~np.isnan(tidal_volumes) & ~tidal_outliers)[0]
            if len(valid_tidal_inds) > 0:
                result['mean_tidal_volume'] = float(np.mean(tidal_volumes[valid_tidal_inds]))

        if hasattr(bm, 'inhaleDurations') and bm.inhaleDurations is not None and len(bm.inhaleDurations) > 0:
            result['inhale_durations'] = np.array(bm.inhaleDurations).flatten()
            valid = bm.inhaleDurations[0][~np.isnan(bm.inhaleDurations[0])]
            if len(valid) > 0:
                result['mean_inhale_duration'] = float(np.mean(valid))
        if hasattr(bm, 'exhaleDurations') and bm.exhaleDurations is not None and len(bm.exhaleDurations) > 0:
            result['exhale_durations'] = np.array(bm.exhaleDurations).flatten()
            valid = bm.exhaleDurations[0][~np.isnan(bm.exhaleDurations[0])]
            if len(valid) > 0:
                result['mean_exhale_duration'] = float(np.mean(valid))

        # Breathing rate — prefer inhale onsets for flow signals, fall back to troughs
        if hasattr(bm, 'inhaleOnsets') and bm.inhaleOnsets is not None and len(bm.inhaleOnsets) > 1:
            rate_markers = np.array(bm.inhaleOnsets)
        elif len(current_troughs) > 1:
            rate_markers = current_troughs
            print("WARNING: Using troughs to estimate breathing rate.")
        else:
            rate_markers = None

        if rate_markers is not None:
            marker_times = rate_markers / sampling_rate
            intervals = np.diff(marker_times)
            if len(intervals) > 0:
                # Compute duration outlier mask first so it can gate the mean interval
                interval_mean = np.mean(intervals)
                interval_std = np.std(intervals)
                duration_outlier_mask = (
                    (intervals < interval_mean - duration_outlier_sd * interval_std) |
                    (intervals > interval_mean + duration_outlier_sd * interval_std)
                )
                result['duration_outliers'] = duration_outlier_mask

                valid_intervals = intervals[~duration_outlier_mask] if exclude_duration_outliers else intervals
                mean_interval = np.mean(valid_intervals) if len(valid_intervals) > 0 else np.mean(intervals)
                result['breathing_rate_bpm'] = 60.0 / mean_interval
                result['mean_breath_interval'] = mean_interval

                breath_times = marker_times[:-1] + intervals / 2
                rates = 60.0 / intervals
                result['breath_times'] = breath_times
                result['rate_values'] = rates
                result['rate_interpolated'] = interp1d(
                    breath_times, rates, kind='linear', bounds_error=False, fill_value=np.nan
                )(bm.time)

                if 'tidal_volumes' in result:
                    min_len = min(len(result['tidal_volumes']), len(rates))
                    minute_ventilation_values = result['tidal_volumes'][:min_len] * rates[:min_len]
                    result['minute_ventilation_values'] = minute_ventilation_values
                    result['minute_ventilation_interpolated'] = interp1d(
                        breath_times[:min_len], minute_ventilation_values,
                        kind='linear', bounds_error=False, fill_value=np.nan
                    )(bm.time)

                    valid_mv_mask = np.isfinite(minute_ventilation_values)
                    if exclude_outliers and 'tidal_outliers' in result:
                        tidal_out = result['tidal_outliers']
                        n = min(min_len, len(tidal_out))
                        valid_mv_mask[:n] &= ~tidal_out[:n]
                    if exclude_duration_outliers and 'duration_outliers' in result:
                        dur_out = result['duration_outliers']
                        n = min(min_len, len(dur_out))
                        valid_mv_mask[:n] &= ~dur_out[:n]
                    valid_mv_inds = np.where(valid_mv_mask)[0]
                    if len(valid_mv_inds) > 0:
                        result['mean_minute_ventilation'] = float(np.mean(minute_ventilation_values[valid_mv_inds]))

        return result

    except Exception as e:
        print(f"breathMetrics processing failed: {e}")
        traceback.print_exc()
        return None
