#!/usr/bin/env python3
"""
pmu_processor.py
PMU Processor - Extract Siemens physiological data aligned to fMRI scans

Handles:
- Parsing PMU files (.resp, .puls, .ext)
- Identifying scans using 5000 markers
- Matching any scan by name (breath, rest, gas, etc.)
- Reconstructing volume timing from BIDS TR
- Graceful handling of timing mismatches

Usage:
    python pmu_processor.py <subject_id> <scan_name>
    python pmu_processor.py 2008 breath
    python pmu_processor.py 2034 rest --show-full
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

BIDS_DIR = Path('/export02/projects/LCS/BIDS')
PHYSIO_DIR = Path('/export02/projects/LCS/01_physio')
OUTPUT_DIR = Path('/export02/users/oreng/ses-b/outputs/pmu')

BIDS_SESSION = 'ses-02'
PHYSIO_SESSION = 'ses-2'
SAMPLING_RATE = 400  # Hz


# ============================================================================
# FILE PARSING
# ============================================================================

def parse_pmu_file(filepath):
    """
    Parse Siemens PMU file (.puls, .resp, .ext).
    
    File format:
    - .ext file: Contains volume markers (5000) and slice triggers (1)
    - .puls/.resp: Physiological signal (0-4095) with cardiac pulses marked as 5000
    
    Returns: dict with 'values' (signal), 'volume_markers' (5000s), 'slice_triggers' (1s)
    """
    with open(filepath, 'rb') as f:
        content = f.read().decode('ascii', errors='ignore')
    
    tokens = [int(x) for x in content.split() if x.strip().lstrip('-').isdigit()]
    
    # Find data section between 6002 and 6003
    start_idx = tokens.index(6002) + 1 if 6002 in tokens else 5
    end_idx = tokens.index(6003) if 6003 in tokens else len(tokens)
    data_section = tokens[start_idx:end_idx]
    
    is_ext_file = filepath.suffix == '.ext'
    
    if is_ext_file:
        # .ext: binary time series where position = sample index
        # value 0 = no activity
        # value 1 = slice acquisition (TTL pulse)
        # value 5000 = volume acquisition marker
        # value 5003 = recording end
        signal = []
        slice_triggers = []    # TTL pulses (1s) - individual slice acquisitions
        volume_markers = []    # 5000 markers - volume acquisitions
        
        for i, value in enumerate(data_section):
            if value == 1:  # Slice TTL pulse
                slice_triggers.append(i)
                signal.append(0)
            elif value == 5000:  # Volume marker
                volume_markers.append(i)
                signal.append(0)
            elif value == 5003:  # Recording end
                signal.append(0)
            else:  # No activity (or metadata values)
                signal.append(0)
        
        return {
            'values': np.array(signal, dtype=float),
            'slice_triggers': np.array(slice_triggers, dtype=int),
            'volume_markers': np.array(volume_markers, dtype=int),
            'filepath': filepath
        }
    
    else:
        # .puls/.resp: physiological signal with embedded markers
        # 5000 in these files = cardiac pulse detection, not scan markers
        signal = []
        cardiac_pulses = []
        
        for value in data_section:
            if value == 5000:  # Cardiac pulse (in physio files)
                cardiac_pulses.append(len(signal))
            elif value == 5003:  # Recording end
                pass
            elif 0 <= value <= 4095:  # Valid physiological data
                signal.append(value)
        
        return {
            'values': np.array(signal, dtype=float),
            'slice_triggers': np.array([], dtype=int),
            'volume_markers': np.array([], dtype=int),  # No scan info in physio files
            'cardiac_pulses': np.array(cardiac_pulses, dtype=int),
            'filepath': filepath
        }


# ============================================================================
# SCAN MATCHING
# ============================================================================

def identify_scans_from_volume_markers(volume_markers, sampling_rate=400, min_gap_seconds=10):
    """
    Identify individual scans by detecting gaps between volume acquisitions.
    
    TAPAS PhysIO approach:
    - 5000 markers indicate VOLUME acquisitions (not scan starts)
    - Scans are separated by gaps (no volumes for several seconds)
    - Large gap (>10s) = end of one scan, start of next
    
    Args:
        volume_markers: Array of sample indices where 5000 markers occur
        sampling_rate: PMU sampling rate (Hz)
        min_gap_seconds: Minimum gap to consider as scan boundary
    
    Returns:
        List of scan dicts with 'start_sample', 'end_sample', 'n_volumes', 'volume_indices'
    """
    if len(volume_markers) == 0:
        return []
    
    # Convert to time
    volume_times = volume_markers / sampling_rate
    
    # Calculate inter-volume intervals
    intervals = np.diff(volume_times)
    
    # Find gaps (scan boundaries)
    gap_threshold = min_gap_seconds
    gap_indices = np.where(intervals > gap_threshold)[0]
    
    # Split volumes into scans
    scan_boundaries = np.concatenate([[0], gap_indices + 1, [len(volume_markers)]])
    
    scans = []
    for i in range(len(scan_boundaries) - 1):
        start_vol_idx = scan_boundaries[i]
        end_vol_idx = scan_boundaries[i + 1] - 1
        
        # Get sample positions of first and last volume
        start_sample = volume_markers[start_vol_idx]
        end_sample = volume_markers[end_vol_idx]
        
        n_volumes = end_vol_idx - start_vol_idx + 1
        
        # Calculate median TR for this scan
        if n_volumes > 1:
            scan_intervals = intervals[start_vol_idx:end_vol_idx]
            median_tr = np.median(scan_intervals)
        else:
            median_tr = 0
        
        scans.append({
            'scan_index': i,
            'start_sample': start_sample,
            'end_sample': end_sample,
            'start_time': start_sample / sampling_rate,
            'end_time': end_sample / sampling_rate,
            'duration': (end_sample - start_sample) / sampling_rate,
            'n_volumes': n_volumes,
            'volume_indices': list(range(start_vol_idx, end_vol_idx + 1)),
            'median_tr': median_tr
        })
    
    return scans


def match_scan_by_properties(scans, target_scan_info, tolerance_seconds=30):
    """
    Match a BIDS scan to a PMU scan by comparing properties.
    
    Matching strategy:
    1. Try exact time match (if BIDS time is within PMU recording)
    2. Fall back to matching by duration and n_volumes
    3. Fall back to scan order
    
    Args:
        scans: List of scan dicts from identify_scans_from_volume_markers
        target_scan_info: Dict with 'offset_seconds', 'duration', 'n_volumes'
        tolerance_seconds: How close times need to be
    
    Returns:
        Matched scan dict, or None
    """
    if len(scans) == 0:
        return None
    
    target_time = target_scan_info['offset_seconds']
    target_duration = target_scan_info['duration']
    target_nvols = target_scan_info['n_volumes']
    
    # Strategy 1: Try time-based matching first
    for scan in scans:
        time_diff = abs(scan['start_time'] - target_time)
        if time_diff < tolerance_seconds:
            return scan
    
    # Strategy 2: Match by duration and number of volumes
    # Allow 10% tolerance on duration, exact match on volumes
    candidates = []
    for scan in scans:
        duration_ratio = scan['duration'] / target_duration
        volume_match = abs(scan['n_volumes'] - target_nvols) / target_nvols if target_nvols > 0 else 1
        
        if 0.9 <= duration_ratio <= 1.1 and volume_match < 0.1:
            candidates.append((scan, abs(duration_ratio - 1.0)))
    
    if candidates:
        # Return best match (closest duration)
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]
    
    # Strategy 3: If nothing matches, return None
    return None


# ============================================================================
# SCAN MATCHING
# ============================================================================

def get_bids_scan_info(subject_id, scan_name):
    """
    Get timing and TR information for a specific scan from BIDS.
    
    Args:
        subject_id: Subject ID (e.g., '2008')
        scan_name: Scan name (e.g., 'breath', 'rest', 'gas')
    
    Returns:
        dict with 'offset_seconds', 'tr', 'n_volumes', 'duration', 'filename'
        or None if scan not found
    """
    scans_file = BIDS_DIR / f'sub-{subject_id}' / BIDS_SESSION / f'sub-{subject_id}_{BIDS_SESSION}_scans.tsv'
    
    if not scans_file.exists():
        print(f"❌ BIDS scans file not found: {scans_file}")
        return None
    
    # Read scans.tsv
    df = pd.read_csv(scans_file, sep='\t')
    
    # Find the scan
    scan_rows = df[df['filename'].str.contains(f'task-{scan_name}', case=False, na=False)]
    if len(scan_rows) == 0:
        print(f"❌ Scan 'task-{scan_name}' not found in BIDS")
        print(f"\n   Available scans:")
        for filename in df['filename']:
            if 'task-' in filename:
                task = filename.split('task-')[1].split('_')[0]
                print(f"      - {task}")
        return None
    
    scan_row = scan_rows.iloc[0]
    
    # Calculate offset from session start
    df['timestamp'] = pd.to_datetime(df['acq_time'])
    session_start = df['timestamp'].min()
    scan_timestamp = pd.to_datetime(scan_row['acq_time'])
    offset_seconds = (scan_timestamp - session_start).total_seconds()
    
    # Get TR and n_volumes from NIfTI header
    nifti_path = BIDS_DIR / f'sub-{subject_id}' / BIDS_SESSION / scan_row['filename']
    
    try:
        import nibabel as nib
        img = nib.load(nifti_path)
        tr = float(img.header.get_zooms()[3])  # 4th dimension is time
        n_volumes = int(img.shape[3])
        duration = tr * n_volumes
    except Exception as e:
        print(f"⚠️  Could not read NIfTI header: {e}")
        print(f"   Using defaults: TR=2.0s, 50 volumes")
        tr, n_volumes, duration = 2.0, 50, 100.0
    
    return {
        'offset_seconds': offset_seconds,
        'tr': tr,
        'n_volumes': n_volumes,
        'duration': duration,
        'filename': scan_row['filename']
    }


def find_matching_scan_marker(scan_markers, target_time, pmu_duration, tolerance_seconds=30):
    """
    Find which 5000 marker corresponds to the target scan time.
    
    Args:
        scan_markers: Array of 5000 marker positions (sample indices)
        target_time: Expected scan start time (seconds from session start)
        pmu_duration: Total PMU recording duration (seconds)
        tolerance_seconds: How far from target to search (default 30s)
    
    Returns:
        Index of closest marker, or None if no match within tolerance
    """
    if len(scan_markers) == 0:
        return None
    
    # Check if target is beyond PMU recording
    if target_time > pmu_duration:
        print(f"⚠️  WARNING: Target scan at {target_time:.1f}s but PMU ends at {pmu_duration:.1f}s")
        print(f"   Gap: {target_time - pmu_duration:.1f}s ({(target_time - pmu_duration)/60:.1f} min)")
        print(f"\n   This suggests:")
        print(f"      - Wrong PMU files (different session?)")
        print(f"      - Incorrect BIDS timestamps")
        print(f"      - PMU recording stopped early")
        return None
    
    # Convert markers to time
    marker_times = scan_markers / SAMPLING_RATE
    
    # Find closest marker
    distances = np.abs(marker_times - target_time)
    closest_idx = np.argmin(distances)
    closest_distance = distances[closest_idx]
    
    if closest_distance > tolerance_seconds:
        print(f"⚠️  WARNING: Closest marker is {closest_distance:.1f}s away (tolerance: {tolerance_seconds}s)")
        print(f"   Target: {target_time:.1f}s")
        print(f"   Closest marker: {marker_times[closest_idx]:.1f}s")
        print(f"\n   Proceeding anyway, but results may be incorrect")
    
    return closest_idx


def reconstruct_volume_timing(scan_start_sample, tr, n_volumes):
    """
    Reconstruct when each volume was acquired.
    
    Args:
        scan_start_sample: Sample index where scan started (from 5000 marker)
        tr: Repetition time (seconds)
        n_volumes: Number of volumes in scan
    
    Returns:
        Array of sample indices for each volume
    """
    tr_samples = tr * SAMPLING_RATE
    volume_samples = scan_start_sample + np.arange(n_volumes) * tr_samples
    return volume_samples.astype(int)


# ============================================================================
# DATA EXTRACTION
# ============================================================================

def extract_scan_window(resp_data, puls_data, scan_window):
    """
    Extract physiological data for a specific scan window.
    
    Args:
        resp_data: Respiratory data dict
        puls_data: Pulse data dict
        scan_window: Dict with 'start_sample', 'end_sample'
    
    Returns:
        DataFrame with time, respiratory, and pulse columns
    """
    start = scan_window['start_sample']
    end = scan_window['end_sample']
    
    # Extract segments
    resp_segment = resp_data['values'][start:end]
    puls_segment = puls_data['values'][start:end]
    
    # Create time axis relative to scan start
    time_segment = np.arange(len(resp_segment)) / SAMPLING_RATE
    
    df = pd.DataFrame({
        'time_sec': time_segment,
        'respiratory': resp_segment,
        'pulse': puls_segment
    })
    
    return df


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_extracted_scan(df, scan_info, subject_id, scan_name, output_dir, show_full_session=False,
                       resp_full=None, puls_full=None, scan_window=None):
    """
    Plot extracted physiological data for a scan.
    
    Args:
        df: DataFrame with time, respiratory, pulse
        scan_info: Dict with scan metadata
        subject_id: Subject ID
        scan_name: Scan name (e.g., 'breath')
        output_dir: Where to save plots
        show_full_session: If True, also plot full session
        resp_full, puls_full: Full data arrays (needed if show_full_session=True)
        scan_window: Dict with start/end times (needed if show_full_session=True)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Full session plot (if requested)
    if show_full_session and resp_full is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))
        fig.suptitle(f'sub-{subject_id} - Full Session', fontsize=16, fontweight='bold')
        
        t_full = np.arange(len(resp_full)) / SAMPLING_RATE
        
        ax1.plot(t_full, resp_full, 'b-', linewidth=0.5, alpha=0.6)
        if scan_window:
            ax1.axvspan(scan_window['start_time'], scan_window['end_time'], 
                       alpha=0.3, color='yellow', label=f'{scan_name} scan')
        ax1.set_ylabel('Respiratory Signal', fontsize=11)
        ax1.set_title('Respiratory (RESP)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.plot(t_full, puls_full, 'r-', linewidth=0.5, alpha=0.6)
        if scan_window:
            ax2.axvspan(scan_window['start_time'], scan_window['end_time'],
                       alpha=0.3, color='yellow', label=f'{scan_name} scan')
        ax2.set_ylabel('Pulse Signal', fontsize=11)
        ax2.set_xlabel('Time (seconds)', fontsize=12)
        ax2.set_title('Pulse Oximetry (PULS)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        full_file = output_dir / f'sub-{subject_id}_{scan_name}_full_session.png'
        fig.savefig(full_file, dpi=150, bbox_inches='tight')
        print(f"   ✓ Full session plot: {full_file.name}")
        plt.close()
    
    # Scan-specific plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))
    fig.suptitle(f'sub-{subject_id} - {scan_name} scan ({scan_info["n_volumes"]} volumes, {scan_info["duration"]:.1f}s)', 
                 fontsize=16, fontweight='bold')
    
    ax1.plot(df['time_sec'], df['respiratory'], 'b-', linewidth=0.8)
    ax1.set_ylabel('Respiratory Signal', fontsize=11)
    ax1.set_title('Respiratory (RESP)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(df['time_sec'], df['pulse'], 'r-', linewidth=0.8)
    ax2.set_ylabel('Pulse Signal', fontsize=11)
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_title('Pulse Oximetry (PULS)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # if breath_task, create a folder named breath-task
    scan_file = output_dir / f'sub-{subject_id}_{scan_name}.png'
    fig.savefig(scan_file, dpi=300, bbox_inches='tight')
    print(f"   ✓ Scan plot: {scan_file.name}")
    plt.show()
    
    return scan_file

# ============================================================================
# UTILITY HELPERS
# ============================================================================

def find_physio_dir(subject_id, physio_session='ses-2'):
    """
    Return the correct Scanner_physio folder path, case-insensitive.
    Checks both 'Scanner_physio' and 'Scanner_Physio'.
    """
    base_dir = PHYSIO_DIR / f'sub-{subject_id}' / physio_session
    for variant in ['Scanner_physio', 'Scanner_Physio']:
        path = base_dir / variant
        if path.exists():
            return path
    raise FileNotFoundError(f"No Scanner_physio folder found for sub-{subject_id} in {base_dir}")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def process_subject(subject_id, scan_name='breath', show_full=False):
    """
    Extract physiological data for a specific scan.
    
    Pipeline:
    1. Parse PMU files
    2. Identify scans from volume markers using gap detection
    3. Get scan info from BIDS (timing, TR, n_volumes)
    4. Match BIDS scan to PMU scan by properties
    5. Extract and save data
    """
    print(f"\n{'='*80}")
    print(f"Processing sub-{subject_id} - {scan_name} scan")
    print(f"{'='*80}\n")
    
    # Find PMU files
    physio_dir = find_physio_dir(subject_id, PHYSIO_SESSION)

    try:
        resp_file = list(physio_dir.glob('*.resp'))[0]
        puls_file = list(physio_dir.glob('*.puls'))[0]
        ext_file = list(physio_dir.glob('*.ext'))[0]
    except IndexError:
        print(f"❌ PMU files not found in: {physio_dir}")
        sys.exit(1)
    
    print(f"📂 Files:")
    print(f"   RESP: {resp_file.name}")
    print(f"   PULS: {puls_file.name}")
    print(f"   EXT:  {ext_file.name}\n")
    
    # Parse files
    print(f"📊 Parsing PMU files...")
    resp_data = parse_pmu_file(resp_file)
    puls_data = parse_pmu_file(puls_file)
    ext_data = parse_pmu_file(ext_file)
    
    pmu_duration = len(resp_data['values']) / SAMPLING_RATE
    
    print(f"   RESP: {len(resp_data['values']):,} samples ({pmu_duration:.1f}s / {pmu_duration/60:.1f} min)")
    print(f"   PULS: {len(puls_data['values']):,} samples")
    print(f"   EXT:  {len(ext_data['volume_markers'])} volume markers (5000)")
    print(f"   EXT:  {len(ext_data['slice_triggers'])} slice triggers (1)\n")
    
    # Identify scans from volume markers
    print(f"🔍 Identifying scans from volume markers...")
    scans = identify_scans_from_volume_markers(ext_data['volume_markers'], SAMPLING_RATE)
    
    print(f"   Found {len(scans)} scans:\n")
    for scan in scans:
        print(f"   Scan {scan['scan_index'] + 1}:")
        print(f"      Time: {scan['start_time']:.1f}s - {scan['end_time']:.1f}s ({scan['duration']:.1f}s)")
        print(f"      Volumes: {scan['n_volumes']}")
        print(f"      Median TR: {scan['median_tr']:.3f}s")
    print()
    
    # Get scan info from BIDS
    print(f"🗂️  Reading BIDS metadata for '{scan_name}' scan...")
    scan_info = get_bids_scan_info(subject_id, scan_name)
    
    if scan_info is None:
        sys.exit(1)
    
    print(f"   Scan starts at: {scan_info['offset_seconds']:.1f}s from session start")
    print(f"   TR: {scan_info['tr']:.3f}s")
    print(f"   Volumes: {scan_info['n_volumes']}")
    print(f"   Duration: {scan_info['duration']:.1f}s\n")
    
    # Check if BIDS time is beyond PMU recording
    if scan_info['offset_seconds'] > pmu_duration:
        print(f"⚠️  WARNING: BIDS timestamp ({scan_info['offset_seconds']:.1f}s) is beyond PMU recording ({pmu_duration:.1f}s)")
        print(f"   Gap: {scan_info['offset_seconds'] - pmu_duration:.1f}s ({(scan_info['offset_seconds'] - pmu_duration)/60:.1f} min)")
        print(f"   BIDS timestamp may be incorrect - will try matching by scan properties instead.\n")
        # Don't exit! Continue to property-based matching below
    else:
        print(f"🎯 BIDS scan time is within PMU recording window.\n")

    # Match BIDS scan to PMU scan (works even if timestamp is wrong)
    print(f"🎯 Matching BIDS scan to PMU scan by properties...")
    print(f"   Looking for: {scan_info['n_volumes']} volumes, {scan_info['duration']:.1f}s duration")
    matched_scan = match_scan_by_properties(scans, scan_info)

    if matched_scan is None:
        print(f"\n❌ Could not find matching scan in PMU data")
        print(f"\n💡 Available scans in PMU:")
        for scan in scans:
            print(f"   Scan {scan['scan_index'] + 1}: {scan['start_time']:.1f}s, {scan['n_volumes']} volumes, {scan['duration']:.1f}s")
        print(f"\n   Try:")
        print(f"   1. Verify scan properties match one of the above")
        print(f"   2. Check if BIDS metadata is correct")
        print(f"   3. Run: python visualize_pmu_raw.py {subject_id}")
        sys.exit(1)

    # Show match details
    print(f"\n   ✓ Matched to PMU Scan #{matched_scan['scan_index'] + 1}")
    print(f"   PMU time:    {matched_scan['start_time']:.1f}s")
    print(f"   BIDS time:   {scan_info['offset_seconds']:.1f}s (may be inaccurate)")
    if abs(matched_scan['start_time'] - scan_info['offset_seconds']) > 30:
        print(f"   ⚠️  Large time discrepancy: {abs(matched_scan['start_time'] - scan_info['offset_seconds']):.1f}s")
        print(f"      Matched by scan properties instead of timestamp")
    
    if matched_scan is None:
        print(f"\n❌ Could not find matching scan in PMU data")
        print(f"\n💡 Try:")
        print(f"   1. Check if scan duration/volumes match any PMU scan")
        print(f"   2. Run: python visualize_pmu_raw.py {subject_id}")
        print(f"   3. Verify BIDS metadata is correct")
        sys.exit(1)
    
    print(f"   ✓ Matched to Scan #{matched_scan['scan_index'] + 1}")
    print(f"   PMU time:    {matched_scan['start_time']:.1f}s")
    print(f"   BIDS time:   {scan_info['offset_seconds']:.1f}s")
    print(f"   Difference:  {abs(matched_scan['start_time'] - scan_info['offset_seconds']):.1f}s")
    print(f"   Volumes:     {matched_scan['n_volumes']} (BIDS: {scan_info['n_volumes']})")
    print(f"   Duration:    {matched_scan['duration']:.1f}s (BIDS: {scan_info['duration']:.1f}s)\n")
    
    # Use matched scan boundaries for extraction
    scan_window = {
        'start_sample': int(matched_scan['start_sample']),
        'end_sample': int(matched_scan['end_sample']),
        'start_time': matched_scan['start_time'],
        'end_time': matched_scan['end_time']
    }
    
    print(f"📐 Scan window:")
    print(f"   Start: {scan_window['start_time']:.1f}s (sample {scan_window['start_sample']:,})")
    print(f"   End:   {scan_window['end_time']:.1f}s (sample {scan_window['end_sample']:,})")
    print(f"   Duration: {scan_window['end_time'] - scan_window['start_time']:.1f}s\n")
    
    # Extract data
    print(f"✂️  Extracting scan window...")
    df = extract_scan_window(resp_data, puls_data, scan_window)
    print(f"   ✓ Extracted {len(df)} samples\n")
    
    # Save outputs
    print(f"💾 Saving outputs...")
    output_dir = OUTPUT_DIR / f'sub-{subject_id}'
    
    # Save Parquet
    parquet_file = output_dir / f'sub-{subject_id}_{scan_name}_data.parquet'
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_file, index=False, engine='pyarrow', compression='snappy')
    print(f"   ✓ Data: {parquet_file.name}")
    
    # Create plots
    plot_extracted_scan(
        df, scan_info, subject_id, scan_name, output_dir,
        show_full_session=show_full,
        resp_full=resp_data['values'] if show_full else None,
        puls_full=puls_data['values'] if show_full else None,
        scan_window=scan_window if show_full else None
    )
    
    print(f"\n{'='*80}")
    print(f"✓ COMPLETE")
    print(f"{'='*80}\n")
    print(f"Outputs saved to: {output_dir}")
    print(f"  - {parquet_file.name}")
    print(f"  - sub-{subject_id}_{scan_name}_scan.png")
    if show_full:
        print(f"  - sub-{subject_id}_{scan_name}_full_session.png")
    print()


# ============================================================================
# CLI
# ============================================================================
def get_all_bids_scans(subject_id):
    """Get all available scan names from BIDS for this subject."""
    scans_file = BIDS_DIR / f'sub-{subject_id}' / BIDS_SESSION / f'sub-{subject_id}_{BIDS_SESSION}_scans.tsv'
    
    if not scans_file.exists():
        return []
    
    df = pd.read_csv(scans_file, sep='\t')
    
    # Extract all task names
    scan_names = []
    for filename in df['filename']:
        if 'task-' in filename:
            task = filename.split('task-')[1].split('_')[0]
            if task not in scan_names:
                scan_names.append(task)
    
    return scan_names


def process_all_scans(subject_id, show_full=False):
    """Process all scans found in PMU recording and match them to BIDS."""
    
    print(f"\n{'='*80}")
    print(f"Processing ALL scans for sub-{subject_id}")
    print(f"{'='*80}\n")
    
    # Find PMU files
    physio_dir = find_physio_dir(subject_id, PHYSIO_SESSION)
    
    try:
        resp_file = list(physio_dir.glob('*.resp'))[0]
        puls_file = list(physio_dir.glob('*.puls'))[0]
        ext_file = list(physio_dir.glob('*.ext'))[0]
    except IndexError:
        print(f"❌ PMU files not found in: {physio_dir}")
        sys.exit(1)
    
    print(f"📂 Files:")
    print(f"   RESP: {resp_file.name}")
    print(f"   PULS: {puls_file.name}")
    print(f"   EXT:  {ext_file.name}\n")
    
    # Parse files
    print(f"📊 Parsing PMU files...")
    resp_data = parse_pmu_file(resp_file)
    puls_data = parse_pmu_file(puls_file)
    ext_data = parse_pmu_file(ext_file)
    
    pmu_duration = len(resp_data['values']) / SAMPLING_RATE
    
    print(f"   RESP: {len(resp_data['values']):,} samples ({pmu_duration:.1f}s / {pmu_duration/60:.1f} min)")
    print(f"   PULS: {len(puls_data['values']):,} samples")
    print(f"   EXT:  {len(ext_data['volume_markers'])} volume markers (5000)")
    print(f"   EXT:  {len(ext_data['slice_triggers'])} slice triggers (1)\n")
    
    # Identify scans from volume markers
    print(f"🔍 Identifying scans from volume markers...")
    scans = identify_scans_from_volume_markers(ext_data['volume_markers'], SAMPLING_RATE)
    
    print(f"   Found {len(scans)} scans in PMU recording:\n")
    for scan in scans:
        print(f"   Scan {scan['scan_index'] + 1}:")
        print(f"      Time: {scan['start_time']:.1f}s - {scan['end_time']:.1f}s ({scan['duration']:.1f}s)")
        print(f"      Volumes: {scan['n_volumes']}")
        print(f"      Median TR: {scan['median_tr']:.3f}s")
    print()
    
    # Get all available BIDS scans
    print(f"🗂️  Reading BIDS metadata...")
    bids_scan_names = get_all_bids_scans(subject_id)
    print(f"   Found {len(bids_scan_names)} task scans in BIDS: {', '.join(bids_scan_names)}\n")
    
    # Try to match each PMU scan to a BIDS scan
    print(f"🎯 Matching PMU scans to BIDS scans...\n")
    
    matched_scans = []
    output_dir = OUTPUT_DIR / f'sub-{subject_id}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for pmu_scan in scans:
        print(f"{'─'*80}")
        print(f"PMU Scan #{pmu_scan['scan_index'] + 1}: {pmu_scan['n_volumes']} volumes, {pmu_scan['duration']:.1f}s")
        print(f"{'─'*80}")
        
        # Try to match to each BIDS scan
        best_match = None
        best_match_score = float('inf')
        
        for scan_name in bids_scan_names:
            scan_info = get_bids_scan_info(subject_id, scan_name)
            if scan_info is None:
                continue
            
            # Calculate match score (lower is better)
            vol_diff = abs(pmu_scan['n_volumes'] - scan_info['n_volumes'])
            dur_diff = abs(pmu_scan['duration'] - scan_info['duration'])
            match_score = vol_diff + dur_diff
            
            # Perfect match: same volumes and duration within 1%
            if vol_diff == 0 and dur_diff < scan_info['duration'] * 0.01:
                best_match = (scan_name, scan_info)
                best_match_score = match_score
                break
            elif match_score < best_match_score:
                best_match = (scan_name, scan_info)
                best_match_score = match_score
        
        if best_match and best_match_score < 10:  # Reasonable match threshold
            scan_name, scan_info = best_match
            print(f"✓ Matched to BIDS scan: {scan_name}")
            print(f"   Volumes: {pmu_scan['n_volumes']} (BIDS: {scan_info['n_volumes']})")
            print(f"   Duration: {pmu_scan['duration']:.1f}s (BIDS: {scan_info['duration']:.1f}s)\n")
            
            # Extract this scan
            scan_window = {
                'start_sample': int(pmu_scan['start_sample']),
                'end_sample': int(pmu_scan['end_sample']),
                'start_time': pmu_scan['start_time'],
                'end_time': pmu_scan['end_time']
            }
            
            print(f"✂️  Extracting scan window...")
            df = extract_scan_window(resp_data, puls_data, scan_window)
            print(f"   ✓ Extracted {len(df)} samples\n")
            
            # Save outputs
            print(f"💾 Saving outputs...")
            parquet_file = output_dir / f'sub-{subject_id}_{scan_name}_data.parquet'
            df.to_parquet(parquet_file, index=False, engine='pyarrow', compression='snappy')
            print(f"   ✓ Data: {parquet_file.name}")
            
            # Create plot
            plot_extracted_scan(
                df, scan_info, subject_id, scan_name, output_dir,
                show_full_session=False
            )
            
            matched_scans.append((pmu_scan['scan_index'] + 1, scan_name))
        else:
            print(f"⚠️  No matching BIDS scan found")
            print(f"   This scan may not have corresponding BIDS data\n")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"✓ PROCESSING COMPLETE")
    print(f"{'='*80}\n")
    print(f"Successfully processed {len(matched_scans)} scans:")
    for scan_num, scan_name in matched_scans:
        print(f"   - PMU Scan #{scan_num} → {scan_name}")
    
    print(f"\nOutputs saved to: {output_dir}")
    for scan_num, scan_name in matched_scans:
        print(f"   - sub-{subject_id}_{scan_name}_data.parquet")
        print(f"   - sub-{subject_id}_{scan_name}_scan.png")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Extract Siemens PMU physiological data for any fMRI scan',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pmu_processor.py 2008              # Process ALL scans
  python pmu_processor.py 2008 breath       # Process specific scan
  python pmu_processor.py 2034 rest --show-full
  
If no scan name is provided, all scans in the PMU recording will be processed.
        """
    )
    
    parser.add_argument('subject_id', help='Subject ID (e.g., 2008, 2034)')
    parser.add_argument('scan_name', nargs='?', default=None,
                       help='Scan name to extract (if omitted, processes all scans)')
    parser.add_argument('--show-full', action='store_true',
                       help='Also plot full session with scan highlighted')
    
    args = parser.parse_args()
    
    try:
        if args.scan_name is None:
            # Process all scans
            process_all_scans(args.subject_id, args.show_full)
        else:
            # Process specific scan
            process_subject(args.subject_id, args.scan_name, args.show_full)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()