#!/usr/bin/env python3
"""
main.py
Breathing Task - PMU Data Visualization
========================================
Simple script to read and plot raw respiratory and pulse data.

Usage:
    python main.py 2034
    python main.py 2045
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from datetime import datetime


# ============================================================================
# CONFIGURATION
# ============================================================================

BIDS_DIR = Path('/export02/projects/LCS/BIDS')
PHYSIO_DIR = Path('/export02/projects/LCS/01_physio')
OUTPUT_DIR = Path('/export02/users/oreng/ses-b/outputs/breathing-task')

BIDS_SESSION = 'ses-02'    # BIDS naming (with zero)
PHYSIO_SESSION = 'ses-2'   # Physio naming (without zero)


# ============================================================================
# FUNCTIONS
# ============================================================================

def read_scans_tsv(subject_id):
    """
    Read scans.tsv file and find breathing task start time.
    
    Returns:
        datetime object of breathing task start time
    """
    scans_file = BIDS_DIR / f'sub-{subject_id}' / BIDS_SESSION / f'sub-{subject_id}_{BIDS_SESSION}_scans.tsv'
    
    print(f"\n📂 Reading: {scans_file.name}")
    
    if not scans_file.exists():
        raise FileNotFoundError(f"Scans file not found: {scans_file}")
    
    # Read TSV
    df = pd.read_csv(scans_file, sep='\t')
    
    # Find breathing task
    breath_rows = df[df['filename'].str.contains('task-breath')]
    
    if len(breath_rows) == 0:
        raise ValueError("No breathing task found in scans.tsv")
    
    # Get first echo's time (all echoes have same time)
    breath_time_str = breath_rows.iloc[0]['acq_time']
    breath_time = datetime.fromisoformat(breath_time_str)
    
    # Get session start time (earliest scan)
    session_start_str = df['acq_time'].min()
    session_start = datetime.fromisoformat(session_start_str)
    
    print(f"\n⏱️  Session timeline:")
    print(f"   Session start: {session_start.strftime('%H:%M:%S')}")
    print(f"   Breathing task: {breath_time.strftime('%H:%M:%S')}")
    
    offset = (breath_time - session_start).total_seconds()
    print(f"   Offset: {offset:.0f} sec ({offset/60:.1f} min)")
    
    return breath_time, session_start


def read_pmu_file(filepath):
    """
    Read Siemens PMU file (.resp or .puls).
    
    Returns:
        numpy array of raw values
    """
    print(f"\n📂 Reading: {filepath.name}")
    
    if not filepath.exists():
        raise FileNotFoundError(f"PMU file not found: {filepath}")
    
    # Read file as binary
    with open(filepath, 'rb') as f:
        content = f.read()
    
    # Parse as ASCII (space-separated integers)
    text = content.decode('ascii', errors='ignore')
    tokens = text.split()
    
    # Extract physiological values (0-8192 range, exclude scan markers)
    values = []
    for token in tokens:
        try:
            val = int(token)
            # Keep physiological data, exclude markers (5002, 5003, 6002, 6003)
            if 0 <= val <= 8192 and val not in [5002, 5003, 6002, 6003]:
                values.append(val)
        except ValueError:
            continue
    
    data = np.array(values, dtype=np.float64)
    
    print(f"   Samples: {len(data)}")
    print(f"   Range: {data.min():.0f} - {data.max():.0f}")
    
    return data


def plot_raw_data(resp_data, puls_data, subject_id, output_dir):
    """
    Plot raw respiratory and pulse data.
    
    X-axis: Sample number
    """
    
    print(f"\n📊 Creating plot...")
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    fig.suptitle(f'sub-{subject_id} - Raw PMU Data', fontsize=16, fontweight='bold')
    
    # SUBPLOT 1: Respiratory
    axes[0].plot(resp_data, 'b-', linewidth=0.3, alpha=0.7)
    axes[0].set_ylabel('Respiratory Signal\n(arbitrary units)', fontsize=11)
    axes[0].set_title('Respiratory (RESP)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, len(resp_data)])
    
    # SUBPLOT 2: Pulse
    axes[1].plot(puls_data, 'r-', linewidth=0.3, alpha=0.7)
    axes[1].set_ylabel('Pulse Signal\n(arbitrary units)', fontsize=11)
    axes[1].set_xlabel('Sample Number', fontsize=12, fontweight='bold')
    axes[1].set_title('Pulse Oximetry (PULS)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, len(puls_data)])
    
    plt.tight_layout()
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'sub-{subject_id}_raw_pmu.png'
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    
    print(f"   ✓ Saved: {output_file}")
    
    plt.show()
    
    return fig


# ============================================================================
# MAIN
# ============================================================================

def main(subject_id):
    """
    Main processing pipeline.
    
    Args:
        subject_id: Subject number (e.g., '2034', '2045')
    """
    
    print("\n" + "="*70)
    print(f"BREATHING TASK - PMU DATA VISUALIZATION")
    print("="*70)
    print(f"\nSubject: sub-{subject_id}")
    
    # Step 1: Read scans.tsv to get breathing task time
    breath_time, session_start = read_scans_tsv(subject_id)
    
    # Step 2: Find PMU files
    physio_dir = PHYSIO_DIR / f'sub-{subject_id}' / PHYSIO_SESSION / 'Scanner_Physio'
    
    if not physio_dir.exists():
        raise FileNotFoundError(f"Physio directory not found: {physio_dir}")
    
    # Find .resp file
    resp_files = list(physio_dir.glob('*.resp'))
    if not resp_files:
        raise FileNotFoundError(f"No .resp file found in {physio_dir}")
    resp_file = resp_files[0]
    
    # Find .puls file
    puls_files = list(physio_dir.glob('*.puls'))
    if not puls_files:
        raise FileNotFoundError(f"No .puls file found in {physio_dir}")
    puls_file = puls_files[0]
    
    # Step 3: Read PMU data
    resp_data = read_pmu_file(resp_file)
    puls_data = read_pmu_file(puls_file)
    
    # Step 4: Plot
    output_dir = OUTPUT_DIR / f'sub-{subject_id}'
    plot_raw_data(resp_data, puls_data, subject_id, output_dir)
    
    print("\n" + "="*70)
    print("✓ COMPLETE")
    print("="*70)
    print(f"\nOutput: {output_dir}")
    print(f"="*70 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python main.py <subject_id>")
        print("\nExamples:")
        print("  python main.py 2034")
        print("  python main.py 2045")
        print()
        sys.exit(1)
    
    subject_id = sys.argv[1]
    
    try:
        main(subject_id)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
