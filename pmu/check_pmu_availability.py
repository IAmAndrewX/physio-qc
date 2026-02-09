#!/usr/bin/env python3
"""check_pmu_availability.py - See which subjects have PMU files"""

from pathlib import Path
import pandas as pd
import numpy as np

PHYSIO_DIR = Path('/export02/projects/LCS/01_physio')
PHYSIO_SESSION = 'ses-2'
SAMPLING_RATE = 400  # Hz

def get_pmu_duration(filepath, verbose=False):
    """Get duration of PMU recording in seconds."""
    try:
        with open(filepath, 'rb') as f:
            content = f.read().decode('ascii', errors='ignore')
        
        # Split into tokens
        tokens = [int(x) for x in content.split() if x.strip().lstrip('-').isdigit()]
        
        if verbose:
            print(f"      Total tokens: {len(tokens)}")
            print(f"      First 20 tokens: {tokens[:20]}")
        
        # Find data section between 6002 and 6003
        if 6002 not in tokens:
            if verbose:
                print(f"      ⚠️  No 6002 marker found, using default start")
            start_idx = 5
        else:
            start_idx = tokens.index(6002) + 1
            
        if 6003 not in tokens:
            if verbose:
                print(f"      ⚠️  No 6003 marker found, using end of file")
            end_idx = len(tokens)
        else:
            end_idx = tokens.index(6003)
        
        data_section = tokens[start_idx:end_idx]
        
        if verbose:
            print(f"      Data section: indices {start_idx} to {end_idx} ({len(data_section)} tokens)")
        
        # For .ext files, count all non-metadata values
        # For .resp/.puls files, count values 0-4095
        is_ext = filepath.suffix == '.ext'
        
        if is_ext:
            # .ext: count everything except 5003
            n_samples = len([x for x in data_section if x != 5003])
        else:
            # .resp/.puls: count physiological values (0-4095) and skip 5000 markers
            n_samples = len([x for x in data_section if 0 <= x <= 4095])
        
        if n_samples == 0:
            if verbose:
                print(f"      ⚠️  No valid samples found!")
            return None, None
        
        duration_sec = n_samples / SAMPLING_RATE
        
        if verbose:
            print(f"      Valid samples: {n_samples:,}")
            print(f"      Duration: {duration_sec:.1f}s ({duration_sec/60:.1f} min)")
        
        return duration_sec, n_samples
        
    except Exception as e:
        if verbose:
            print(f"      ❌ ERROR: {type(e).__name__}: {e}")
        return None, None

def get_n_volume_markers(ext_file, verbose=False):
    """Count volume markers (5000) in .ext file."""
    try:
        with open(ext_file, 'rb') as f:
            content = f.read().decode('ascii', errors='ignore')
        
        tokens = [int(x) for x in content.split() if x.strip().lstrip('-').isdigit()]
        
        # Find data section
        start_idx = tokens.index(6002) + 1 if 6002 in tokens else 5
        end_idx = tokens.index(6003) if 6003 in tokens else len(tokens)
        data_section = tokens[start_idx:end_idx]
        
        # Count 5000 markers
        n_volumes = sum(1 for x in data_section if x == 5000)
        
        if verbose:
            print(f"      Volume markers (5000): {n_volumes}")
        
        return n_volumes
    except Exception as e:
        if verbose:
            print(f"      ❌ ERROR counting volumes: {e}")
        return None

# Get all subject directories
subject_dirs = sorted(PHYSIO_DIR.glob('sub-*'))

print(f"\n{'='*80}")
print(f"PMU FILE AVAILABILITY CHECK")
print(f"{'='*80}\n")

available = []
missing = []
durations = []
failed_parsing = []

for subject_dir in subject_dirs:
    subject_id = subject_dir.name.replace('sub-', '')
    for variant in ['Scanner_physio', 'Scanner_Physio']:
        physio_path = subject_dir / PHYSIO_SESSION / variant
        if physio_path.exists():
            break
    
    # Check for PMU files
    resp_files = list(physio_path.glob('*.resp'))
    puls_files = list(physio_path.glob('*.puls'))
    ext_files = list(physio_path.glob('*.ext'))
    
    if resp_files and puls_files and ext_files:
        available.append(subject_id)
        
        # Get duration info
        duration_sec, n_samples = get_pmu_duration(resp_files[0], verbose=False)
        n_volumes = get_n_volume_markers(ext_files[0], verbose=False)
        
        if duration_sec is not None:
            duration_min = duration_sec / 60
            durations.append(duration_sec)
            vol_info = f", {n_volumes} vols" if n_volumes else ""
            print(f"✓ sub-{subject_id}: {duration_sec:7.1f}s ({duration_min:5.1f} min{vol_info})")
        else:
            failed_parsing.append(subject_id)
            print(f"⚠️  sub-{subject_id}: FILES EXIST but PARSING FAILED")
            print(f"     Files: {resp_files[0].name}")
            # Re-run with verbose to see what went wrong
            print(f"     Attempting verbose parse...")
            get_pmu_duration(resp_files[0], verbose=True)
            get_n_volume_markers(ext_files[0], verbose=True)
    else:
        missing.append(subject_id)
        print(f"✗ sub-{subject_id}: PMU files MISSING")

print(f"\n{'='*80}")
print(f"SUMMARY:")
print(f"  Successfully parsed: {len(durations)} subjects")
if durations:
    print(f"     IDs: {[available[i] for i in range(len(available)) if i < len(durations)]}")
    print(f"     Duration range: {min(durations)/60:.1f} - {max(durations)/60:.1f} min")
    print(f"     Mean duration: {np.mean(durations)/60:.1f} min")

if failed_parsing:
    print(f"\n  ⚠️  Files exist but parsing failed: {len(failed_parsing)} subjects")
    print(f"     IDs: {failed_parsing}")
    print(f"     These files may have a different format - check verbose output above")

print(f"\n  Missing PMU files: {len(missing)} subjects")
print(f"     IDs: {missing[:10]}{'...' if len(missing) > 10 else ''}")
print(f"{'='*80}\n")