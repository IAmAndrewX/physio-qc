#!/usr/bin/env python3
"""
visualize_pmu_recording.py
PMU Raw Data Visualizer - Diagnostic tool to see the full picture

Creates comprehensive plots showing:
- Full respiratory and pulse traces
- All 5000 scan markers
- All TTL trigger positions
- BIDS scan times overlaid
- Zoom into first scan for detailed inspection

Use this BEFORE trying to extract data - helps identify timing issues.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402
from utils.pmu_integration import DEFAULT_SCANNER_DIR_VARIANTS  # noqa: E402

# ============================================================================
# CONFIGURATION
# ============================================================================

BIDS_DIR = Path(config.PMU_BIDS_BASE_PATH)
PHYSIO_DIR = Path(config.BASE_DATA_PATH)
OUTPUT_DIR = REPO_ROOT / "outputs" / "pmu" / "diagnostics"

BIDS_SESSION = config.PMU_BIDS_SESSION
PHYSIO_SESSION = config.PMU_PHYSIO_SESSION
SAMPLING_RATE = config.PMU_SAMPLING_RATE


# ============================================================================
# FILE PARSING
# ============================================================================


def parse_pmu_file(filepath):
    """
    Parse Siemens PMU file (.puls, .resp, .ext).

    Returns: dict with 'values', 'volume_markers' (5000s), 'slice_triggers' (1s)
    """
    with open(filepath, "rb") as f:
        content = f.read().decode("ascii", errors="ignore")

    tokens = [int(x) for x in content.split() if x.strip().lstrip("-").isdigit()]

    # Find data section between 6002 and 6003
    start_idx = tokens.index(6002) + 1 if 6002 in tokens else 5
    end_idx = tokens.index(6003) if 6003 in tokens else len(tokens)
    data_section = tokens[start_idx:end_idx]

    is_ext_file = filepath.suffix == ".ext"

    if is_ext_file:
        # .ext: binary time series (position = sample index)
        # value 1 = slice trigger (TTL pulse)
        # value 5000 = volume marker
        signal = []
        slice_triggers = []
        volume_markers = []

        for i, value in enumerate(data_section):
            if value == 1:  # Slice trigger
                slice_triggers.append(i)
                signal.append(0)
            elif value == 5000:  # Volume marker
                volume_markers.append(i)
                signal.append(0)
            else:  # No trigger
                signal.append(0)

        return {
            "values": np.array(signal, dtype=float),
            "slice_triggers": np.array(slice_triggers, dtype=int),
            "volume_markers": np.array(volume_markers, dtype=int),
            "filepath": filepath,
            "duration": len(signal) / SAMPLING_RATE,
        }

    else:
        # .puls/.resp: physiological signal
        signal = []
        volume_markers = []  # Not used in physio files

        for value in data_section:
            if value == 5000:  # Cardiac pulse in physio files
                pass  # Skip for now
            elif 0 <= value <= 4095:
                signal.append(value)

        return {
            "values": np.array(signal, dtype=float),
            "slice_triggers": np.array([], dtype=int),
            "volume_markers": np.array(volume_markers, dtype=int),
            "filepath": filepath,
            "duration": len(signal) / SAMPLING_RATE,
        }


def get_bids_scan_times(subject_id):
    """
    Read all scan times from BIDS scans.tsv.

    Returns: DataFrame with scan names and offsets from session start
    """
    scans_file = BIDS_DIR / f"sub-{subject_id}" / BIDS_SESSION / f"sub-{subject_id}_{BIDS_SESSION}_scans.tsv"

    if not scans_file.exists():
        print(f"⚠️  BIDS scans file not found: {scans_file}")
        return None

    df = pd.read_csv(scans_file, sep="\t")

    # Parse timestamps
    df["timestamp"] = pd.to_datetime(df["acq_time"])
    session_start = df["timestamp"].min()
    df["offset_seconds"] = (df["timestamp"] - session_start).dt.total_seconds()

    # Extract scan names (task names)
    df["scan_name"] = df["filename"].str.extract(r"task-(\w+)")[0]

    return df[["scan_name", "offset_seconds", "filename"]].dropna()


# ============================================================================
# VISUALIZATION
# ============================================================================


def create_diagnostic_plots(resp_data, puls_data, ext_data, subject_id, bids_times=None):
    """
    Create comprehensive diagnostic visualization with improved clarity.

    7 panels showing:
    1. Full respiratory trace + scan boundaries
    2. Full pulse trace + scan boundaries
    3. Volume marker timeline with scan groupings
    4. Inter-volume intervals showing scan gaps
    5. Zoom into scan boundary transition
    6. Zoom into volume markers within a scan
    7. Histogram of inter-volume intervals
    """
    output_dir = OUTPUT_DIR / f"sub-{subject_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Time axes
    t_resp = np.arange(len(resp_data["values"])) / SAMPLING_RATE
    t_puls = np.arange(len(puls_data["values"])) / SAMPLING_RATE

    volume_times = ext_data["volume_markers"] / SAMPLING_RATE

    # Detect scans from volume markers
    if len(volume_times) > 1:
        intervals = np.diff(volume_times)
        gap_indices = np.where(intervals > 10)[0]
        scan_boundaries = np.concatenate([[0], gap_indices + 1, [len(volume_times)]])
    else:
        scan_boundaries = [0, len(volume_times)]

    n_scans = len(scan_boundaries) - 1

    # Create figure
    fig = plt.figure(figsize=(24, 20))
    gs = fig.add_gridspec(7, 1, hspace=0.5, height_ratios=[1, 1, 0.8, 1, 1, 1, 0.8])

    fig.suptitle(
        f"sub-{subject_id} - PMU Data Diagnostic\n" + f"{len(volume_times)} volumes across {n_scans} scans",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )

    # ========== Panel 1: Respiratory trace with scan boundaries ==========
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t_resp, resp_data["values"], "b-", linewidth=0.3, alpha=0.7)

    # Add scan boundary shading
    colors = ["lightblue", "lightcoral", "lightgreen", "lightyellow", "lightpink", "lightcyan"]
    for i in range(n_scans):
        start_idx = scan_boundaries[i]
        end_idx = scan_boundaries[i + 1] - 1
        start_time = volume_times[start_idx]
        end_time = volume_times[end_idx]
        ax1.axvspan(start_time, end_time, alpha=0.2, color=colors[i % len(colors)], label=f"Scan {i + 1}")

    # Add BIDS times
    if bids_times is not None:
        for _, row in bids_times.iterrows():
            ax1.axvline(
                row["offset_seconds"],
                color="green",
                alpha=0.7,
                linewidth=2,
                linestyle="--",
                label="BIDS scan" if _ == 0 else "",
            )
            ax1.text(
                row["offset_seconds"],
                ax1.get_ylim()[1] * 0.95,
                row["scan_name"],
                rotation=90,
                va="top",
                fontsize=8,
                color="darkgreen",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.7},
            )

    ax1.set_ylabel("Respiratory\nSignal", fontsize=11, fontweight="bold")
    ax1.set_title("Panel 1: Respiratory Trace (shaded regions = detected scans)", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, t_resp[-1]])
    ax1.legend(loc="upper right", fontsize=8)

    # ========== Panel 2: Pulse trace with scan boundaries ==========
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(t_puls, puls_data["values"], "r-", linewidth=0.3, alpha=0.7)

    # Add scan boundary shading
    for i in range(n_scans):
        start_idx = scan_boundaries[i]
        end_idx = scan_boundaries[i + 1] - 1
        start_time = volume_times[start_idx]
        end_time = volume_times[end_idx]
        ax2.axvspan(start_time, end_time, alpha=0.2, color=colors[i % len(colors)])

    ax2.set_ylabel("Pulse\nSignal", fontsize=11, fontweight="bold")
    ax2.set_title("Panel 2: Pulse Oximetry Trace", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, t_puls[-1]])

    # ========== Panel 3: Volume markers with scan groupings ==========
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    # Plot volume markers colored by scan
    for i in range(n_scans):
        start_idx = scan_boundaries[i]
        end_idx = scan_boundaries[i + 1]
        scan_vol_times = volume_times[start_idx:end_idx]
        ax3.scatter(
            scan_vol_times,
            np.ones_like(scan_vol_times) * (i + 1),
            marker="|",
            s=100,
            linewidths=2,
            color=colors[i % len(colors)],
            label=f"Scan {i + 1} ({len(scan_vol_times)} vols)",
        )

    ax3.set_ylabel("Scan ID", fontsize=11, fontweight="bold")
    ax3.set_title(
        f"Panel 3: Volume Markers Grouped by Scan ({len(volume_times)} total volumes)", fontsize=12, fontweight="bold"
    )
    ax3.set_ylim([0.5, n_scans + 0.5])
    ax3.set_yticks(range(1, n_scans + 1))
    ax3.grid(True, alpha=0.3, axis="x")
    ax3.legend(loc="upper right", fontsize=8, ncol=n_scans)

    # ========== Panel 4: Inter-volume intervals ==========
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    if len(volume_times) > 1:
        intervals = np.diff(volume_times)
        ax4.semilogy(volume_times[:-1], intervals, "purple", linewidth=1, alpha=0.7)
        ax4.set_ylabel("Interval (s)\n[log scale]", fontsize=11, fontweight="bold")
        ax4.set_title("Panel 4: Inter-Volume Intervals (peaks = scan boundaries)", fontsize=12, fontweight="bold")
        ax4.grid(True, alpha=0.3)

        # Add threshold lines
        median_interval = np.median(intervals[intervals < 5])  # Exclude gaps
        ax4.axhline(
            median_interval,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median TR: {median_interval:.3f}s",
            zorder=5,
        )
        ax4.axhline(10, color="red", linestyle="--", linewidth=2, label="Gap threshold (10s)", zorder=5)

        # Annotate gaps
        for i in range(n_scans - 1):
            gap_idx = scan_boundaries[i + 1] - 1
            gap_time = volume_times[gap_idx]
            gap_size = intervals[gap_idx]
            ax4.plot(gap_time, gap_size, "ro", markersize=10, zorder=10)
            ax4.text(
                gap_time,
                gap_size * 1.5,
                f"{gap_size:.1f}s gap",
                ha="center",
                fontsize=8,
                color="red",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
            )

        ax4.legend(loc="upper right", fontsize=9)
        ax4.set_ylim([0.5, max(intervals.max(), 20)])

    # ========== Panel 5: Zoom into scan boundary ==========
    ax5 = fig.add_subplot(gs[4])
    if n_scans > 1:
        # Show first scan boundary
        gap_idx = scan_boundaries[1] - 1
        gap_time = volume_times[gap_idx]

        zoom_start = gap_time - 10
        zoom_end = gap_time + 10

        # Respiratory signal
        resp_mask = (t_resp >= zoom_start) & (t_resp <= zoom_end)
        ax5.plot(t_resp[resp_mask], resp_data["values"][resp_mask], "b-", linewidth=1, alpha=0.7, label="Respiratory")

        # Volume markers before gap (Scan 1)
        vol_mask_before = (volume_times >= zoom_start) & (volume_times <= gap_time)
        for vt in volume_times[vol_mask_before]:
            ax5.axvline(vt, color=colors[0], alpha=0.5, linewidth=2)

        # Volume markers after gap (Scan 2)
        vol_mask_after = (volume_times > gap_time) & (volume_times <= zoom_end)
        for vt in volume_times[vol_mask_after]:
            ax5.axvline(vt, color=colors[1 % len(colors)], alpha=0.5, linewidth=2)

        # Mark the gap
        ax5.axvline(
            gap_time,
            color="red",
            linewidth=3,
            linestyle="--",
            label=f"Scan boundary ({intervals[gap_idx]:.1f}s gap)",
            zorder=10,
        )
        ax5.axvspan(gap_time, gap_time + intervals[gap_idx], alpha=0.3, color="red")

        ax5.set_xlabel("Time (seconds)", fontsize=11)
        ax5.set_ylabel("Respiratory Signal", fontsize=11, fontweight="bold")
        ax5.set_title("Panel 5: Scan Boundary Transition (Scan 1 → Scan 2)", fontsize=12, fontweight="bold")
        ax5.set_xlim([zoom_start, zoom_end])
        ax5.grid(True, alpha=0.3)
        ax5.legend(loc="upper right", fontsize=9)

    # ========== Panel 6: Zoom into volume markers within scan ==========
    ax6 = fig.add_subplot(gs[5])
    if len(volume_times) > 10:
        # Show first 10 volumes of first scan
        first_vol_time = volume_times[0]
        tenth_vol_time = volume_times[min(9, len(volume_times) - 1)]

        zoom_start = first_vol_time - 2
        zoom_end = tenth_vol_time + 2

        # Respiratory signal (normalized)
        resp_mask = (t_resp >= zoom_start) & (t_resp <= zoom_end)
        resp_norm = resp_data["values"][resp_mask]
        resp_norm = (resp_norm - resp_norm.min()) / (resp_norm.max() - resp_norm.min() + 1e-8)
        ax6.plot(t_resp[resp_mask], resp_norm, "b-", linewidth=1, alpha=0.7, label="Respiratory (normalized)")

        # Pulse signal (normalized + offset)
        puls_mask = (t_puls >= zoom_start) & (t_puls <= zoom_end)
        puls_norm = puls_data["values"][puls_mask]
        puls_norm = (puls_norm - puls_norm.min()) / (puls_norm.max() - puls_norm.min() + 1e-8)
        ax6.plot(t_puls[puls_mask], puls_norm + 1.2, "r-", linewidth=1, alpha=0.7, label="Pulse (normalized)")

        # Volume markers
        vol_mask = (volume_times >= zoom_start) & (volume_times <= zoom_end)
        for i, vt in enumerate(volume_times[vol_mask]):
            ax6.axvline(vt, color="darkblue", linewidth=2, alpha=0.6, linestyle="--")
            ax6.text(
                vt,
                2.3,
                f"V{i + 1}",
                ha="center",
                fontsize=8,
                color="darkblue",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.7},
            )

        # Show TR
        if len(volume_times[vol_mask]) > 1:
            tr = volume_times[vol_mask][1] - volume_times[vol_mask][0]
            ax6.annotate(
                "",
                xy=(volume_times[vol_mask][1], 2.5),
                xytext=(volume_times[vol_mask][0], 2.5),
                arrowprops={"arrowstyle": "<->", "color": "black", "lw": 2},
            )
            ax6.text(
                (volume_times[vol_mask][0] + volume_times[vol_mask][1]) / 2,
                2.6,
                f"TR = {tr:.3f}s",
                ha="center",
                fontsize=9,
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "yellow", "alpha": 0.7},
            )

        ax6.set_xlabel("Time (seconds)", fontsize=11, fontweight="bold")
        ax6.set_ylabel("Normalized Signal", fontsize=11, fontweight="bold")
        ax6.set_title("Panel 6: Volume Markers Within Scan (first 10 volumes)", fontsize=12, fontweight="bold")
        ax6.set_xlim([zoom_start, zoom_end])
        ax6.set_ylim([-0.1, 2.8])
        ax6.grid(True, alpha=0.3)
        ax6.legend(loc="upper left", fontsize=9)

    # ========== Panel 7: Interval histogram ==========
    ax7 = fig.add_subplot(gs[6])
    if len(volume_times) > 1:
        intervals = np.diff(volume_times)

        # Separate scan intervals from gaps
        scan_intervals = intervals[intervals < 5]  # Normal TRs
        gaps = intervals[intervals >= 10]  # Scan boundaries

        ax7.hist(scan_intervals, bins=50, color="blue", alpha=0.7, label=f"Within-scan TRs (n={len(scan_intervals)})")
        if len(gaps) > 0:
            ax7.hist(gaps, bins=20, color="red", alpha=0.7, label=f"Scan gaps (n={len(gaps)})")

        ax7.axvline(
            np.median(scan_intervals),
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median TR: {np.median(scan_intervals):.3f}s",
        )
        ax7.set_xlabel("Interval (seconds)", fontsize=11, fontweight="bold")
        ax7.set_ylabel("Count", fontsize=11, fontweight="bold")
        ax7.set_title("Panel 7: Distribution of Inter-Volume Intervals", fontsize=12, fontweight="bold")
        ax7.grid(True, alpha=0.3, axis="y")
        ax7.legend(loc="upper right", fontsize=9)

    plt.tight_layout()

    output_file = output_dir / f"sub-{subject_id}_pmu_raw_diagnostic.png"
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"✓ Diagnostic plot saved: {output_file}")

    return output_file


def print_summary_stats(resp_data, puls_data, ext_data, subject_id, bids_times=None):
    """Print comprehensive summary of all data."""
    print(f"\n{'=' * 80}")
    print(f"PMU DATA SUMMARY - sub-{subject_id}")
    print(f"{'=' * 80}\n")

    # Physiological data
    print("📊 PHYSIOLOGICAL DATA:")
    print("   Respiratory:")
    print(f"      Samples: {len(resp_data['values']):,}")
    print(f"      Duration: {resp_data['duration']:.1f}s ({resp_data['duration'] / 60:.1f} min)")
    print(f"      Range: {resp_data['values'].min():.0f} - {resp_data['values'].max():.0f}")
    print("   Pulse:")
    print(f"      Samples: {len(puls_data['values']):,}")
    print(f"      Duration: {puls_data['duration']:.1f}s ({puls_data['duration'] / 60:.1f} min)")
    print(f"      Range: {puls_data['values'].min():.0f} - {puls_data['values'].max():.0f}\n")

    # Markers
    print("📍 MARKERS (.ext file):")
    print(f"   Volume markers (5000): {len(ext_data['volume_markers'])}")
    if len(ext_data["volume_markers"]) > 0:
        vol_times = ext_data["volume_markers"] / SAMPLING_RATE
        print(f"      First at: {vol_times[0]:.1f}s")
        print(f"      Last at: {vol_times[-1]:.1f}s")
        if len(vol_times) > 1:
            intervals = np.diff(vol_times)
            print(f"      Median TR: {np.median(intervals):.3f}s")
            print(f"      TR range: {intervals.min():.3f}s - {intervals.max():.3f}s")

    print(f"   Slice triggers (1): {len(ext_data['slice_triggers'])}")
    if len(ext_data["slice_triggers"]) > 0:
        slice_times = ext_data["slice_triggers"] / SAMPLING_RATE
        print(f"      First at: {slice_times[0]:.1f}s")
        print(f"      Last at: {slice_times[-1]:.1f}s")

        # Calculate slices per volume
        if len(ext_data["volume_markers"]) > 0:
            slices_per_volume = len(ext_data["slice_triggers"]) / len(ext_data["volume_markers"])
            print(f"      Slices per volume: ~{slices_per_volume:.1f}")
    print()

    # Identify scans from volume markers
    if len(ext_data["volume_markers"]) > 1:
        print("🔍 SCAN DETECTION (from volume markers):")
        vol_times = ext_data["volume_markers"] / SAMPLING_RATE
        intervals = np.diff(vol_times)

        # Find gaps > 10s (scan boundaries)
        gap_indices = np.where(intervals > 10)[0]
        scan_boundaries = np.concatenate([[0], gap_indices + 1, [len(ext_data["volume_markers"])]])
        n_scans = len(scan_boundaries) - 1

        print(f"   Detected {n_scans} scans:\n")
        for i in range(n_scans):
            start_idx = scan_boundaries[i]
            end_idx = scan_boundaries[i + 1] - 1
            start_time = vol_times[start_idx]
            end_time = vol_times[end_idx]
            n_vols = end_idx - start_idx + 1
            duration = end_time - start_time

            if n_vols > 1:
                scan_tr = np.median(intervals[start_idx:end_idx])
            else:
                scan_tr = 0

            print(
                f"   Scan {i + 1}: {start_time:6.1f}s - {end_time:6.1f}s ({duration:5.1f}s), "
                f"{n_vols:4d} volumes, TR={scan_tr:.3f}s"
            )
        print()

    # BIDS times
    if bids_times is not None:
        print("🗂️  BIDS SCAN TIMES:")
        print(f"   PMU recording ends at: {resp_data['duration']:.1f}s ({resp_data['duration'] / 60:.1f} min)")
        print("\n   Scan schedule:")
        for _, row in bids_times.iterrows():
            status = "✓ Within PMU" if row["offset_seconds"] <= resp_data["duration"] else "✗ After PMU ends"
            print(f"      {row['offset_seconds']:6.1f}s - {row['scan_name']:15s} {status}")

        # Check for timing issues
        scans_after_pmu = bids_times[bids_times["offset_seconds"] > resp_data["duration"]]
        if len(scans_after_pmu) > 0:
            print(f"\n   ⚠️  WARNING: {len(scans_after_pmu)} scan(s) scheduled AFTER PMU recording ended!")
            print("      This suggests either:")
            print("         - Wrong PMU files (different session?)")
            print("         - Incorrect BIDS timestamps")
            print("         - PMU recording stopped early")
    else:
        print("⚠️  BIDS timing data not available\n")

    print(f"{'=' * 80}\n")


# ============================================================================
# UTILITY HELPERS
# ============================================================================


def find_physio_dir(subject_id, physio_session=None):
    """Return the first existing scanner directory variant for this subject/session."""
    if physio_session is None:
        physio_session = PHYSIO_SESSION
    base_dir = PHYSIO_DIR / f"sub-{subject_id}" / physio_session
    for variant in DEFAULT_SCANNER_DIR_VARIANTS:
        candidate = base_dir / variant
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No scanner PMU folder found for sub-{subject_id} in {base_dir}. "
        f"Checked: {', '.join(DEFAULT_SCANNER_DIR_VARIANTS)}"
    )


# ============================================================================
# MAIN
# ============================================================================


def main():
    global BIDS_DIR, PHYSIO_DIR, OUTPUT_DIR, BIDS_SESSION, PHYSIO_SESSION, SAMPLING_RATE

    parser = argparse.ArgumentParser(
        description="Visualize raw PMU data with all markers and timing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This diagnostic tool shows the complete picture of PMU data BEFORE extraction.
Use this to:
  - Verify scan markers are detected correctly
  - Check if BIDS times align with PMU recording
  - Identify timing mismatches
  - See TTL trigger patterns

Examples:
  python scripts/pmu/visualize_pmu_recording.py 2008
  python scripts/pmu/visualize_pmu_recording.py 2034
        """,
    )

    parser.add_argument("subject_id", help="Subject ID (e.g., 2008, 2034)")
    parser.add_argument("--bids-dir", default=str(BIDS_DIR), help="BIDS root path")
    parser.add_argument("--physio-dir", default=str(PHYSIO_DIR), help="Physio root path")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output base directory")
    parser.add_argument("--bids-session", default=BIDS_SESSION, help="BIDS session label (default from config)")
    parser.add_argument("--physio-session", default=PHYSIO_SESSION, help="Physio session label (default from config)")
    parser.add_argument("--sampling-rate", type=float, default=SAMPLING_RATE, help="Sampling rate in Hz")
    args = parser.parse_args()

    BIDS_DIR = Path(args.bids_dir)
    PHYSIO_DIR = Path(args.physio_dir)
    OUTPUT_DIR = Path(args.output_dir)
    BIDS_SESSION = args.bids_session
    PHYSIO_SESSION = args.physio_session
    SAMPLING_RATE = float(args.sampling_rate)

    subject_id = args.subject_id

    print(f"\n{'=' * 80}")
    print(f"PMU RAW DATA VISUALIZATION - sub-{subject_id}")
    print(f"{'=' * 80}\n")

    # Find files
    try:
        physio_dir = find_physio_dir(subject_id, PHYSIO_SESSION)
    except FileNotFoundError as exc:
        print(f"❌ {exc}")
        return

    try:
        resp_file = list(physio_dir.glob("*.resp"))[0]
        puls_file = list(physio_dir.glob("*.puls"))[0]
        ext_file = list(physio_dir.glob("*.ext"))[0]
    except IndexError:
        print(f"❌ PMU files not found in: {physio_dir}")
        return

    print("📂 Loading PMU files...")
    print(f"   RESP: {resp_file.name}")
    print(f"   PULS: {puls_file.name}")
    print(f"   EXT:  {ext_file.name}\n")

    # Parse files
    print("📊 Parsing files...")
    resp_data = parse_pmu_file(resp_file)
    puls_data = parse_pmu_file(puls_file)
    ext_data = parse_pmu_file(ext_file)
    print("   ✓ Parsing complete\n")

    # Get BIDS timing
    print("🗂️  Reading BIDS metadata...")
    bids_times = get_bids_scan_times(subject_id)
    if bids_times is not None:
        print(f"   ✓ Found {len(bids_times)} scans in BIDS\n")
    else:
        print("   ⚠️  BIDS metadata not found\n")

    # Print summary
    print_summary_stats(resp_data, puls_data, ext_data, subject_id, bids_times)

    # Create plots
    print("📈 Creating diagnostic plots...")
    output_file = create_diagnostic_plots(resp_data, puls_data, ext_data, subject_id, bids_times)

    print(f"\n{'=' * 80}")
    print("✓ COMPLETE")
    print(f"{'=' * 80}\n")
    print("Next steps:")
    print(f"  1. Open: {output_file}")
    print("  2. Check if BIDS times (green) align with 5000 markers (red)")
    print("  3. Identify which scan marker corresponds to your target scan")
    print("  4. If times don't align, investigate timing mismatch\n")


if __name__ == "__main__":
    main()
