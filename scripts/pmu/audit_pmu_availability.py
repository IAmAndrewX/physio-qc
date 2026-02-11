#!/usr/bin/env python3
"""Audit which participants have parseable PMU files for a given physio session."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402
from utils.pmu_integration import DEFAULT_SCANNER_DIR_VARIANTS  # noqa: E402

PHYSIO_DIR = Path(config.BASE_DATA_PATH)
PHYSIO_SESSION = config.PMU_PHYSIO_SESSION
SAMPLING_RATE = config.PMU_SAMPLING_RATE


def get_pmu_duration(filepath: Path, sampling_rate: float, verbose: bool = False):
    """Get PMU duration in seconds and number of valid samples."""
    try:
        with open(filepath, "rb") as f:
            content = f.read().decode("ascii", errors="ignore")

        tokens = [int(x) for x in content.split() if x.strip().lstrip("-").isdigit()]

        if verbose:
            print(f"      Total tokens: {len(tokens)}")
            print(f"      First 20 tokens: {tokens[:20]}")

        start_idx = tokens.index(6002) + 1 if 6002 in tokens else 5
        end_idx = tokens.index(6003) if 6003 in tokens else len(tokens)
        data_section = tokens[start_idx:end_idx]

        if verbose:
            print(f"      Data section: indices {start_idx} to {end_idx} ({len(data_section)} tokens)")

        if filepath.suffix == ".ext":
            n_samples = len([x for x in data_section if x != 5003])
        else:
            n_samples = len([x for x in data_section if 0 <= x <= 4095])

        if n_samples == 0:
            if verbose:
                print("      No valid samples found")
            return None, None

        duration_sec = n_samples / sampling_rate
        if verbose:
            print(f"      Valid samples: {n_samples:,}")
            print(f"      Duration: {duration_sec:.1f}s ({duration_sec / 60:.1f} min)")

        return duration_sec, n_samples
    except Exception as exc:
        if verbose:
            print(f"      Parse error: {type(exc).__name__}: {exc}")
        return None, None


def get_n_volume_markers(ext_file: Path, verbose: bool = False):
    """Count 5000 volume markers in an .ext file."""
    try:
        with open(ext_file, "rb") as f:
            content = f.read().decode("ascii", errors="ignore")

        tokens = [int(x) for x in content.split() if x.strip().lstrip("-").isdigit()]
        start_idx = tokens.index(6002) + 1 if 6002 in tokens else 5
        end_idx = tokens.index(6003) if 6003 in tokens else len(tokens)
        data_section = tokens[start_idx:end_idx]
        n_volumes = sum(1 for x in data_section if x == 5000)

        if verbose:
            print(f"      Volume markers (5000): {n_volumes}")
        return n_volumes
    except Exception as exc:
        if verbose:
            print(f"      Marker parse error: {exc}")
        return None


def find_scanner_dir(subject_dir: Path, session: str, variants: list[str]):
    """Return first scanner directory match for a participant/session."""
    for variant in variants:
        candidate = subject_dir / session / variant
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Audit PMU file availability across participants")
    parser.add_argument("--physio-dir", default=str(PHYSIO_DIR), help="Physio root directory")
    parser.add_argument("--session", default=PHYSIO_SESSION, help="Physio session label")
    parser.add_argument("--sampling-rate", type=float, default=SAMPLING_RATE, help="PMU sampling rate")
    parser.add_argument(
        "--scanner-variants",
        nargs="+",
        default=DEFAULT_SCANNER_DIR_VARIANTS,
        help="Scanner directory names to try",
    )
    parser.add_argument("--verbose-errors", action="store_true", help="Print verbose parse diagnostics on failures")
    return parser.parse_args()


def main():
    args = parse_args()
    physio_dir = Path(args.physio_dir)
    session = str(args.session)
    sampling_rate = float(args.sampling_rate)
    scanner_variants = list(args.scanner_variants)

    subject_dirs = sorted(physio_dir.glob("sub-*"))

    print(f"\n{'=' * 80}")
    print("PMU FILE AVAILABILITY CHECK")
    print(f"physio_dir={physio_dir}")
    print(f"session={session}")
    print(f"scanner_variants={', '.join(scanner_variants)}")
    print(f"{'=' * 80}\n")

    available = []
    missing = []
    durations = []
    failed_parsing = []

    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        scanner_dir = find_scanner_dir(subject_dir, session, scanner_variants)

        if scanner_dir is None:
            missing.append(subject_id)
            print(f"MISS {subject_id}: scanner folder not found")
            continue

        resp_files = sorted(scanner_dir.glob("*.resp"))
        puls_files = sorted(scanner_dir.glob("*.puls"))
        ext_files = sorted(scanner_dir.glob("*.ext"))

        if not (resp_files and puls_files and ext_files):
            missing.append(subject_id)
            print(
                f"MISS {subject_id}: incomplete PMU set "
                f"(resp={len(resp_files)}, puls={len(puls_files)}, ext={len(ext_files)})"
            )
            continue

        available.append(subject_id)
        duration_sec, _ = get_pmu_duration(resp_files[0], sampling_rate=sampling_rate, verbose=False)
        n_volumes = get_n_volume_markers(ext_files[0], verbose=False)

        if duration_sec is not None:
            durations.append(duration_sec)
            vol_info = f", {n_volumes} vols" if n_volumes is not None else ""
            print(f"OK   {subject_id}: {duration_sec:7.1f}s ({duration_sec / 60:5.1f} min{vol_info})")
        else:
            failed_parsing.append(subject_id)
            print(f"WARN {subject_id}: PMU files found but parsing failed")
            if args.verbose_errors:
                print(f"     RESP file: {resp_files[0].name}")
                get_pmu_duration(resp_files[0], sampling_rate=sampling_rate, verbose=True)
                get_n_volume_markers(ext_files[0], verbose=True)

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"participants scanned: {len(subject_dirs)}")
    print(f"available PMU sets : {len(available)}")
    if durations:
        print(f"duration range      : {min(durations) / 60:.1f} - {max(durations) / 60:.1f} min")
        print(f"mean duration       : {np.mean(durations) / 60:.1f} min")
    print(f"parse failures      : {len(failed_parsing)}")
    print(f"missing/incomplete  : {len(missing)}")
    if missing:
        preview = ", ".join(missing[:10])
        suffix = " ..." if len(missing) > 10 else ""
        print(f"missing preview     : {preview}{suffix}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
