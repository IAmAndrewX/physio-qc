#!/usr/bin/env python3
"""Diagnose PMU enrichment failures for one participant/session/task."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from pprint import pprint

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402
from utils.pmu_integration import (  # noqa: E402
    DEFAULT_SCANNER_DIR_VARIANTS,
    _normalize_task_name,
    extract_pmu_task_signals,
    infer_bids_session_label,
    infer_pmu_session_label,
)


def _uniq(seq):
    seen = set()
    out = []
    for item in seq:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _normalize_participant(participant: str) -> str:
    participant = str(participant).strip()
    return participant if participant.startswith("sub-") else f"sub-{participant}"


def _session_variants(label: str) -> list[str]:
    values = [label]
    if label.startswith("ses-"):
        token = label[4:]
        if token.isdigit():
            values.extend([f"ses-{int(token)}", f"ses-{int(token):02d}"])
    return _uniq(values)


def _print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose why PMU enrichment did not apply for a subject/session/task."
    )
    parser.add_argument("--participant", required=True, help="Participant ID (e.g., sub-1027 or 1027)")
    parser.add_argument("--session", required=True, help="Session label (e.g., ses-2)")
    parser.add_argument("--task", required=True, help="Task label (e.g., rest)")
    parser.add_argument("--base-physio", default=config.BASE_DATA_PATH, help="Base physio path")
    parser.add_argument("--bids-base", default=config.PMU_BIDS_BASE_PATH, help="BIDS base path")
    parser.add_argument("--pmu-session", default=config.PMU_PHYSIO_SESSION, help="Preferred PMU session label")
    parser.add_argument("--bids-session", default=config.PMU_BIDS_SESSION, help="Preferred BIDS session label")
    parser.add_argument("--sampling-rate", type=float, default=config.PMU_SAMPLING_RATE, help="PMU sampling rate")
    parser.add_argument(
        "--scan-gap-seconds", type=float, default=config.PMU_SCAN_GAP_SECONDS, help="Scan gap threshold"
    )
    parser.add_argument(
        "--time-tolerance-seconds",
        type=float,
        default=config.PMU_TIME_MATCH_TOLERANCE_SECONDS,
        help="Scan-time matching tolerance",
    )
    args = parser.parse_args()

    participant = _normalize_participant(args.participant)
    task_norm = _normalize_task_name(args.task)
    base_physio = Path(args.base_physio)
    bids_base = Path(args.bids_base)

    pmu_candidates = []
    if args.pmu_session:
        pmu_candidates.extend(_session_variants(args.pmu_session))
    pmu_candidates.extend(_session_variants(infer_pmu_session_label(args.session)))
    pmu_candidates = _uniq(pmu_candidates)

    bids_candidates = []
    if args.bids_session:
        bids_candidates.extend(_session_variants(args.bids_session))
    bids_candidates.extend(_session_variants(infer_bids_session_label(args.session)))
    bids_candidates = _uniq(bids_candidates)

    _print_header("Input")
    print(f"participant={participant}")
    print(f"session={args.session}")
    print(f"task={args.task} (normalized={task_norm})")

    _print_header("Configured Paths")
    print(f"BASE physio: {base_physio}")
    print(f"BIDS base : {bids_base}")

    _print_header("PMU Session Candidates")
    print(", ".join(pmu_candidates) if pmu_candidates else "(none)")

    _print_header("Scanner Folder Checks")
    found_scanner_dirs = []
    for ses in pmu_candidates:
        for variant in DEFAULT_SCANNER_DIR_VARIANTS:
            candidate = base_physio / participant / ses / variant
            exists = candidate.exists()
            print(f"[{'OK' if exists else 'MISS'}] {candidate}")
            if exists:
                found_scanner_dirs.append(candidate)

    _print_header("PMU Files (.resp/.puls/.ext)")
    if not found_scanner_dirs:
        print("No scanner folders found in expected locations.")
    else:
        for scanner_dir in found_scanner_dirs:
            resp = sorted(scanner_dir.glob("*.resp"))
            puls = sorted(scanner_dir.glob("*.puls"))
            ext = sorted(scanner_dir.glob("*.ext"))
            print(f"\n{scanner_dir}")
            print(f"  resp: {len(resp)}")
            print(f"  puls: {len(puls)}")
            print(f"  ext : {len(ext)}")
            for f in resp[:3] + puls[:3] + ext[:3]:
                print(f"   - {f.name}")

    _print_header("Possible Naming Mismatches")
    participant_root = base_physio / participant
    if not participant_root.exists():
        print(f"Participant folder not found: {participant_root}")
    else:
        fuzzy_dirs = []
        for p in participant_root.glob("ses-*/*"):
            if not p.is_dir():
                continue
            name = p.name.lower()
            if "scanner" in name or ("physio" in name and "pmu" in name):
                fuzzy_dirs.append(p)
        if fuzzy_dirs:
            print("Potential scanner-related folders found (non-standard names):")
            for d in sorted(fuzzy_dirs):
                print(f" - {d}")
        else:
            print("No obvious non-standard scanner folder names found under ses-*/*")

        pmu_files_anywhere = (
            sorted(participant_root.rglob("*.resp"))
            + sorted(participant_root.rglob("*.puls"))
            + sorted(participant_root.rglob("*.ext"))
        )
        if pmu_files_anywhere:
            print("PMU-like files found somewhere under participant (showing up to 20):")
            for f in pmu_files_anywhere[:20]:
                print(f" - {f}")
        else:
            print("No .resp/.puls/.ext files found anywhere under participant folder.")

    _print_header("BIDS scans.tsv Checks")
    any_scans = False
    for ses in bids_candidates:
        scans_tsv = bids_base / participant / ses / f"{participant}_{ses}_scans.tsv"
        if not scans_tsv.exists():
            print(f"[MISS] {scans_tsv}")
            continue
        any_scans = True
        print(f"[OK]   {scans_tsv}")
        try:
            df = pd.read_csv(scans_tsv, sep="\t")
        except Exception as exc:
            print(f"  could not parse scans.tsv: {exc}")
            continue
        if "filename" not in df.columns:
            print("  missing 'filename' column")
            continue
        task_rows = df[df["filename"].astype(str).str.contains("task-", case=False, na=False)].copy()
        if task_rows.empty:
            print("  no task rows found in scans.tsv")
            continue
        task_rows["task"] = task_rows["filename"].astype(str).str.extract(r"task-([^_]+)", expand=False)
        task_rows["task_norm"] = task_rows["task"].astype(str).apply(_normalize_task_name)
        exact = task_rows[task_rows["task_norm"] == task_norm]
        partial = task_rows[
            task_rows["task_norm"].str.contains(task_norm, regex=False)
            | pd.Series([task_norm in x for x in task_rows["task_norm"]], index=task_rows.index)
        ]
        print(f"  task rows: {len(task_rows)}")
        print(f"  exact matches for '{task_norm}': {len(exact)}")
        print(f"  partial matches for '{task_norm}': {len(partial)}")
        preview = (exact if not exact.empty else partial).head(5)
        for _, row in preview.iterrows():
            acq_time = row["acq_time"] if "acq_time" in row else "n/a"
            print(f"   - task={row['task']} filename={row['filename']} acq_time={acq_time}")
    if not any_scans:
        print("No candidate scans.tsv files found.")

    _print_header("App Matcher Result (extract_pmu_task_signals)")
    result = extract_pmu_task_signals(
        base_physio_path=base_physio,
        bids_base_path=bids_base,
        participant=participant,
        session=args.session,
        task=args.task,
        pmu_session=args.pmu_session,
        bids_session=args.bids_session,
        sampling_rate=args.sampling_rate,
        scan_gap_seconds=args.scan_gap_seconds,
        time_tolerance_seconds=args.time_tolerance_seconds,
    )
    pprint(result)

    return 0 if result.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
