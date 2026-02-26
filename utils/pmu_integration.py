"""
PMU integration helpers for injecting Siemens scanner respiration/pulse into app data.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_SCANNER_DIR_VARIANTS = (
    "Scanner_physio",
    "Scanner_Physio",
    "scanner_physio",
    "scanner_Physio",
)
_CANONICAL_SCANNER_FOLDER_TOKEN = "scannerphysio"


def _normalize_task_name(task_name: str) -> str:
    task = str(task_name).strip().lower()
    return task[5:] if task.startswith("task-") else task


def _normalize_session_token(session_label: str) -> str:
    token = str(session_label).strip().lower().replace("_", "-")
    for prefix in ("session-", "session", "ses-"):
        if token.startswith(prefix):
            token = token[len(prefix):]
            break
    return token.strip("- ")


def infer_bids_session_label(session_label: str) -> str:
    token = _normalize_session_token(session_label)
    alias = {"a": "01", "b": "02"}.get(token, token)
    if alias.isdigit():
        return f"ses-{int(alias):02d}"
    return f"ses-{alias}"


def infer_pmu_session_label(session_label: str) -> str:
    token = _normalize_session_token(session_label)
    alias = {"a": "1", "b": "2"}.get(token, token)
    if alias.isdigit():
        return f"ses-{int(alias)}"
    return f"ses-{alias}"


def session_matches_alias(session_label: str, aliases: list[str]) -> bool:
    normalized = _normalize_session_token(session_label)
    normalized_aliases = {_normalize_session_token(alias) for alias in aliases}
    return normalized in normalized_aliases


def _normalize_scanner_folder_token(folder_name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(folder_name).strip().lower())


def _is_scanner_folder_name(folder_name: str, folder_variants: tuple[str, ...]) -> bool:
    normalized = _normalize_scanner_folder_token(folder_name)
    if normalized == _CANONICAL_SCANNER_FOLDER_TOKEN:
        return True
    normalized_variants = {_normalize_scanner_folder_token(name) for name in folder_variants}
    return normalized in normalized_variants


def _find_scanner_folder_in_session(
    session_dir: Path,
    folder_variants: tuple[str, ...] = DEFAULT_SCANNER_DIR_VARIANTS,
) -> Path | None:
    if not session_dir.exists() or not session_dir.is_dir():
        return None

    child_dirs = [child for child in sorted(session_dir.iterdir()) if child.is_dir()]

    # Fast path: explicit variant names first (exact directory-name matches).
    for variant in folder_variants:
        for child in child_dirs:
            if child.name == variant:
                return child

    # Flexible fallback: case-insensitive + separator-insensitive token matching.
    for child in child_dirs:
        if child.is_dir() and _is_scanner_folder_name(child.name, folder_variants):
            return child
    return None


def find_scanner_folder_in_session(
    session_dir: Path,
    folder_variants: tuple[str, ...] = DEFAULT_SCANNER_DIR_VARIANTS,
) -> Path | None:
    """Public helper: resolve scanner PMU folder with flexible name matching."""
    return _find_scanner_folder_in_session(session_dir, folder_variants=folder_variants)


def _parse_tokens(filepath: Path) -> list[int]:
    with open(filepath, "rb") as f:
        content = f.read().decode("ascii", errors="ignore")
    return [int(x) for x in content.split() if x.strip().lstrip("-").isdigit()]


def parse_pmu_file(filepath: Path) -> dict[str, Any]:
    """Parse Siemens PMU file (.resp, .puls, .ext)."""
    tokens = _parse_tokens(filepath)
    start_idx = tokens.index(6002) + 1 if 6002 in tokens else 5
    end_idx = tokens.index(6003) if 6003 in tokens else len(tokens)
    data_section = tokens[start_idx:end_idx]
    is_ext = filepath.suffix.lower() == ".ext"

    if is_ext:
        volume_markers = [i for i, value in enumerate(data_section) if value == 5000]
        slice_triggers = [i for i, value in enumerate(data_section) if value == 1]
        return {
            "values": np.zeros(len(data_section), dtype=float),
            "volume_markers": np.array(volume_markers, dtype=int),
            "slice_triggers": np.array(slice_triggers, dtype=int),
        }

    signal = []
    cardiac_pulses = []
    for value in data_section:
        if value == 5000:
            cardiac_pulses.append(len(signal))
        elif value == 5003:
            continue
        elif 0 <= value <= 4095:
            signal.append(value)

    return {
        "values": np.array(signal, dtype=float),
        "volume_markers": np.array([], dtype=int),
        "slice_triggers": np.array([], dtype=int),
        "cardiac_pulses": np.array(cardiac_pulses, dtype=int),
    }


def identify_scans_from_volume_markers(
    volume_markers: np.ndarray,
    sampling_rate: float = 400.0,
    min_gap_seconds: float = 10.0,
) -> list[dict[str, Any]]:
    """Split PMU recording into scans using large inter-volume gaps."""
    if len(volume_markers) == 0:
        return []

    volume_times = volume_markers / float(sampling_rate)
    intervals = np.diff(volume_times)
    gap_indices = np.where(intervals > min_gap_seconds)[0]
    boundaries = np.concatenate(([0], gap_indices + 1, [len(volume_markers)]))

    scans: list[dict[str, Any]] = []
    for i in range(len(boundaries) - 1):
        start_idx = int(boundaries[i])
        end_idx = int(boundaries[i + 1] - 1)
        start_sample = int(volume_markers[start_idx])
        end_sample = int(volume_markers[end_idx])
        n_volumes = int(end_idx - start_idx + 1)

        if n_volumes > 1:
            scan_intervals = intervals[start_idx:end_idx]
            median_tr = float(np.median(scan_intervals))
        else:
            median_tr = 0.0

        scans.append(
            {
                "scan_index": i,
                "start_sample": start_sample,
                "end_sample": end_sample,
                "start_time": float(start_sample / sampling_rate),
                "end_time": float(end_sample / sampling_rate),
                "duration": float((end_sample - start_sample) / sampling_rate),
                "n_volumes": n_volumes,
                "median_tr": median_tr,
            }
        )

    return scans


def _safe_float(value: Any) -> float | None:
    try:
        val = float(value)
        return val if np.isfinite(val) else None
    except Exception:
        return None


def _load_bids_task_table(bids_base_path: Path, participant: str, bids_session: str) -> pd.DataFrame | None:
    scans_tsv = bids_base_path / participant / bids_session / f"{participant}_{bids_session}_scans.tsv"
    if not scans_tsv.exists():
        return None

    table = pd.read_csv(scans_tsv, sep="\t")
    if "filename" not in table.columns or "acq_time" not in table.columns:
        return None

    task_rows = table[table["filename"].astype(str).str.contains("task-", case=False, na=False)].copy()
    if task_rows.empty:
        return None

    task_rows["task"] = task_rows["filename"].astype(str).str.extract(r"task-([^_]+)", expand=False)
    task_rows["task"] = task_rows["task"].astype(str)
    task_rows["task_normalized"] = task_rows["task"].apply(_normalize_task_name)
    task_rows["timestamp"] = pd.to_datetime(task_rows["acq_time"], errors="coerce")
    task_rows = task_rows.dropna(subset=["timestamp", "task"])
    if task_rows.empty:
        return None

    task_rows = task_rows.sort_values("timestamp")
    session_start = task_rows["timestamp"].min()
    task_rows["offset_seconds"] = (task_rows["timestamp"] - session_start).dt.total_seconds().astype(float)

    first_rows = task_rows.drop_duplicates(subset=["task_normalized"], keep="first").copy().reset_index(drop=True)
    first_rows["task_order_index"] = np.arange(len(first_rows))
    return first_rows


def _enrich_with_nifti_metadata(
    task_table: pd.DataFrame,
    bids_base_path: Path,
    participant: str,
    bids_session: str,
) -> pd.DataFrame:
    try:
        import nibabel as nib  # type: ignore
    except Exception:
        task_table["n_volumes"] = np.nan
        task_table["duration"] = np.nan
        return task_table

    n_volumes: list[float] = []
    durations: list[float] = []

    for _, row in task_table.iterrows():
        rel_path = str(row["filename"]).lstrip("/")
        nifti_path = bids_base_path / participant / bids_session / rel_path
        if not nifti_path.exists():
            n_volumes.append(np.nan)
            durations.append(np.nan)
            continue

        try:
            img = nib.load(str(nifti_path))
            shape = img.shape
            tr = float(img.header.get_zooms()[3]) if len(img.header.get_zooms()) > 3 else np.nan
            n_vol = float(shape[3]) if len(shape) > 3 else np.nan
            n_volumes.append(n_vol)
            durations.append(float(n_vol * tr) if np.isfinite(tr) and np.isfinite(n_vol) else np.nan)
        except Exception:
            n_volumes.append(np.nan)
            durations.append(np.nan)

    task_table = task_table.copy()
    task_table["n_volumes"] = n_volumes
    task_table["duration"] = durations
    return task_table


def _match_scan(
    scans: list[dict[str, Any]],
    task_info: dict[str, Any] | None,
    time_tolerance_seconds: float,
) -> tuple[dict[str, Any], str]:
    if not scans:
        raise ValueError("No PMU scans detected")

    if not task_info:
        return scans[0], "fallback_first_scan"

    n_volumes = _safe_float(task_info.get("n_volumes"))
    duration = _safe_float(task_info.get("duration"))
    offset_seconds = _safe_float(task_info.get("offset_seconds"))
    task_order_index = task_info.get("task_order_index")

    # Most robust path when NIfTI metadata is available.
    if n_volumes and n_volumes > 0 and duration and duration > 0:
        candidates: list[tuple[float, dict[str, Any]]] = []
        for scan in scans:
            vol_error = abs(scan["n_volumes"] - n_volumes) / n_volumes
            dur_error = abs(scan["duration"] - duration) / duration
            if vol_error <= 0.12 and dur_error <= 0.20:
                score = (2.0 * vol_error) + dur_error
                candidates.append((score, scan))
        if candidates:
            candidates.sort(key=lambda item: item[0])
            return candidates[0][1], "properties"

    if offset_seconds is not None:
        best_scan = min(scans, key=lambda scan: abs(scan["start_time"] - offset_seconds))
        if abs(best_scan["start_time"] - offset_seconds) <= time_tolerance_seconds:
            return best_scan, "time_tolerance"

    if isinstance(task_order_index, (int, np.integer)):
        idx = int(task_order_index)
        if 0 <= idx < len(scans):
            return scans[idx], "task_order"

    if offset_seconds is not None:
        best_scan = min(scans, key=lambda scan: abs(scan["start_time"] - offset_seconds))
        return best_scan, "closest_time"

    return scans[0], "fallback_first_scan"


def _find_scanner_physio_dir(
    base_physio_path: Path,
    participant: str,
    pmu_sessions: list[str],
    folder_variants: tuple[str, ...] = DEFAULT_SCANNER_DIR_VARIANTS,
) -> tuple[Path | None, str | None, str | None]:
    for pmu_session in pmu_sessions:
        session_dir = base_physio_path / participant / pmu_session
        matched = _find_scanner_folder_in_session(session_dir, folder_variants=folder_variants)
        if matched is not None:
            return matched, pmu_session, matched.name
    return None, None, None


def _find_available_scanner_sessions(
    base_physio_path: Path,
    participant: str,
    folder_variants: tuple[str, ...] = DEFAULT_SCANNER_DIR_VARIANTS,
) -> list[str]:
    """Return participant sessions that contain a scanner PMU folder variant."""
    participant_root = base_physio_path / participant
    if not participant_root.exists():
        return []

    available_sessions: list[str] = []
    for ses_dir in sorted(participant_root.glob("ses-*")):
        if not ses_dir.is_dir():
            continue
        if _find_scanner_folder_in_session(ses_dir, folder_variants=folder_variants) is not None:
            available_sessions.append(ses_dir.name)
    return available_sessions


def extract_pmu_task_signals(
    *,
    base_physio_path: str | Path,
    bids_base_path: str | Path,
    participant: str,
    session: str,
    task: str,
    pmu_session: str | None = None,
    bids_session: str | None = None,
    sampling_rate: float = 400.0,
    scan_gap_seconds: float = 10.0,
    time_tolerance_seconds: float = 30.0,
) -> dict[str, Any]:
    """Extract PMU respiration/pulse arrays for one participant-session-task."""
    participant_label = participant if str(participant).startswith("sub-") else f"sub-{participant}"
    base_physio_path = Path(base_physio_path)
    bids_base_path = Path(bids_base_path)
    task_name = _normalize_task_name(task)

    pmu_sessions = []
    if pmu_session:
        pmu_sessions.append(pmu_session)
    pmu_sessions.append(infer_pmu_session_label(session))
    pmu_sessions = list(dict.fromkeys(pmu_sessions))

    physio_dir, resolved_pmu_session, resolved_pmu_folder = _find_scanner_physio_dir(
        base_physio_path,
        participant_label,
        pmu_sessions,
    )
    if physio_dir is None:
        available_sessions = _find_available_scanner_sessions(base_physio_path, participant_label)
        if available_sessions:
            availability_msg = f" Available scanner PMU sessions for this participant: {', '.join(available_sessions)}."
        else:
            availability_msg = " No scanner PMU folders were found for this participant."
        return {
            "success": False,
            "message": (
                f"No PMU scanner folder found for {participant_label} "
                f"in sessions: {', '.join(pmu_sessions)}.{availability_msg}"
            ),
        }

    resp_files = sorted(physio_dir.glob("*.resp"))
    puls_files = sorted(physio_dir.glob("*.puls"))
    ext_files = sorted(physio_dir.glob("*.ext"))
    if not (resp_files and puls_files and ext_files):
        return {
            "success": False,
            "message": f"Missing PMU files in {physio_dir}",
        }

    try:
        resp_data = parse_pmu_file(resp_files[0])
        puls_data = parse_pmu_file(puls_files[0])
        ext_data = parse_pmu_file(ext_files[0])
    except PermissionError as e:
        return {
            "success": False,
            "message": f"Permission denied reading PMU file: {e.filename}",
        }

    scans = identify_scans_from_volume_markers(
        ext_data["volume_markers"],
        sampling_rate=sampling_rate,
        min_gap_seconds=scan_gap_seconds,
    )
    if not scans:
        return {"success": False, "message": "No PMU scan windows detected from .ext markers"}

    bids_sessions = []
    if bids_session:
        bids_sessions.append(bids_session)
    bids_sessions.append(infer_bids_session_label(session))
    bids_sessions = list(dict.fromkeys(bids_sessions))

    task_info = None
    resolved_bids_session = None
    for candidate_bids_session in bids_sessions:
        task_table = _load_bids_task_table(bids_base_path, participant_label, candidate_bids_session)
        if task_table is None:
            continue
        task_table = _enrich_with_nifti_metadata(task_table, bids_base_path, participant_label, candidate_bids_session)

        exact = task_table[task_table["task_normalized"] == task_name]
        if exact.empty:
            partial = task_table[
                task_table["task_normalized"].str.contains(task_name, regex=False)
                | pd.Series([task_name in x for x in task_table["task_normalized"]])
            ]
            task_row = partial.iloc[0].to_dict() if not partial.empty else None
        else:
            task_row = exact.iloc[0].to_dict()

        if task_row is not None:
            task_info = task_row
            resolved_bids_session = candidate_bids_session
            break

    if task_info is None and len(scans) > 1:
        return {
            "success": False,
            "message": (
                f"Task '{task_name}' not matched in BIDS scans.tsv; "
                "multiple PMU scans found so integration was skipped to avoid wrong alignment."
            ),
        }

    matched_scan, strategy = _match_scan(scans, task_info, time_tolerance_seconds)

    start = max(0, int(matched_scan["start_sample"]))
    end = int(matched_scan["end_sample"])
    if end <= start:
        return {"success": False, "message": "Invalid PMU extraction window"}

    resp_segment = resp_data["values"][start:end]
    puls_segment = puls_data["values"][start:end]
    n = min(len(resp_segment), len(puls_segment))
    if n <= 1:
        return {"success": False, "message": "Extracted PMU segment is empty"}

    resp_segment = resp_segment[:n]
    puls_segment = puls_segment[:n]

    return {
        "success": True,
        "message": f"PMU matched with strategy: {strategy}",
        "rsp": resp_segment,
        "ppg": puls_segment,
        "pmu_sampling_rate": float(sampling_rate),
        "match_strategy": strategy,
        "scan_index": int(matched_scan["scan_index"]) + 1,
        "scan_start_time_sec": float(matched_scan["start_time"]),
        "scan_duration_sec": float(matched_scan["duration"]),
        "resolved_pmu_session": resolved_pmu_session,
        "resolved_pmu_folder": resolved_pmu_folder,
        "resolved_bids_session": resolved_bids_session,
    }


def resample_signal_to_length(signal: np.ndarray, target_length: int) -> np.ndarray:
    """Linearly resample one signal to a target sample count."""
    if target_length <= 0:
        return np.array([], dtype=float)

    signal = np.asarray(signal, dtype=float)
    if len(signal) == 0:
        return np.zeros(target_length, dtype=float)
    if len(signal) == target_length:
        return signal.copy()
    if len(signal) == 1:
        return np.full(target_length, signal[0], dtype=float)

    old_x = np.linspace(0.0, 1.0, len(signal))
    new_x = np.linspace(0.0, 1.0, target_length)
    return np.interp(new_x, old_x, signal).astype(float)
