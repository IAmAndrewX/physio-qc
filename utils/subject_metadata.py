"""
Helpers for loading participant-level metadata and experiment notes from CSV exports.
"""

from __future__ import annotations

import csv
import re
import xml.etree.ElementTree as ET
import zipfile
from datetime import date, datetime, time
from functools import lru_cache
from pathlib import Path
from typing import Any

import config

_MISSING_NOTE_TOKENS = {"", "na", "n/a", "nan", "none", "null", "?", "-"}

_SCHEDULE_CANCEL_KEYWORDS = (
    "cancel",
    "no-show",
    "withdrew",
    "withdrawn",
    "rescheduled",
)

_SESSION_A_TASK_NOTE_COLUMNS = {
    "rest": "Resting State - Notes",
    "gas": "Gas Test - Notes",
    "breath": "Breathing Task - Notes",
    "sts": "Supine-to-Stand - Notes",
    "valsalva": "Valsalva - Notes",
    "coldpress": "Cold Pressor - Notes",
}

_SESSION_B_TASK_NOTE_COLUMNS = {
    "rest": "Resting State - Notes",
    "gas": "Gas Test - Notes",
    "breath": "Breathing Task - Notes",
}

_XLSX_NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_XLSX_NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"

_XLSX_NS = {"a": _XLSX_NS_MAIN}
_XLSX_CELL_REF_RE = re.compile(r"([A-Z]+)")


def normalize_participant_id(value: Any) -> str:
    """Normalize participant identifiers to sub-xxxx lowercase format when possible."""
    if value is None:
        return ""
    token = str(value).strip().lower()
    if not token:
        return ""
    token = token.replace(" ", "")
    if token.startswith("sub-"):
        return token
    if token.startswith("sub"):
        return f"sub-{token[3:].lstrip('-_')}"
    return token


def infer_session_class(session_label: Any) -> str | None:
    """Map session labels to A/B classes."""
    if session_label is None:
        return None
    token = str(session_label).strip().lower().replace("_", "-")
    for prefix in ("session-", "session", "ses-"):
        if token.startswith(prefix):
            token = token[len(prefix):]
            break
    token = token.strip("- ")
    if not token:
        return None
    if token.startswith("a"):
        return "A"
    if token.startswith("b"):
        return "B"
    if token in {"1", "01"}:
        return "A"
    if token in {"2", "02"}:
        return "B"
    return None


def normalize_task_key(task_name: Any) -> str | None:
    """Map a task name from filenames/selectors to canonical task keys."""
    if task_name is None:
        return None
    normalised = str(task_name).strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    if not normalised:
        return None

    key = config.TASK_EVENT_ALIASES.get(normalised)
    if key:
        return key

    for alias, mapped in config.TASK_EVENT_ALIASES.items():
        if alias in normalised or normalised in alias:
            return mapped

    if "valsalva" in normalised:
        return "valsalva"
    if "coldpress" in normalised or "coldpressor" in normalised:
        return "coldpress"
    return None


def _xlsx_column_index_from_ref(cell_ref: str) -> int | None:
    """Convert an XLSX cell reference (e.g., 'C12') into 0-based column index."""
    match = _XLSX_CELL_REF_RE.match(str(cell_ref or ""))
    if not match:
        return None
    letters = match.group(1)
    index = 0
    for char in letters:
        index = (index * 26) + (ord(char) - ord("A") + 1)
    return index - 1


def _xlsx_cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    value_node = cell.find(f"{{{_XLSX_NS_MAIN}}}v")
    if value_node is None:
        inline_node = cell.find(f"{{{_XLSX_NS_MAIN}}}is/{{{_XLSX_NS_MAIN}}}t")
        return (inline_node.text or "").strip() if inline_node is not None else ""

    raw = (value_node.text or "").strip()
    if cell.attrib.get("t") == "s":
        try:
            return str(shared_strings[int(raw)]).strip()
        except Exception:
            return raw
    return raw


def _xlsx_row_to_cells(row: ET.Element, shared_strings: list[str]) -> dict[int, str]:
    values: dict[int, str] = {}
    fallback_index = 0
    for cell in row.findall(f"{{{_XLSX_NS_MAIN}}}c"):
        col_index = _xlsx_column_index_from_ref(cell.attrib.get("r", ""))  # e.g., A3/B3
        if col_index is None:
            col_index = fallback_index
        values[col_index] = _xlsx_cell_value(cell, shared_strings)
        fallback_index = max(fallback_index + 1, col_index + 1)
    return values


def _normalize_definition_field_name(field_name: str) -> str:
    return str(field_name or "").strip().lower()


def _normalize_choice_key(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).strip().lower()
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else str(value)

    text = str(value).strip()
    if re.fullmatch(r"[-+]?\d+\.0+", text):
        try:
            return str(int(float(text)))
        except Exception:
            return text
    return text


def _field_matches_numeric_range(field_name: str, range_start: str, range_end: str) -> bool:
    """Match field against REDCap range labels like `phq9_1 - phq9_9`."""
    field_match = re.fullmatch(r"([a-z_]+)(\d+)", field_name)
    start_match = re.fullmatch(r"([a-z_]+)(\d+)", range_start)
    end_match = re.fullmatch(r"([a-z_]+)(\d+)", range_end)
    if not (field_match and start_match and end_match):
        return False
    if not (
        field_match.group(1) == start_match.group(1) == end_match.group(1)
    ):
        return False

    field_num = int(field_match.group(2))
    start_num = int(start_match.group(2))
    end_num = int(end_match.group(2))
    return min(start_num, end_num) <= field_num <= max(start_num, end_num)


def _lookup_redcap_definition_entry(
    field_name: str,
    definitions_map: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    key = _normalize_definition_field_name(field_name)
    if not key:
        return None

    if key in definitions_map:
        return definitions_map[key]

    for candidate_key, candidate_entry in definitions_map.items():
        if " - " not in candidate_key:
            continue
        left, right = [part.strip() for part in candidate_key.split(" - ", 1)]
        if _field_matches_numeric_range(key, left, right):
            return candidate_entry
    return None


def _load_redcap_definitions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    with zipfile.ZipFile(path) as workbook_zip:
        workbook = ET.fromstring(workbook_zip.read("xl/workbook.xml"))
        rels = ET.fromstring(workbook_zip.read("xl/_rels/workbook.xml.rels"))

        rel_map = {
            rel.attrib["Id"]: rel.attrib.get("Target", "")
            for rel in rels.findall(f"{{{_PKG_REL_NS}}}Relationship")
        }

        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in workbook_zip.namelist():
            shared_root = ET.fromstring(workbook_zip.read("xl/sharedStrings.xml"))
            for item in shared_root.findall("a:si", _XLSX_NS):
                pieces = [
                    text_node.text or ""
                    for text_node in item.iter(f"{{{_XLSX_NS_MAIN}}}t")
                ]
                shared_strings.append("".join(pieces))

        output: dict[str, dict[str, Any]] = {}
        for sheet in workbook.findall("a:sheets/a:sheet", _XLSX_NS):
            sheet_name = sheet.attrib.get("name", "").strip()
            rid = sheet.attrib.get(f"{{{_XLSX_NS_REL}}}id", "")
            target = rel_map.get(rid, "")
            if not target:
                continue
            sheet_path = target if target.startswith("xl/") else f"xl/{target}"
            if sheet_path not in workbook_zip.namelist():
                continue

            sheet_root = ET.fromstring(workbook_zip.read(sheet_path))
            rows = sheet_root.findall(".//a:sheetData/a:row", _XLSX_NS)
            if not rows:
                continue

            header_cells = _xlsx_row_to_cells(rows[0], shared_strings)
            if not header_cells:
                continue
            max_col = max(header_cells.keys())
            headers = [str(header_cells.get(i, "")).strip() for i in range(max_col + 1)]

            var_idx = None
            if "Var" in headers:
                var_idx = headers.index("Var")
            else:
                for idx, header in enumerate(headers):
                    if "Variable" in header:
                        var_idx = idx
                        break
            if var_idx is None:
                continue

            value_idx = headers.index("Value") if "Value" in headers else None
            field_type_idx = headers.index("Field Type") if "Field Type" in headers else None

            definition_idx = None
            for idx, header in enumerate(headers):
                if "Definition" in header:
                    definition_idx = idx
                    break
            if definition_idx is None and len(headers) > 3:
                definition_idx = 3

            current_field = ""
            for row in rows[1:]:
                cells = _xlsx_row_to_cells(row, shared_strings)
                field_name = str(cells.get(var_idx, "")).strip()
                if field_name:
                    current_field = field_name
                if not current_field:
                    continue

                normalized_field = _normalize_definition_field_name(current_field)
                if not normalized_field:
                    continue

                entry = output.setdefault(
                    normalized_field,
                    {
                        "source_sheet": sheet_name,
                        "field_type": "",
                        "choices": {},
                    },
                )

                field_type = str(cells.get(field_type_idx, "")).strip() if field_type_idx is not None else ""
                if field_type:
                    entry["field_type"] = field_type

                value = str(cells.get(value_idx, "")).strip() if value_idx is not None else ""
                definition = str(cells.get(definition_idx, "")).strip() if definition_idx is not None else ""
                if value and definition:
                    entry["choices"][_normalize_choice_key(value)] = definition

        return output


@lru_cache(maxsize=8)
def _load_redcap_definitions_cached(path_text: str) -> dict[str, dict[str, Any]]:
    return _load_redcap_definitions(Path(path_text))


def _interpret_redcap_value(
    field_name: str,
    value: Any,
    definitions_map: dict[str, dict[str, Any]],
) -> str | None:
    if value is None:
        return None
    entry = _lookup_redcap_definition_entry(field_name, definitions_map)
    if not entry:
        return None
    return entry.get("choices", {}).get(_normalize_choice_key(value))


def _format_scale_anchors(
    field_name: str,
    definitions_map: dict[str, dict[str, Any]],
) -> str | None:
    entry = _lookup_redcap_definition_entry(field_name, definitions_map)
    if not entry:
        return None
    choices = entry.get("choices", {})
    if not isinstance(choices, dict) or len(choices) < 2:
        return None

    numeric_choices: list[tuple[float, str, str]] = []
    for raw_key, raw_value in choices.items():
        try:
            numeric_key = float(raw_key)
        except Exception:
            continue
        numeric_choices.append((numeric_key, str(raw_key), str(raw_value)))

    if len(numeric_choices) < 2:
        return None
    numeric_choices.sort(key=lambda item: item[0])

    low = numeric_choices[0]
    high = numeric_choices[-1]
    if low[1] == high[1]:
        return None
    return f"{low[1]}={low[2]}; {high[1]}={high[2]}"


def _build_interpreted_questionnaires(
    questionnaire_values: dict[str, Any],
    definitions_map: dict[str, dict[str, Any]],
) -> dict[str, dict[str, str]]:
    interpreted: dict[str, dict[str, str]] = {}
    for field_name, raw_value in questionnaire_values.items():
        details: dict[str, str] = {}
        interpreted_value = _interpret_redcap_value(field_name, raw_value, definitions_map)
        if interpreted_value:
            details["interpreted_value"] = interpreted_value

        anchors = _format_scale_anchors(field_name, definitions_map)
        if anchors:
            details["scale_anchors"] = anchors

        if details:
            interpreted[field_name] = details
    return interpreted


def _read_rows_with_fallback(path: Path) -> list[list[str]]:
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            with path.open("r", encoding=encoding, newline="") as f:
                return list(csv.reader(f))
        except UnicodeDecodeError:
            continue
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.reader(f))


def _coerce_scalar(value: Any) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lower = text.lower()
    if lower in {"na", "n/a", "nan", "null"}:
        return None
    if re.fullmatch(r"[-+]?\d+", text):
        try:
            return int(text)
        except Exception:
            return text
    if re.fullmatch(r"[-+]?\d*\.\d+", text):
        try:
            return float(text)
        except Exception:
            return text
    return text


def _clean_note(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in _MISSING_NOTE_TOKENS:
        return None
    return text or None


def _parse_iso_date(text: str) -> date | None:
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except Exception:
        return None


def _parse_datetime_from_string(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %I:%M %p",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
    ):
        try:
            return datetime.strptime(text, fmt)
        except Exception:
            continue
    return None


def _parse_date_value(value: Any, default_year: int = 2025) -> date | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    month_fixes = {
        "fev": "feb",
        "fév": "feb",
        "aout": "aug",
        "août": "aug",
        "dec": "dec",
        "déc": "dec",
        "avr": "apr",
        "mai": "may",
        "juin": "jun",
        "juillet": "jul",
        "sept": "sep",
        "octobre": "oct",
        "novembre": "nov",
    }
    normalized = text.lower()
    for src, dst in month_fixes.items():
        normalized = normalized.replace(src, dst)
    normalized = normalized.replace("  ", " ").strip()

    formats = [
        ("%Y-%m-%d", False),
        ("%Y/%m/%d", False),
        ("%d-%b-%Y", False),
        ("%d-%b-%y", False),
        ("%d-%b", True),
        ("%d/%m/%Y", False),
        ("%m/%d/%Y", False),
        ("%d/%m/%y", False),
        ("%m/%d/%y", False),
        ("%d-%m-%Y", False),
        ("%d-%m-%y", False),
    ]
    for fmt, needs_year in formats:
        try:
            dt = datetime.strptime(normalized, fmt)
            if needs_year:
                return date(default_year, dt.month, dt.day)
            return dt.date()
        except Exception:
            continue
    return None


def _parse_time_value(value: Any) -> time | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    # Excel-exported time values can appear as "1/12/1900 10:15"
    match = re.search(r"(\d{1,2}:\d{2}(?::\d{2})?\s*[APMapm]{0,2})$", text)
    if match:
        text = match.group(1).strip()

    for fmt in ("%I:%M %p", "%I:%M%p", "%H:%M", "%H:%M:%S", "%I%p"):
        try:
            return datetime.strptime(text, fmt).time()
        except Exception:
            continue
    return None


def _compose_datetime(date_raw: Any, time_raw: Any, default_year: int = 2025) -> datetime | None:
    parsed_date = _parse_date_value(date_raw, default_year=default_year)
    if parsed_date is None:
        return None
    parsed_time = _parse_time_value(time_raw) or time(0, 0)
    return datetime.combine(parsed_date, parsed_time)


def _split_researchers(raw_values: list[str]) -> list[str]:
    canonical = {
        "jack": "Jack",
        "sophia": "Sophia",
        "oren": "Oren",
        "amanda": "Amanda",
        "andrew": "Andrew",
        "michelle": "Michelle",
        "mary": "Mary",
        "nesrine": "Nesrine",
        "dina": "Dina",
    }

    out: list[str] = []
    for raw in raw_values:
        text = _clean_note(raw)
        if not text:
            continue
        text = re.sub(r"\([^)]*\)", "", text)
        text = re.sub(r"\band\b", "|", text, flags=re.IGNORECASE)
        for sep in ("/", "+", ",", ";"):
            text = text.replace(sep, "|")
        for token in (x.strip() for x in text.split("|")):
            if not token:
                continue
            key = token.lower()
            normalized = canonical.get(key, token.title())
            if normalized and normalized not in out:
                out.append(normalized)
    return out[:2]


def _load_schedule_entries(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {}
    rows = _read_rows_with_fallback(path)
    if not rows:
        return {}

    header = rows[0]
    idx = {name: i for i, name in enumerate(header)}

    def _value(row: list[str], key: str) -> str:
        pos = idx.get(key)
        if pos is None or pos >= len(row):
            return ""
        return str(row[pos]).strip()

    by_participant: dict[str, list[dict[str, Any]]] = {}
    for row in rows[1:]:
        participant = normalize_participant_id(_value(row, "ID "))
        if not participant:
            participant = normalize_participant_id(_value(row, "ID"))
        if not participant:
            continue

        session_raw = _value(row, "Session")
        session_class = infer_session_class(session_raw)
        notes = _clean_note(_value(row, "Notes"))
        researchers = _split_researchers(
            [
                _value(row, "Slot 1 (if ses-A : Doppler/Equipment Setup)"),
                _value(row, "Slot 2 (if ses-A : Computerized Tasks/Spirometry)"),
                _value(row, "Slot 3 (Shadowing)"),
            ]
        )
        dt = _compose_datetime(_value(row, "Date"), _value(row, "Time"))
        entry = {
            "participant": participant,
            "session_raw": session_raw,
            "session_class": session_class,
            "date_raw": _value(row, "Date"),
            "time_raw": _value(row, "Time"),
            "datetime": dt,
            "date": dt.date() if dt else _parse_date_value(_value(row, "Date")),
            "notes": notes,
            "researchers": researchers,
        }
        by_participant.setdefault(participant, []).append(entry)
    return by_participant


def _is_cancelled_without_staff(entry: dict[str, Any]) -> bool:
    note = (entry.get("notes") or "").lower()
    has_cancel_keyword = any(keyword in note for keyword in _SCHEDULE_CANCEL_KEYWORDS)
    return has_cancel_keyword and not entry.get("researchers")


def _select_schedule_entry(
    entries: list[dict[str, Any]],
    session_class: str | None,
) -> dict[str, Any] | None:
    if not entries:
        return None
    candidates = entries
    if session_class:
        by_session = [e for e in entries if e.get("session_class") == session_class]
        if by_session:
            candidates = by_session

    ordered = sorted(candidates, key=lambda e: e.get("datetime") or datetime.min)
    for entry in reversed(ordered):
        if not _is_cancelled_without_staff(entry):
            return entry
    return ordered[-1] if ordered else None


def _load_redcap_metadata(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = _read_rows_with_fallback(path)
    if not rows:
        return {}
    header = rows[0]
    idx = {name: i for i, name in enumerate(header)}

    def _value(row: list[str], key: str) -> str:
        pos = idx.get(key)
        if pos is None or pos >= len(row):
            return ""
        return str(row[pos]).strip()

    out: dict[str, dict[str, Any]] = {}
    for row in rows[1:]:
        participant = normalize_participant_id(_value(row, "redcap_survey_identifier"))
        if not participant:
            continue
        demographics_dt = _parse_datetime_from_string(_value(row, "demographics_timestamp"))
        consent_dt = _parse_datetime_from_string(_value(row, "consent_form_timestamp"))
        recording_dt = demographics_dt or consent_dt

        questionnaires: dict[str, Any] = {}
        for field in getattr(config, "CORE_QUESTIONNAIRE_FIELDS", []):
            questionnaires[field] = _coerce_scalar(_value(row, field))

        out[participant] = {
            "age": _coerce_scalar(_value(row, "age")),
            "sex_asab": _coerce_scalar(_value(row, "asab")),
            "gender": _coerce_scalar(_value(row, "gender")),
            "recording_datetime": recording_dt,
            "questionnaires": questionnaires,
        }
    return out


def _load_group_info(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = _read_rows_with_fallback(path)
    if not rows:
        return {}

    header = rows[0]
    participant_cols: dict[int, str] = {}
    for idx, col_name in enumerate(header):
        if idx < 3:
            continue
        participant = normalize_participant_id(col_name)
        if participant:
            participant_cols[idx] = participant

    out: dict[str, dict[str, Any]] = {}
    for row in rows[1:]:
        if len(row) < 2:
            continue
        variable = str(row[1]).strip()
        if not variable:
            continue
        for idx, participant in participant_cols.items():
            if idx >= len(row):
                continue
            value = _coerce_scalar(row[idx])
            if value is None:
                continue
            out.setdefault(participant, {})[variable] = value
    return out


def _find_header_row(rows: list[list[str]]) -> int | None:
    for i, row in enumerate(rows[:20]):
        for cell in row:
            if str(cell).strip().lower() == "id":
                return i
    return None


def _load_experiment_notes_entries(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {}
    rows = _read_rows_with_fallback(path)
    if not rows:
        return {}
    header_idx = _find_header_row(rows)
    if header_idx is None:
        return {}

    header = [str(x).strip() for x in rows[header_idx]]
    id_idx = next((i for i, col in enumerate(header) if col.lower() == "id"), None)
    if id_idx is None:
        return {}

    by_participant: dict[str, list[dict[str, Any]]] = {}
    for row in rows[header_idx + 1:]:
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        participant = normalize_participant_id(row[id_idx])
        if not participant:
            continue

        fields: dict[str, str] = {}
        additional: list[str] = []
        for i, col in enumerate(header):
            value = str(row[i]).strip() if i < len(row) else ""
            if col:
                fields[col] = value
            else:
                note_value = _clean_note(value)
                if note_value:
                    additional.append(note_value)

        notes: dict[str, str] = {}
        for col, value in fields.items():
            if "notes" in col.lower() or col == "Overall Notes":
                cleaned = _clean_note(value)
                if cleaned:
                    notes[col] = cleaned

        dt = _compose_datetime(fields.get("Date", ""), fields.get("Time", ""))
        date_only = dt.date() if dt else _parse_date_value(fields.get("Date", ""))
        note_count = len([v for v in notes.values() if v])

        entry = {
            "participant": participant,
            "date_raw": fields.get("Date", ""),
            "time_raw": fields.get("Time", ""),
            "datetime": dt,
            "date": date_only,
            "fields": fields,
            "notes": notes,
            "researchers": _split_researchers([fields.get("Researchers", "")]),
            "additional_session_notes": additional,
            "note_count": note_count,
        }
        by_participant.setdefault(participant, []).append(entry)
    return by_participant


def _select_notes_entry(
    entries: list[dict[str, Any]],
    target_date: date | None = None,
) -> dict[str, Any] | None:
    if not entries:
        return None

    candidates = entries
    if target_date:
        matched = [entry for entry in entries if entry.get("date") == target_date]
        if matched:
            candidates = matched

    ordered = sorted(
        candidates,
        key=lambda entry: (
            entry.get("datetime") or datetime.min,
            int(entry.get("note_count") or 0),
        ),
    )
    return ordered[-1] if ordered else None


def _resolve_neuropsych_summary(participant: str, neuro_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source = neuro_map.get(participant, {})
    if not source:
        return {}

    moca_subscores: dict[str, Any] = {}
    for key, value in source.items():
        if key.startswith("MoCA_") and key != "MoCA_Total":
            moca_subscores[key] = value

    core_tests: dict[str, Any] = {}
    for key in getattr(config, "CORE_NEUROPSYCH_FIELDS", []):
        if key in source:
            core_tests[key] = source[key]

    return {
        "NP_Date": source.get("NP_Date"),
        "MoCA_Total": source.get("MoCA_Total"),
        "MoCA_Subscores": moca_subscores,
        "CoreTests": core_tests,
    }


def _resolve_task_notes(entry: dict[str, Any] | None, session_class: str | None) -> dict[str, str]:
    if not entry:
        return {}
    columns = _SESSION_A_TASK_NOTE_COLUMNS if session_class == "A" else _SESSION_B_TASK_NOTE_COLUMNS
    notes = entry.get("notes", {})
    out: dict[str, str] = {}
    for task_key, column in columns.items():
        value = _clean_note(notes.get(column))
        if value:
            out[task_key] = value
    return out


def _append_note(existing: str | None, extra: str) -> str:
    if not existing:
        return extra
    return f"{existing}\n\n{extra}"


def _resolve_ecg_config(recording_date: date | None) -> str:
    if recording_date is None:
        return "Unknown"
    cutoff = _parse_iso_date(getattr(config, "ECG_CONFIG_CUTOFF_DATE", "2025-10-31"))
    if cutoff is None:
        return "Unknown"
    if recording_date >= cutoff:
        return getattr(config, "ECG_CONFIG_NEW_LABEL", "New")
    return getattr(config, "ECG_CONFIG_OLD_LABEL", "Old")


def _resolve_recording_date(
    schedule_entry: dict[str, Any] | None,
    notes_entry: dict[str, Any] | None,
    redcap_entry: dict[str, Any] | None,
) -> tuple[date | None, str]:
    if schedule_entry and schedule_entry.get("date"):
        return schedule_entry["date"], "schedule"
    if notes_entry and notes_entry.get("date"):
        return notes_entry["date"], "experiment_notes"
    if redcap_entry and redcap_entry.get("recording_datetime"):
        return redcap_entry["recording_datetime"].date(), "redcap_demographics_timestamp"
    return None, "unknown"


def build_subject_metadata(participant: str, session: str, task: str) -> dict[str, Any]:
    """
    Build one merged metadata payload for the selected participant/session/task.
    """
    participant_id = normalize_participant_id(participant)
    session_class = infer_session_class(session)

    definitions_path_text = str(getattr(config, "PHENOTYPE_REDCAP_DEFINITIONS_PATH", "")).strip()
    redcap_definitions_path = Path(definitions_path_text) if definitions_path_text else None
    definitions_load_error = None
    if redcap_definitions_path is not None:
        try:
            redcap_definitions = _load_redcap_definitions_cached(str(redcap_definitions_path))
        except Exception as exc:
            redcap_definitions = {}
            definitions_load_error = str(exc)
    else:
        redcap_definitions = {}

    redcap_data = _load_redcap_metadata(Path(config.PHENOTYPE_REDCAP_PATH))
    group_cc = _load_group_info(Path(config.PHENOTYPE_GROUP_INFO_CC_PATH))
    group_lc = _load_group_info(Path(config.PHENOTYPE_GROUP_INFO_LC_PATH))
    schedule_data = _load_schedule_entries(Path(config.PHENOTYPE_TESTING_SCHEDULE_PATH))
    notes_a_data = _load_experiment_notes_entries(Path(config.PHENOTYPE_NOTES_SESSION_A_PATH))
    notes_b_data = _load_experiment_notes_entries(Path(config.PHENOTYPE_NOTES_SESSION_B_PATH))

    neuro_map: dict[str, dict[str, Any]] = {}
    neuro_map.update(group_cc)
    neuro_map.update(group_lc)

    schedule_entry = _select_schedule_entry(schedule_data.get(participant_id, []), session_class=session_class)

    notes_map = notes_a_data if session_class == "A" else notes_b_data
    notes_entry = _select_notes_entry(
        notes_map.get(participant_id, []),
        target_date=schedule_entry.get("date") if schedule_entry else None,
    )

    redcap_entry = redcap_data.get(participant_id, {})
    questionnaire_values = redcap_entry.get("questionnaires", {})
    questionnaires_interpreted = _build_interpreted_questionnaires(
        questionnaire_values,
        redcap_definitions,
    )
    sex_asab_raw = redcap_entry.get("sex_asab")
    gender_raw = redcap_entry.get("gender")
    sex_asab_label = _interpret_redcap_value("asab", sex_asab_raw, redcap_definitions)
    gender_label = _interpret_redcap_value("gender", gender_raw, redcap_definitions)

    recording_date, recording_date_source = _resolve_recording_date(schedule_entry, notes_entry, redcap_entry)
    ecg_configuration = _resolve_ecg_config(recording_date)

    researchers: list[str] = []
    if schedule_entry and schedule_entry.get("researchers"):
        researchers = list(schedule_entry["researchers"])
    elif notes_entry and notes_entry.get("researchers"):
        researchers = list(notes_entry["researchers"])

    task_notes = _resolve_task_notes(notes_entry, session_class=session_class)

    valsalva_cutoff = _parse_iso_date(getattr(config, "VALSALVA_OLD_SETUP_CUTOFF_DATE", "2025-10-02"))
    if session_class == "A" and valsalva_cutoff and recording_date:
        if recording_date <= valsalva_cutoff:
            setup_note = getattr(config, "VALSALVA_OLD_SETUP_NOTE", "").strip()
        else:
            setup_note = getattr(config, "VALSALVA_NEW_SETUP_NOTE", "").strip()
        if setup_note:
            task_notes["valsalva"] = _append_note(task_notes.get("valsalva"), setup_note)

    overall_notes = None
    setup_notes = None
    mri_notes = None
    additional_notes: list[str] = []
    if notes_entry:
        overall_notes = _clean_note(notes_entry["fields"].get("Overall Notes"))
        setup_notes = _clean_note(notes_entry["notes"].get("Setup - Notes"))
        mri_notes = _clean_note(notes_entry["notes"].get("MRI Notes"))
        additional_notes = list(notes_entry.get("additional_session_notes", []))

    schedule_notes = _clean_note(schedule_entry.get("notes") if schedule_entry else None)
    session_notes_parts = [part for part in (setup_notes, schedule_notes) if part]
    session_notes = "\n\n".join(session_notes_parts) if session_notes_parts else None

    result = {
        "participant": participant_id,
        "session": session,
        "session_class": session_class,
        "task": task,
        "recording_date": recording_date.isoformat() if recording_date else None,
        "recording_date_source": recording_date_source,
        "sex_asab": sex_asab_raw,
        "sex_asab_label": sex_asab_label,
        "gender": gender_raw,
        "gender_label": gender_label,
        "age": redcap_entry.get("age"),
        "ecg_configuration": ecg_configuration,
        "researchers": researchers,
        "neuropsych": _resolve_neuropsych_summary(participant_id, neuro_map),
        "questionnaires": questionnaire_values,
        "questionnaires_interpreted": questionnaires_interpreted,
        "experiment_notes": {
            "overall_notes": overall_notes,
            "session_notes": session_notes,
            "setup_notes": setup_notes,
            "schedule_notes": schedule_notes,
            "task_notes": task_notes,
            "mri_notes": mri_notes,
            "additional_session_notes": additional_notes,
        },
        "sources": {
            "redcap_found": participant_id in redcap_data,
            "redcap_definitions_found": bool(redcap_definitions),
            "redcap_definitions_error": definitions_load_error,
            "neuropsych_found": participant_id in neuro_map,
            "schedule_found": schedule_entry is not None,
            "session_notes_found": notes_entry is not None,
        },
        "definition_sources": {
            "redcap_definitions_path": str(redcap_definitions_path) if redcap_definitions_path is not None else None,
        },
    }
    return result
