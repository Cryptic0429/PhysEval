"""Benchmark metadata loading, normalization, and row selection."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .video import discover_videos
from .reports import json_safe

def read_metadata(path: str | Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    required = [
        "Index",
        "Prompt_ID",
        "Video_File",
        "Tracking_JSON",
        "Module",
        "Metric",
        "Evaluator",
        "Measurement_Method",
        "Scale_Mode",
        "Target_Type",
        "Target_Value",
        "Target_Unit",
        "Known_Parameters_JSON",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"metadata is missing required columns: {missing}")
    return df


def clean_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return value


def row_to_dict(row: pd.Series) -> dict[str, Any]:
    return {str(k): clean_value(v) for k, v in row.to_dict().items()}


def parse_known_params(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("Known_Parameters_JSON")
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        raise ValueError("Known_Parameters_JSON must be JSON text or empty")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid Known_Parameters_JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Known_Parameters_JSON must decode to an object")
    return parsed


def load_metadata(path: Path | None, video_root: Path) -> list[dict[str, Any]]:
    if path is None:
        rows = []
        for index, video_path in enumerate(discover_videos(video_root), start=1):
            rows.append({
                "Index": index,
                "Prompt_ID": video_path.stem,
                "Video_File": video_path.name,
                "Metric": "tracking_only",
                "Num_Objects": 1,
                "Tracking_Objects": "object_center",
                "Calibration_Object": "none",
                "Known_Parameters_JSON": "{}",
            })
        return rows

    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    if path.suffix.lower() == ".csv":
        table = pd.read_csv(path)
    else:
        workbook = pd.ExcelFile(path)
        sheet = "metadata" if "metadata" in workbook.sheet_names else workbook.sheet_names[0]
        table = pd.read_excel(path, sheet_name=sheet)
    table = table.where(pd.notna(table), None)
    return [json_safe(row) for row in table.to_dict(orient="records")]


def filter_rows(
    rows: Iterable[dict[str, Any]],
    *,
    index: int | None,
    prompt_id: str | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if index is not None and int(row.get("Index") or -1) != index:
            continue
        if prompt_id is not None and str(row.get("Prompt_ID")) != prompt_id:
            continue
        selected.append(row)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def infer_target_hint(row: dict[str, Any]) -> str:
    known_raw = row.get("Known_Parameters_JSON")
    known: dict[str, Any] = {}
    if isinstance(known_raw, dict):
        known = known_raw
    elif known_raw:
        try:
            known = json.loads(str(known_raw))
        except json.JSONDecodeError:
            known = {}
    for key in ("object", "tracking_target", "target_object"):
        value = known.get(key)
        if value and str(value).lower() != "none":
            return str(value)
    calibration = row.get("Calibration_Object")
    if calibration and str(calibration).lower() != "none":
        return str(calibration)
    tracking = str(row.get("Tracking_Objects") or "object")
    return tracking.replace("_center", "")


def expected_objects(row: dict[str, Any]) -> int:
    try:
        return max(1, int(float(row.get("Num_Objects") or 1)))
    except (TypeError, ValueError):
        return 1
