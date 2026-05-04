from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


RESULT_COLUMNS = [
    "Index",
    "Prompt_ID",
    "Video_File",
    "Tracking_JSON",
    "Module",
    "Metric",
    "Evaluator",
    "Measurement_Method",
    "Target_Type",
    "Target_Value",
    "Target_Unit",
    "Measured_Value",
    "Measured_Unit",
    "Absolute_Error",
    "Relative_Error",
    "Fit_R2",
    "Scale_m_per_px",
    "Status",
    "Failure_Reason",
    "Extra_JSON",
]


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


def resolve_tracking_json(row: dict[str, Any], metadata_path: Path, json_root: Path | None) -> Path:
    raw = row.get("Tracking_JSON")
    if not raw:
        raise FileNotFoundError("Tracking_JSON is empty")

    candidate = Path(str(raw))
    candidates = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        if json_root is not None:
            candidates.append(json_root / candidate)
            candidates.append(json_root / candidate.name)
        candidates.append(metadata_path.parent / candidate)
        candidates.append(Path.cwd() / candidate)

    for path in candidates:
        if path.exists():
            return path.resolve()

    searched = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Tracking_JSON not found. searched: {searched}")


def make_base_result(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "Index": row.get("Index"),
        "Prompt_ID": row.get("Prompt_ID"),
        "Video_File": row.get("Video_File"),
        "Tracking_JSON": row.get("Tracking_JSON"),
        "Module": row.get("Module"),
        "Metric": row.get("Metric"),
        "Evaluator": row.get("Evaluator"),
        "Measurement_Method": row.get("Measurement_Method"),
        "Target_Type": row.get("Target_Type"),
        "Target_Value": row.get("Target_Value"),
        "Target_Unit": row.get("Target_Unit"),
        "Measured_Value": None,
        "Measured_Unit": None,
        "Absolute_Error": None,
        "Relative_Error": None,
        "Fit_R2": None,
        "Scale_m_per_px": None,
        "Status": "failed",
        "Failure_Reason": "",
        "Extra_JSON": "{}",
    }


def finish_result(
    row: dict[str, Any],
    measured_value: float | None,
    measured_unit: str | None,
    fit_r2: float | None,
    scale_m_per_px: float | None,
    status: str,
    failure_reason: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = make_base_result(row)
    result["Measured_Value"] = measured_value
    result["Measured_Unit"] = measured_unit
    result["Fit_R2"] = fit_r2
    result["Scale_m_per_px"] = scale_m_per_px
    result["Status"] = status
    result["Failure_Reason"] = failure_reason
    result["Extra_JSON"] = json.dumps(extra or {}, ensure_ascii=False, sort_keys=True)

    target = row.get("Target_Value")
    if measured_value is not None and isinstance(target, (int, float)) and math.isfinite(float(target)):
        abs_err = abs(float(measured_value) - float(target))
        result["Absolute_Error"] = abs_err
        denom = abs(float(target))
        result["Relative_Error"] = abs_err / denom if denom > 1e-12 else abs_err
    return result


def failure_result(row: dict[str, Any], reason: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    return finish_result(
        row=row,
        measured_value=None,
        measured_unit=None,
        fit_r2=None,
        scale_m_per_px=None,
        status="failed",
        failure_reason=reason,
        extra=extra,
    )


def write_results(results: list[dict[str, Any]], output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    for col in RESULT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[RESULT_COLUMNS]
    df.to_csv(out / "results.csv", index=False, encoding="utf-8-sig")
    (out / "results.json").write_text(
        json.dumps(df.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
