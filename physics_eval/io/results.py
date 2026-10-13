"""Physical estimation result schema and table serialization."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ..reporting import save_trajectory_plot

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


def scene_name_for_row(row: dict[str, Any]) -> str:
    prompt_id = str(row.get("Prompt_ID") or "").strip()
    if prompt_id:
        return prompt_id
    return f"index_{int(row.get('Index')):04d}"


def slug_value(value: Any, fallback: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        raw = fallback
    out = []
    for ch in raw:
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out).strip("_")
    return slug or fallback


def metric_dir_name(row: dict[str, Any]) -> str:
    return slug_value(row.get("Metric") or row.get("Evaluator"), "unknown_metric")


def prompt_dir_name(row: dict[str, Any]) -> str:
    return slug_value(row.get("Prompt_ID"), scene_name_for_row(row))


def per_video_output_dir(output_dir: Path, row: dict[str, Any], layout: str) -> Path:
    if layout == "flat":
        return output_dir
    return output_dir / metric_dir_name(row) / prompt_dir_name(row)


def output_tracking_json(output_dir: Path, row: dict[str, Any], layout: str = "organized") -> Path:
    if layout == "flat":
        return output_dir / "tracking_json" / f"{scene_name_for_row(row)}.json"
    return per_video_output_dir(output_dir, row, layout) / "tracking_points.json"


def write_resolved_metadata(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    path = output_dir / "metadata_with_resolved_tracking.xlsx"
    pd.DataFrame(rows).to_excel(path, index=False)
    return path


def json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def parse_extra_json(result: dict[str, Any]) -> dict[str, Any]:
    raw = result.get("Extra_JSON")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"raw_extra_json": raw}
    return parsed if isinstance(parsed, dict) else {"extra": parsed}


def _embed_initialization_provenance(path: Path, provenance: dict[str, Any]) -> None:
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["batch_initialization"] = provenance
    if path.name == "run_config.json":
        payload["semantic_verification"] = provenance.get("semantic_verification")
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def write_per_video_result(
    output_dir: Path,
    row: dict[str, Any],
    result: dict[str, Any],
    tracking_json: Path | None,
    area_summaries: list[dict[str, Any]],
    area_paths: dict[str, Any],
    args: argparse.Namespace,
) -> Path | None:
    if args.output_layout != "organized":
        return None

    row_dir = per_video_output_dir(output_dir, row, args.output_layout)
    row_dir.mkdir(parents=True, exist_ok=True)
    trajectory_plot = save_trajectory_plot(row_dir, tracking_json, args.fps) if tracking_json is not None and tracking_json.exists() else None
    extra = parse_extra_json(result)
    tracking_quality = None
    tracking_initialization = None
    if tracking_json is not None and tracking_json.exists():
        try:
            tracking_data = json.loads(tracking_json.read_text(encoding="utf-8"))
            tracking_quality = (tracking_data.get("quality_checks") or {}).get("tracking_quality")
            tracking_initialization = tracking_data.get("batch_initialization")
        except Exception:
            tracking_quality = None
            tracking_initialization = None
    if tracking_initialization is None:
        raw_initialization = row.get("Resolved_Tracking_Initialization_JSON")
        if isinstance(raw_initialization, str) and raw_initialization:
            try:
                tracking_initialization = json.loads(raw_initialization)
            except json.JSONDecodeError:
                tracking_initialization = {"status": "invalid_provenance_json"}

    result_compact = {k: v for k, v in result.items() if k != "Extra_JSON"}
    sam2_scene_dir = Path(args.workspace_root) / "sam2_tracks" / scene_name_for_row(row)
    local_vis = row_dir / "tracking_vis.mp4"
    source_vis = sam2_scene_dir / "tracking_vis.mp4"
    local_debug = row_dir / "auto_init_debug.jpg"
    source_debug = sam2_scene_dir / "auto_init_debug.jpg"
    artifact_paths = {
        "tracking_json": str(tracking_json) if tracking_json is not None else None,
        "mask_qc_json": area_paths.get("area_report_json"),
        "trajectory_plot": str(trajectory_plot) if trajectory_plot is not None else None,
        "tracking_visualization": str(local_vis if local_vis.exists() else source_vis) if (local_vis.exists() or source_vis.exists()) else None,
        "auto_init_debug": str(local_debug if local_debug.exists() else source_debug) if (local_debug.exists() or source_debug.exists()) else None,
        "resolved_init_prompts": str(row_dir / "resolved_init_prompts.json")
        if (row_dir / "resolved_init_prompts.json").exists()
        else None,
        "sam2_mask_directory": str(sam2_scene_dir / "masks"),
        "mask_qc_plots": [
            item.get("plot")
            for item in area_summaries
            if item.get("plot") is not None
        ],
    }

    payload = {
        "schema_version": 1,
        "index": row.get("Index"),
        "prompt_id": row.get("Prompt_ID"),
        "metric": row.get("Metric"),
        "video_file": row.get("Video_File"),
        "metadata": row,
        "physics_result": result_compact,
        "physics_extra": extra,
        "mask_qc": {
            "diagnostic_version": "main_mask_diagnostics_v1",
            "evidence_schema_version": 1,
            "evidence_source": "tracking_and_masks",
            "status": (
                "not_evaluated"
                if not area_summaries
                else "pass" if all(item.get("decision") == "keep" for item in area_summaries) else "flagged"
            ),
            "objects": area_summaries,
        },
        "tracking_quality": tracking_quality,
        "tracking_initialization": tracking_initialization,
        "semantic_verification": (
            tracking_initialization.get("semantic_verification")
            if isinstance(tracking_initialization, dict)
            else None
        ),
        "artifacts": artifact_paths,
    }
    path = row_dir / "result.json"
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return path
