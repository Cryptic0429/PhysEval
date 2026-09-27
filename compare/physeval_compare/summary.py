from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import MODE_SPECS
from .io_utils import dump_json, json_safe
from .metrics import compare_tracking_files


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize paired PhysEval comparison outputs.")
    parser.add_argument("--result-root", default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def _resolve_model_root(result_root: Path, model_name: str | None) -> tuple[Path, str]:
    if model_name:
        return result_root / model_name, model_name
    if any((result_root / mode).is_dir() for mode in MODE_SPECS):
        return result_root, result_root.name
    candidates = [
        path for path in result_root.iterdir()
        if path.is_dir() and any((path / mode).is_dir() for mode in MODE_SPECS)
    ] if result_root.exists() else []
    if len(candidates) != 1:
        raise RuntimeError("Pass --model-name when result-root contains zero or multiple model directories.")
    return candidates[0], candidates[0].name


def _read_results(model_root: Path) -> tuple[pd.DataFrame, dict[tuple[str, str, str], dict[str, Any]]]:
    rows = []
    indexed = {}
    for mode_name in MODE_SPECS:
        for path in (model_root / mode_name).glob("*/*/result.json"):
            result = json.loads(path.read_text(encoding="utf-8"))
            metadata = result.get("metadata") or {}
            prompt_id = str(metadata.get("Prompt_ID") or path.parent.name)
            metric = str(metadata.get("Metric") or path.parent.parent.name)
            key = (mode_name, metric, prompt_id)
            indexed[key] = result
            tracking = result.get("tracking_quality") or {}
            mask_qc = result.get("mask_qc") or {}
            provenance = result.get("prompt_provenance") or {}
            rows.append({
                "qmask_protocol_id": mask_qc.get("protocol_id", "compare_legacy_v1") if result.get("status") == "ok" else None,
                "qmask_protocol_hash": mask_qc.get("protocol_hash"),
                "mode": mode_name,
                "label": MODE_SPECS[mode_name].label,
                "metric": metric,
                "prompt_id": prompt_id,
                "status": result.get("status"),
                "success": result.get("status") == "ok",
                "tracking_pass": tracking.get("status") == "pass",
                "coverage": tracking.get("coverage"),
                "motion_extent_norm_by_diameter": tracking.get("motion_extent_norm_by_diameter"),
                "mask_quality_score": mask_qc.get("score"),
                "mask_qc_pass": mask_qc.get("status") == "pass",
                "used_motion_fallback": provenance.get("used_motion_fallback"),
                "detector_used": provenance.get("detector_used"),
                "result_path": str(path.resolve()),
                "tracking_path": (result.get("artifacts") or {}).get("tracking_json"),
                "error": result.get("error"),
            })
    return pd.DataFrame(rows), indexed


def _rate_ci(successes: int, total: int) -> tuple[float | None, float | None]:
    if total == 0:
        return None, None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = z * np.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) / denominator
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def _mode_summary(frame: pd.DataFrame) -> pd.DataFrame:
    output = []
    for mode_name, group in frame.groupby("mode", sort=False):
        total = len(group)
        successful = group[group["success"]]
        tracking_passes = int(successful["tracking_pass"].sum())
        low, high = _rate_ci(tracking_passes, total)
        fallback_values = successful["used_motion_fallback"].dropna().astype(bool)
        output.append({
            "mode": mode_name,
            "label": MODE_SPECS[mode_name].label,
            "n_total": total,
            "n_success": len(successful),
            "run_success_rate": len(successful) / total if total else None,
            "tracking_pass_rate_all": tracking_passes / total if total else None,
            "tracking_pass_wilson_low": low,
            "tracking_pass_wilson_high": high,
            "motion_fallback_rate": float(fallback_values.mean()) if len(fallback_values) else None,
            "median_coverage": successful["coverage"].median(),
            "median_motion_extent_norm_by_diameter": successful["motion_extent_norm_by_diameter"].median(),
            "median_mask_quality_score": successful["mask_quality_score"].median(),
            "mask_qc_pass_rate": successful["mask_qc_pass"].mean() if len(successful) else None,
        })
    return pd.DataFrame(output)


def _bootstrap_mean_ci(values: np.ndarray, iterations: int, rng: np.random.Generator) -> tuple[float | None, float | None]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None, None
    if len(values) == 1:
        return float(values[0]), float(values[0])
    sampled = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    low, high = np.percentile(sampled, [2.5, 97.5])
    return float(low), float(high)


def _pairwise(
    frame: pd.DataFrame,
    *,
    iterations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    indexed = {
        (row.mode, row.metric, row.prompt_id): row
        for row in frame.itertuples(index=False)
    }
    pairs = [
        ("fasterrcnn_sam2", "yolo_sam2"),
        ("yolo_tam", "yolo_sam2"),
    ]
    per_video = []
    for left_mode, baseline_mode in pairs:
        left_keys = {(metric, prompt) for mode, metric, prompt in indexed if mode == left_mode}
        right_keys = {(metric, prompt) for mode, metric, prompt in indexed if mode == baseline_mode}
        for metric, prompt_id in sorted(left_keys & right_keys):
            left = indexed[(left_mode, metric, prompt_id)]
            baseline = indexed[(baseline_mode, metric, prompt_id)]
            row = {
                "alternative_mode": left_mode,
                "baseline_mode": baseline_mode,
                "metric": metric,
                "prompt_id": prompt_id,
                "alternative_success": bool(left.success),
                "baseline_success": bool(baseline.success),
                "tracking_pass_difference": float(left.tracking_pass) - float(baseline.tracking_pass),
                "mask_quality_difference": _difference(left.mask_quality_score, baseline.mask_quality_score),
            }
            left_tracking_path = _optional_path(left.tracking_path)
            baseline_tracking_path = _optional_path(baseline.tracking_path)
            if left_tracking_path and baseline_tracking_path and left_tracking_path.exists() and baseline_tracking_path.exists():
                row.update(compare_tracking_files(left_tracking_path, baseline_tracking_path))
            per_video.append(row)
    per_video_frame = pd.DataFrame(per_video)
    aggregates = []
    rng = np.random.default_rng(seed)
    if not per_video_frame.empty:
        for (alternative, baseline), group in per_video_frame.groupby(["alternative_mode", "baseline_mode"]):
            aggregate = {
                "alternative_mode": alternative,
                "baseline_mode": baseline,
                "n_paired": len(group),
            }
            for column in (
                "tracking_pass_difference",
                "mask_quality_difference",
                "centroid_nrmse_by_diameter",
                "bbox_iou_mean",
                "mask_iou_mean",
            ):
                if column not in group:
                    continue
                values = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float)
                finite = values[np.isfinite(values)]
                aggregate[f"mean_{column}"] = float(np.mean(finite)) if len(finite) else None
                low, high = _bootstrap_mean_ci(values, iterations, rng)
                aggregate[f"{column}_bootstrap_low"] = low
                aggregate[f"{column}_bootstrap_high"] = high
            aggregates.append(aggregate)
    return per_video_frame, pd.DataFrame(aggregates)


def _difference(left: Any, right: Any) -> float | None:
    try:
        if pd.isna(left) or pd.isna(right):
            return None
        return float(left) - float(right)
    except (TypeError, ValueError):
        return None


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    raw = str(value).strip()
    return Path(raw) if raw else None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result_root = Path(args.result_root).expanduser().resolve()
    model_root, model_name = _resolve_model_root(result_root, args.model_name)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else model_root / "comparison_reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    per_run, _ = _read_results(model_root)
    if per_run.empty:
        raise RuntimeError(f"No comparison result.json files found below: {model_root}")
    versions = per_run["qmask_protocol_id"].dropna().unique()
    hashes = per_run["qmask_protocol_hash"].dropna().unique()
    if len(versions) > 1 or len(hashes) > 1:
        raise ValueError("Refusing to aggregate mixed Qmask protocols")
    if per_run.loc[per_run["success"], "mask_quality_score"].isna().any():
        raise ValueError("Successful tracking runs contain unavailable Qmask evidence; complete diagnostics before summarizing")
    mode_summary = _mode_summary(per_run)
    per_video_pairwise, pair_summary = _pairwise(
        per_run, iterations=max(100, args.bootstrap), seed=args.seed,
    )
    per_run.to_csv(output_dir / "per_run_results.csv", index=False)
    mode_summary.to_csv(output_dir / "mode_summary.csv", index=False)
    per_video_pairwise.to_csv(output_dir / "per_video_pairwise.csv", index=False)
    pair_summary.to_csv(output_dir / "pairwise_summary.csv", index=False)
    dump_json(output_dir / "comparison_summary.json", json_safe({
        "qmask_protocol_id": str(versions[0]) if len(versions) else None,
        "qmask_population": "tracking_runs_without_physics_gate",
        "model_name": model_name,
        "mode_summary": mode_summary.astype(object).where(pd.notna(mode_summary), None).to_dict(orient="records"),
        "pairwise_summary": pair_summary.astype(object).where(pd.notna(pair_summary), None).to_dict(orient="records"),
        "bootstrap_iterations": max(100, args.bootstrap),
        "bootstrap_seed": args.seed,
    }))
    print(f"Wrote comparison reports to: {output_dir}")
    return 0
