#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from physics_eval.utils.tracking_quality import analyze_tracking_quality_file
from physics_eval.quality.protocol import load_protocol, protocol_hash
from physics_eval.quality.scoring import compute_qmask
from physics_eval.quality.adapters import required_ids


DEFAULT_TOLERANCES = {
    "velocity": 0.30,
    "acceleration": 0.35,
    "gravity": 0.30,
    "friction": 0.35,
    "friction_coefficient": 0.35,
    "restitution": 0.30,
    "restitution_coefficient": 0.30,
    "density": 0.35,
    "density_acceleration": 0.35,
    "density_from_initial_acceleration": 0.35,
    "viscosity": 0.40,
    "fluid_viscosity": 0.40,
    "spring": 0.35,
    "spring_constant": 0.35,
    "energy": 0.35,
    "mechanical_energy_conservation": 0.35,
    "momentum": 0.35,
    "momentum_1d": 0.35,
    "momentum_conservation_1d": 0.35,
}

STATUS_MULTIPLIERS = {
    "valid": 1.0,
    "weak_valid": 0.8,
    "invalid": 0.0,
    "failed": 0.0,
}

PHYSICS_VALID_STATUSES = {"valid", "weak_valid"}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate compact per-video result.json files into metric/model scores."
    )
    parser.add_argument("--result-root", default="batch_eval_results",
                        help="Root containing <metric>/<prompt_id>/result.json")
    parser.add_argument("--scoring-protocol", default="grouped_v2_candidate", help="Built-in Qmask protocol name or JSON path")
    parser.add_argument("--metadata", default=str(WORKSPACE_ROOT / "benchmark/metadata/csv/phys_t2v_bench_metadata__metadata.csv"), help="Expected benchmark CSV/XLSX; use none for explicitly partial/custom scoring")
    parser.add_argument("--metric", default=None,
                        help="Only score one metric directory, e.g. friction_coefficient")
    parser.add_argument("--model-name", default=None,
                        help="Name written into the score summary")
    parser.add_argument("--output-dir", default=None,
                        help="Default: <result-root>/score_reports[/<metric>]")
    parser.add_argument("--tau", type=float, default=None,
                        help="Override tolerance for all metrics")
    parser.add_argument("--tolerance-json", default=None,
                        help="Optional JSON object mapping metric names to tau values")
    parser.add_argument("--penalty-multiplier", type=float, default=None,
                        help="Convenience multiplier for weak_valid; also used by legacy --mask-policy penalty")
    parser.add_argument("--weak-valid-multiplier", type=float, default=0.8,
                        help="Multiplier for physics status weak_valid. Default: 0.8 (paper protocol)")
    parser.add_argument("--mask-policy", choices=["continuous", "penalty", "hard_fail", "ignore"], default="continuous",
                        help="How to handle mask-QC. Default: continuous 0-1 mask quality score")
    parser.add_argument("--qc-flag-multiplier", type=float, default=0.7,
                        help="Legacy multiplier for mask-QC flagged videos when --mask-policy penalty. Default: 0.7")
    parser.add_argument("--strict-qc", action="store_true",
                        help="Alias for --mask-policy hard_fail")
    parser.add_argument("--tracking-policy", choices=["hard_fail", "penalty", "ignore"], default="hard_fail",
                        help="How to handle tracking-quality flags. Default: hard_fail")
    parser.add_argument("--quality-flag-multiplier", type=float, default=0.0,
                        help="Multiplier for tracking-quality flags when --tracking-policy penalty. Default: 0.0")
    parser.add_argument("--min-motion-extent-ratio", type=float, default=0.75,
                        help="Flag if primary object motion extent is below this many object diameters. Default: 0.75")
    parser.add_argument("--min-object-coverage", type=float, default=0.50,
                        help="Object must be valid in at least this fraction of frames to count. Default: 0.50")
    parser.add_argument("--recompute-tracking-quality", action="store_true",
                        help=("Recompute tracking-quality status from tracking_points.json with the requested "
                              "thresholds instead of reusing an embedded status. Required for threshold sensitivity."))
    parser.add_argument("--allow-contract-errors", action="store_true",
                        help="Collect evaluator_contract_error rows into an incomplete report instead of failing at the first row; no formal aggregate is published")
    parser.add_argument("--no-plots", action="store_true",
                        help="Do not generate plots")
    args = parser.parse_args()
    if args.penalty_multiplier is not None:
        args.weak_valid_multiplier = args.penalty_multiplier
        if args.mask_policy == "penalty":
            args.qc_flag_multiplier = args.penalty_multiplier
    if args.tau is not None and args.tau <= 0:
        parser.error("--tau must be positive")
    for name in ("penalty_multiplier", "weak_valid_multiplier", "qc_flag_multiplier", "quality_flag_multiplier"):
        value = getattr(args, name)
        if value is not None and not (0.0 <= value <= 1.0):
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 1")
    args.qmask_protocol = load_protocol(args.scoring_protocol)
    if args.qmask_protocol["missing_policy"] != "legacy" and (args.mask_policy != "continuous" or args.strict_qc or args.tracking_policy != "hard_fail"):
        parser.error("Candidate protocols require continuous masks and hard-fail tracking; no legacy policy overrides")
    return args


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_tolerances(args: argparse.Namespace) -> dict[str, float]:
    tolerances = dict(DEFAULT_TOLERANCES)
    if args.tolerance_json:
        custom = load_json(Path(args.tolerance_json).expanduser())
        if not isinstance(custom, dict):
            raise ValueError("--tolerance-json must point to a JSON object")
        for key, value in custom.items():
            tolerances[str(key)] = float(value)
    return tolerances


def discover_result_files(result_root: Path, metric: str | None) -> list[Path]:
    if metric:
        return sorted((result_root / metric).glob("*/result.json"))
    return sorted(result_root.glob("*/*/result.json"))


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def metric_key(result: dict[str, Any], result_path: Path) -> str:
    metric = result.get("metric")
    if metric:
        return str(metric)
    physics = result.get("physics_result", {}) or {}
    if physics.get("Metric"):
        return str(physics["Metric"])
    return result_path.parent.parent.name


def prompt_id(result: dict[str, Any], result_path: Path) -> str:
    return str(result.get("prompt_id") or result_path.parent.name)


def relative_error(physics: dict[str, Any]) -> float | None:
    rel = as_float(physics.get("Relative_Error"))
    if rel is not None:
        return rel

    measured = as_float(physics.get("Measured_Value"))
    if measured is None:
        return None

    measured_unit = str(physics.get("Measured_Unit") or "").lower()
    target_unit = str(physics.get("Target_Unit") or "").lower()
    target_type = str(physics.get("Target_Type") or "").lower()
    metric = str(physics.get("Metric") or "").lower()
    if "relative_error" in {measured_unit, target_unit}:
        return abs(measured)

    target = as_float(physics.get("Target_Value"))
    if target is None and (
        metric in {"energy", "mechanical_energy_conservation"}
        or target_type == "energy_conservation_ratio"
    ):
        target = 1.0
    if target is None:
        return None
    denom = max(abs(target), 1e-12)
    return abs(measured - target) / denom


def base_score(rel_error: float | None, tau: float) -> float | None:
    if rel_error is None or tau <= 0:
        return None
    return float(100.0 * math.exp(-rel_error / tau))


def mask_qc_status(result: dict[str, Any]) -> tuple[str, list[str], list[dict[str, Any]]]:
    mask_qc = result.get("mask_qc") or {}
    objects = mask_qc.get("objects") or []
    reasons: list[str] = []
    flagged = False
    for item in objects:
        decision = str(item.get("decision") or "")
        reason = str(item.get("reasons") or "")
        if decision and decision != "keep":
            flagged = True
        if reason:
            reasons.extend(part for part in reason.split(";") if part)
    if flagged or str(mask_qc.get("status") or "") == "flagged":
        return "flagged", sorted(set(reasons)), objects
    return "pass", [], objects


def mask_quality(result: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    qc_status, qc_reasons, objects = mask_qc_status(result)
    policy = "hard_fail" if args.strict_qc else args.mask_policy
    if policy == "ignore":
        return {
            "status": "pass",
            "score": 1.0,
            "reasons": qc_reasons,
            "objects": [],
        }
    if policy == "penalty":
        score = 1.0 if qc_status == "pass" else float(args.qc_flag_multiplier)
        return {
            "status": "pass" if score >= 1.0 else ("weak" if score > 0 else "fail"),
            "score": score,
            "reasons": qc_reasons,
            "objects": [],
        }
    if policy == "hard_fail":
        score = 1.0 if qc_status == "pass" else 0.0
        return {
            "status": "pass" if score >= 1.0 else "fail",
            "score": score,
            "reasons": qc_reasons,
            "objects": [],
        }

    protocol = getattr(args, "qmask_protocol", None) or load_protocol("grouped_v2_candidate")
    return compute_qmask(result.get("mask_qc") or {}, protocol, required_ids(result))


def tracking_path_from_result(result: dict[str, Any], result_path: Path) -> Path | None:
    artifacts = result.get("artifacts") or {}
    raw = artifacts.get("tracking_json")
    if raw:
        path = Path(str(raw)).expanduser()
        if path.exists():
            return path

    physics = result.get("physics_result", {}) or {}
    raw = physics.get("Tracking_JSON")
    if raw:
        path = Path(str(raw)).expanduser()
        if path.exists():
            return path

    fallback = result_path.parent / "tracking_points.json"
    return fallback if fallback.exists() else None


def expected_object_count(result: dict[str, Any]) -> int:
    metadata = result.get("metadata") or {}
    for key in ("Num_Objects", "Tracking_Objects"):
        try:
            value = int(metadata.get(key))
        except (TypeError, ValueError):
            continue
        return max(1, value)
    return 1


def analyze_tracking_quality(
    result: dict[str, Any],
    result_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not args.recompute_tracking_quality:
        embedded = result.get("tracking_quality")
        if isinstance(embedded, dict):
            return embedded

    tracking_path = tracking_path_from_result(result, result_path)
    expected_objects = expected_object_count(result)
    if tracking_path is None:
        return {
            "status": "flagged",
            "expected_objects": expected_objects,
            "actual_objects": 0,
            "reasons": ["tracking_json_missing"],
        }

    if not args.recompute_tracking_quality:
        try:
            data = load_json(tracking_path)
            quality = ((data.get("quality_checks") or {}).get("tracking_quality")
                       if isinstance(data, dict) else None)
            if isinstance(quality, dict):
                return quality
        except Exception:
            pass

    return analyze_tracking_quality_file(
        tracking_path,
        expected_objects=expected_objects,
        min_motion_extent_ratio=args.min_motion_extent_ratio,
        min_object_coverage=args.min_object_coverage,
    )


def multiplier_for_tracking(status: str, args: argparse.Namespace) -> float:
    if status == "pass" or args.tracking_policy == "ignore":
        return 1.0
    if args.tracking_policy == "hard_fail":
        return 0.0
    return float(args.quality_flag_multiplier)


def multiplier_for_status(status: str, args: argparse.Namespace) -> float:
    if status == "valid":
        return 1.0
    if status == "weak_valid":
        return float(args.weak_valid_multiplier)
    if status in {"invalid", "failed"}:
        return 0.0
    return 0.0


def status_multipliers_for_report(args: argparse.Namespace) -> dict[str, float]:
    return {
        "valid": 1.0,
        "weak_valid": float(args.weak_valid_multiplier),
        "invalid": 0.0,
        "failed": 0.0,
    }


def exclusion_reasons(
    both_gates_pass: bool,
    tracking_passes_selected_policy: bool,
    evaluator_contract_error: bool = False,
) -> list[str]:
    reasons: list[str] = []
    if not tracking_passes_selected_policy:
        reasons.append("tracking_quality_failed")
        return reasons
    if evaluator_contract_error:
        reasons.append("evaluator_contract_error")
    if not both_gates_pass:
        reasons.append("physics_status_not_valid")
    return reasons


def score_one(path: Path, tolerances: dict[str, float], args: argparse.Namespace) -> dict[str, Any]:
    result = load_json(path)
    physics = result.get("physics_result", {}) or {}
    metric = metric_key(result, path)
    tau = float(args.tau if args.tau is not None else tolerances.get(metric, 0.25))

    tracking_quality = analyze_tracking_quality(result, path, args)
    tracking_status = str(tracking_quality.get("status") or "flagged")
    tracking_mult = multiplier_for_tracking(tracking_status, args)
    tracking_passes_selected_policy = tracking_status == "pass" or tracking_mult > 0

    status = str(physics.get("Status") or "failed")
    physics_status_pass = status in PHYSICS_VALID_STATUSES
    tracking_quality_pass = tracking_status == "pass"
    tracking_problem = not tracking_quality_pass
    physics_fitting_problem = not physics_status_pass
    if tracking_problem and physics_fitting_problem:
        discard_overlap_category = "both_tracking_and_physics"
    elif tracking_problem:
        discard_overlap_category = "tracking_only"
    elif physics_fitting_problem:
        discard_overlap_category = "physics_only"
    else:
        discard_overlap_category = "neither"
    both_gates_pass = tracking_passes_selected_policy and physics_status_pass
    status_mult = multiplier_for_status(status, args) if tracking_passes_selected_policy else 0.0

    rel = None
    raw = None
    mask_info = {}
    mask_status = "not_evaluated"
    mask_score = None
    qc_reasons: list[str] = []
    mask_objects: list[dict[str, Any]] = []
    measurement_quality_mult = 0.0 if not tracking_passes_selected_policy else None
    measurement_quality_status = "fail" if not tracking_passes_selected_policy else "not_evaluated"

    # Q_mask is a measurement-quality score, not an attrition diagnostic.
    # It is evaluated only after both eligibility gates have passed.
    if both_gates_pass:
        rel = relative_error(physics)
        raw = base_score(rel, tau)

        mask_info = mask_quality(result, args)
        mask_status = str(mask_info.get("status") or "fail")
        mask_score = mask_info.get("score")
        qc_reasons = list(mask_info.get("reasons") or [])
        mask_objects = list(mask_info.get("objects") or [])

        measurement_quality_mult = tracking_mult * mask_score if mask_score is not None else None
        if measurement_quality_mult is None:
            measurement_quality_status = "score_unavailable"
        elif measurement_quality_mult <= 0:
            measurement_quality_status = "fail"
        elif tracking_status == "pass" and mask_status == "pass" and measurement_quality_mult >= 0.999:
            measurement_quality_status = "pass"
        else:
            measurement_quality_status = "weak"

    evaluator_contract_error = both_gates_pass and raw is None
    if evaluator_contract_error and not args.allow_contract_errors:
        raise ValueError(
            f"{path}: physics_result.Status is {status!r} and tracking passes the selected policy, "
            "but no score can be computed. An effective video must have Relative_Error or computable "
            "Measured_Value/Target_Value fields."
        )

    reasons = exclusion_reasons(
        both_gates_pass=both_gates_pass,
        tracking_passes_selected_policy=tracking_passes_selected_policy,
        evaluator_contract_error=evaluator_contract_error,
    )
    effective_for_score = not reasons
    adjusted = (
        raw * status_mult * tracking_mult * float(mask_score)
        if effective_for_score and raw is not None and mask_score is not None
        else 0.0
    )

    score_available = not both_gates_pass or (raw is not None and mask_score is not None)
    if both_gates_pass and not score_available:
        adjusted = None
    return {
        "score_available": score_available,
        "qmask_protocol_id": mask_info.get("protocol_id"),
        "qmask_protocol_hash": mask_info.get("protocol_hash"),
        "qmask_details": mask_info,
        "model": args.model_name or Path(args.result_root).expanduser().resolve().name,
        "metric": metric,
        "prompt_id": prompt_id(result, path),
        "result_json": str(path),
        "target_value": physics.get("Target_Value"),
        "measured_value": physics.get("Measured_Value"),
        "relative_error": rel,
        "tau": tau,
        "base_score": raw,
        "raw_score": raw,
        "status": status,
        "both_gates_pass": both_gates_pass,
        "q_mask_evaluated": both_gates_pass,
        "physics_status_pass": physics_status_pass,
        "physics_fitting_problem": physics_fitting_problem,
        "status_multiplier": status_mult,
        "estimation_validity_multiplier": status_mult,
        "mask_qc_status": mask_status,
        "mask_qc_pass": mask_status == "pass",
        "mask_quality_score": mask_score,
        "mask_qc_multiplier": mask_score,
        "tracking_quality_status": tracking_status,
        "tracking_quality_pass": tracking_quality_pass,
        "tracking_problem": tracking_problem,
        "tracking_quality_multiplier": tracking_mult,
        "measurement_quality_status": measurement_quality_status,
        "measurement_quality_multiplier": measurement_quality_mult,
        "effective_for_score": effective_for_score,
        "adjusted_score": adjusted,
        "expected_objects": tracking_quality.get("expected_objects"),
        "actual_objects": tracking_quality.get("actual_objects"),
        "motion_extent_norm_by_diameter": tracking_quality.get("motion_extent_norm_by_diameter"),
        "path_length_norm_by_diameter": tracking_quality.get("path_length_norm_by_diameter"),
        "failure_reason": physics.get("Failure_Reason"),
        "score_exclusion_reasons": ";".join(reasons),
        "mask_qc_reasons": ";".join(qc_reasons),
        "mask_quality_components_json": json.dumps(mask_objects, ensure_ascii=False, sort_keys=True),
        "tracking_quality_reasons": ";".join(tracking_quality.get("reasons") or []),
        "discard_overlap_category": discard_overlap_category,
    }


def numeric_values(values: list[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(val):
            out.append(val)
    return out


def mean(values: list[Any]) -> float | None:
    vals = numeric_values(values)
    return float(sum(vals) / len(vals)) if vals else None


def median(values: list[Any]) -> float | None:
    vals = sorted(numeric_values(values))
    if not vals:
        return None
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return float((vals[mid - 1] + vals[mid]) / 2.0)


def percentile(values: list[Any], q: float) -> float | None:
    vals = sorted(numeric_values(values))
    if not vals:
        return None
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return float(vals[lo] * (1.0 - frac) + vals[hi] * frac)


def count_reason_tokens(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        raw = str(row.get(key) or "")
        for token in raw.split(";"):
            if token:
                counts[token] += 1
    return dict(sorted(counts.items()))


def discard_overlap_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "tracking_only": 0,
        "physics_only": 0,
        "both_tracking_and_physics": 0,
        "neither": 0,
    }
    for row in rows:
        key = str(row.get("discard_overlap_category") or "neither")
        if key not in counts:
            key = "neither"
        counts[key] += 1
    return counts


def discard_overlap_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n_total = len(rows)
    counts = discard_overlap_counts(rows)
    return {
        "counts": counts,
        "rates": {key: count / max(n_total, 1) for key, count in counts.items()},
    }


def summarize_metric(metric: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    n_total = len(items)
    both_gates_pass_items = [r for r in items if r["both_gates_pass"]]
    physics_status_pass_items = [r for r in items if r["physics_status_pass"]]
    tracking_pass_items = [r for r in items if r["tracking_quality_pass"]]
    mask_pass_items = [r for r in items if r["mask_qc_pass"]]
    measurement_pass_items = [r for r in items if r["measurement_quality_status"] == "pass"]
    effective_items = [r for r in items if r["effective_for_score"]]

    adjusted_all = [float(r["adjusted_score"] or 0.0) for r in items]
    adjusted_effective = [r["adjusted_score"] for r in effective_items]
    base_effective = [r["base_score"] for r in effective_items]
    rel_effective = [r["relative_error"] for r in effective_items]
    rel_both_gates_pass = [
        r["relative_error"]
        for r in both_gates_pass_items
        if r["relative_error"] is not None
    ]

    status_counts = Counter(str(item["status"]) for item in items)
    qc_counts = Counter(str(item["mask_qc_status"]) for item in items)
    tracking_counts = Counter(str(item["tracking_quality_status"]) for item in items)
    measurement_counts = Counter(str(item["measurement_quality_status"]) for item in items)
    exclusion_counts = count_reason_tokens(items, "score_exclusion_reasons")
    overlap = discard_overlap_summary(items)

    effective_score = mean(adjusted_effective)
    end_to_end_score = mean(adjusted_all)
    return {
        "metric": metric,
        "n_total": n_total,
        "n_both_gates_pass": len(both_gates_pass_items),
        "n_physics_status_pass": len(physics_status_pass_items),
        "n_tracking_quality_pass": len(tracking_pass_items),
        "n_mask_qc_pass": len(mask_pass_items),
        "n_measurement_quality_pass": len(measurement_pass_items),
        "n_effective_for_score": len(effective_items),
        "both_gates_pass_rate": len(both_gates_pass_items) / max(n_total, 1),
        "physics_status_pass_rate": len(physics_status_pass_items) / max(n_total, 1),
        "tracking_quality_pass_rate": len(tracking_pass_items) / max(n_total, 1),
        "mask_qc_pass_rate": len(mask_pass_items) / max(n_total, 1),
        "measurement_quality_pass_rate": len(measurement_pass_items) / max(n_total, 1),
        "effective_video_rate": len(effective_items) / max(n_total, 1),
        "valid_only_score": effective_score,
        "effective_video_score": effective_score,
        "effective_base_score": mean(base_effective),
        "end_to_end_score": end_to_end_score,
        "mean_mask_quality_score": mean([r["mask_quality_score"] for r in items]),
        "mean_measurement_quality_multiplier": mean([r["measurement_quality_multiplier"] for r in items]),
        "median_relative_error_effective": median(rel_effective),
        "median_relative_error_both_gates_pass": median(rel_both_gates_pass),
        "p90_relative_error_effective": percentile(rel_effective, 0.90),
        "status_counts": dict(sorted(status_counts.items())),
        "mask_qc_counts": dict(sorted(qc_counts.items())),
        "tracking_quality_counts": dict(sorted(tracking_counts.items())),
        "measurement_quality_counts": dict(sorted(measurement_counts.items())),
        "score_exclusion_counts": exclusion_counts,
        "failure_reason_counts": count_reason_tokens(items, "failure_reason"),
        "mask_qc_reason_counts": count_reason_tokens(items, "mask_qc_reasons"),
        "tracking_quality_reason_counts": count_reason_tokens(items, "tracking_quality_reasons"),
        "discard_overlap_counts": overlap["counts"],
        "discard_overlap_rates": overlap["rates"],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_metric: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_metric.setdefault(str(row["metric"]), []).append(row)

    metric_summaries = [
        summarize_metric(metric, items)
        for metric, items in sorted(by_metric.items())
    ]

    overlap = discard_overlap_summary(rows)
    return {
        "n_results": len(rows),
        "n_both_gates_pass": sum(1 for r in rows if r["both_gates_pass"]),
        "n_physics_status_pass": sum(1 for r in rows if r["physics_status_pass"]),
        "n_tracking_quality_pass": sum(1 for r in rows if r["tracking_quality_pass"]),
        "n_mask_qc_pass": sum(1 for r in rows if r["mask_qc_pass"]),
        "n_measurement_quality_pass": sum(1 for r in rows if r["measurement_quality_status"] == "pass"),
        "n_effective_for_score": sum(1 for r in rows if r["effective_for_score"]),
        "both_gates_pass_rate": sum(1 for r in rows if r["both_gates_pass"]) / max(len(rows), 1),
        "physics_status_pass_rate": sum(1 for r in rows if r["physics_status_pass"]) / max(len(rows), 1),
        "tracking_quality_pass_rate": sum(1 for r in rows if r["tracking_quality_pass"]) / max(len(rows), 1),
        "mask_qc_pass_rate": sum(1 for r in rows if r["mask_qc_pass"]) / max(len(rows), 1),
        "measurement_quality_pass_rate": sum(1 for r in rows if r["measurement_quality_status"] == "pass") / max(len(rows), 1),
        "effective_video_rate": sum(1 for r in rows if r["effective_for_score"]) / max(len(rows), 1),
        "overall_valid_only_score": mean([m["valid_only_score"] for m in metric_summaries]),
        "overall_effective_video_score": mean([m["effective_video_score"] for m in metric_summaries]),
        "overall_end_to_end_score": mean([m["end_to_end_score"] for m in metric_summaries]),
        "mean_mask_quality_score": mean([r["mask_quality_score"] for r in rows]),
        "mean_measurement_quality_multiplier": mean([r["measurement_quality_multiplier"] for r in rows]),
        "global_exclusion_counts": count_reason_tokens(rows, "score_exclusion_reasons"),
        "global_failure_reason_counts": count_reason_tokens(rows, "failure_reason"),
        "global_mask_qc_reason_counts": count_reason_tokens(rows, "mask_qc_reasons"),
        "global_tracking_quality_reason_counts": count_reason_tokens(rows, "tracking_quality_reasons"),
        "discard_overlap_counts": overlap["counts"],
        "discard_overlap_rates": overlap["rates"],
        "metrics": metric_summaries,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_reason_csv(path: Path, name: str, counts: dict[str, int]) -> None:
    rows = [{"reason_type": name, "reason": reason, "count": count}
            for reason, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    write_csv(path, rows)


def plot_bar(path: Path, labels: list[str], values: list[float], title: str, ylabel: str,
             color: str = "#64748b", ylim: tuple[float, float] | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(5, 0.75 * len(labels)), 4))
    ax.bar(labels, values, color=color)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.tick_params(axis="x", rotation=35)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_discard_overlap(plots_dir: Path, summary: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rates = summary.get("discard_overlap_rates") or {}
    counts = summary.get("discard_overlap_counts") or {}
    keys = ["tracking_only", "physics_only", "both_tracking_and_physics"]
    labels = ["Tracking only", "Physics only", "Both"]
    values = [float(rates.get(key, 0.0)) * 100.0 for key in keys]

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#3b82f6", "#f59e0b", "#ef4444"]
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0, max(100.0, max(values, default=0.0) * 1.15))
    ax.set_ylabel("rate (%)")
    ax.set_title("Discard diagnostic overlap")
    for bar, key, value in zip(bars, keys, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}%\n(n={int(counts.get(key, 0))})",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(plots_dir / "discard_overlap_overall.png", dpi=150)
    plt.close(fig)


def plot_reports(out_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for old_plot in plots_dir.glob("*.png"):
        old_plot.unlink()

    metrics = [item["metric"] for item in summary["metrics"]]
    e2e = [item["end_to_end_score"] or 0.0 for item in summary["metrics"]]
    effective = [item["effective_video_score"] or 0.0 for item in summary["metrics"]]
    if metrics:
        fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(metrics)), 4))
        x = list(range(len(metrics)))
        ax.bar([i - 0.18 for i in x], effective, width=0.36, label="effective-only")
        ax.bar([i + 0.18 for i in x], e2e, width=0.36, label="end-to-end")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=35, ha="right")
        ax.set_ylim(0, 100)
        ax.set_ylabel("score")
        ax.set_title("Scores by metric")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(plots_dir / "metric_scores.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(metrics)), 4))
        x = list(range(len(metrics)))
        for offset, key, label in [
            (-0.30, "physics_status_pass_rate", "physics status pass"),
            (-0.10, "tracking_quality_pass_rate", "tracking pass"),
            (0.10, "both_gates_pass_rate", "both gates pass"),
            (0.30, "effective_video_rate", "effective"),
        ]:
            ax.bar([i + offset for i in x], [m[key] * 100 for m in summary["metrics"]],
                   width=0.18, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=35, ha="right")
        ax.set_ylim(0, 100)
        ax.set_ylabel("rate (%)")
        ax.set_title("Gate pass and effective-video rates by metric")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(plots_dir / "metric_rates.png", dpi=150)
        plt.close(fig)

    plot_discard_overlap(plots_dir, summary)


def check_completeness(paths, metadata, metric=None):
    import re
    def canonical(value):
        match = re.search(r"(?:^|[_-])([A-Z]+\d{3})$", str(value))
        return match.group(1) if match else str(value)
    actual = []
    for path in paths:
        result = load_json(path)
        actual.append(canonical((result.get("metadata") or {}).get("Prompt_ID") or prompt_id(result, path)))
    counts = Counter(actual)
    if str(metadata).lower() == "none":
        expected = set(actual)
        scope = "explicit_custom_scope"
    else:
        path = Path(metadata)
        if path.suffix.lower() == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as handle:
                records = list(csv.DictReader(handle))
        else:
            import pandas as pd
            records = pd.read_excel(path).to_dict("records")
        expected_list = [canonical(row["Prompt_ID"]) for row in records if metric is None or str(row["Metric"]) == metric]
        if len(expected_list) != len(set(expected_list)):
            raise ValueError("Metadata contains duplicate Prompt_ID values")
        expected = set(expected_list)
        scope = "metadata_verified"
    return {"n_expected": len(expected), "n_present": len(actual), "scope": scope,
            "missing_ids": sorted(expected-set(actual)), "unexpected_ids": sorted(set(actual)-expected),
            "duplicate_ids": sorted(k for k,v in counts.items() if v>1)}


def main() -> None:
    args = parse_args()
    result_root = Path(args.result_root).expanduser().resolve()
    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else result_root / "score_reports" / args.qmask_protocol["protocol_id"] / (args.metric if args.metric else "all_metrics")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    tolerances = load_tolerances(args)
    result_files = discover_result_files(result_root, args.metric)
    if not result_files:
        raise SystemExit(f"No result.json files found under {result_root}")

    rows = [score_one(path, tolerances, args) for path in result_files]
    unavailable = [row for row in rows if not row["score_available"]]
    if unavailable:
        (out_dir / "incomplete_report.json").write_text(json.dumps({"status": "incomplete", "unavailable": unavailable}, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit(f"Refusing formal scores: {len(unavailable)} results have unavailable quality/physical evidence; see incomplete_report.json")
    completeness = check_completeness(result_files, args.metadata, args.metric)
    if completeness["missing_ids"] or completeness["unexpected_ids"] or completeness["duplicate_ids"]:
        (out_dir / "incomplete_report.json").write_text(json.dumps(completeness, indent=2), encoding="utf-8")
        raise SystemExit("Result set does not match metadata; see incomplete_report.json")
    summary = summarize(rows)
    summary.update(completeness)
    summary["n_score_available"] = sum(row["score_available"] for row in rows)
    summary["n_score_unavailable"] = 0
    effective_mask_policy = "hard_fail" if args.strict_qc else args.mask_policy
    payload = {
        "model_name": args.model_name or result_root.name,
        "result_root": str(result_root),
        "scoring": {
            "qmask_protocol": args.qmask_protocol,
            "qmask_protocol_hash": protocol_hash(args.qmask_protocol),
            "valid_video_rule": (
                "The scorer applies gates in order: tracking quality first, then physics status. "
                "If tracking fails under the selected policy, the video is excluded and later scoring steps are skipped. "
                "If tracking passes, effective_for_score is true only when physics status is valid/weak_valid. "
                "Only effective candidates are required by evaluator contract to provide a computable physical accuracy score. "
                "Mask quality is computed after those gates and applied only as a 0-1 score multiplier."
            ),
            "formula": (
                "relative_error = physics_result.Relative_Error when available; "
                "otherwise abs(measured-target)/max(abs(target), eps), "
                "or measured when Measured_Unit/Target_Unit is relative_error; "
                "physical_accuracy_score = 100*exp(-relative_error/tau); "
                "measurement_quality_multiplier = tracking_quality_multiplier*mask_quality_score; "
                "adjusted_score = physical_accuracy_score*estimation_validity_multiplier*measurement_quality_multiplier"
            ),
            "status_multipliers": status_multipliers_for_report(args),
            "penalty_multiplier": args.penalty_multiplier,
            "weak_valid_multiplier": args.weak_valid_multiplier,
            "mask_policy": effective_mask_policy,
            "qc_flag_multiplier": 0.0 if effective_mask_policy == "hard_fail" else args.qc_flag_multiplier,
            "mask_score_thresholds": args.qmask_protocol["components"],
            "mask_score_gamma": args.qmask_protocol["gamma"],
            "tracking_policy": args.tracking_policy,
            "quality_flag_multiplier": 0.0 if args.tracking_policy == "hard_fail" else args.quality_flag_multiplier,
            "min_motion_extent_ratio": args.min_motion_extent_ratio,
            "min_object_coverage": args.min_object_coverage,
            "tracking_quality_recomputed": args.recompute_tracking_quality,
            "tolerances": tolerances,
        },
        "summary": summary,
        "per_video": rows,
    }

    (out_dir / "score_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(out_dir / "per_video_scores.csv", rows)
    write_csv(out_dir / "metric_scores.csv", summary["metrics"])
    write_reason_csv(out_dir / "score_exclusion_reasons.csv", "score_exclusion",
                     summary["global_exclusion_counts"])
    if not args.no_plots:
        plot_reports(out_dir, rows, summary)

    print(f"Scored {len(rows)} result(s)")
    print(f"Effective videos: {summary['n_effective_for_score']}/{summary['n_results']}")
    print(f"Overall effective-only score: {summary['overall_effective_video_score']}")
    print(f"Overall end-to-end score: {summary['overall_end_to_end_score']}")
    overlap_counts = summary["discard_overlap_counts"]
    overlap_rates = summary["discard_overlap_rates"]
    print("Overall discard diagnostics:")
    print(
        "  tracking only: "
        f"{overlap_counts['tracking_only']}/{summary['n_results']} "
        f"({overlap_rates['tracking_only']:.4f})"
    )
    print(
        "  physics only: "
        f"{overlap_counts['physics_only']}/{summary['n_results']} "
        f"({overlap_rates['physics_only']:.4f})"
    )
    print(
        "  both tracking and physics: "
        f"{overlap_counts['both_tracking_and_physics']}/{summary['n_results']} "
        f"({overlap_rates['both_tracking_and_physics']:.4f})"
    )
    print(f"Summary JSON: {out_dir / 'score_summary.json'}")
    print(f"Per-video CSV: {out_dir / 'per_video_scores.csv'}")
    print(f"Metric CSV: {out_dir / 'metric_scores.csv'}")
    if not args.no_plots:
        print(f"Plots: {out_dir / 'plots'}")


if __name__ == "__main__":
    main()
