"""Metadata overrides and provenance for the fixed formal tracking protocol.

Prompt validation, renderer-mask selection and detector fallback are provided by
initialization.py. This module resolves row metadata into that public protocol;
it has no dependency on pipeline orchestration.
"""
from __future__ import annotations

import argparse
import math
from typing import Any

from physics_eval.config import INITIALIZATION_POLICY
from physics_eval.io.metadata import parse_known_params
from physics_eval.schemas import PromptSpec
from .initialization import (
    normalize_init_mode,
    normalize_manual_prompts,
    parse_json_value,
    renderer_gt_eligibility,
    renderer_gt_prompts,
)


def num_objects_for_row(row: dict[str, Any]) -> int:
    for key in ("_Resolved_Tracking_Num_Objects", "Tracking_Num_Objects", "Num_Objects"):
        try:
            value = row.get(key)
            if value is not None and value != "":
                return max(1, int(value))
        except (TypeError, ValueError):
            continue
    return 1


_TRACKING_OVERRIDE_ALIASES = {
    "auto_init_num_objects": "num_objects",
    "tracking_num_objects": "num_objects",
    "auto_init_detector": "detector",
    "scan_frames": "auto_init_scan_frames",
    "scan_mode": "auto_init_scan_mode",
    "threshold": "auto_init_threshold",
    "min_area": "auto_init_min_area",
    "max_area_ratio": "auto_init_max_area_ratio",
    "min_fill_ratio": "auto_init_min_fill_ratio",
    "max_aspect_ratio": "auto_init_max_aspect_ratio",
    "object_refine": "auto_init_object_refine",
    "box_expand": "auto_init_box_expand",
    "target_hint": "auto_init_target_hint",
    "shadow_filter_disabled": "disable_auto_init_shadow_filter",
    "negative_points_disabled": "no_auto_init_negative_points",
    "box_disabled": "no_auto_init_box",
    "bidirectional_propagation_disabled": "no_bidirectional_propagation",
}

_TRACKING_OVERRIDE_KEYS = {
    "num_objects",
    "detector",
    "yolo_conf",
    "yolo_iou",
    "yolo_classes",
    "auto_init_scan_frames",
    "auto_init_scan_mode",
    "auto_init_threshold",
    "auto_init_min_area",
    "auto_init_max_area_ratio",
    "auto_init_min_fill_ratio",
    "auto_init_max_aspect_ratio",
    "auto_init_object_refine",
    "auto_init_box_expand",
    "auto_init_target_hint",
    "disable_auto_init_shadow_filter",
    "no_auto_init_negative_points",
    "no_auto_init_box",
    "no_bidirectional_propagation",
    "offload_video_to_cpu",
}


def _bool_setting(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    raise ValueError(f"{label} must be a boolean")


def _finite_setting(value: Any, label: str) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be a finite number")
    return parsed


def tracking_overrides_for_row(row: dict[str, Any]) -> dict[str, Any]:
    params = parse_known_params(row)
    raw = row.get("Tracking_Overrides_JSON")
    if raw is None or raw == "":
        raw = params.get("tracking_overrides", {})
    parsed = parse_json_value(raw, "Tracking_Overrides_JSON") or {}
    if not isinstance(parsed, dict):
        raise ValueError("Tracking_Overrides_JSON must decode to an object")

    normalized: dict[str, Any] = {}
    for raw_key, value in parsed.items():
        key = _TRACKING_OVERRIDE_ALIASES.get(str(raw_key), str(raw_key))
        if key not in _TRACKING_OVERRIDE_KEYS:
            raise ValueError(
                f"unsupported Tracking_Overrides_JSON key {raw_key!r}; "
                f"allowed keys are {sorted(_TRACKING_OVERRIDE_KEYS)}"
            )
        if key in normalized:
            raise ValueError(f"duplicate tracking override after alias normalization: {key}")
        normalized[key] = value

    if "num_objects" in normalized:
        value = _finite_setting(normalized["num_objects"], "tracking override num_objects")
        if value < 1 or not value.is_integer():
            raise ValueError("tracking override num_objects must be a positive integer")
        normalized["num_objects"] = int(value)
    if "detector" in normalized and normalized["detector"] != INITIALIZATION_POLICY:
        raise ValueError("Formal evaluation fixes initialization to YOLO then motion fallback")
    if "auto_init_scan_mode" in normalized and normalized["auto_init_scan_mode"] not in {"start", "middle", "uniform"}:
        raise ValueError("tracking override auto_init_scan_mode must be start, middle, or uniform")
    for key in ("auto_init_scan_frames", "auto_init_min_area"):
        if key in normalized:
            value = _finite_setting(normalized[key], f"tracking override {key}")
            if value < 1 or not value.is_integer():
                raise ValueError(f"tracking override {key} must be a positive integer")
            normalized[key] = int(value)
    for key in ("yolo_conf", "yolo_iou", "auto_init_max_area_ratio", "auto_init_min_fill_ratio"):
        if key in normalized:
            value = _finite_setting(normalized[key], f"tracking override {key}")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"tracking override {key} must be in [0, 1]")
            normalized[key] = value
    for key in ("auto_init_threshold", "auto_init_max_aspect_ratio", "auto_init_box_expand"):
        if key in normalized:
            value = _finite_setting(normalized[key], f"tracking override {key}")
            if value <= 0.0:
                raise ValueError(f"tracking override {key} must be > 0")
            normalized[key] = value
    for key in (
        "auto_init_object_refine",
        "disable_auto_init_shadow_filter",
        "no_auto_init_negative_points",
        "no_auto_init_box",
        "no_bidirectional_propagation",
        "offload_video_to_cpu",
    ):
        if key in normalized:
            normalized[key] = _bool_setting(normalized[key], f"tracking override {key}")
    for key in ("yolo_classes", "auto_init_target_hint"):
        if key in normalized and normalized[key] is not None and not isinstance(normalized[key], str):
            raise ValueError(f"tracking override {key} must be a string or null")
    return normalized


def effective_tracking_settings(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    settings = {
        key: getattr(args, key, False) if key == "offload_video_to_cpu" else getattr(args, key)
        for key in _TRACKING_OVERRIDE_KEYS
        if key != "num_objects"
    }
    settings["detector"] = INITIALIZATION_POLICY
    settings["num_objects"] = num_objects_for_row(row)
    settings.update(tracking_overrides_for_row(row))
    return settings


def resolve_tracking_initialization(
    row: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[PromptSpec]]:
    params = parse_known_params(row)
    nested = params.get("tracking_init", {})
    if nested is None:
        nested = {}
    if not isinstance(nested, dict):
        raise ValueError("Known_Parameters_JSON.tracking_init must be an object")
    mode_raw = row.get("Tracking_Init_Mode")
    if mode_raw is None or mode_raw == "":
        mode_raw = nested.get("mode", "auto")
    mode = normalize_init_mode(mode_raw)
    settings = effective_tracking_settings(row, args)
    expected_objects = int(settings["num_objects"])

    prompts: list[PromptSpec] = []
    if mode == "manual":
        raw_prompts = row.get("Tracking_Prompts_JSON")
        if raw_prompts is None or raw_prompts == "":
            raw_prompts = nested.get("prompts")
        prompts = normalize_manual_prompts(raw_prompts, expected_objects)
    elif mode == "renderer_gt":
        if not args.allow_renderer_gt_init:
            raise ValueError(
                "renderer_gt initialization is GT-assisted and requires --allow-renderer-gt-init"
            )
        prompts = renderer_gt_prompts(row, nested)
        if len(prompts) != expected_objects:
            raise ValueError(
                f"renderer_gt initialization resolved {len(prompts)} prompt(s), "
                f"but the row expects {expected_objects} object(s)"
            )

    if prompts:
        box_presence = [item.box is not None for item in prompts]
        if any(box_presence) and not all(box_presence):
            raise ValueError(
                "all metadata prompts in a row must consistently provide boxes or omit boxes; "
                "the SAM2 prompt adapter applies one prompt policy per run"
            )

    boxes_used = bool(prompts) and all(item.box is not None for item in prompts) and not settings["no_auto_init_box"]
    if mode == "auto":
        protocol = "video_only_auto"
    elif mode == "manual":
        protocol = "one_frame_manual_point_box" if boxes_used else "one_frame_manual_point"
    else:
        eligibility = renderer_gt_eligibility(nested)
        selector = (
            "first_visible"
            if eligibility == renderer_gt_eligibility()
            else "first_eligible"
        )
        protocol = f"renderer_{selector}_gt_point_box" if boxes_used else f"renderer_{selector}_gt_point"
    provenance = {
        "schema_version": 1,
        "mode": mode,
        "protocol": protocol,
        "num_objects": expected_objects,
        "prompts": [item.to_dict() for item in prompts],
        "tracking_overrides": tracking_overrides_for_row(row),
        "uses_renderer_ground_truth": mode == "renderer_gt",
        "uses_ground_truth_after_initialization": False,
        "box_prompts_used": boxes_used,
        "renderer_gt_eligibility": renderer_gt_eligibility(nested) if mode == "renderer_gt" else None,
        "semantic_verification": {
            "status": "declared",
            "target_description": row.get("Tracking_Objects") or row.get("Calibration_Object"),
            "expected_object_count": expected_objects,
            "resolved_prompt_count": len(prompts) if mode != "auto" else None,
            "prompt_count_matches_expected": len(prompts) == expected_objects if mode != "auto" else None,
            "initialization_supervision": protocol,
            "future_frame_gt_used_for_initialization": False,
        },
    }
    return {**settings, "initialization": provenance}, prompts
