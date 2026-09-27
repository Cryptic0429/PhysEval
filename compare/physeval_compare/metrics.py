from __future__ import annotations

import math
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np


MASK_SCORE_THRESHOLDS = {
    "area_ratio_p95_p05": (2.00, 4.00, "high_bad"),
    "area_cv": (0.25, 0.70, "high_bad"),
    "bbox_width_cv": (0.20, 0.60, "high_bad"),
    "bbox_height_cv": (0.20, 0.60, "high_bad"),
    "bbox_aspect_cv": (0.20, 0.60, "high_bad"),
    "max_centroid_step_norm_by_diameter": (2.50, 5.00, "high_bad"),
    "translation_aligned_iou_median": (0.30, 0.65, "low_bad"),
    "middle_coverage": (0.50, 0.70, "low_bad"),
}
MASK_SCORE_GAMMA = 0.35


def mask_geometry(mask: np.ndarray) -> tuple[float | None, list[int] | None, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None, None, 0
    centroid = [float(np.mean(xs)), float(np.mean(ys))]
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    return centroid, bbox, int(len(xs))


def bbox_iou(a: list[float] | None, b: list[float] | None) -> float | None:
    if a is None or b is None:
        return None
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(intersection / max(area_a + area_b - intersection, 1e-9))


def _cv(values: list[float]) -> float | None:
    if not values:
        return None
    mean = float(np.mean(values))
    return float(np.std(values) / mean) if mean > 0 else None


def _ramp(value: float | None, good: float, bad: float, direction: str) -> float | None:
    if value is None:
        return None
    if direction == "high_bad":
        if value <= good:
            return 1.0
        if value >= bad:
            return 0.0
        normalized = (value - good) / (bad - good)
    else:
        if value >= bad:
            return 1.0
        if value <= good:
            return 0.0
        normalized = (bad - value) / (bad - good)
    return float(max(0.0, min(1.0, 1.0 - normalized ** MASK_SCORE_GAMMA)))


def _shift_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    height, width = mask.shape
    output = np.zeros_like(mask, dtype=bool)
    source_x1 = max(0, -dx)
    source_x2 = min(width, width - dx)
    source_y1 = max(0, -dy)
    source_y2 = min(height, height - dy)
    if source_x1 >= source_x2 or source_y1 >= source_y2:
        return output
    destination_x1 = source_x1 + dx
    destination_x2 = source_x2 + dx
    destination_y1 = source_y1 + dy
    destination_y2 = source_y2 + dy
    output[destination_y1:destination_y2, destination_x1:destination_x2] = mask[
        source_y1:source_y2, source_x1:source_x2
    ]
    return output


def _aligned_iou(reference: np.ndarray, current: np.ndarray) -> float | None:
    ref_center, _, _ = mask_geometry(reference)
    cur_center, _, _ = mask_geometry(current)
    if ref_center is None or cur_center is None:
        return None
    shifted = _shift_mask(
        current,
        int(round(ref_center[0] - cur_center[0])),
        int(round(ref_center[1] - cur_center[1])),
    )
    intersection = int(np.count_nonzero(reference & shifted))
    union = int(np.count_nonzero(reference | shifted))
    return float(intersection / union) if union else None


def tracking_quality(records, *, expected_objects, total_frames, min_coverage, min_motion_ratio):
    from physics_eval.quality.adapters import normalize_records, nested_frames
    from physics_eval.utils.tracking_quality import analyze_tracking_quality_data
    canonical, total = normalize_records({"frames": records, "total_frames": total_frames}, "exclusive")
    return analyze_tracking_quality_data(
        {"frames": nested_frames(canonical, total)}, expected_objects,
        min_motion_extent_ratio=min_motion_ratio, min_object_coverage=min_coverage,
    )


def legacy_compare_mask_quality(
    records: list[dict[str, Any]],
    masks: dict[int, dict[int, np.ndarray]],
    *,
    total_frames: int,
) -> dict[str, Any]:
    summaries = []
    for object_id, object_masks in sorted(masks.items()):
        items = {int(item["frame_idx"]): item for item in records if int(item["object_id"]) == object_id}
        middle_start = int(math.floor(total_frames * 0.15))
        middle_end = max(middle_start + 1, int(math.ceil(total_frames * 0.85)))
        middle_indices = list(range(middle_start, min(total_frames, middle_end)))
        valid_middle = [idx for idx in middle_indices if idx in items and items[idx].get("valid")]
        middle_coverage = len(valid_middle) / max(len(middle_indices), 1)
        valid_items = [items[idx] for idx in sorted(items) if items[idx].get("valid")]
        areas = [float(item["area"]) for item in valid_items]
        widths = [float(item["bbox"][2] - item["bbox"][0]) for item in valid_items]
        heights = [float(item["bbox"][3] - item["bbox"][1]) for item in valid_items]
        aspects = [width / max(height, 1e-9) for width, height in zip(widths, heights, strict=True)]
        p05 = float(np.percentile(areas, 5)) if areas else 0.0
        p95 = float(np.percentile(areas, 95)) if areas else 0.0
        steps = []
        diameters = []
        for previous, current in zip(valid_items, valid_items[1:]):
            steps.append(float(math.dist(previous["centroid"], current["centroid"])))
            diameters.append(math.sqrt(4.0 * float(previous["area"]) / math.pi))
        step_norm = max(steps, default=0.0) / max(float(median(diameters)) if diameters else 0.0, 1e-9)
        sampled_indices = valid_middle[::max(1, len(valid_middle) // 12)] if valid_middle else []
        aligned = []
        if sampled_indices:
            reference = object_masks[sampled_indices[len(sampled_indices) // 2]]
            for idx in sampled_indices:
                value = _aligned_iou(reference, object_masks[idx])
                if value is not None:
                    aligned.append(value)
        summary = {
            "object_id": object_id,
            "total_frames": total_frames,
            "valid_frames_total": len(valid_items),
            "middle_coverage": middle_coverage,
            "area_ratio_p95_p05": p95 / max(p05, 1e-9) if areas else None,
            "area_cv": _cv(areas),
            "bbox_width_cv": _cv(widths),
            "bbox_height_cv": _cv(heights),
            "bbox_aspect_cv": _cv(aspects),
            "max_centroid_step_norm_by_diameter": step_norm,
            "translation_aligned_iou_median": float(median(aligned)) if aligned else None,
        }
        components = {}
        for key, (good, bad, direction) in MASK_SCORE_THRESHOLDS.items():
            components[key] = _ramp(summary.get(key), good, bad, direction)
        available = [value for value in components.values() if value is not None]
        score = min(available) if available else 0.0
        reasons = []
        if middle_coverage < 0.5:
            reasons.append("low_middle_coverage")
        if score <= 0:
            reasons.append("unstable_mask_geometry")
        summary.update({
            "score": score,
            "components": components,
            "status": "pass" if not reasons else "flagged",
            "decision": "keep" if not reasons else "review",
            "reasons": ";".join(reasons),
        })
        summaries.append(summary)
    overall = min((float(item["score"]) for item in summaries), default=0.0)
    return {
        "status": "pass" if summaries and all(item["status"] == "pass" for item in summaries) else "flagged",
        "score": overall,
        "objects": summaries,
    }


def mask_quality(records, masks, *, total_frames, protocol=None, required_object_ids=None):
    from physics_eval.quality.diagnostics import extract_quality_evidence
    from physics_eval.quality.protocol import load_protocol
    from physics_eval.quality.scoring import compute_qmask
    if protocol == "compare_legacy_v1":
        result = legacy_compare_mask_quality(records, masks, total_frames=total_frames)
        result["protocol_id"] = "compare_legacy_v1"
        return result
    selected = load_protocol(protocol or "grouped_v2_candidate") if not isinstance(protocol, dict) else protocol
    evidence = extract_quality_evidence(records, masks, total_frames=total_frames, bbox_convention="exclusive")
    result = compute_qmask(evidence, selected, required_object_ids if required_object_ids is not None else sorted(masks))
    result["evidence_objects"] = evidence["objects"]
    result["evidence_schema_version"] = 1
    result["population"] = "tracking_runs_without_physics_gate"
    return result


def compare_tracking_files(left_path: Path, right_path: Path) -> dict[str, Any]:
    import json

    left = json.loads(left_path.read_text(encoding="utf-8"))
    right = json.loads(right_path.read_text(encoding="utf-8"))
    from physics_eval.quality.adapters import normalize_records
    left_flat, _ = normalize_records(left, left.get("bbox_convention", "exclusive"))
    right_flat, _ = normalize_records(right, right.get("bbox_convention", "exclusive"))
    # Pairwise IoU helper expects exclusive upper bounds.
    for rec in left_flat + right_flat:
        if rec.get("bbox") is not None:
            rec["bbox"][2] += 1
            rec["bbox"][3] += 1
    left_records = {(int(item["object_id"]), int(item["frame_idx"])): item for item in left_flat if item.get("valid")}
    right_records = {(int(item["object_id"]), int(item["frame_idx"])): item for item in right_flat if item.get("valid")}
    keys = sorted(set(left_records) & set(right_records))
    distances = []
    ious = []
    mask_ious = []
    diameters = []
    for key in keys:
        left_item, right_item = left_records[key], right_records[key]
        distances.append(math.dist(left_item["centroid"], right_item["centroid"]))
        value = bbox_iou(left_item.get("bbox"), right_item.get("bbox"))
        if value is not None:
            ious.append(value)
        left_mask_path = left_item.get("mask_path")
        right_mask_path = right_item.get("mask_path")
        if left_mask_path and right_mask_path:
            from PIL import Image

            try:
                left_mask = np.asarray(Image.open(left_mask_path).convert("L"))
                right_mask = np.asarray(Image.open(right_mask_path).convert("L"))
            except (FileNotFoundError, OSError):
                left_mask = right_mask = None
            if left_mask is not None and right_mask is not None and left_mask.shape == right_mask.shape:
                left_bool = left_mask > 0
                right_bool = right_mask > 0
                union = int(np.count_nonzero(left_bool | right_bool))
                if union:
                    mask_ious.append(float(np.count_nonzero(left_bool & right_bool) / union))
        for item in (left_item, right_item):
            if float(item.get("area") or 0) > 0:
                diameters.append(math.sqrt(4.0 * float(item["area"]) / math.pi))
    rmse = math.sqrt(float(np.mean(np.square(distances)))) if distances else None
    diameter = float(median(diameters)) if diameters else None
    return {
        "common_object_frames": len(keys),
        "centroid_rmse_px": rmse,
        "centroid_nrmse_by_diameter": rmse / diameter if rmse is not None and diameter else None,
        "bbox_iou_mean": float(np.mean(ious)) if ious else None,
        "mask_iou_mean": float(np.mean(mask_ious)) if mask_ious else None,
    }
