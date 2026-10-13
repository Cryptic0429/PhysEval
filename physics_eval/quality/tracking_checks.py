from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def bbox_equivalent_diameter(record: dict[str, Any]) -> float | None:
    area = as_float(record.get("area", record.get("area_px2")))
    if area is not None and area > 0:
        return float(math.sqrt(4.0 * area / math.pi))
    bbox = record.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        except (TypeError, ValueError):
            return None
        w = max(0.0, x2 - x1 + 1.0)
        h = max(0.0, y2 - y1 + 1.0)
        if w > 0 and h > 0:
            return float(math.sqrt(w * h))
    return None


def record_center(record: dict[str, Any]) -> tuple[float, float] | None:
    if record.get("valid", True) is False:
        return None
    centroid = record.get("centroid")
    if isinstance(centroid, (list, tuple)) and len(centroid) >= 2:
        x, y = as_float(centroid[0]), as_float(centroid[1])
    else:
        x = as_float(record.get("x", record.get("cx", record.get("cx_px"))))
        y = as_float(record.get("y", record.get("cy", record.get("cy_px"))))
    if x is None or y is None:
        return None
    return float(x), float(y)


def percentile(values: list[float], q: float) -> float | None:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return None
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return float(vals[lo] * (1.0 - frac) + vals[hi] * frac)


def load_tracking(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {"frames": data}
    if not isinstance(data, dict):
        raise ValueError("tracking JSON root must be an object or list")
    return data


def analyze_tracking_quality_data(
    data: dict[str, Any],
    expected_objects: int,
    min_motion_extent_ratio: float = 0.75,
    min_object_coverage: float = 0.50,
) -> dict[str, Any]:
    frames = data.get("frames", [])
    expected_objects = max(1, int(expected_objects or 1))
    if not isinstance(frames, list) or not frames:
        return {
            "status": "flagged",
            "expected_objects": expected_objects,
            "actual_objects": 0,
            "reasons": ["tracking_frames_missing"],
        }

    total_frames = len(frames)
    objects: dict[str, dict[str, Any]] = {}
    default_oid = str(data.get("object_id", 1))
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        records = frame.get("objects")
        if not isinstance(records, list):
            records = [frame]
        for rec in records:
            if not isinstance(rec, dict):
                continue
            oid = str(rec.get("object_id", rec.get("obj_id", default_oid)))
            if oid not in objects:
                objects[oid] = {"points": [], "diameters": [], "valid_frames": 0}

            diameter = bbox_equivalent_diameter(rec)
            if diameter is not None and diameter > 0:
                objects[oid]["diameters"].append(diameter)

            center = record_center(rec)
            if center is not None:
                objects[oid]["points"].append(center)
                objects[oid]["valid_frames"] += 1

    counted = {
        oid: item
        for oid, item in objects.items()
        if item["valid_frames"] / max(total_frames, 1) >= min_object_coverage
    }
    actual_objects = len(counted)
    reasons: list[str] = []
    if actual_objects != expected_objects:
        reasons.append("object_count_mismatch")

    primary_motion: dict[str, Any] = {}
    if counted:
        primary_oid, primary = max(counted.items(), key=lambda kv: kv[1]["valid_frames"])
        pts = primary["points"]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        extent = float(math.hypot(max(xs) - min(xs), max(ys) - min(ys))) if len(pts) >= 2 else 0.0
        path_len = float(sum(math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]) for i in range(1, len(pts))))
        median_diameter = percentile(primary["diameters"], 0.5)
        extent_norm = extent / median_diameter if median_diameter and median_diameter > 0 else None
        path_norm = path_len / median_diameter if median_diameter and median_diameter > 0 else None
        primary_motion = {
            "primary_object_id": primary_oid,
            "valid_frames": primary["valid_frames"],
            "coverage": primary["valid_frames"] / max(total_frames, 1),
            "motion_extent_px": extent,
            "path_length_px": path_len,
            "median_equivalent_diameter_px": median_diameter,
            "motion_extent_norm_by_diameter": extent_norm,
            "path_length_norm_by_diameter": path_norm,
        }
        if extent_norm is not None and extent_norm < min_motion_extent_ratio:
            reasons.append("no_meaningful_motion")
    else:
        reasons.append("no_counted_object")

    return {
        "status": "flagged" if reasons else "pass",
        "expected_objects": expected_objects,
        "actual_objects": actual_objects,
        "total_frames": total_frames,
        "min_object_coverage": float(min_object_coverage),
        "min_motion_extent_ratio": float(min_motion_extent_ratio),
        "counted_object_ids": sorted(counted),
        "all_object_ids": sorted(objects),
        "reasons": sorted(set(reasons)),
        **primary_motion,
    }


def analyze_tracking_quality_file(
    path: str | Path,
    expected_objects: int,
    min_motion_extent_ratio: float = 0.75,
    min_object_coverage: float = 0.50,
) -> dict[str, Any]:
    path = Path(path)
    try:
        data = load_tracking(path)
    except Exception as exc:
        return {
            "status": "flagged",
            "tracking_json": str(path),
            "expected_objects": max(1, int(expected_objects or 1)),
            "actual_objects": 0,
            "reasons": [f"tracking_json_unreadable:{type(exc).__name__}"],
        }
    result = analyze_tracking_quality_data(
        data=data,
        expected_objects=expected_objects,
        min_motion_extent_ratio=min_motion_extent_ratio,
        min_object_coverage=min_object_coverage,
    )
    result["tracking_json"] = str(path)
    return result


def write_tracking_quality(
    path: str | Path,
    expected_objects: int,
    min_motion_extent_ratio: float = 0.75,
    min_object_coverage: float = 0.50,
) -> dict[str, Any]:
    path = Path(path)
    data = load_tracking(path)
    result = analyze_tracking_quality_data(
        data=data,
        expected_objects=expected_objects,
        min_motion_extent_ratio=min_motion_extent_ratio,
        min_object_coverage=min_object_coverage,
    )
    result["tracking_json"] = str(path)
    quality_checks = data.get("quality_checks")
    if not isinstance(quality_checks, dict):
        quality_checks = {}
    quality_checks["tracking_quality"] = result
    data["quality_checks"] = quality_checks
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
