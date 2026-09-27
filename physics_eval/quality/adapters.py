"""Convert flat/nested tracks to inclusive pixel boxes and real frame counts."""
from __future__ import annotations

from collections import defaultdict
import math


def required_ids(result: dict) -> list[str]:
    metadata = result.get("metadata") or {}
    expected = int(metadata.get("Num_Objects") or metadata.get("Tracking_Objects") or 1)
    explicit = metadata.get("Required_Object_IDs")
    if explicit is not None:
        if not isinstance(explicit, list) or len(explicit) != expected or len(set(map(str, explicit))) != expected:
            raise ValueError("Required_Object_IDs must contain exactly the required unique IDs")
        return list(map(str, explicit))
    tracked = (result.get("tracking_quality") or {}).get("counted_object_ids")
    if isinstance(tracked, list) and len(tracked) == expected:
        return list(map(str, tracked))
    return [str(i) for i in range(1, expected + 1)]


def normalize_records(data: dict, bbox_convention: str | None = None) -> tuple[list[dict], int]:
    convention = bbox_convention or data.get("bbox_convention", "inclusive")
    if convention not in {"inclusive", "exclusive"}:
        raise ValueError("unknown bbox convention")
    records = []
    seen = set()
    for i, frame in enumerate(data.get("frames") or []):
        frame_idx = int(frame.get("frame_idx", frame.get("frame", i)))
        for item in frame.get("objects", [frame]):
            rec = dict(item)
            idx = int(rec.get("frame_idx", frame_idx))
            oid = int(rec.get("object_id", rec.get("obj_id", data.get("object_id", 1))))
            if (oid, idx) in seen or idx < 0:
                raise ValueError("duplicate object/frame or negative frame index")
            seen.add((oid, idx))
            box = rec.get("bbox")
            if box is not None:
                box = list(map(float, box))
                if convention == "exclusive":
                    box[2] -= 1
                    box[3] -= 1
            center = rec.get("centroid")
            if center is None and rec.get("x") is not None and rec.get("y") is not None:
                center = [rec["x"], rec["y"]]
            valid = rec.get("valid", True) is not False and center is not None
            if valid:
                valid = all(math.isfinite(float(v)) for v in center)
            rec.update(object_id=oid, frame_idx=idx, bbox=box, centroid=center if valid else None, valid=valid)
            rec.setdefault("time_sec", frame.get("time_sec"))
            records.append(rec)
    inferred = max((r["frame_idx"] for r in records), default=-1) + 1
    info = data.get("video_info") or {}
    total = int(info.get("total_frames") or info.get("num_frames") or data.get("total_frames") or inferred)
    if total < inferred:
        raise ValueError("declared frame count is smaller than observed frame indices")
    return sorted(records, key=lambda r: (r["frame_idx"], r["object_id"])), total


def nested_frames(records: list[dict], total_frames: int) -> list[dict]:
    grouped = defaultdict(list)
    for rec in records:
        grouped[int(rec["frame_idx"])].append(rec)
    return [{"frame_idx": i, "objects": grouped[i]} for i in range(total_frames)]
