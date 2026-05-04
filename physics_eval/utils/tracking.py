from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class TrackingFormatError(ValueError):
    pass


def _bbox_fields(bbox: Any) -> dict[str, float | None]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return {"bbox_x": None, "bbox_y": None, "bbox_w": None, "bbox_h": None}
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    return {
        "bbox_x": x1,
        "bbox_y": y1,
        "bbox_w": max(0.0, x2 - x1 + 1.0),
        "bbox_h": max(0.0, y2 - y1 + 1.0),
    }


def _center_from_record(rec: dict[str, Any], bbox_data: dict[str, float | None]) -> tuple[float | None, float | None]:
    centroid = rec.get("centroid")
    if isinstance(centroid, (list, tuple)) and len(centroid) >= 2:
        return float(centroid[0]), float(centroid[1])
    x = rec.get("x", rec.get("cx", rec.get("cx_px")))
    y = rec.get("y", rec.get("cy", rec.get("cy_px")))
    if x is not None and y is not None:
        return float(x), float(y)
    if bbox_data["bbox_x"] is not None and bbox_data["bbox_w"] is not None:
        return (
            float(bbox_data["bbox_x"]) + float(bbox_data["bbox_w"]) / 2.0,
            float(bbox_data["bbox_y"]) + float(bbox_data["bbox_h"]) / 2.0,
        )
    return None, None


def _record_to_row(
    rec: dict[str, Any],
    default_object_id: int | str,
    frame_idx: int,
    t: float | None,
) -> dict[str, Any] | None:
    valid = rec.get("valid", True)
    if valid is False:
        return None

    bbox_data = _bbox_fields(rec.get("bbox"))
    cx, cy = _center_from_record(rec, bbox_data)
    if cx is None or cy is None:
        return None

    area = rec.get("area", rec.get("area_px2"))
    area_f = float(area) if area is not None else None
    eq_d = rec.get("equivalent_diameter_px")
    if eq_d is None and area_f is not None and area_f > 0:
        eq_d = float(np.sqrt(4.0 * area_f / np.pi))
    if eq_d is None and bbox_data["bbox_w"] is not None and bbox_data["bbox_h"] is not None:
        eq_d = float(np.sqrt(float(bbox_data["bbox_w"]) * float(bbox_data["bbox_h"])))

    return {
        "frame": int(frame_idx),
        "t": float(t) if t is not None else None,
        "object_id": str(rec.get("object_id", rec.get("obj_id", default_object_id))),
        "cx_px": float(cx),
        "cy_px": float(cy),
        "area_px2": area_f,
        "bbox_x": bbox_data["bbox_x"],
        "bbox_y": bbox_data["bbox_y"],
        "bbox_w": bbox_data["bbox_w"],
        "bbox_h": bbox_data["bbox_h"],
        "equivalent_diameter_px": float(eq_d) if eq_d is not None else None,
    }


def normalize_tracking_json(path: str | Path, fps_override: float | None = None) -> pd.DataFrame:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TrackingFormatError(f"invalid JSON in {path}: {exc}") from exc

    if isinstance(data, list):
        frames = data
        video_info: dict[str, Any] = {}
        default_object_id = 1
    elif isinstance(data, dict):
        frames = data.get("frames")
        video_info = data.get("video_info", {}) or {}
        default_object_id = data.get("object_id", 1)
    else:
        raise TrackingFormatError("tracking JSON root must be an object or list")

    if not isinstance(frames, list):
        raise TrackingFormatError("tracking JSON must contain a frames list")

    fps = video_info.get("fps", fps_override)
    if fps is None:
        fps = fps_override
    if fps is not None:
        fps = float(fps)
        if fps <= 0:
            raise TrackingFormatError("fps must be positive")

    rows: list[dict[str, Any]] = []
    for i, frame_rec in enumerate(frames):
        if not isinstance(frame_rec, dict):
            continue
        frame_idx = int(frame_rec.get("frame_idx", frame_rec.get("frame", i)))
        t = frame_rec.get("time_sec", frame_rec.get("t"))
        if t is None and fps is not None:
            t = frame_idx / fps

        objects = frame_rec.get("objects")
        if isinstance(objects, list):
            for obj_rec in objects:
                if isinstance(obj_rec, dict):
                    row = _record_to_row(obj_rec, default_object_id, frame_idx, t)
                    if row is not None:
                        rows.append(row)
        else:
            row = _record_to_row(frame_rec, default_object_id, frame_idx, t)
            if row is not None:
                rows.append(row)

    if not rows:
        raise TrackingFormatError("no valid tracking points found")

    df = pd.DataFrame(rows)
    if df["t"].isna().any():
        raise TrackingFormatError("tracking JSON has no time values and no fps was provided")

    required = ["frame", "t", "object_id", "cx_px", "cy_px"]
    missing = [c for c in required if c not in df.columns or df[c].isna().all()]
    if missing:
        raise TrackingFormatError(f"normalized tracking is missing required fields: {missing}")

    df = df.sort_values(["object_id", "frame"]).reset_index(drop=True)
    return df


def get_object_track(df: pd.DataFrame, object_id: str | int | None = None) -> pd.DataFrame:
    if df.empty:
        raise TrackingFormatError("tracking dataframe is empty")
    if object_id is None:
        counts = df.groupby("object_id").size().sort_values(ascending=False)
        object_id = counts.index[0]
    obj = df[df["object_id"].astype(str) == str(object_id)].copy()
    if obj.empty:
        available = sorted(df["object_id"].astype(str).unique())
        raise TrackingFormatError(f"object_id {object_id!r} not found; available={available}")
    return obj.sort_values("frame").reset_index(drop=True)


def get_two_object_tracks(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = df.groupby("object_id").size().sort_values(ascending=False)
    if len(counts) < 2:
        raise TrackingFormatError("two-object evaluator requires at least two object_id values")
    return get_object_track(df, counts.index[0]), get_object_track(df, counts.index[1])
