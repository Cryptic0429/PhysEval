from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ScaleEstimate:
    scale_m_per_px: float | None
    measured_pixel_size: float | None
    source: str
    status: str
    reason: str = ""


def _median_positive(values: pd.Series) -> float | None:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if len(arr) == 0:
        return None
    return float(np.median(arr))


def _diameter_px(track: pd.DataFrame) -> tuple[float | None, str]:
    if "equivalent_diameter_px" in track:
        val = _median_positive(track["equivalent_diameter_px"])
        if val is not None:
            return val, "equivalent_diameter_px"
    if "bbox_w" in track and "bbox_h" in track:
        w = pd.to_numeric(track["bbox_w"], errors="coerce")
        h = pd.to_numeric(track["bbox_h"], errors="coerce")
        val = _median_positive(np.sqrt(w * h))
        if val is not None:
            return val, "sqrt_bbox_area"
    return None, "unavailable"


def _side_px(track: pd.DataFrame) -> tuple[float | None, str]:
    if "bbox_w" in track and "bbox_h" in track:
        w = _median_positive(track["bbox_w"])
        h = _median_positive(track["bbox_h"])
        if w is not None and h is not None:
            return float((w + h) / 2.0), "mean_bbox_side"
    return _diameter_px(track)


def estimate_scale(row: dict, tracking_df: pd.DataFrame) -> ScaleEstimate:
    mode = str(row.get("Scale_Mode") or "none").strip().lower()
    if mode in {"none", "ratio_only", ""}:
        return ScaleEstimate(None, None, "none", "ok")

    value = row.get("Calibration_Value_m")
    try:
        value_m = float(value)
    except (TypeError, ValueError):
        return ScaleEstimate(None, None, "metadata", "failed", "calibration_value_missing")
    if not np.isfinite(value_m) or value_m <= 0:
        return ScaleEstimate(None, None, "metadata", "failed", "calibration_value_missing")

    if mode == "explicit_object_size" and str(row.get("Calibration_Dimension") or "").lower() == "side_length":
        measured_px, source = _side_px(tracking_df)
    elif mode in {"standard_object", "object_diameter", "explicit_object_size"}:
        measured_px, source = _diameter_px(tracking_df)
    elif mode == "explicit_height":
        return ScaleEstimate(None, None, "explicit_height", "failed", "explicit_height_not_implemented")
    else:
        return ScaleEstimate(None, None, mode, "failed", f"unknown_scale_mode:{mode}")

    if measured_px is None or measured_px <= 0:
        return ScaleEstimate(None, None, source, "failed", "measured_pixel_size_unavailable")
    return ScaleEstimate(value_m / measured_px, measured_px, source, "ok")


def require_scale(row: dict, tracking_df: pd.DataFrame) -> ScaleEstimate:
    scale = estimate_scale(row, tracking_df)
    if scale.scale_m_per_px is None:
        raise ValueError(scale.reason or "scale_unavailable")
    return scale

