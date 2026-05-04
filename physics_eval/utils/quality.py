from __future__ import annotations

import numpy as np
import pandas as pd


def tracking_coverage(track: pd.DataFrame) -> float:
    if track.empty:
        return 0.0
    frames = pd.to_numeric(track["frame"], errors="coerce").dropna()
    if frames.empty:
        return 0.0
    span = int(frames.max() - frames.min() + 1)
    return float(len(track) / max(span, 1))


def require_min_frames(track: pd.DataFrame, minimum: int = 6) -> None:
    if len(track) < minimum:
        raise ValueError("insufficient_frames")


def check_coverage(track: pd.DataFrame, minimum: float = 0.5) -> None:
    if tracking_coverage(track) < minimum:
        raise ValueError("tracking_missing")


def size_stability(track: pd.DataFrame) -> float | None:
    if "equivalent_diameter_px" not in track:
        return None
    values = pd.to_numeric(track["equivalent_diameter_px"], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) < 3:
        return None
    return float(np.std(values) / max(np.mean(values), 1e-12))


def one_dimensional_score(x: np.ndarray, y: np.ndarray) -> float:
    pts = np.column_stack([x, y]).astype(float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 3:
        return 0.0
    pts = pts - np.mean(pts, axis=0)
    _, s, _ = np.linalg.svd(pts, full_matrices=False)
    if len(s) < 2 or s[0] <= 1e-12:
        return 1.0
    return float(1.0 - (s[1] / s[0]) ** 2)


def status_from_r2(r2: float | None, strict: bool = False) -> tuple[str, str]:
    if r2 is None:
        return "weak_valid", ""
    if r2 >= 0.85:
        return "valid", ""
    if r2 >= 0.65 and not strict:
        return "weak_valid", "low_fit_r2"
    return "invalid", "fit_failed"

