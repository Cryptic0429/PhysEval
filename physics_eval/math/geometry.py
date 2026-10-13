from __future__ import annotations

import numpy as np


def distance_xy(x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray) -> np.ndarray:
    return np.sqrt((np.asarray(x1) - np.asarray(x2)) ** 2 + (np.asarray(y1) - np.asarray(y2)) ** 2)


def align_tracks_on_time(track1, track2):
    merged = track1[["frame", "t", "cx_px", "cy_px"]].merge(
        track2[["frame", "cx_px", "cy_px"]],
        on="frame",
        suffixes=("_1", "_2"),
    )
    if merged.empty:
        raise ValueError("tracking_missing")
    return merged
