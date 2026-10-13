"""Mask geometry and PNG encoding for tracking artifacts."""
from pathlib import Path

import cv2
import numpy as np


def mask_to_centroid(mask_bool: np.ndarray):
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def mask_to_bbox(mask_bool: np.ndarray):
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return None
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max())
    y2 = int(ys.max())
    return [x1, y1, x2, y2]


def save_mask_png(mask_path: Path, mask_bool: np.ndarray):
    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    cv2.imwrite(str(mask_path), mask_u8)
