from __future__ import annotations

import math

import numpy as np

from physeval_compare.config import MODE_SPECS
from physeval_compare.detectors import bbox_iou, scan_indices
from physeval_compare.metrics import mask_geometry, mask_quality, tracking_quality


def test_mode_roles_are_paired() -> None:
    assert MODE_SPECS["fasterrcnn_sam2"].tracker == "sam2"
    assert MODE_SPECS["yolo_sam2"].detector == "yolo"
    assert MODE_SPECS["yolo_tam"].detector == "yolo"
    assert MODE_SPECS["yolo_tam"].tracker == "tam"


def test_uniform_scan_includes_endpoints() -> None:
    indices = scan_indices(121, 60, "uniform")
    assert indices[0] == 0
    assert indices[-1] == 120
    assert len(indices) == 60


def test_bbox_iou_identity_and_disjoint() -> None:
    assert bbox_iou([1, 2, 5, 8], [1, 2, 5, 8]) == 1.0
    assert bbox_iou([0, 0, 2, 2], [3, 3, 5, 5]) == 0.0


def test_mask_geometry_uses_exclusive_max_corner() -> None:
    mask = np.zeros((8, 9), dtype=bool)
    mask[2:5, 3:7] = True
    centroid, bbox, area = mask_geometry(mask)
    assert bbox == [3, 2, 7, 5]
    assert area == 12
    assert centroid == [4.5, 3.0]


def _moving_records(total: int = 10):
    records = []
    masks = {1: {}}
    for frame_idx in range(total):
        mask = np.zeros((30, 50), dtype=bool)
        mask[10:15, 5 + frame_idx:10 + frame_idx] = True
        masks[1][frame_idx] = mask
        records.append({
            "object_id": 1,
            "frame_idx": frame_idx,
            "valid": True,
            "centroid": [7.0 + frame_idx, 12.0],
            "bbox": [5 + frame_idx, 10, 10 + frame_idx, 15],
            "area": 25,
        })
    return records, masks


def test_tracking_and_mask_quality_pass_stable_motion() -> None:
    records, masks = _moving_records()
    tracking = tracking_quality(
        records,
        expected_objects=1,
        total_frames=10,
        min_coverage=0.5,
        min_motion_ratio=0.75,
    )
    quality = mask_quality(records, masks, total_frames=10)
    assert tracking["status"] == "pass"
    assert tracking["motion_extent_norm_by_diameter"] > 0.75
    assert quality["status"] == "pass"
    assert math.isclose(quality["score"], 1.0)


def test_static_track_fails_motion_gate() -> None:
    records, _ = _moving_records()
    for record in records:
        record["centroid"] = [7.0, 12.0]
    tracking = tracking_quality(
        records,
        expected_objects=1,
        total_frames=10,
        min_coverage=0.5,
        min_motion_ratio=0.75,
    )
    assert tracking["status"] == "flagged"
    assert "no_meaningful_motion" in tracking["reasons"]
