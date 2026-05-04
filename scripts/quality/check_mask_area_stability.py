#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class AreaSeries:
    scene: str
    object_id: int | None
    source: str
    frames: list[int]
    areas: list[float | None]
    bbox_areas: list[float | None]
    bboxes: list[list[float] | None]
    centroids: list[tuple[float, float] | None]
    mask_paths: list[str | None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether SAM2 mask area is stable in the middle part of a video. "
            "Large middle-window area changes are flagged for tracking review or rejection."
        )
    )
    parser.add_argument(
        "--workspace-root",
        type=str,
        default=".",
        help="Workspace root. Default: current directory.",
    )
    parser.add_argument(
        "--tracks-root",
        type=str,
        default=None,
        help="Root containing scene folders, default: <workspace-root>/sam2_tracks.",
    )
    parser.add_argument(
        "--scene-name",
        type=str,
        nargs="*",
        default=None,
        help="Scene names under tracks-root. If omitted in batch mode, all scenes with tracking_points.json or masks are checked.",
    )
    parser.add_argument(
        "--sam2-json",
        type=str,
        default=None,
        help="Single SAM2 tracking_points.json to check.",
    )
    parser.add_argument(
        "--mask-dir",
        type=str,
        default=None,
        help="Single mask directory to check. Used directly, or as an override for --sam2-json.",
    )
    parser.add_argument(
        "--mask-pattern",
        type=str,
        default="{frame_idx:05d}.png",
        help="Mask file pattern when mask-dir is given. Default: {frame_idx:05d}.png",
    )
    parser.add_argument(
        "--object-id",
        type=int,
        default=None,
        help="Only check this object id for multi-object tracking jsons.",
    )
    parser.add_argument(
        "--prefer-mask-area",
        action="store_true",
        help="Always count pixels from mask pngs when available, instead of trusting json area.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: <workspace-root>/runs/quality/mask_area_stability",
    )
    parser.add_argument(
        "--middle-start",
        type=float,
        default=0.15,
        help="Fraction of the series to trim from the beginning. Default: 0.15",
    )
    parser.add_argument(
        "--middle-end",
        type=float,
        default=0.85,
        help="Fraction of the series kept until this point. Default: 0.85",
    )
    parser.add_argument(
        "--min-valid-frames",
        type=int,
        default=6,
        help="Minimum valid middle-window frames required. Default: 6",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.70,
        help="Minimum valid-area coverage inside the middle window. Default: 0.70",
    )
    parser.add_argument(
        "--area-ratio-threshold",
        type=float,
        default=2.00,
        help="Flag if middle p95(area)/p5(area) exceeds this value. Default: 2.00",
    )
    parser.add_argument(
        "--area-cv-threshold",
        type=float,
        default=0.25,
        help="Flag if middle std(area)/mean(area) exceeds this value. Default: 0.25",
    )
    parser.add_argument(
        "--max-log-jump-threshold",
        type=float,
        default=0.45,
        help="Flag if max abs diff(log(area)) between adjacent middle frames exceeds this value. Default: 0.45",
    )
    parser.add_argument(
        "--smooth-scale-ratio-threshold",
        type=float,
        default=1.60,
        help="Classify as possible depth motion if smooth trend changes by this ratio. Default: 1.60",
    )
    parser.add_argument(
        "--trend-r2-threshold",
        type=float,
        default=0.65,
        help="Minimum linear R2 in log-area vs time to call the change smooth/monotonic. Default: 0.65",
    )
    parser.add_argument(
        "--bbox-size-cv-threshold",
        type=float,
        default=0.20,
        help="Flag if middle-window bbox width or height CV exceeds this value. Default: 0.20",
    )
    parser.add_argument(
        "--bbox-aspect-cv-threshold",
        type=float,
        default=0.20,
        help="Flag if middle-window bbox aspect-ratio CV exceeds this value. Default: 0.20",
    )
    parser.add_argument(
        "--aligned-iou-threshold",
        type=float,
        default=0.65,
        help=(
            "Flag if median centroid-aligned mask IoU to the reference mask is below this value. "
            "Only used when mask PNGs are available. Default: 0.65"
        ),
    )
    parser.add_argument(
        "--aligned-iou-sample-stride",
        type=int,
        default=5,
        help="Frame stride for centroid-aligned IoU sampling in the middle window. Default: 5",
    )
    parser.add_argument(
        "--max-centroid-step-ratio-threshold",
        type=float,
        default=2.50,
        help=(
            "Flag if max adjacent centroid step divided by median equivalent diameter exceeds this value. "
            "Default: 2.50"
        ),
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save area curve pngs for visual inspection.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing report files.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_mask_area(path: Path) -> int:
    with Image.open(path) as im:
        arr = np.asarray(im)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return int(np.count_nonzero(arr))


def read_mask_bool(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.asarray(im)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > 0


def mask_geometry(path: Path) -> tuple[int, list[float] | None, tuple[float, float] | None]:
    mask = read_mask_bool(path)
    ys, xs = np.nonzero(mask)
    area = int(len(xs))
    if area <= 0:
        return 0, None, None
    bbox = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
    centroid = (float(xs.mean()), float(ys.mean()))
    return area, bbox, centroid


def normalize_bbox(bbox: Any) -> list[float] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return None
    if x2 < x1 or y2 < y1:
        return None
    return [x1, y1, x2, y2]


def bbox_stats(bbox: Any) -> tuple[float | None, float | None, float | None, float | None]:
    box = normalize_bbox(bbox)
    if box is None:
        return None, None, None, None
    x1, y1, x2, y2 = box
    w = max(0.0, x2 - x1 + 1.0)
    h = max(0.0, y2 - y1 + 1.0)
    if w <= 0 or h <= 0:
        return None, None, None, None
    area = w * h
    aspect = max(w / h, h / w)
    return area, w, h, aspect


def bbox_area(bbox: Any) -> float | None:
    area, _, _, _ = bbox_stats(bbox)
    return area


def centroid_from_record(record: dict[str, Any], bbox: list[float] | None) -> tuple[float, float] | None:
    centroid = record.get("centroid")
    if isinstance(centroid, (list, tuple)) and len(centroid) >= 2:
        try:
            x, y = float(centroid[0]), float(centroid[1])
            if math.isfinite(x) and math.isfinite(y):
                return x, y
        except (TypeError, ValueError):
            pass
    x = record.get("x", record.get("cx", record.get("cx_px")))
    y = record.get("y", record.get("cy", record.get("cy_px")))
    if x is not None and y is not None:
        try:
            xf, yf = float(x), float(y)
            if math.isfinite(xf) and math.isfinite(yf):
                return xf, yf
        except (TypeError, ValueError):
            pass
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        return 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    return None


def parse_frame_idx(path: Path) -> int | None:
    try:
        return int(path.stem)
    except ValueError:
        return None


def resolve_mask_path(
    frame_idx: int,
    object_id: int | None,
    json_path: Path | None,
    mask_dir: Path | None,
    mask_pattern: str,
    record: dict[str, Any] | None = None,
) -> Path | None:
    candidates: list[Path] = []
    if record is not None:
        raw_mask_path = record.get("mask_path")
        if raw_mask_path:
            candidates.append(Path(str(raw_mask_path)).expanduser())
    rel_name = mask_pattern.format(frame_idx=frame_idx, object_id=object_id or 0)
    if mask_dir is not None:
        candidates.append(mask_dir / rel_name)
    if json_path is not None:
        base = json_path.parent / "masks"
        if object_id is not None:
            candidates.append(base / f"obj_{object_id:06d}" / rel_name)
        candidates.append(base / rel_name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def area_from_record(
    record: dict[str, Any],
    frame_idx: int,
    object_id: int | None,
    json_path: Path | None,
    mask_dir: Path | None,
    mask_pattern: str,
    prefer_mask_area: bool,
) -> float | None:
    mask_path = resolve_mask_path(frame_idx, object_id, json_path, mask_dir, mask_pattern, record)
    if prefer_mask_area and mask_path is not None:
        return float(read_mask_area(mask_path))

    area = record.get("area", None)
    if area is not None:
        try:
            area_f = float(area)
            if math.isfinite(area_f) and area_f > 0:
                return area_f
        except (TypeError, ValueError):
            pass

    if mask_path is not None:
        return float(read_mask_area(mask_path))
    return None


def load_series_from_masks(mask_dir: Path, scene: str, mask_pattern: str) -> AreaSeries:
    del mask_pattern
    files = sorted(p for p in mask_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    frames: list[int] = []
    areas: list[float | None] = []
    bboxes: list[list[float] | None] = []
    centroids: list[tuple[float, float] | None] = []
    mask_paths: list[str | None] = []
    for path in files:
        frame_idx = parse_frame_idx(path)
        if frame_idx is None:
            continue
        area, bbox, centroid = mask_geometry(path)
        frames.append(frame_idx)
        areas.append(float(area) if area > 0 else None)
        bboxes.append(bbox)
        centroids.append(centroid)
        mask_paths.append(str(path))
    if not frames:
        raise ValueError(f"No numbered mask images found in {mask_dir}")
    order = np.argsort(np.asarray(frames))
    return AreaSeries(
        scene=scene,
        object_id=None,
        source=str(mask_dir),
        frames=[frames[i] for i in order],
        areas=[areas[i] for i in order],
        bbox_areas=[bbox_area(bboxes[i]) for i in order],
        bboxes=[bboxes[i] for i in order],
        centroids=[centroids[i] for i in order],
        mask_paths=[mask_paths[i] for i in order],
    )


def infer_scene_name(path: Path) -> str:
    if path.name == "masks":
        return path.parent.name
    if path.name == "tracking_points.json":
        return path.parent.name
    return path.name


def load_series_from_json(
    json_path: Path,
    mask_dir: Path | None,
    mask_pattern: str,
    object_id: int | None,
    prefer_mask_area: bool,
) -> list[AreaSeries]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scene = data.get("video_info", {}).get("scene_name") or infer_scene_name(json_path)
    frames_in = data.get("frames", [])
    if not isinstance(frames_in, list) or not frames_in:
        raise ValueError(f"No frames found in {json_path}")

    object_ids: set[int | None] = set()
    for fr in frames_in:
        objects = fr.get("objects", None)
        if isinstance(objects, list):
            for obj in objects:
                oid = obj.get("object_id", None)
                if oid is not None:
                    object_ids.add(int(oid))
        else:
            oid = fr.get("object_id", data.get("object_id", None))
            object_ids.add(int(oid) if oid is not None else None)

    if object_id is not None:
        object_ids = {object_id}

    series_list: list[AreaSeries] = []
    for oid in sorted(object_ids, key=lambda x: -1 if x is None else x):
        frames: list[int] = []
        areas: list[float | None] = []
        bbox_areas: list[float | None] = []
        bboxes: list[list[float] | None] = []
        centroids: list[tuple[float, float] | None] = []
        mask_paths: list[str | None] = []
        for fr in frames_in:
            if "frame_idx" not in fr:
                continue
            frame_idx = int(fr["frame_idx"])
            objects = fr.get("objects", None)
            if isinstance(objects, list):
                obj = next((o for o in objects if int(o.get("object_id", -999999)) == oid), None)
                if obj is None:
                    continue
                record = obj
            else:
                record = fr

            box = normalize_bbox(record.get("bbox", None))
            mask_path = resolve_mask_path(
                frame_idx,
                oid,
                json_path,
                mask_dir,
                mask_pattern,
                record,
            )
            if record.get("valid", True) is False and record.get("area", 0) in (None, 0):
                area = None
            else:
                area = area_from_record(
                    record,
                    frame_idx,
                    oid,
                    json_path,
                    mask_dir,
                    mask_pattern,
                    prefer_mask_area,
                )
            centroid = centroid_from_record(record, box)
            frames.append(frame_idx)
            areas.append(area)
            bbox_areas.append(bbox_area(box))
            bboxes.append(box)
            centroids.append(centroid)
            mask_paths.append(str(mask_path) if mask_path is not None else None)

        if frames:
            series_list.append(
                AreaSeries(
                    scene=str(scene),
                    object_id=oid,
                    source=str(json_path),
                    frames=frames,
                    areas=areas,
                    bbox_areas=bbox_areas,
                    bboxes=bboxes,
                    centroids=centroids,
                    mask_paths=mask_paths,
                )
            )
    return series_list


def linear_trend_r2(y: np.ndarray) -> tuple[float, float]:
    if len(y) < 3:
        return 0.0, 0.0
    x = np.linspace(0.0, 1.0, len(y), dtype=np.float64)
    coef = np.polyfit(x, y, deg=1)
    pred = np.polyval(coef, x)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 if ss_tot <= 1e-12 else max(0.0, 1.0 - ss_res / ss_tot)
    return float(coef[0]), float(r2)


def safe_ratio(p95: float, p05: float) -> float:
    if not math.isfinite(p95) or not math.isfinite(p05) or p05 <= 0:
        return float("nan")
    return float(p95 / p05)


def finite_positive(values: list[float | None]) -> np.ndarray:
    arr = np.asarray([np.nan if v is None else float(v) for v in values], dtype=np.float64)
    return arr[np.isfinite(arr) & (arr > 0)]


def finite_values(values: list[float | None]) -> np.ndarray:
    arr = np.asarray([np.nan if v is None else float(v) for v in values], dtype=np.float64)
    return arr[np.isfinite(arr)]


def coefficient_of_variation(values: list[float | None]) -> float:
    arr = finite_positive(values)
    if len(arr) < 2:
        return float("nan")
    return float(np.std(arr) / max(float(np.mean(arr)), 1e-12))


def shift_mask(mask: np.ndarray, dx: int, dy: int, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    src_h, src_w = mask.shape
    dst_h, dst_w = shape
    src_x1 = max(0, -dx)
    src_y1 = max(0, -dy)
    dst_x1 = max(0, dx)
    dst_y1 = max(0, dy)
    width = min(src_w - src_x1, dst_w - dst_x1)
    height = min(src_h - src_y1, dst_h - dst_y1)
    if width <= 0 or height <= 0:
        return out
    out[dst_y1:dst_y1 + height, dst_x1:dst_x1 + width] = mask[src_y1:src_y1 + height, src_x1:src_x1 + width]
    return out


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    if union <= 0:
        return float("nan")
    return float(inter / union)


def aligned_iou_stats(series: AreaSeries, indices: list[int], stride: int) -> tuple[float, float, int, int | None]:
    usable = [
        i for i in indices
        if series.mask_paths[i] is not None
        and series.centroids[i] is not None
        and Path(str(series.mask_paths[i])).exists()
    ]
    if len(usable) < 2:
        return float("nan"), float("nan"), 0, None

    ref_i = usable[len(usable) // 2]
    ref_mask = read_mask_bool(Path(str(series.mask_paths[ref_i])))
    ref_centroid = series.centroids[ref_i]
    if ref_centroid is None:
        return float("nan"), float("nan"), 0, None

    stride = max(1, int(stride))
    sampled = usable[::stride]
    if ref_i not in sampled:
        sampled.append(ref_i)
    ious: list[float] = []
    for i in sorted(set(sampled)):
        if i == ref_i:
            continue
        centroid = series.centroids[i]
        if centroid is None or series.mask_paths[i] is None:
            continue
        mask = read_mask_bool(Path(str(series.mask_paths[i])))
        dx = int(round(ref_centroid[0] - centroid[0]))
        dy = int(round(ref_centroid[1] - centroid[1]))
        shifted = shift_mask(mask, dx, dy, ref_mask.shape)
        iou = mask_iou(shifted, ref_mask)
        if math.isfinite(iou):
            ious.append(iou)
    if not ious:
        return float("nan"), float("nan"), 0, ref_i
    vals = np.asarray(ious, dtype=np.float64)
    return float(np.median(vals)), float(np.min(vals)), int(len(vals)), ref_i


def centroid_step_stats(centroids: list[tuple[float, float] | None], equiv_diameters: list[float | None]) -> tuple[float, float, float, int | None, int | None]:
    steps: list[float] = []
    step_from: list[int] = []
    step_to: list[int] = []
    for i in range(1, len(centroids)):
        prev = centroids[i - 1]
        curr = centroids[i]
        if prev is None or curr is None:
            continue
        steps.append(float(np.hypot(curr[0] - prev[0], curr[1] - prev[1])))
        step_from.append(i - 1)
        step_to.append(i)
    if not steps:
        return float("nan"), float("nan"), float("nan"), None, None
    diam = finite_positive(equiv_diameters)
    median_diam = float(np.median(diam)) if len(diam) else float("nan")
    max_step = float(np.max(steps))
    median_step = float(np.median(steps))
    max_pos = int(np.argmax(steps))
    max_ratio = max_step / median_diam if math.isfinite(median_diam) and median_diam > 0 else float("nan")
    return max_step, median_step, float(max_ratio), step_from[max_pos], step_to[max_pos]


def analyze_series(series: AreaSeries, args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frames = list(series.frames)
    areas_all = np.asarray([np.nan if a is None else float(a) for a in series.areas], dtype=np.float64)
    bbox_stats_all = [bbox_stats(box) for box in series.bboxes]
    bbox_widths_all = [item[1] for item in bbox_stats_all]
    bbox_heights_all = [item[2] for item in bbox_stats_all]
    bbox_aspects_all = [item[3] for item in bbox_stats_all]
    equiv_diameters_all = [
        float(np.sqrt(4.0 * area / np.pi)) if math.isfinite(area) and area > 0 else None
        for area in areas_all
    ]
    n_total = len(frames)
    start_i = int(math.floor(n_total * args.middle_start))
    end_i = int(math.ceil(n_total * args.middle_end))
    start_i = max(0, min(start_i, n_total))
    end_i = max(start_i, min(end_i, n_total))

    mid_frames = frames[start_i:end_i]
    mid_indices = list(range(start_i, end_i))
    mid_areas_all = areas_all[start_i:end_i]
    valid_mask = np.isfinite(mid_areas_all) & (mid_areas_all > 0)
    mid_areas = mid_areas_all[valid_mask]
    mid_valid_frames = [mid_frames[i] for i, ok in enumerate(valid_mask) if ok]
    middle_count = len(mid_areas_all)
    valid_count = len(mid_areas)
    coverage = float(valid_count / max(middle_count, 1))

    reasons: list[str] = []
    status = "pass"
    decision = "keep"

    summary: dict[str, Any] = {
        "scene": series.scene,
        "object_id": series.object_id,
        "source": series.source,
        "total_frames": n_total,
        "valid_frames_total": int(np.sum(np.isfinite(areas_all) & (areas_all > 0))),
        "middle_start_index": start_i,
        "middle_end_index": max(start_i, end_i - 1),
        "middle_start_frame": mid_frames[0] if mid_frames else None,
        "middle_end_frame": mid_frames[-1] if mid_frames else None,
        "middle_frames": middle_count,
        "middle_valid_frames": valid_count,
        "middle_coverage": coverage,
    }

    if valid_count < args.min_valid_frames:
        reasons.append("insufficient_valid_middle_frames")
    if coverage < args.min_coverage:
        reasons.append("low_middle_coverage")

    if valid_count >= 2:
        logs = np.log(mid_areas)
        diffs = np.abs(np.diff(logs))
        max_jump = float(np.max(diffs)) if len(diffs) else 0.0
        max_jump_pos = int(np.argmax(diffs)) if len(diffs) else None
        jump_from = mid_valid_frames[max_jump_pos] if max_jump_pos is not None else None
        jump_to = mid_valid_frames[max_jump_pos + 1] if max_jump_pos is not None else None
    else:
        logs = np.asarray([], dtype=np.float64)
        max_jump = float("nan")
        jump_from = None
        jump_to = None

    if valid_count >= args.min_valid_frames:
        p05 = float(np.percentile(mid_areas, 5))
        p50 = float(np.percentile(mid_areas, 50))
        p95 = float(np.percentile(mid_areas, 95))
        mean = float(np.mean(mid_areas))
        std = float(np.std(mid_areas))
        cv = float(std / max(mean, 1e-12))
        ratio = safe_ratio(p95, p05)
        trend_slope, trend_r2 = linear_trend_r2(logs)
        trend_scale_ratio = float(math.exp(abs(logs[-1] - logs[0]))) if len(logs) >= 2 else float("nan")
    else:
        p05 = p50 = p95 = mean = std = cv = ratio = trend_slope = trend_r2 = trend_scale_ratio = float("nan")

    mid_bbox_width_cv = coefficient_of_variation(bbox_widths_all[start_i:end_i])
    mid_bbox_height_cv = coefficient_of_variation(bbox_heights_all[start_i:end_i])
    mid_bbox_aspect_cv = coefficient_of_variation(bbox_aspects_all[start_i:end_i])
    mid_bbox_width_median_vals = finite_positive(bbox_widths_all[start_i:end_i])
    mid_bbox_height_median_vals = finite_positive(bbox_heights_all[start_i:end_i])
    mid_bbox_aspect_median_vals = finite_positive(bbox_aspects_all[start_i:end_i])
    mid_bbox_width_median = float(np.median(mid_bbox_width_median_vals)) if len(mid_bbox_width_median_vals) else float("nan")
    mid_bbox_height_median = float(np.median(mid_bbox_height_median_vals)) if len(mid_bbox_height_median_vals) else float("nan")
    mid_bbox_aspect_median = float(np.median(mid_bbox_aspect_median_vals)) if len(mid_bbox_aspect_median_vals) else float("nan")

    aligned_iou_median, aligned_iou_min, aligned_iou_count, aligned_ref_idx = aligned_iou_stats(
        series,
        mid_indices,
        getattr(args, "aligned_iou_sample_stride", 5),
    )
    max_centroid_step, median_centroid_step, max_centroid_step_ratio, step_from_i, step_to_i = centroid_step_stats(
        series.centroids[start_i:end_i],
        equiv_diameters_all[start_i:end_i],
    )
    step_from_frame = mid_frames[step_from_i] if step_from_i is not None and step_from_i < len(mid_frames) else None
    step_to_frame = mid_frames[step_to_i] if step_to_i is not None and step_to_i < len(mid_frames) else None
    aligned_ref_frame = frames[aligned_ref_idx] if aligned_ref_idx is not None else None

    if math.isfinite(ratio) and ratio > args.area_ratio_threshold:
        reasons.append("large_area_ratio")
    if math.isfinite(cv) and cv > args.area_cv_threshold:
        reasons.append("large_area_cv")
    if math.isfinite(max_jump) and max_jump > args.max_log_jump_threshold:
        reasons.append("sudden_area_jump")
    if math.isfinite(mid_bbox_width_cv) and mid_bbox_width_cv > getattr(args, "bbox_size_cv_threshold", 0.20):
        reasons.append("large_bbox_width_cv")
    if math.isfinite(mid_bbox_height_cv) and mid_bbox_height_cv > getattr(args, "bbox_size_cv_threshold", 0.20):
        reasons.append("large_bbox_height_cv")
    if math.isfinite(mid_bbox_aspect_cv) and mid_bbox_aspect_cv > getattr(args, "bbox_aspect_cv_threshold", 0.20):
        reasons.append("large_bbox_aspect_cv")
    if math.isfinite(aligned_iou_median) and aligned_iou_median < getattr(args, "aligned_iou_threshold", 0.65):
        reasons.append("low_translation_aligned_mask_iou")
    if (
        math.isfinite(max_centroid_step_ratio)
        and max_centroid_step_ratio > getattr(args, "max_centroid_step_ratio_threshold", 2.50)
    ):
        reasons.append("large_centroid_jump")

    smooth_depth_like = (
        math.isfinite(trend_scale_ratio)
        and trend_scale_ratio >= args.smooth_scale_ratio_threshold
        and trend_r2 >= args.trend_r2_threshold
        and "sudden_area_jump" not in reasons
    )
    if smooth_depth_like:
        reasons.append("smooth_scale_change_possible_depth_motion")

    if reasons:
        status = "fail" if any(r in reasons for r in ("insufficient_valid_middle_frames", "low_middle_coverage")) else "warn"
        if "large_centroid_jump" in reasons:
            decision = "rerun_tracking"
        elif "sudden_area_jump" in reasons:
            decision = "rerun_tracking_or_reject"
        elif "smooth_scale_change_possible_depth_motion" in reasons:
            decision = "reject_or_verify_plane_motion"
        elif "low_translation_aligned_mask_iou" in reasons:
            decision = "review_segmentation_or_object_deformation"
        elif any(r in reasons for r in ("large_area_ratio", "large_area_cv", "large_bbox_width_cv", "large_bbox_height_cv", "large_bbox_aspect_cv")):
            decision = "review_or_reject_plane_motion"
        else:
            decision = "rerun_tracking"

    summary.update(
        {
            "median_area": p50,
            "mean_area": mean,
            "std_area": std,
            "p05_area": p05,
            "p95_area": p95,
            "area_ratio_p95_p05": ratio,
            "area_cv": cv,
            "max_log_jump": max_jump,
            "max_log_jump_frame_from": jump_from,
            "max_log_jump_frame_to": jump_to,
            "trend_log_slope": trend_slope,
            "trend_r2": trend_r2,
            "trend_scale_ratio": trend_scale_ratio,
            "bbox_width_median": mid_bbox_width_median,
            "bbox_height_median": mid_bbox_height_median,
            "bbox_aspect_median": mid_bbox_aspect_median,
            "bbox_width_cv": mid_bbox_width_cv,
            "bbox_height_cv": mid_bbox_height_cv,
            "bbox_aspect_cv": mid_bbox_aspect_cv,
            "translation_aligned_iou_median": aligned_iou_median,
            "translation_aligned_iou_min": aligned_iou_min,
            "translation_aligned_iou_count": aligned_iou_count,
            "translation_aligned_iou_ref_frame": aligned_ref_frame,
            "max_centroid_step_px": max_centroid_step,
            "median_centroid_step_px": median_centroid_step,
            "max_centroid_step_norm_by_diameter": max_centroid_step_ratio,
            "max_centroid_step_frame_from": step_from_frame,
            "max_centroid_step_frame_to": step_to_frame,
            "status": status,
            "decision": decision,
            "reasons": ";".join(reasons),
        }
    )

    frame_rows: list[dict[str, Any]] = []
    finite_mid = finite_positive([float(a) for a in mid_areas])
    median_area = float(np.median(finite_mid)) if len(finite_mid) else float("nan")
    prev_centroid: tuple[float, float] | None = None
    for i, (frame_idx, area, bbox_a, box, centroid) in enumerate(
        zip(frames, series.areas, series.bbox_areas, series.bboxes, series.centroids)
    ):
        area_f = None if area is None else float(area)
        norm_area = (
            area_f / median_area
            if area_f is not None and math.isfinite(area_f) and median_area > 0
            else None
        )
        _, bbox_w, bbox_h, bbox_aspect = bbox_stats(box)
        centroid_step = None
        if prev_centroid is not None and centroid is not None:
            centroid_step = float(np.hypot(centroid[0] - prev_centroid[0], centroid[1] - prev_centroid[1]))
        if centroid is not None:
            prev_centroid = centroid
        equiv_d = equiv_diameters_all[i]
        centroid_step_norm = (
            centroid_step / equiv_d
            if centroid_step is not None and equiv_d is not None and math.isfinite(equiv_d) and equiv_d > 0
            else None
        )
        frame_rows.append(
            {
                "scene": series.scene,
                "object_id": series.object_id,
                "frame_idx": frame_idx,
                "area": area_f,
                "area_norm_to_middle_median": norm_area,
                "bbox_area": bbox_a,
                "bbox_width": bbox_w,
                "bbox_height": bbox_h,
                "bbox_aspect": bbox_aspect,
                "centroid_x": centroid[0] if centroid is not None else None,
                "centroid_y": centroid[1] if centroid is not None else None,
                "centroid_step_px": centroid_step,
                "centroid_step_norm_by_diameter": centroid_step_norm,
                "mask_path": series.mask_paths[i],
                "in_middle_window": start_i <= i < end_i,
            }
        )
    return summary, frame_rows


def discover_inputs(args: argparse.Namespace) -> list[tuple[Path | None, Path | None, str]]:
    workspace_root = Path(args.workspace_root).expanduser().resolve()
    tracks_root = Path(args.tracks_root).expanduser().resolve() if args.tracks_root else workspace_root / "sam2_tracks"
    inputs: list[tuple[Path | None, Path | None, str]] = []

    if args.sam2_json:
        json_path = Path(args.sam2_json).expanduser().resolve()
        mask_dir = Path(args.mask_dir).expanduser().resolve() if args.mask_dir else None
        inputs.append((json_path, mask_dir, infer_scene_name(json_path)))
        return inputs

    if args.mask_dir:
        mask_dir = Path(args.mask_dir).expanduser().resolve()
        inputs.append((None, mask_dir, infer_scene_name(mask_dir)))
        return inputs

    scene_names = args.scene_name
    scene_dirs = []
    if scene_names:
        scene_dirs = [tracks_root / name for name in scene_names]
    elif tracks_root.exists():
        scene_dirs = sorted(p for p in tracks_root.iterdir() if p.is_dir())

    for scene_dir in scene_dirs:
        json_path = scene_dir / "tracking_points.json"
        masks_dir = scene_dir / "masks"
        if json_path.exists():
            inputs.append((json_path, None, scene_dir.name))
        elif masks_dir.exists():
            inputs.append((None, masks_dir, scene_dir.name))
    return inputs


def write_csv(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {path}. Use --overwrite to replace.")
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_plot(series: AreaSeries, summary: dict[str, Any], frame_rows: list[dict[str, Any]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    frames = [r["frame_idx"] for r in frame_rows]
    areas = [np.nan if r["area"] is None else float(r["area"]) for r in frame_rows]
    middle_frames = [r["frame_idx"] for r in frame_rows if r["in_middle_window"]]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(frames, areas, color="#2563eb", linewidth=1.8, label="mask area")
    if middle_frames:
        ax.axvspan(middle_frames[0], middle_frames[-1], color="#f59e0b", alpha=0.15, label="middle window")
    if math.isfinite(float(summary.get("median_area", float("nan")))):
        ax.axhline(float(summary["median_area"]), color="#111827", linestyle="--", linewidth=1.0, label="middle median")
    title_obj = "" if series.object_id is None else f" obj={series.object_id}"
    ax.set_title(f"{series.scene}{title_obj} area stability: {summary['status']} / {summary['decision']}")
    ax.set_xlabel("frame")
    ax.set_ylabel("mask area (px)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    workspace_root = Path(args.workspace_root).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else workspace_root / "runs" / "quality" / "mask_area_stability"
    )
    ensure_dir(output_dir)

    inputs = discover_inputs(args)
    if not inputs:
        raise RuntimeError("No inputs found. Provide --sam2-json, --mask-dir, or valid --tracks-root/--scene-name.")

    summaries: list[dict[str, Any]] = []
    detail_rows_all: list[dict[str, Any]] = []
    series_outputs: list[dict[str, Any]] = []

    for json_path, mask_dir, scene in inputs:
        if json_path is not None:
            series_list = load_series_from_json(
                json_path=json_path,
                mask_dir=mask_dir,
                mask_pattern=args.mask_pattern,
                object_id=args.object_id,
                prefer_mask_area=args.prefer_mask_area,
            )
        elif mask_dir is not None:
            series_list = [load_series_from_masks(mask_dir=mask_dir, scene=scene, mask_pattern=args.mask_pattern)]
        else:
            continue

        for series in series_list:
            summary, frame_rows = analyze_series(series, args)
            summaries.append(summary)
            detail_rows_all.extend(frame_rows)
            obj_suffix = "obj_unknown" if series.object_id is None else f"obj_{series.object_id:06d}"
            base_name = f"{series.scene}_{obj_suffix}"
            detail_path = output_dir / f"{base_name}_area_series.csv"
            write_csv(detail_path, frame_rows, overwrite=args.overwrite)
            plot_path = None
            if args.save_plots:
                plot_dir = output_dir / "plots"
                ensure_dir(plot_dir)
                plot_path = plot_dir / f"{base_name}_area.png"
                if plot_path.exists() and not args.overwrite:
                    raise FileExistsError(f"Output exists: {plot_path}. Use --overwrite to replace.")
                save_plot(series, summary, frame_rows, plot_path)

            series_outputs.append(
                {
                    "scene": series.scene,
                    "object_id": series.object_id,
                    "detail_csv": str(detail_path),
                    "plot": str(plot_path) if plot_path is not None else None,
                }
            )

    report_csv = output_dir / "mask_area_stability_report.csv"
    report_json = output_dir / "mask_area_stability_report.json"
    detail_csv = output_dir / "mask_area_stability_frame_details.csv"

    write_csv(report_csv, summaries, overwrite=args.overwrite)
    write_csv(detail_csv, detail_rows_all, overwrite=args.overwrite)

    if report_json.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {report_json}. Use --overwrite to replace.")
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "middle_start": args.middle_start,
                    "middle_end": args.middle_end,
                    "min_valid_frames": args.min_valid_frames,
                    "min_coverage": args.min_coverage,
                    "area_ratio_threshold": args.area_ratio_threshold,
                    "area_cv_threshold": args.area_cv_threshold,
                    "max_log_jump_threshold": args.max_log_jump_threshold,
                    "smooth_scale_ratio_threshold": args.smooth_scale_ratio_threshold,
                    "trend_r2_threshold": args.trend_r2_threshold,
                    "bbox_size_cv_threshold": args.bbox_size_cv_threshold,
                    "bbox_aspect_cv_threshold": args.bbox_aspect_cv_threshold,
                    "aligned_iou_threshold": args.aligned_iou_threshold,
                    "aligned_iou_sample_stride": args.aligned_iou_sample_stride,
                    "max_centroid_step_ratio_threshold": args.max_centroid_step_ratio_threshold,
                },
                "summaries": summaries,
                "series_outputs": series_outputs,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    counts: dict[str, int] = {}
    for item in summaries:
        decision = str(item["decision"])
        counts[decision] = counts.get(decision, 0) + 1

    print("=" * 72)
    print("Mask area stability check completed")
    print("=" * 72)
    print(f"Checked series: {len(summaries)}")
    for decision, count in sorted(counts.items()):
        print(f"{decision}: {count}")
    print(f"Report CSV: {report_csv}")
    print(f"Report JSON: {report_json}")
    print(f"Frame details CSV: {detail_csv}")

    warn_items = [s for s in summaries if s["decision"] != "keep"]
    if warn_items:
        print("\nFlagged:")
        for item in warn_items:
            obj = "" if item["object_id"] is None else f" obj={item['object_id']}"
            print(
                f"- {item['scene']}{obj}: {item['decision']} "
                f"(ratio={item['area_ratio_p95_p05']:.3f}, cv={item['area_cv']:.3f}, "
                f"bbox_w_cv={item['bbox_width_cv']:.3f}, bbox_h_cv={item['bbox_height_cv']:.3f}, "
                f"aligned_iou={item['translation_aligned_iou_median']:.3f}, "
                f"centroid_jump={item['max_centroid_step_norm_by_diameter']:.3f}, "
                f"reasons={item['reasons']})"
            )


if __name__ == "__main__":
    main()
