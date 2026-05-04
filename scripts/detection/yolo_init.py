#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


COCO_HINT_CLASSES = {
    "ball": {"sports ball"},
    "sphere": {"sports ball"},
    "tennis": {"sports ball"},
    "basket": {"sports ball"},
    "basketball": {"sports ball"},
    "person": {"person"},
}


def _list_jpg_frames(frames_dir: Path) -> list[Path]:
    return sorted(frames_dir.glob("*.jpg"))


def _select_scan_indices(total: int, scan_frames: int, scan_mode: str) -> list[int]:
    n_scan = min(max(2, int(scan_frames)), total)
    mode = (scan_mode or "uniform").lower()
    if mode == "start" or total <= n_scan:
        return list(range(n_scan))
    if mode == "middle":
        center = total // 2
        start = max(0, center - n_scan // 2)
        end = min(total, start + n_scan)
        return list(range(max(0, end - n_scan), end))
    return sorted(set(np.linspace(0, total - 1, n_scan, dtype=int).tolist()))


def _parse_classes(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {part.strip().lower() for part in raw.split(",") if part.strip()}


def _hint_tokens(target_hint: str) -> set[str]:
    hint = (target_hint or "").lower()
    for ch in "-_/":
        hint = hint.replace(ch, " ")
    ignored = {"standard", "solid", "object", "calibration", "tracking"}
    return {tok for tok in hint.split() if tok and tok not in ignored}


def _class_allowed(
    class_id: int,
    class_name: str,
    explicit_classes: set[str],
    target_hint: str,
) -> bool:
    name = class_name.lower()
    if explicit_classes:
        return str(class_id) in explicit_classes or name in explicit_classes

    hint = (target_hint or "").lower()
    for key, allowed_names in COCO_HINT_CLASSES.items():
        if key in hint:
            return name in allowed_names

    tokens = _hint_tokens(target_hint)
    if tokens:
        return any(tok in name or name in tok for tok in tokens)
    return True


def _bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(inter / max(area_a + area_b - inter, 1e-9))


def _select_distinct(candidates: list[dict[str, Any]], num_objects: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for cand in sorted(candidates, key=lambda item: item["score"], reverse=True):
        if all(_bbox_iou(cand["bbox"], prev["bbox"]) < 0.35 for prev in selected):
            selected.append(cand)
            if len(selected) >= num_objects:
                break
    return selected


def detect_yolo_prompts(
    frames_dir: Path,
    width: int,
    height: int,
    num_objects: int,
    scan_frames: int,
    scan_mode: str,
    weights: str | None,
    conf: float,
    iou: float,
    target_hint: str,
    classes: str | None = None,
) -> list[dict[str, Any]]:
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - depends on server env
        raise RuntimeError("YOLO detector requires `pip install ultralytics`.") from exc

    frame_paths = _list_jpg_frames(frames_dir)
    if not frame_paths:
        raise RuntimeError(f"No jpg frames found in: {frames_dir}")

    model = YOLO(weights or "yolov8n.pt")
    names = getattr(model, "names", {}) or {}
    explicit_classes = _parse_classes(classes)
    indices = _select_scan_indices(len(frame_paths), scan_frames, scan_mode)
    image_area = max(1.0, float(width * height))
    candidates: list[dict[str, Any]] = []

    for frame_idx in indices:
        frame_path = frame_paths[frame_idx]
        results = model.predict(
            source=str(frame_path),
            conf=float(conf),
            iou=float(iou),
            verbose=False,
        )
        if not results:
            continue
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            xyxy = box.xyxy.detach().cpu().numpy().reshape(-1).tolist()
            if len(xyxy) < 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in xyxy[:4]]
            cls_id = int(box.cls.detach().cpu().numpy().reshape(-1)[0])
            cls_name = str(names.get(cls_id, cls_id))
            if not _class_allowed(cls_id, cls_name, explicit_classes, target_hint):
                continue
            confidence = float(box.conf.detach().cpu().numpy().reshape(-1)[0])
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            bbox_area = bw * bh
            if bbox_area <= 4 or bbox_area > 0.70 * image_area:
                continue
            aspect = max(bw / bh, bh / bw)
            if aspect > 8.0:
                continue
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            centrality = 1.0 - min(
                1.0,
                float(np.hypot(cx - width / 2.0, cy - height / 2.0) / max(np.hypot(width, height) / 2.0, 1.0)),
            )
            size_prior = min(1.0, bbox_area / (0.18 * image_area))
            score = confidence * (0.75 + 0.25 * centrality) * (0.70 + 0.30 * size_prior)
            candidates.append({
                "frame_idx": int(frame_idx),
                "point": [float(cx), float(cy)],
                "bbox": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
                "motion_bbox": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
                "area": int(round(bbox_area)),
                "bbox_area": int(round(bbox_area)),
                "fill_ratio": 1.0,
                "aspect_ratio": float(aspect),
                "motion_strength": 0.0,
                "intensity_delta": 0.0,
                "color_change": None,
                "shadow_score": 0.0,
                "texture_strength": 0.0,
                "score": float(score),
                "cluster_score": float(score),
                "method": "yolo",
                "detector": "yolo",
                "yolo_class_id": cls_id,
                "yolo_class_name": cls_name,
                "yolo_confidence": confidence,
                "n_candidates": 0,
                "scan_frames": len(indices),
                "scan_mode": str(scan_mode),
                "scan_indices": [int(i) for i in indices],
                "threshold": None,
                "min_area": None,
                "max_area_ratio": None,
                "min_fill_ratio": None,
                "max_aspect_ratio": None,
                "shadow_filter": None,
                "object_refine": True,
                "object_refined": True,
                "object_refine_source": "yolo_bbox",
                "box_expand": None,
                "target_hint": str(target_hint),
            })

    if not candidates:
        raise RuntimeError(
            "YOLO did not find a usable target. Try custom --yolo-weights, "
            "--yolo-classes, lower --yolo-conf, or fallback to motion detector."
        )

    chosen = _select_distinct(candidates, max(1, int(num_objects)))
    for item in chosen:
        item["n_candidates"] = len(candidates)
    return chosen


def draw_yolo_debug(frame_path: Path, prompts: list[dict[str, Any]], out_path: Path) -> None:
    if not prompts:
        return
    frame = cv2.imread(str(frame_path))
    if frame is None:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for i, item in enumerate(prompts):
        x1, y1, x2, y2 = item["bbox"]
        cx, cy = item["point"]
        color = (0, 255, 255) if i == 0 else (255, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (int(round(cx)), int(round(cy))), 5, (0, 0, 255), -1)
        label = f"YOLO {item.get('yolo_class_name')} {item.get('yolo_confidence', 0.0):.2f}"
        cv2.putText(frame, label, (max(0, x1), max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imwrite(str(out_path), frame)
