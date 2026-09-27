from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .types import Prompt, VideoInfo


BALL_TOKENS = {
    "ball", "sphere", "tennis", "basketball", "baseball", "soccer",
    "volleyball", "golf", "ping pong", "bearing",
}


def scan_indices(total: int, count: int, mode: str = "uniform") -> list[int]:
    if total <= 0:
        return []
    count = min(total, max(1, int(count)))
    if mode == "start" or count == total:
        return list(range(count))
    if mode == "middle":
        start = max(0, total // 2 - count // 2)
        return list(range(start, min(total, start + count)))
    return sorted(set(np.linspace(0, total - 1, count, dtype=int).tolist()))


def bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(intersection / max(area_a + area_b - intersection, 1e-9))


def negative_points(box: list[float], width: int, height: int, margin: int = 12) -> list[list[float]]:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return [
        [cx, max(0.0, y1 - margin)],
        [cx, min(float(height - 1), y2 + margin)],
        [max(0.0, x1 - margin), cy],
        [min(float(width - 1), x2 + margin), cy],
    ]


def _class_allowed(class_name: str, target_hint: str) -> bool:
    name = class_name.lower().replace("_", " ")
    hint = target_hint.lower().replace("_", " ").replace("-", " ")
    if any(token in hint for token in BALL_TOKENS):
        return name == "sports ball"
    if "person" in hint:
        return name == "person"
    ignored = {"standard", "solid", "object", "calibration", "tracking", "center", "none"}
    tokens = [token for token in hint.split() if token not in ignored]
    if not tokens:
        return True
    return any(token in name or name in token for token in tokens)


def _candidate(
    *,
    frame_idx: int,
    box: list[float],
    confidence: float,
    class_id: int | None,
    class_name: str | None,
    detector: str,
    info: VideoInfo,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    x1, y1, x2, y2 = [float(value) for value in box]
    x1 = max(0.0, min(x1, info.width - 1.0))
    x2 = max(0.0, min(x2, info.width - 1.0))
    y1 = max(0.0, min(y1, info.height - 1.0))
    y2 = max(0.0, min(y2, info.height - 1.0))
    bw, bh = x2 - x1, y2 - y1
    if bw <= 2 or bh <= 2:
        return None
    area = bw * bh
    image_area = float(info.width * info.height)
    aspect = max(bw / bh, bh / bw)
    if area > 0.70 * image_area or aspect > 8.0:
        return None
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    max_radius = max(math.hypot(info.width, info.height) / 2.0, 1.0)
    centrality = 1.0 - min(1.0, math.hypot(cx - info.width / 2.0, cy - info.height / 2.0) / max_radius)
    size_prior = min(1.0, area / max(0.18 * image_area, 1.0))
    score = float(confidence) * (0.75 + 0.25 * centrality) * (0.70 + 0.30 * size_prior)
    return {
        "frame_idx": int(frame_idx),
        "bbox": [x1, y1, x2, y2],
        "point": [cx, cy],
        "score": score,
        "confidence": float(confidence),
        "class_id": class_id,
        "class_name": class_name,
        "detector": detector,
        "area": area,
        "aspect_ratio": aspect,
        **(extra or {}),
    }


def _distinct(items: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda value: value["score"], reverse=True):
        if all(bbox_iou(item["bbox"], previous["bbox"]) < 0.35 for previous in chosen):
            chosen.append(item)
        if len(chosen) >= count:
            break
    return chosen


def _same_frame_prompts(
    candidates: list[dict[str, Any]],
    *,
    count: int,
    info: VideoInfo,
    detector: str,
    scan: list[int],
) -> list[Prompt]:
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        by_frame[int(item["frame_idx"])].append(item)
    groups = []
    for frame_idx, items in by_frame.items():
        chosen = _distinct(items, count)
        if len(chosen) == count:
            groups.append((sum(float(item["score"]) for item in chosen), -frame_idx, chosen))
    if not groups:
        return []
    _, _, selected = max(groups, key=lambda value: (value[0], value[1]))
    selected = sorted(selected, key=lambda item: item["point"][0])
    prompts = []
    for obj_id, item in enumerate(selected, start=1):
        prompts.append(Prompt(
            obj_id=obj_id,
            frame_idx=int(item["frame_idx"]),
            point=[float(value) for value in item["point"]],
            box=[float(value) for value in item["bbox"]],
            score=float(item["score"]),
            detector=detector,
            class_id=item.get("class_id"),
            class_name=item.get("class_name"),
            confidence=item.get("confidence"),
            negative_points=negative_points(item["bbox"], info.width, info.height),
            details={
                key: value for key, value in item.items()
                if key not in {"frame_idx", "bbox", "point", "score", "confidence", "class_id", "class_name", "detector"}
            } | {"scan_indices": scan},
        ))
    return prompts


class YoloDetector:
    name = "yolo"

    def __init__(self, weights: str, device: str, confidence: float, iou: float) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("YOLO mode requires the `ultralytics` package.") from exc
        self.model = YOLO(weights)
        self.device = device
        self.confidence = confidence
        self.iou = iou

    def detect(self, info: VideoInfo, target_hint: str, count: int, scan_frames: int, scan_mode: str) -> list[Prompt]:
        paths = sorted(info.frames_dir.glob("*.jpg"))
        indices = scan_indices(len(paths), scan_frames, scan_mode)
        candidates: list[dict[str, Any]] = []
        names = getattr(self.model, "names", {}) or {}
        for frame_idx in indices:
            results = self.model.predict(
                source=str(paths[frame_idx]), conf=self.confidence, iou=self.iou,
                device=self.device, verbose=False,
            )
            if not results or getattr(results[0], "boxes", None) is None:
                continue
            for box in results[0].boxes:
                class_id = int(box.cls.detach().cpu().reshape(-1)[0])
                class_name = str(names.get(class_id, class_id))
                if not _class_allowed(class_name, target_hint):
                    continue
                item = _candidate(
                    frame_idx=frame_idx,
                    box=box.xyxy.detach().cpu().reshape(-1).tolist()[:4],
                    confidence=float(box.conf.detach().cpu().reshape(-1)[0]),
                    class_id=class_id,
                    class_name=class_name,
                    detector=self.name,
                    info=info,
                )
                if item is not None:
                    candidates.append(item)
        return _same_frame_prompts(candidates, count=count, info=info, detector=self.name, scan=indices)


class FasterRcnnDetector:
    name = "fasterrcnn"

    def __init__(self, device: str, confidence: float) -> None:
        try:
            import torch
            from torchvision.models.detection import (
                FasterRCNN_ResNet50_FPN_Weights,
                fasterrcnn_resnet50_fpn,
            )
        except ImportError as exc:
            raise RuntimeError("Faster R-CNN mode requires torch and torchvision.") from exc
        self.torch = torch
        self.weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        self.model = fasterrcnn_resnet50_fpn(weights=self.weights).to(device).eval()
        self.categories = self.weights.meta["categories"]
        self.device = device
        self.confidence = confidence

    def detect(self, info: VideoInfo, target_hint: str, count: int, scan_frames: int, scan_mode: str) -> list[Prompt]:
        from PIL import Image
        from torchvision.transforms.functional import to_tensor

        paths = sorted(info.frames_dir.glob("*.jpg"))
        indices = scan_indices(len(paths), scan_frames, scan_mode)
        candidates: list[dict[str, Any]] = []
        with self.torch.inference_mode():
            for frame_idx in indices:
                image = Image.open(paths[frame_idx]).convert("RGB")
                prediction = self.model([to_tensor(image).to(self.device)])[0]
                for box, label, score in zip(
                    prediction["boxes"], prediction["labels"], prediction["scores"], strict=True,
                ):
                    confidence = float(score.detach().cpu())
                    if confidence < self.confidence:
                        continue
                    class_id = int(label.detach().cpu())
                    class_name = str(self.categories[class_id])
                    if not _class_allowed(class_name, target_hint):
                        continue
                    item = _candidate(
                        frame_idx=frame_idx,
                        box=box.detach().cpu().tolist(),
                        confidence=confidence,
                        class_id=class_id,
                        class_name=class_name,
                        detector=self.name,
                        info=info,
                    )
                    if item is not None:
                        candidates.append(item)
        return _same_frame_prompts(candidates, count=count, info=info, detector=self.name, scan=indices)


class MotionDetector:
    name = "motion"

    def __init__(self, threshold: float = 25.0, min_area: int = 40, max_area_ratio: float = 0.20) -> None:
        self.threshold = float(threshold)
        self.min_area = int(min_area)
        self.max_area_ratio = float(max_area_ratio)

    def detect(self, info: VideoInfo, target_hint: str, count: int, scan_frames: int, scan_mode: str) -> list[Prompt]:
        import cv2

        del target_hint
        paths = sorted(info.frames_dir.glob("*.jpg"))
        indices = scan_indices(len(paths), scan_frames, scan_mode)
        gray_frames = []
        for frame_idx in indices:
            gray = cv2.imread(str(paths[frame_idx]), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise RuntimeError(f"Cannot read frame: {paths[frame_idx]}")
            gray_frames.append(gray)
        background = np.median(np.stack(gray_frames, axis=0), axis=0).astype(np.uint8)
        candidates: list[dict[str, Any]] = []
        image_area = float(info.width * info.height)
        kernel = np.ones((3, 3), dtype=np.uint8)
        for frame_idx, gray in zip(indices, gray_frames, strict=True):
            difference = cv2.absdiff(gray, background)
            binary = (difference >= self.threshold).astype(np.uint8) * 255
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < self.min_area or area > self.max_area_ratio * image_area:
                    continue
                x, y, width, height = cv2.boundingRect(contour)
                fill_ratio = area / max(float(width * height), 1.0)
                aspect = max(width / max(height, 1), height / max(width, 1))
                if fill_ratio < 0.08 or aspect > 6.0:
                    continue
                roi = difference[y:y + height, x:x + width]
                motion_strength = float(np.mean(roi)) / 255.0 if roi.size else 0.0
                item = _candidate(
                    frame_idx=frame_idx,
                    box=[x - 8, y - 8, x + width + 8, y + height + 8],
                    confidence=min(1.0, 0.35 + 2.5 * motion_strength + 0.25 * fill_ratio),
                    class_id=None,
                    class_name=None,
                    detector=self.name,
                    info=info,
                    extra={"motion_strength": motion_strength, "fill_ratio": fill_ratio},
                )
                if item is not None:
                    candidates.append(item)
        return _same_frame_prompts(candidates, count=count, info=info, detector=self.name, scan=indices)


def detect_with_fallback(
    detector: Any,
    motion_detector: MotionDetector,
    *,
    info: VideoInfo,
    target_hint: str,
    count: int,
    scan_frames: int,
    scan_mode: str,
) -> tuple[list[Prompt], dict[str, Any]]:
    error: str | None = None
    try:
        prompts = detector.detect(info, target_hint, count, scan_frames, scan_mode)
    except Exception as exc:
        prompts = []
        error = f"{type(exc).__name__}: {exc}"
    if len(prompts) == count:
        return prompts, {
            "requested_detector": detector.name,
            "detector_used": detector.name,
            "used_motion_fallback": False,
            "fallback_reason": None,
        }
    fallback_reason = error or f"{detector.name} returned {len(prompts)}/{count} same-frame prompts"
    prompts = motion_detector.detect(info, target_hint, count, scan_frames, scan_mode)
    if len(prompts) != count:
        raise RuntimeError(
            f"Both {detector.name} and motion fallback failed: {fallback_reason}; "
            f"motion returned {len(prompts)}/{count} prompts"
        )
    for prompt in prompts:
        prompt.details["fallback_from"] = detector.name
        prompt.details["fallback_reason"] = fallback_reason
    return prompts, {
        "requested_detector": detector.name,
        "detector_used": "motion",
        "used_motion_fallback": True,
        "fallback_reason": fallback_reason,
    }
