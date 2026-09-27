#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import cv2
import json
import time
import shutil
import argparse
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

WORKSPACE_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT_FOR_IMPORTS))


SAM2_PRECISION_CHOICES = ("auto", "fp32", "fp16", "bf16")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SAM2 video tracking inside the workspace and save shared frames + tracking outputs."
    )

    parser.add_argument("--workspace-root", type=str, default=str(WORKSPACE_ROOT_FOR_IMPORTS),
                        help="Workspace root. Default: inferred from this script.")
    parser.add_argument("--scene-name", type=str, required=True,
                        help="Scene / video id, e.g. video_001")
    parser.add_argument("--video-path", type=str, required=True,
                        help="Input video path")
    parser.add_argument("--copy-video-to-workspace", action="store_true",
                        help="Copy input video into data/custom/<scene>/raw/input.mp4")

    parser.add_argument("--sam2-repo", type=str, default=str(WORKSPACE_ROOT_FOR_IMPORTS / "repo" / "sam2"),
                        help="Path to SAM2 repo root")
    parser.add_argument("--model-cfg", type=str, required=True,
                        help="SAM2 model config path, e.g. configs/sam2.1/sam2.1_hiera_b+.yaml")
    parser.add_argument("--model-weights", type=str, required=True,
                        help="Path to SAM2 checkpoint")

    parser.add_argument("--init-frame-idx", type=int, default=0,
                        help="Manual prompt frame. In auto mode this is replaced by the detected frame.")
    parser.add_argument("--init-obj-id", type=int, default=1)
    parser.add_argument("--init-point-x", type=float, default=None)
    parser.add_argument("--init-point-y", type=float, default=None)
    parser.add_argument("--init-label", type=int, default=1,
                        help="1 = foreground point")
    parser.add_argument("--init-box", type=float, nargs=4, default=None,
                        metavar=("X1", "Y1", "X2", "Y2"),
                        help="Optional manual box prompt in image coordinates.")

    parser.add_argument("--auto-init", action="store_true",
                        help="Automatically find the main moving object and use it as the SAM2 prompt.")
    parser.add_argument("--auto-init-detector", type=str, default="motion",
                        choices=["motion", "yolo", "yolo_then_motion"],
                        help="Detector used to create the initial SAM2 prompt. Default: motion.")
    parser.add_argument("--auto-init-num-objects", type=int, default=1,
                        help="Number of moving objects to initialize automatically. Default tracks one object.")
    parser.add_argument("--auto-init-scan-frames", type=int, default=60,
                        help="Number of frames used to estimate motion.")
    parser.add_argument("--auto-init-scan-mode", type=str, default="uniform",
                        choices=["start", "middle", "uniform"],
                        help="Which part of the video to scan for automatic initialization.")
    parser.add_argument("--no-bidirectional-propagation", action="store_true",
                        help="Only propagate forward from the init frame. By default nonzero init frames track both directions.")
    parser.add_argument("--auto-init-threshold", type=float, default=25.0,
                        help="Grayscale difference threshold for motion segmentation.")
    parser.add_argument("--auto-init-min-area", type=int, default=40,
                        help="Minimum moving component area in pixels.")
    parser.add_argument("--auto-init-max-area-ratio", type=float, default=0.20,
                        help="Reject moving components larger than this fraction of the image.")
    parser.add_argument("--auto-init-box-pad", type=int, default=8,
                        help="Padding around the detected moving-object box.")
    parser.add_argument("--auto-init-min-fill-ratio", type=float, default=0.08,
                        help="Reject sparse/slender motion components below this component bbox fill ratio.")
    parser.add_argument("--auto-init-max-aspect-ratio", type=float, default=6.0,
                        help="Reject very elongated motion components, e.g. springs or rods.")
    parser.add_argument("--auto-init-min-separation-ratio", type=float, default=0.03,
                        help="Minimum point separation for selecting multiple auto-init objects, as image diagonal ratio.")
    parser.add_argument("--auto-init-object-refine", action=argparse.BooleanOptionalAction, default=True,
                        help="Refine motion candidates to the surrounding whole visible object before prompting SAM2.")
    parser.add_argument("--auto-init-box-expand", type=float, default=2.8,
                        help="Expansion factor around a motion component when searching for the whole object.")
    parser.add_argument("--auto-init-target-hint", type=str, default="auto",
                        help="Optional target hint from metadata, e.g. standard_tennis_ball, basketball, hockey_puck.")
    parser.add_argument("--disable-auto-init-shadow-filter", action="store_true",
                        help="Disable the shadow-like motion penalty used during auto initialization.")
    parser.add_argument("--no-auto-init-negative-points", action="store_true",
                        help="Do not add automatic background negative points around the auto-init box.")
    parser.add_argument("--no-auto-init-box", action="store_true",
                        help="Use only a point prompt for auto initialization.")
    parser.add_argument("--save-auto-init-debug", action="store_true",
                        help="Save a debug image showing the detected auto-init prompt.")
    parser.add_argument("--yolo-weights", type=str, default=None,
                        help="YOLO weights, e.g. yolo11n.pt or a custom trained .pt.")
    parser.add_argument("--yolo-conf", type=float, default=0.25,
                        help="YOLO confidence threshold.")
    parser.add_argument("--yolo-iou", type=float, default=0.50,
                        help="YOLO NMS IoU threshold.")
    parser.add_argument("--yolo-classes", type=str, default=None,
                        help="Optional comma-separated YOLO class names or ids to allow.")

    parser.add_argument("--save-vis-video", action="store_true",
                        help="Save tracking visualization video")
    parser.add_argument("--save-mask-png", action="store_true",
                        help="Save binary mask png for each frame")
    parser.add_argument("--overwrite-frames", action="store_true",
                        help="Re-extract frames even if frames dir already has jpgs")
    parser.add_argument("--overwrite-outputs", action="store_true",
                        help="Overwrite tracking json/video/masks if already exist")

    parser.add_argument("--vos-optimized", action="store_true")
    parser.add_argument("--offload-video-to-cpu", action="store_true")
    parser.add_argument("--offload-state-to-cpu", action="store_true")
    parser.add_argument(
        "--precision",
        type=str,
        default="auto",
        choices=SAM2_PRECISION_CHOICES,
        help=(
            "SAM2 inference precision. 'auto' uses BF16 when supported and FP16 otherwise; "
            "'fp32' disables autocast. The selected policy covers initialization, prompt "
            "injection, and every propagation iteration."
        ),
    )

    return parser.parse_args()


def resolve_precision_policy(
    requested: str,
    *,
    cuda_available: bool,
    bf16_supported: bool,
):
    """Resolve a requested SAM2 precision without touching CUDA.

    Keeping this decision function pure makes the precision behavior testable on
    CPU-only machines. Runtime CUDA checks are supplied by ``main``.
    """
    requested = str(requested).strip().lower()
    if requested not in SAM2_PRECISION_CHOICES:
        choices = ", ".join(SAM2_PRECISION_CHOICES)
        raise ValueError(f"Unsupported precision '{requested}'. Choose one of: {choices}.")

    if requested == "auto":
        if not cuda_available:
            resolved = "fp32"
        else:
            resolved = "bf16" if bf16_supported else "fp16"
    else:
        resolved = requested

    if resolved in ("fp16", "bf16") and not cuda_available:
        raise RuntimeError(
            f"--precision {resolved} requires CUDA. Use --precision fp32 on CPU."
        )
    if resolved == "bf16" and not bf16_supported:
        raise RuntimeError(
            "--precision bf16 was requested, but the active CUDA device does not "
            "report BF16 support. Use --precision fp16 or --precision fp32."
        )

    return {
        "requested": requested,
        "resolved": resolved,
        "autocast_enabled": resolved in ("fp16", "bf16"),
        "autocast_device_type": "cuda" if resolved in ("fp16", "bf16") else None,
        "autocast_dtype": resolved if resolved in ("fp16", "bf16") else None,
    }


@contextmanager
def sam2_inference_context(precision_policy):
    """Apply one inference/autocast policy to a complete SAM2 operation."""
    with torch.inference_mode():
        if not precision_policy["autocast_enabled"]:
            yield
            return

        dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[precision_policy["resolved"]]
        with torch.autocast(
            device_type=precision_policy["autocast_device_type"],
            dtype=dtype,
        ):
            yield


def initialize_predictor_state(
    predictor,
    *,
    frames_dir: Path,
    offload_video_to_cpu: bool,
    offload_state_to_cpu: bool,
    prompts,
    inference_context_factory,
):
    """Initialize SAM2 and inject prompts inside the supplied inference context."""
    with inference_context_factory():
        try:
            inference_state = predictor.init_state(
                video_path=str(frames_dir),
                offload_video_to_cpu=offload_video_to_cpu,
                offload_state_to_cpu=offload_state_to_cpu,
            )
        except TypeError:
            inference_state = predictor.init_state(video_path=str(frames_dir))

        for prompt in prompts:
            positive_points = prompt.get("positive_points", prompt.get("points"))
            if positive_points is None:
                positive_points = [prompt["point"]]
            negative_points = list(prompt.get("negative_points", []))
            points_in = list(positive_points) + negative_points
            labels_in = [prompt["label"]] * len(positive_points) + [0] * len(negative_points)
            point = np.array(points_in, dtype=np.float32)
            label = np.array(labels_in, dtype=np.int32)

            prompt_kwargs = {
                "inference_state": inference_state,
                "frame_idx": int(prompt["frame_idx"]),
                "obj_id": int(prompt["obj_id"]),
                "points": point,
                "labels": label,
            }
            if prompt["box"] is not None:
                prompt_kwargs["box"] = np.array(prompt["box"], dtype=np.float32)

            predictor.add_new_points_or_box(**prompt_kwargs)

    return inference_state


def propagate_in_video_with_context(
    predictor,
    inference_state,
    *,
    start_frame_idx: int,
    reverse: bool,
    inference_context_factory,
):
    """Yield a complete lazy SAM2 propagation while its precision context is active."""
    with inference_context_factory():
        try:
            iterator = predictor.propagate_in_video(
                inference_state,
                start_frame_idx=int(start_frame_idx),
                reverse=bool(reverse),
            )
        except TypeError:
            if reverse:
                print("Current SAM2 predictor does not support reverse propagation; skipping reverse pass.")
                return
            iterator = predictor.propagate_in_video(inference_state)

        # SAM2 returns a lazy generator. The context must remain entered while
        # every item is produced, not only while the generator is constructed.
        yield from iterator


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def list_jpg_frames(frames_dir: Path):
    return sorted(frames_dir.glob("*.jpg"))


def read_video_meta(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return float(fps), frame_count, width, height


def clamp_box(box, width: int, height: int):
    x1, y1, x2, y2 = box
    x1 = int(np.clip(np.floor(x1), 0, width - 1))
    y1 = int(np.clip(np.floor(y1), 0, height - 1))
    x2 = int(np.clip(np.ceil(x2), 0, width - 1))
    y2 = int(np.clip(np.ceil(y2), 0, height - 1))
    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)
    return [x1, y1, x2, y2]


def expand_box(box, width: int, height: int, factor: float, min_pad: int = 10):
    x1, y1, x2, y2 = [float(v) for v in box]
    bw = max(1.0, x2 - x1 + 1.0)
    bh = max(1.0, y2 - y1 + 1.0)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    side_w = max(bw * factor, bw + 2 * min_pad)
    side_h = max(bh * factor, bh + 2 * min_pad)
    return clamp_box(
        [cx - 0.5 * side_w, cy - 0.5 * side_h, cx + 0.5 * side_w, cy + 0.5 * side_h],
        width,
        height,
    )


def component_candidates(
    mask: np.ndarray,
    diff: np.ndarray,
    gray_frame: np.ndarray,
    background_gray: np.ndarray,
    color_frame: np.ndarray | None,
    background_color: np.ndarray | None,
    frame_idx: int,
    width: int,
    height: int,
    min_area: int,
    max_area_ratio: float,
    box_pad: int,
    min_fill_ratio: float,
    max_aspect_ratio: float,
    shadow_filter: bool,
):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    max_area = int(width * height * max_area_ratio)
    candidates = []

    for label_idx in range(1, num_labels):
        x = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y = int(stats[label_idx, cv2.CC_STAT_TOP])
        w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_idx, cv2.CC_STAT_AREA])

        if area < min_area or area > max_area:
            continue
        if w <= 1 or h <= 1:
            continue

        comp_mask = labels == label_idx
        motion_strength = float(diff[comp_mask].mean()) if area > 0 else 0.0
        intensity_delta = float((gray_frame.astype(np.float32) - background_gray.astype(np.float32))[comp_mask].mean())
        bbox_area = max(w * h, 1)
        fill_ratio = area / float(bbox_area)
        aspect_ratio = float(max(w / float(h), h / float(w)))

        if fill_ratio < min_fill_ratio:
            continue
        if aspect_ratio > max_aspect_ratio:
            continue

        roi = gray_frame[y:y + h, x:x + w]
        roi_mask = comp_mask[y:y + h, x:x + w]
        grad_x = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(grad_x, grad_y)
        texture_strength = float(np.percentile(grad[roi_mask], 75)) if roi_mask.any() else 0.0

        shape_score = (fill_ratio ** 0.75) / np.sqrt(max(aspect_ratio, 1.0))
        texture_score = 0.35 + 0.65 * min(texture_strength / 35.0, 1.0)
        shadow_score = 0.0
        color_change = None
        if shadow_filter and color_frame is not None and background_color is not None:
            roi_color = color_frame[y:y + h, x:x + w].astype(np.float32)
            roi_bg = background_color[y:y + h, x:x + w].astype(np.float32)
            if roi_mask.any():
                color_change = float(np.linalg.norm(roi_color[roi_mask] - roi_bg[roi_mask], axis=1).mean())
                darkening = max(0.0, -intensity_delta)
                chroma_ratio = color_change / max(abs(intensity_delta), 1.0)
                # Shadows often darken the background without adding much chromatic or edge change.
                if darkening > 4.0 and chroma_ratio < 2.5:
                    shadow_score = min(1.0, darkening / 35.0) * (1.0 - min(chroma_ratio / 2.5, 1.0))

        shadow_penalty = 1.0 - 0.75 * shadow_score
        score = area * motion_strength * shape_score * texture_score * shadow_penalty
        cx, cy = centroids[label_idx]
        box = clamp_box(
            [x - box_pad, y - box_pad, x + w - 1 + box_pad, y + h - 1 + box_pad],
            width,
            height,
        )

        candidates.append({
            "frame_idx": int(frame_idx),
            "point": [float(cx), float(cy)],
            "bbox": box,
            "motion_bbox": box,
            "area": area,
            "bbox_area": int(bbox_area),
            "fill_ratio": float(fill_ratio),
            "aspect_ratio": aspect_ratio,
            "motion_strength": motion_strength,
            "intensity_delta": intensity_delta,
            "color_change": color_change,
            "shadow_score": float(shadow_score),
            "texture_strength": texture_strength,
            "score": float(score),
        })

    return candidates


def load_gray_frame(frame_path: Path):
    frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
    if frame is None:
        raise RuntimeError(f"Cannot read frame: {frame_path}")
    return cv2.GaussianBlur(frame, (5, 5), 0)


def load_color_frame(frame_path: Path):
    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Cannot read frame: {frame_path}")
    return cv2.GaussianBlur(frame, (5, 5), 0)


def target_hint_mask(hsv: np.ndarray, hint: str):
    hint = (hint or "auto").lower()
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    if "tennis" in hint:
        # Fluorescent tennis balls: yellow-green, saturated, bright. This rejects most floor shadows/reflections.
        return ((h >= 18) & (h <= 48) & (s >= 55) & (v >= 95)).astype(np.uint8) * 255
    if "basket" in hint:
        return (((h <= 18) | ((h >= 160) & (h <= 179))) & (s >= 45) & (v >= 60)).astype(np.uint8) * 255
    if "hockey" in hint or "puck" in hint:
        return ((s <= 80) & (v <= 120)).astype(np.uint8) * 255
    if any(k in hint for k in ["slider", "cube", "block", "box", "square"]):
        sat_thr = max(55, int(np.percentile(s, 72)))
        val_floor = max(45, int(np.percentile(v, 18)))
        return ((s >= sat_thr) & (v >= val_floor)).astype(np.uint8) * 255
    return None


def generic_object_mask(hsv: np.ndarray):
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    del h

    sat_thr = max(35, int(np.percentile(s, 68)))
    bright_thr = max(45, int(np.percentile(v, 72)))
    dark_thr = min(140, int(np.percentile(v, 18)))

    saturated = (s >= sat_thr) & (v >= 35)
    bright_object = (v >= bright_thr) & (s >= 25)
    dark_solid = (v <= dark_thr) & (s >= 20)
    mask = saturated | bright_object | dark_solid

    # Suppress typical floor shadows: dark but nearly achromatic regions.
    shadow_like = (v <= dark_thr) & (s < 25)
    mask &= ~shadow_like
    return mask.astype(np.uint8) * 255


def is_box_like_hint(hint: str) -> bool:
    hint = (hint or "").lower()
    return any(k in hint for k in ["slider", "cube", "block", "box", "square"])


def is_ball_like_hint(hint: str) -> bool:
    hint = (hint or "").lower()
    return any(k in hint for k in ["ball", "sphere", "puck", "standard", "bearing"])


def component_edge_stats(gray_roi: np.ndarray, component: np.ndarray):
    if gray_roi.size == 0 or not component.any():
        return 0.0, 0.0
    grad_x = cv2.Sobel(gray_roi, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_roi, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(grad_x, grad_y)
    vals = grad[component]
    if len(vals) == 0:
        return 0.0, 0.0
    return float(np.percentile(vals, 75)), float((vals > 12.0).mean())


def reflection_penalty_for_component(hsv: np.ndarray, component: np.ndarray, cy_global: float, seed_y_global: float):
    if not component.any():
        return 0.0
    s = hsv[:, :, 1][component].astype(np.float32)
    v = hsv[:, :, 2][component].astype(np.float32)
    low_detail_lower = cy_global > seed_y_global and float(np.mean(s)) < 55 and float(np.mean(v)) < 170
    if not low_detail_lower:
        return 0.0
    return 0.35


def background_spread_penalty(area: int, bbox_area: int, roi_area: int, aspect: float, fill_ratio: float):
    if roi_area <= 0:
        return 0.0
    area_ratio = area / float(roi_area)
    penalty = 0.0
    if area_ratio > 0.45:
        penalty += min(0.85, (area_ratio - 0.45) / 0.35)
    if bbox_area / float(roi_area) > 0.55 and fill_ratio < 0.55:
        penalty += 0.35
    if aspect > 3.0 and area_ratio > 0.20:
        penalty += 0.25
    return float(min(0.95, penalty))


def hough_circle_refinement(candidate, gray_roi: np.ndarray, search_box, width: int, height: int):
    motion_box = candidate.get("motion_bbox", candidate["bbox"])
    sx1, sy1, _, _ = search_box
    mx1, my1, mx2, my2 = [float(v) for v in motion_box]
    motion_w = max(2.0, mx2 - mx1 + 1.0)
    motion_h = max(2.0, my2 - my1 + 1.0)
    min_radius = max(5, int(0.35 * min(motion_w, motion_h)))
    max_radius = max(min_radius + 2, int(2.2 * max(motion_w, motion_h)))
    max_radius = min(max_radius, int(0.48 * min(gray_roi.shape[:2])))
    if max_radius <= min_radius:
        return None

    blurred = cv2.medianBlur(gray_roi, 5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(8, int(min(gray_roi.shape[:2]) * 0.25)),
        param1=80,
        param2=14,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return None

    motion_cx = 0.5 * (mx1 + mx2) - sx1
    motion_cy = 0.5 * (my1 + my2) - sy1
    best = None
    best_score = -1.0
    for cx, cy, radius in np.round(circles[0, :]).astype(float):
        if not (0 <= cx < gray_roi.shape[1] and 0 <= cy < gray_roi.shape[0]):
            continue
        dist = float(np.hypot(cx - motion_cx, cy - motion_cy))
        if dist > 1.4 * radius:
            continue
        x1 = sx1 + cx - radius
        y1 = sy1 + cy - radius
        x2 = sx1 + cx + radius
        y2 = sy1 + cy + radius
        circle_box = clamp_box([x1, y1, x2, y2], width, height)
        overlap = box_iou(circle_box, motion_box)
        score = 2.0 * overlap + radius / max(dist, 1.0)
        if score > best_score:
            best_score = score
            best = circle_box, [float(sx1 + cx), float(sy1 + cy)], float(radius)
    if best is None or best_score <= 0.0:
        return None
    return best


def refine_candidate_to_visible_object(candidate, color_frame: np.ndarray, width: int, height: int,
                                       expand_factor: float, target_hint: str = "auto"):
    if color_frame is None:
        return candidate

    search_box = expand_box(candidate["bbox"], width, height, factor=expand_factor, min_pad=12)
    sx1, sy1, sx2, sy2 = search_box
    roi = color_frame[sy1:sy2 + 1, sx1:sx2 + 1]
    if roi.size == 0:
        return candidate

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hinted_mask = target_hint_mask(hsv, target_hint)
    if hinted_mask is not None and int(hinted_mask.sum()) > 0:
        obj_mask = hinted_mask
        mask_source = f"target_hint:{target_hint}"
    else:
        obj_mask = generic_object_mask(hsv)
        mask_source = "generic_objectness"

    kernel = np.ones((3, 3), dtype=np.uint8)
    obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(obj_mask, connectivity=8)
    if num_labels <= 1:
        candidate["search_bbox"] = search_box
        candidate["object_refined"] = False
        return candidate

    px = int(np.clip(round(candidate["point"][0]) - sx1, 0, roi.shape[1] - 1))
    py = int(np.clip(round(candidate["point"][1]) - sy1, 0, roi.shape[0] - 1))
    motion_box = candidate.get("motion_bbox", candidate["bbox"])
    mx1, my1, mx2, my2 = motion_box
    mx1r = int(np.clip(mx1 - sx1, 0, roi.shape[1] - 1))
    mx2r = int(np.clip(mx2 - sx1, 0, roi.shape[1] - 1))
    my1r = int(np.clip(my1 - sy1, 0, roi.shape[0] - 1))
    my2r = int(np.clip(my2 - sy1, 0, roi.shape[0] - 1))

    best_label = None
    best_score = -1.0
    seed_y_global = float(candidate["point"][1])
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    object_hint = (target_hint or "").lower()
    likely_ball_like = is_ball_like_hint(object_hint)
    likely_box_like = is_box_like_hint(object_hint)
    circle_candidate = hough_circle_refinement(candidate, gray_roi, search_box, width, height) if likely_ball_like else None
    roi_area = max(1, roi.shape[0] * roi.shape[1])
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < max(12, int(candidate["area"] * 0.35)):
            continue
        component = labels == label_idx
        contains_seed = bool(component[py, px])
        overlap = int(component[my1r:my2r + 1, mx1r:mx2r + 1].sum())
        x = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y = int(stats[label_idx, cv2.CC_STAT_TOP])
        w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        bbox_area = max(1, w * h)
        fill_ratio = area / float(bbox_area)
        aspect = max(w / max(h, 1), h / max(w, 1))
        if aspect > (3.2 if likely_box_like else 4.5):
            continue
        if likely_box_like and area > 0.50 * roi_area:
            continue
        if likely_box_like and fill_ratio < 0.35:
            continue
        cx, cy = centroids[label_idx]
        cy_global = sy1 + float(cy)
        circularity = min(w, h) / max(max(w, h), 1)
        edge_strength, edge_density = component_edge_stats(gray_roi, component)
        reflection_penalty = reflection_penalty_for_component(hsv, component, cy_global, seed_y_global)
        upper_bonus = 0.35 * area if likely_ball_like and cy_global <= seed_y_global else 0.0
        lower_penalty = 0.45 * area if likely_ball_like and cy_global > seed_y_global + 0.15 * h else 0.0
        if likely_box_like:
            upper_bonus += 0.55 * area if cy_global <= seed_y_global + 0.10 * h else 0.0
            lower_penalty += 0.70 * area if cy_global > seed_y_global + 0.20 * h else 0.0
        spread_penalty = background_spread_penalty(area, bbox_area, roi_area, aspect, fill_ratio)
        boxness = fill_ratio * min(1.0, 1.8 / max(aspect, 1.0))
        objectness = (
            0.30 * area
            + 0.40 * area * circularity
            + 0.18 * area * min(edge_strength / 35.0, 1.0)
            + 0.12 * area * min(edge_density / 0.25, 1.0)
        )
        if likely_box_like:
            objectness += 0.55 * area * boxness
        score = overlap + (area if contains_seed else 0) + objectness + upper_bonus - lower_penalty
        score *= 1.0 - reflection_penalty
        score *= 1.0 - spread_penalty
        if score > best_score:
            best_label = label_idx
            best_score = score

    if best_label is None:
        if circle_candidate is not None:
            circle_box, circle_point, circle_radius = circle_candidate
            candidate = dict(candidate)
            candidate["bbox"] = circle_box
            candidate["point"] = circle_point
            candidate["object_area"] = float(np.pi * circle_radius * circle_radius)
            candidate["object_refined"] = True
            candidate["object_refine_source"] = "circle_hough"
            candidate["search_bbox"] = search_box
            candidate["score"] = float(candidate["score"] * 1.12)
            return candidate
        candidate["search_bbox"] = search_box
        candidate["object_refined"] = False
        return candidate

    x = int(stats[best_label, cv2.CC_STAT_LEFT])
    y = int(stats[best_label, cv2.CC_STAT_TOP])
    w = int(stats[best_label, cv2.CC_STAT_WIDTH])
    h = int(stats[best_label, cv2.CC_STAT_HEIGHT])
    area = int(stats[best_label, cv2.CC_STAT_AREA])
    cx, cy = centroids[best_label]
    refined_box = clamp_box([sx1 + x, sy1 + y, sx1 + x + w - 1, sy1 + y + h - 1], width, height)

    # Only accept a refinement that actually expands or recenters the motion cue.
    old_area = max(1, (candidate["bbox"][2] - candidate["bbox"][0] + 1) * (candidate["bbox"][3] - candidate["bbox"][1] + 1))
    new_area = max(1, (refined_box[2] - refined_box[0] + 1) * (refined_box[3] - refined_box[1] + 1))
    if circle_candidate is not None:
        circle_box, circle_point, circle_radius = circle_candidate
        circle_area = max(1, (circle_box[2] - circle_box[0] + 1) * (circle_box[3] - circle_box[1] + 1))
        if circle_area > 1.25 * new_area and box_iou(circle_box, refined_box) > 0.05:
            candidate = dict(candidate)
            candidate["bbox"] = circle_box
            candidate["point"] = circle_point
            candidate["object_area"] = float(np.pi * circle_radius * circle_radius)
            candidate["object_refined"] = True
            candidate["object_refine_source"] = "circle_hough"
            candidate["search_bbox"] = search_box
            candidate["score"] = float(candidate["score"] * 1.12)
            return candidate
    if new_area < 0.75 * old_area:
        if circle_candidate is not None:
            circle_box, circle_point, circle_radius = circle_candidate
            candidate = dict(candidate)
            candidate["bbox"] = circle_box
            candidate["point"] = circle_point
            candidate["object_area"] = float(np.pi * circle_radius * circle_radius)
            candidate["object_refined"] = True
            candidate["object_refine_source"] = "circle_hough"
            candidate["search_bbox"] = search_box
            candidate["score"] = float(candidate["score"] * 1.12)
            return candidate
        candidate["search_bbox"] = search_box
        candidate["object_refined"] = False
        return candidate

    candidate = dict(candidate)
    candidate["bbox"] = refined_box
    candidate["point"] = [float(sx1 + cx), float(sy1 + cy)]
    candidate["object_area"] = area
    candidate["object_refined"] = True
    candidate["object_refine_source"] = mask_source
    candidate["search_bbox"] = search_box
    candidate["score"] = float(candidate["score"] * (1.15 if new_area > old_area else 1.05))
    return candidate


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1 + 1)
    ih = max(0, iy2 - iy1 + 1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1 + 1) * max(0, ay2 - ay1 + 1)
    area_b = max(0, bx2 - bx1 + 1) * max(0, by2 - by1 + 1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else inter / float(denom)


def select_distinct_candidates(candidates, num_objects: int, min_distance_px: float, max_iou: float = 0.20):
    selected = []
    for cand in sorted(candidates, key=lambda c: c["score"], reverse=True):
        cx, cy = cand["point"]
        keep = True
        for prev in selected:
            px, py = prev["point"]
            if np.hypot(cx - px, cy - py) < min_distance_px:
                keep = False
                break
            if box_iou(cand["bbox"], prev["bbox"]) > max_iou:
                keep = False
                break
        if keep:
            selected.append(cand)
            if len(selected) >= num_objects:
                break
    return selected


def cluster_motion_candidates(candidates, min_distance_px: float, preferred_frame: int | None = None):
    clusters = []
    for cand in sorted(candidates, key=lambda c: c["score"], reverse=True):
        cx, cy = cand["point"]
        assigned = False
        for cluster in clusters:
            ccx, ccy = cluster["center"]
            if np.hypot(cx - ccx, cy - ccy) <= min_distance_px:
                cluster["items"].append(cand)
                weights = np.array([max(item["score"], 1e-6) for item in cluster["items"]], dtype=np.float64)
                pts = np.array([item["point"] for item in cluster["items"]], dtype=np.float64)
                cluster["center"] = list((pts * weights[:, None]).sum(axis=0) / weights.sum())
                assigned = True
                break
        if not assigned:
            clusters.append({"center": list(cand["point"]), "items": [cand]})

    out = []
    for cluster in clusters:
        items = cluster["items"]
        frames = sorted({int(item["frame_idx"]) for item in items})
        best_item = max(items, key=lambda item: item["score"])
        stable_items = [item for item in items if item["score"] >= 0.65 * best_item["score"]]
        if preferred_frame is None:
            stable_item = min(stable_items, key=lambda item: int(item["frame_idx"]))
        else:
            stable_item = min(
                stable_items,
                key=lambda item: (
                    abs(int(item["frame_idx"]) - int(preferred_frame)),
                    -float(item["score"]),
                ),
            )
        cluster_score = float(np.median([item["score"] for item in items]) * np.log1p(len(frames)))
        item_out = dict(stable_item)
        item_out["cluster_score"] = cluster_score
        item_out["cluster_frames"] = frames
        item_out["cluster_n_frames"] = len(frames)
        item_out["cluster_n_candidates"] = len(items)
        out.append(item_out)
    return sorted(out, key=lambda item: item["cluster_score"], reverse=True)


def draw_auto_init_debug(frame_paths, chosen, debug_path: Path):
    if debug_path is None or not chosen:
        return
    ensure_dir(debug_path.parent)
    frame_idx = int(chosen[0]["frame_idx"])
    frame = cv2.imread(str(frame_paths[frame_idx]))
    if frame is None:
        return
    colors = [(0, 255, 255), (255, 0, 255), (255, 180, 0), (0, 180, 255)]
    for i, item in enumerate(chosen):
        color = colors[i % len(colors)]
        x1, y1, x2, y2 = item["bbox"]
        cx, cy = item["point"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (int(round(cx)), int(round(cy))), 5, (0, 0, 255), -1)
        cv2.putText(
            frame,
            f"obj={i + 1} score={item['score']:.1f}",
            (max(0, x1), max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(debug_path), frame)


def make_negative_points_around_box(box, width: int, height: int, margin: int = 12):
    if box is None:
        return []
    x1, y1, x2, y2 = [float(v) for v in box]
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    candidates = [
        (cx, y1 - margin),
        (cx, y2 + margin),
        (x1 - margin, cy),
        (x2 + margin, cy),
    ]
    points = []
    for x, y in candidates:
        x = float(np.clip(x, 0, width - 1))
        y = float(np.clip(y, 0, height - 1))
        if x1 <= x <= x2 and y1 <= y <= y2:
            continue
        points.append([x, y])
    return points


def select_scan_frames(frame_paths, scan_frames: int, scan_mode: str):
    total = len(frame_paths)
    n_scan = min(max(2, int(scan_frames)), total)
    mode = (scan_mode or "uniform").lower()
    if mode == "start" or total <= n_scan:
        indices = list(range(n_scan))
    elif mode == "middle":
        center = total // 2
        start = max(0, center - n_scan // 2)
        end = min(total, start + n_scan)
        start = max(0, end - n_scan)
        indices = list(range(start, end))
    else:
        indices = sorted(set(np.linspace(0, total - 1, n_scan, dtype=int).tolist()))
        if len(indices) < n_scan:
            missing = [i for i in range(total) if i not in set(indices)]
            indices.extend(missing[: n_scan - len(indices)])
            indices = sorted(indices)
    return indices, [frame_paths[i] for i in indices]


def detect_moving_object_prompts(
    frames_dir: Path,
    width: int,
    height: int,
    num_objects: int,
    scan_frames: int,
    scan_mode: str,
    threshold: float,
    min_area: int,
    max_area_ratio: float,
    box_pad: int,
    min_fill_ratio: float,
    max_aspect_ratio: float,
    min_separation_ratio: float,
    shadow_filter: bool,
    object_refine: bool,
    box_expand: float,
    target_hint: str,
    debug_path=None,
):
    frame_paths = list_jpg_frames(frames_dir)
    if not frame_paths:
        raise RuntimeError(f"No jpg frames found in: {frames_dir}")

    scan_indices, scan_paths = select_scan_frames(frame_paths, scan_frames, scan_mode)
    n_scan = len(scan_paths)
    gray_frames = [load_gray_frame(p) for p in scan_paths]
    color_frames = [load_color_frame(p) for p in scan_paths]

    background = np.median(np.stack(gray_frames, axis=0), axis=0).astype(np.uint8)
    background_color = np.median(np.stack(color_frames, axis=0), axis=0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    all_candidates = []

    for local_idx, gray in enumerate(gray_frames):
        frame_idx = int(scan_indices[local_idx])
        diff = cv2.absdiff(gray, background)
        _, motion = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, kernel, iterations=1)
        motion = cv2.morphologyEx(motion, cv2.MORPH_CLOSE, kernel, iterations=2)
        candidates = component_candidates(
            mask=motion,
            diff=diff,
            gray_frame=gray,
            background_gray=background,
            color_frame=color_frames[local_idx],
            background_color=background_color,
            frame_idx=frame_idx,
            width=width,
            height=height,
            min_area=min_area,
            max_area_ratio=max_area_ratio,
            box_pad=box_pad,
            min_fill_ratio=min_fill_ratio,
            max_aspect_ratio=max_aspect_ratio,
            shadow_filter=shadow_filter,
        )
        if object_refine:
            candidates = [
                refine_candidate_to_visible_object(c, color_frames[local_idx], width, height, box_expand, target_hint)
                for c in candidates
            ]
        all_candidates.extend(candidates)

    method = "median_background"
    if not all_candidates and len(gray_frames) >= 2:
        for local_idx in range(1, len(gray_frames)):
            frame_idx = int(scan_indices[local_idx])
            diff = cv2.absdiff(gray_frames[local_idx], gray_frames[local_idx - 1])
            _, motion = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
            motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, kernel, iterations=1)
            motion = cv2.morphologyEx(motion, cv2.MORPH_CLOSE, kernel, iterations=2)
            candidates = component_candidates(
                mask=motion,
                diff=diff,
                gray_frame=gray_frames[local_idx],
                background_gray=gray_frames[local_idx - 1],
                color_frame=color_frames[local_idx],
                background_color=color_frames[local_idx - 1],
                frame_idx=frame_idx,
                width=width,
                height=height,
                min_area=min_area,
                max_area_ratio=max_area_ratio,
                box_pad=box_pad,
                min_fill_ratio=min_fill_ratio,
                max_aspect_ratio=max_aspect_ratio,
                shadow_filter=shadow_filter,
            )
            if object_refine:
                candidates = [
                    refine_candidate_to_visible_object(c, color_frames[local_idx], width, height, box_expand, target_hint)
                    for c in candidates
                ]
            all_candidates.extend(candidates)
        method = "pairwise_frame_difference"

    if not all_candidates:
        raise RuntimeError(
            "Auto init failed: no moving object was detected. "
            "Try lowering --auto-init-threshold or --auto-init-min-area."
        )

    num_objects = max(1, int(num_objects))
    min_distance_px = float(np.hypot(width, height) * min_separation_ratio)
    mode = (scan_mode or "uniform").lower()
    preferred_frame = None if mode == "start" else len(frame_paths) // 2
    clustered_candidates = cluster_motion_candidates(
        all_candidates,
        min_distance_px,
        preferred_frame=preferred_frame,
    )
    max_score = max(c["cluster_score"] for c in clustered_candidates)
    confident = [c for c in clustered_candidates if c["cluster_score"] >= 0.35 * max_score]

    chosen = []
    if num_objects == 1:
        top = [c for c in clustered_candidates if c["cluster_score"] >= 0.60 * max_score]
        chosen = [top[0]]
    else:
        by_frame = {}
        for cand in confident:
            by_frame.setdefault(cand["frame_idx"], []).append(cand)
        for frame_idx in sorted(by_frame):
            frame_chosen = select_distinct_candidates(
                by_frame[frame_idx],
                num_objects=num_objects,
                min_distance_px=min_distance_px,
            )
            if len(frame_chosen) >= num_objects:
                chosen = frame_chosen
                break
        if not chosen:
            chosen = select_distinct_candidates(
                confident,
                num_objects=num_objects,
                min_distance_px=min_distance_px,
            )

    if len(chosen) < num_objects:
        print(
            f"[WARN] Requested {num_objects} objects, but auto-init found only {len(chosen)} distinct candidate(s)."
        )

    for item in chosen:
        item["method"] = method
        item["n_candidates"] = len(all_candidates)
        item["scan_frames"] = n_scan
        item["scan_mode"] = str(scan_mode)
        item["scan_indices"] = [int(i) for i in scan_indices]
        item["threshold"] = float(threshold)
        item["min_area"] = int(min_area)
        item["max_area_ratio"] = float(max_area_ratio)
        item["min_fill_ratio"] = float(min_fill_ratio)
        item["max_aspect_ratio"] = float(max_aspect_ratio)
        item["shadow_filter"] = bool(shadow_filter)
        item["object_refine"] = bool(object_refine)
        item["box_expand"] = float(box_expand)
        item["target_hint"] = str(target_hint)

    draw_auto_init_debug(frame_paths, chosen, debug_path)
    return chosen


def detect_moving_object_prompt(
    frames_dir: Path,
    width: int,
    height: int,
    scan_frames: int,
    threshold: float,
    min_area: int,
    max_area_ratio: float,
    box_pad: int,
    debug_path=None,
    scan_mode: str = "uniform",
    min_fill_ratio: float = 0.08,
    max_aspect_ratio: float = 6.0,
    min_separation_ratio: float = 0.03,
    shadow_filter: bool = True,
    object_refine: bool = True,
    box_expand: float = 2.8,
    target_hint: str = "auto",
):
    prompts = detect_moving_object_prompts(
        frames_dir=frames_dir,
        width=width,
        height=height,
        num_objects=1,
        scan_frames=scan_frames,
        scan_mode=scan_mode,
        threshold=threshold,
        min_area=min_area,
        max_area_ratio=max_area_ratio,
        box_pad=box_pad,
        min_fill_ratio=min_fill_ratio,
        max_aspect_ratio=max_aspect_ratio,
        min_separation_ratio=min_separation_ratio,
        shadow_filter=shadow_filter,
        object_refine=object_refine,
        box_expand=box_expand,
        target_hint=target_hint,
        debug_path=debug_path,
    )
    return prompts[0]


def extract_video_to_frames(video_path: Path, frames_dir: Path, overwrite: bool = False):
    ensure_dir(frames_dir)

    existing = list_jpg_frames(frames_dir)
    if existing and not overwrite:
        fps, _, width, height = read_video_meta(video_path)
        return fps, len(existing), width, height, False

    if overwrite:
        for p in frames_dir.glob("*.jpg"):
            p.unlink()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_name = f"{idx:05d}.jpg"
        frame_path = frames_dir / frame_name
        cv2.imwrite(str(frame_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        idx += 1

    cap.release()
    return float(fps), idx, width, height, True


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


def draw_tracking_result(frame, cx, cy, obj_id, color=(0, 255, 0)):
    cx_i, cy_i = int(round(cx)), int(round(cy))
    cv2.circle(frame, (cx_i, cy_i), 5, color, -1)
    cv2.putText(
        frame,
        f"ID:{obj_id} ({cx_i},{cy_i})",
        (cx_i + 8, cy_i - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def alpha_blend_mask(frame_bgr: np.ndarray, mask_bool: np.ndarray, color_bgr=(0, 255, 0), alpha=0.4):
    if mask_bool.ndim != 2:
        return
    if mask_bool.shape[:2] != frame_bgr.shape[:2]:
        return

    frame_f = frame_bgr.astype(np.float32)
    color = np.array(color_bgr, dtype=np.float32).reshape(1, 1, 3)
    frame_f[mask_bool] = frame_f[mask_bool] * (1.0 - alpha) + color * alpha
    frame_bgr[:] = np.clip(frame_f, 0, 255).astype(np.uint8)


def save_mask_png(mask_path: Path, mask_bool: np.ndarray):
    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    cv2.imwrite(str(mask_path), mask_u8)


def load_sam2_predictor(sam2_repo: Path, model_cfg: str, model_weights: str, vos_optimized: bool):
    sam2_repo = sam2_repo.expanduser().resolve()
    if not sam2_repo.exists():
        raise RuntimeError(f"SAM2 repo not found: {sam2_repo}")

    if str(sam2_repo) not in sys.path:
        sys.path.insert(0, str(sam2_repo))

    from sam2.build_sam import build_sam2_video_predictor

    try:
        predictor = build_sam2_video_predictor(
            model_cfg,
            model_weights,
            vos_optimized=vos_optimized,
        )
    except TypeError:
        predictor = build_sam2_video_predictor(
            model_cfg,
            model_weights,
        )

    return predictor


def main():
    args = parse_args()

    workspace_root = Path(args.workspace_root).expanduser().resolve()
    scene_name = args.scene_name
    video_path = Path(args.video_path).expanduser().resolve()
    sam2_repo = Path(args.sam2_repo).expanduser().resolve()
    model_weights = Path(args.model_weights).expanduser().resolve()

    if not video_path.exists():
        raise RuntimeError(f"Input video not found: {video_path}")
    if not model_weights.exists():
        raise RuntimeError(f"Model weights not found: {model_weights}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for SAM2 video inference.")

    bf16_supported = bool(
        hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
    )
    precision_policy = resolve_precision_policy(
        args.precision,
        cuda_available=True,
        bf16_supported=bf16_supported,
    )
    inference_context_factory = lambda: sam2_inference_context(precision_policy)

    data_scene_dir = workspace_root / "data" / "custom" / scene_name
    raw_dir = data_scene_dir / "raw"
    frames_dir = data_scene_dir / "frames"

    sam2_out_dir = workspace_root / "sam2_tracks" / scene_name
    masks_dir = sam2_out_dir / "masks"
    json_path = sam2_out_dir / "tracking_points.json"
    vis_video_path = sam2_out_dir / "tracking_vis.mp4"
    run_config_path = sam2_out_dir / "run_config.json"

    ensure_dir(workspace_root)
    ensure_dir(data_scene_dir)
    ensure_dir(raw_dir)
    ensure_dir(frames_dir)
    ensure_dir(sam2_out_dir)
    if args.save_mask_png:
        ensure_dir(masks_dir)

    workspace_video_path = raw_dir / "input.mp4"
    if args.copy_video_to_workspace:
        if (not workspace_video_path.exists()) or args.overwrite_outputs:
            shutil.copy2(video_path, workspace_video_path)

    tf32_enabled = precision_policy["resolved"] != "fp32"
    torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
    torch.backends.cudnn.allow_tf32 = tf32_enabled
    torch.set_float32_matmul_precision("high" if tf32_enabled else "highest")

    print("=" * 70)
    print("SAM2 tracking inside workspace")
    print("=" * 70)
    print(
        "Precision policy: "
        f"requested={precision_policy['requested']}, "
        f"resolved={precision_policy['resolved']}, "
        f"autocast={precision_policy['autocast_enabled']}, "
        f"tf32={tf32_enabled}"
    )

    print("\n[1] Preparing shared frames ...")
    fps, total_frames, W, H, extracted_now = extract_video_to_frames(
        video_path=video_path,
        frames_dir=frames_dir,
        overwrite=args.overwrite_frames,
    )

    if extracted_now:
        print(f"Frames extracted to: {frames_dir}")
    else:
        print(f"Reusing existing frames in: {frames_dir}")

    print(f"total_frames={total_frames}, size={W}x{H}, fps={fps:.3f}")

    prompt_mode = "manual"
    auto_init_result = None
    auto_init_results = []
    init_frame_idx = int(args.init_frame_idx)
    init_obj_id = int(args.init_obj_id)
    init_label = int(args.init_label)
    init_box = list(args.init_box) if args.init_box is not None else None
    prompts = []

    need_auto_init = bool(args.auto_init) or args.init_point_x is None or args.init_point_y is None
    if need_auto_init:
        print("\n[1b] Auto-detecting target for SAM2 prompt ...")
        debug_path = sam2_out_dir / "auto_init_debug.jpg" if args.save_auto_init_debug else None
        detector_used = "motion"
        yolo_error = None
        if args.auto_init_detector in ("yolo", "yolo_then_motion"):
            try:
                from scripts.detection.yolo_init import detect_yolo_prompts, draw_yolo_debug

                auto_init_results = detect_yolo_prompts(
                    frames_dir=frames_dir,
                    width=W,
                    height=H,
                    num_objects=args.auto_init_num_objects,
                    scan_frames=args.auto_init_scan_frames,
                    scan_mode=args.auto_init_scan_mode,
                    weights=args.yolo_weights,
                    conf=args.yolo_conf,
                    iou=args.yolo_iou,
                    target_hint=args.auto_init_target_hint,
                    classes=args.yolo_classes,
                )
                detector_used = "yolo"
                if debug_path is not None:
                    first_frame = frames_dir / f"{int(auto_init_results[0]['frame_idx']):05d}.jpg"
                    draw_yolo_debug(first_frame, auto_init_results, debug_path)
            except Exception as exc:
                yolo_error = str(exc)
                if args.auto_init_detector == "yolo":
                    raise RuntimeError(f"YOLO auto-init failed: {yolo_error}") from exc
                print(f"[WARN] YOLO auto-init failed, falling back to motion detector: {yolo_error}")

        if not auto_init_results:
            auto_init_results = detect_moving_object_prompts(
                frames_dir=frames_dir,
                width=W,
                height=H,
                num_objects=args.auto_init_num_objects,
                scan_frames=args.auto_init_scan_frames,
                scan_mode=args.auto_init_scan_mode,
                threshold=args.auto_init_threshold,
                min_area=args.auto_init_min_area,
                max_area_ratio=args.auto_init_max_area_ratio,
                box_pad=args.auto_init_box_pad,
                min_fill_ratio=args.auto_init_min_fill_ratio,
                max_aspect_ratio=args.auto_init_max_aspect_ratio,
                min_separation_ratio=args.auto_init_min_separation_ratio,
                shadow_filter=not args.disable_auto_init_shadow_filter,
                object_refine=args.auto_init_object_refine,
                box_expand=args.auto_init_box_expand,
                target_hint=args.auto_init_target_hint,
                debug_path=debug_path,
            )
            detector_used = "motion"
            if yolo_error:
                for item in auto_init_results:
                    item["yolo_fallback_reason"] = yolo_error

        auto_init_result = auto_init_results[0]
        init_frame_idx = int(auto_init_result["frame_idx"])
        prompt_mode = f"auto_{detector_used}"
        for i, item in enumerate(auto_init_results):
            positive_points = [
                [float(value) for value in point]
                for point in item.get("positive_points", [item["point"]])
            ]
            point_x = float(positive_points[0][0])
            point_y = float(positive_points[0][1])
            if "prompt_box" in item:
                box = (
                    [float(value) for value in item["prompt_box"]]
                    if item["prompt_box"] is not None
                    else None
                )
            else:
                box = [float(v) for v in item["bbox"]] if not args.no_auto_init_box else None
            if "negative_points" in item:
                negative_points = [
                    [float(value) for value in point]
                    for point in item["negative_points"]
                ]
            elif args.no_auto_init_negative_points or not item.get("object_refined", False):
                negative_points = []
            else:
                negative_points = make_negative_points_around_box(box, W, H)
            obj_id = init_obj_id + i
            prompts.append({
                "obj_id": obj_id,
                "frame_idx": int(item["frame_idx"]),
                "point": [point_x, point_y],
                "positive_points": positive_points,
                "label": init_label,
                "box": box,
                "negative_points": negative_points,
                "auto_init": item,
            })
            print(
                "Auto prompt: "
                f"obj_id={obj_id}, frame={item['frame_idx']}, "
                f"point=({point_x:.2f}, {point_y:.2f}), pos_pts={len(positive_points)}, bbox={box}, "
                f"score={item['score']:.2f}, fill={item['fill_ratio']:.2f}, "
                f"aspect={item['aspect_ratio']:.2f}, texture={item['texture_strength']:.2f}, "
                f"shadow={item.get('shadow_score', 0.0):.2f}, refined={item.get('object_refined', False)}, "
                f"refine_source={item.get('object_refine_source', 'none')}, "
                f"neg_pts={len(negative_points)}, "
                f"method={item['method']}"
            )
        if debug_path is not None:
            print(f"Auto-init debug image: {debug_path}")
    elif args.init_point_x is None or args.init_point_y is None:
        raise RuntimeError(
            "Manual mode requires --init-point-x and --init-point-y. "
            "Use --auto-init to detect the moving object automatically."
        )
    else:
        prompts.append({
            "obj_id": init_obj_id,
            "frame_idx": init_frame_idx,
            "point": [float(args.init_point_x), float(args.init_point_y)],
            "positive_points": [[float(args.init_point_x), float(args.init_point_y)]],
            "label": init_label,
            "box": init_box,
            "negative_points": [],
            "auto_init": None,
        })

    if not prompts:
        raise RuntimeError("No SAM2 prompts were created.")
    args.init_point_x = float(prompts[0]["point"][0])
    args.init_point_y = float(prompts[0]["point"][1])
    init_box = prompts[0]["box"]
    init_frame_idx = int(prompts[0]["frame_idx"])

    print("\n[2] Loading SAM2 predictor ...")
    predictor = load_sam2_predictor(
        sam2_repo=sam2_repo,
        model_cfg=args.model_cfg,
        model_weights=str(model_weights),
        vos_optimized=args.vos_optimized,
    )
    print("SAM2 loaded.")

    print("\n[3] Initializing inference state ...")
    inference_state = initialize_predictor_state(
        predictor,
        frames_dir=frames_dir,
        offload_video_to_cpu=args.offload_video_to_cpu,
        offload_state_to_cpu=args.offload_state_to_cpu,
        prompts=prompts,
        inference_context_factory=inference_context_factory,
    )

    for prompt in prompts:
        print(
            f"Added prompt: mode={prompt_mode}, frame={prompt['frame_idx']}, "
            f"obj_id={prompt['obj_id']}, point={prompt['point']}, box={prompt['box']}"
        )

    writer = None
    if args.save_vis_video:
        if vis_video_path.exists() and not args.overwrite_outputs:
            raise RuntimeError(f"Output exists: {vis_video_path}. Use --overwrite-outputs to replace.")
        writer = cv2.VideoWriter(
            str(vis_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (W, H),
        )

    if json_path.exists() and not args.overwrite_outputs:
        raise RuntimeError(f"Output exists: {json_path}. Use --overwrite-outputs to replace.")

    object_ids = [int(p["obj_id"]) for p in prompts]
    prompt_records = [
        {
            "obj_id": int(p["obj_id"]),
            "init_frame_idx": int(p["frame_idx"]),
            "init_point": [float(p["point"][0]), float(p["point"][1])],
            "positive_points": [
                [float(value) for value in point]
                for point in p.get("positive_points", [p["point"]])
            ],
            "init_label": int(p["label"]),
            "init_box": p["box"],
            "negative_points": p.get("negative_points", []),
            "auto_init": p["auto_init"],
        }
        for p in prompts
    ]

    tracking_data = {
        "video_info": {
            "scene_name": scene_name,
            "video_path_original": str(video_path),
            "video_path_workspace": str(workspace_video_path) if args.copy_video_to_workspace else None,
            "frames_dir": str(frames_dir),
            "fps": fps,
            "width": W,
            "height": H,
            "total_frames": total_frames,
            "sam2_repo": str(sam2_repo),
            "model_cfg": args.model_cfg,
            "model_weights": str(model_weights),
        },
        "object_id": object_ids[0],
        "object_ids": object_ids,
        "objects": prompt_records,
        "prompt": {
            "mode": prompt_mode,
            "init_frame_idx": init_frame_idx,
            "init_point": [args.init_point_x, args.init_point_y],
            "positive_points": prompt_records[0]["positive_points"],
            "negative_points": prompt_records[0]["negative_points"],
            "init_label": init_label,
            "init_box": init_box,
            "auto_init": auto_init_result,
            "auto_init_results": auto_init_results,
        },
        "frames": []
    }

    run_config = {
        "workspace_root": str(workspace_root),
        "scene_name": scene_name,
        "video_path": str(video_path),
        "frames_dir": str(frames_dir),
        "sam2_output_dir": str(sam2_out_dir),
        "save_vis_video": bool(args.save_vis_video),
        "save_mask_png": bool(args.save_mask_png),
        "prompt_mode": prompt_mode,
        "init_frame_idx": init_frame_idx,
        "init_obj_id": init_obj_id,
        "init_point_x": args.init_point_x,
        "init_point_y": args.init_point_y,
        "init_positive_points": prompt_records[0]["positive_points"],
        "init_negative_points": prompt_records[0]["negative_points"],
        "init_label": init_label,
        "init_box": init_box,
        "auto_init": auto_init_result,
        "auto_init_results": auto_init_results,
        "prompts": prompt_records,
        "auto_init_detector": args.auto_init_detector,
        "yolo_weights": args.yolo_weights,
        "yolo_conf": args.yolo_conf,
        "yolo_iou": args.yolo_iou,
        "yolo_classes": args.yolo_classes,
        "auto_init_num_objects": args.auto_init_num_objects,
        "auto_init_scan_frames": args.auto_init_scan_frames,
        "auto_init_scan_mode": args.auto_init_scan_mode,
        "bidirectional_propagation": not args.no_bidirectional_propagation,
        "auto_init_threshold": args.auto_init_threshold,
        "auto_init_min_area": args.auto_init_min_area,
        "auto_init_max_area_ratio": args.auto_init_max_area_ratio,
        "auto_init_min_fill_ratio": args.auto_init_min_fill_ratio,
        "auto_init_max_aspect_ratio": args.auto_init_max_aspect_ratio,
        "auto_init_min_separation_ratio": args.auto_init_min_separation_ratio,
        "auto_init_shadow_filter": not args.disable_auto_init_shadow_filter,
        "auto_init_negative_points": not args.no_auto_init_negative_points,
        "auto_init_object_refine": bool(args.auto_init_object_refine),
        "auto_init_box_expand": args.auto_init_box_expand,
        "auto_init_target_hint": args.auto_init_target_hint,
        "model_cfg": args.model_cfg,
        "model_weights": str(model_weights),
        "precision": precision_policy["resolved"],
        "precision_requested": precision_policy["requested"],
        "precision_policy": precision_policy,
        "tf32_enabled": tf32_enabled,
    }

    with open(run_config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    print("\n[4] Tracking ...")
    t0 = time.time()
    processed = 0
    tracking_records_by_frame = {}
    vis_frames_by_frame = {}
    prompt_frame_indices = [int(p["frame_idx"]) for p in prompts] or [int(init_frame_idx)]
    first_prompt_frame = min(prompt_frame_indices)
    last_prompt_frame = max(prompt_frame_indices)
    propagation_jobs = [("forward", False, first_prompt_frame)]
    if not args.no_bidirectional_propagation and last_prompt_frame > 0:
        propagation_jobs.append(("reverse", True, last_prompt_frame))

    for direction_name, reverse, start_frame_idx in propagation_jobs:
        print(f"Propagating {direction_name} from frame {start_frame_idx} ...")
        iterator = propagate_in_video_with_context(
            predictor,
            inference_state,
            start_frame_idx=int(start_frame_idx),
            reverse=bool(reverse),
            inference_context_factory=inference_context_factory,
        )

        for out_frame_idx, out_obj_ids, out_mask_logits in iterator:
            frame_idx = int(out_frame_idx)
            if frame_idx in tracking_records_by_frame:
                continue

            frame_name = f"{frame_idx:05d}.jpg"
            frame_path = frames_dir / frame_name
            masks_by_id = {int(oid): out_mask_logits[i] for i, oid in enumerate(out_obj_ids)}
            object_records = []
            vis_frame = cv2.imread(str(frame_path)) if writer is not None else None
            colors = [(0, 255, 0), (255, 0, 255), (255, 180, 0), (0, 180, 255)]

            for obj_pos, obj_id in enumerate(object_ids):
                target_mask = masks_by_id.get(int(obj_id))
                if target_mask is None:
                    object_records.append({
                        "object_id": int(obj_id),
                        "valid": False,
                        "x": None,
                        "y": None,
                        "centroid": None,
                        "bbox": None,
                        "area": 0,
                        "mask_path": None,
                    })
                    continue

                mask_bool = (target_mask > 0.0).detach().cpu().numpy().squeeze().astype(bool)
                area = int(mask_bool.sum())
                centroid = mask_to_centroid(mask_bool)
                bbox = mask_to_bbox(mask_bool)

                mask_path_str = None
                if args.save_mask_png:
                    if len(object_ids) == 1:
                        mask_path = masks_dir / f"{frame_idx:05d}.png"
                    else:
                        obj_mask_dir = masks_dir / f"obj_{int(obj_id):06d}"
                        ensure_dir(obj_mask_dir)
                        mask_path = obj_mask_dir / f"{frame_idx:05d}.png"
                    save_mask_png(mask_path, mask_bool)
                    mask_path_str = str(mask_path)

                object_record = {
                    "object_id": int(obj_id),
                    "valid": centroid is not None and area > 0,
                    "x": float(centroid[0]) if centroid is not None else None,
                    "y": float(centroid[1]) if centroid is not None else None,
                    "centroid": [float(centroid[0]), float(centroid[1])] if centroid is not None else None,
                    "bbox": bbox,
                    "area": area,
                    "mask_path": mask_path_str,
                    "propagation_direction": direction_name,
                }
                object_records.append(object_record)

                if vis_frame is not None:
                    color = colors[obj_pos % len(colors)]
                    if area > 0:
                        alpha_blend_mask(vis_frame, mask_bool, color_bgr=color, alpha=0.4)
                    if centroid is not None:
                        draw_tracking_result(vis_frame, centroid[0], centroid[1], obj_id, color=color)

            if len(object_ids) == 1:
                record = dict(object_records[0])
                record.update({
                    "frame_idx": frame_idx,
                    "frame_name": frame_name,
                    "frame_path": str(frame_path),
                    "time_sec": frame_idx / fps,
                })
            else:
                record = {
                    "frame_idx": frame_idx,
                    "frame_name": frame_name,
                    "frame_path": str(frame_path),
                    "time_sec": frame_idx / fps,
                    "objects": object_records,
                    "propagation_direction": direction_name,
                }
            tracking_records_by_frame[frame_idx] = record

            if writer is not None and vis_frame is not None:
                vis_frames_by_frame[frame_idx] = vis_frame

            processed += 1
            if processed % 10 == 0:
                avg_fps = processed / max(time.time() - t0, 1e-6)
                print(
                    f"Processed {processed}/{total_frames}, avg_fps={avg_fps:.2f}",
                    end="\r",
                )

    tracking_data["frames"] = [
        tracking_records_by_frame[i]
        for i in sorted(tracking_records_by_frame)
    ]

    if writer is not None:
        for frame_idx in sorted(vis_frames_by_frame):
            writer.write(vis_frames_by_frame[frame_idx])
        writer.release()

    recorded = {int(fr["frame_idx"]) for fr in tracking_data["frames"]}
    if len(recorded) < total_frames:
        for frame_idx in range(total_frames):
            if frame_idx in recorded:
                continue
            frame_name = f"{frame_idx:05d}.jpg"
            frame_path = frames_dir / frame_name
            reason = "not_tracked_before_init" if frame_idx < init_frame_idx else "not_tracked"
            if len(object_ids) == 1:
                tracking_data["frames"].append({
                    "frame_idx": frame_idx,
                    "frame_name": frame_name,
                    "frame_path": str(frame_path),
                    "time_sec": frame_idx / fps,
                    "object_id": int(object_ids[0]),
                    "valid": False,
                    "x": None,
                    "y": None,
                    "centroid": None,
                    "bbox": None,
                    "area": 0,
                    "mask_path": None,
                    "reason": reason,
                })
            else:
                tracking_data["frames"].append({
                    "frame_idx": frame_idx,
                    "frame_name": frame_name,
                    "frame_path": str(frame_path),
                    "time_sec": frame_idx / fps,
                    "objects": [
                        {
                            "object_id": int(obj_id),
                            "valid": False,
                            "x": None,
                            "y": None,
                            "centroid": None,
                            "bbox": None,
                            "area": 0,
                            "mask_path": None,
                            "reason": reason,
                        }
                        for obj_id in object_ids
                    ],
                })
        tracking_data["frames"].sort(key=lambda fr: int(fr["frame_idx"]))

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(tracking_data, f, ensure_ascii=False, indent=2)

    print("\nTracking completed.")
    print(f"Frames dir: {frames_dir}")
    print(f"Tracking JSON: {json_path}")
    if args.save_vis_video:
        print(f"Visualization video: {vis_video_path}")
    if args.save_mask_png:
        print(f"Masks dir: {masks_dir}")
    print(f"Run config: {run_config_path}")


if __name__ == "__main__":
    main()
