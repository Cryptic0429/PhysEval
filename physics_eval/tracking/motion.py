"""Motion candidates and visible-object refinement for formal initialization.

These thresholds, clustering, and refinement operations retain the published
formal pipeline policy.
"""
from pathlib import Path

import cv2
import numpy as np

from .video import list_jpg_frames


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
