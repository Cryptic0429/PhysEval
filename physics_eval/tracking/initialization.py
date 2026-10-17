"""Validated initialization prompts for metadata-driven SAM2 tracking."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


from physics_eval.schemas import PromptSpec


def parse_json_value(value: Any, field_name: str = "JSON value") -> Any:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field_name} is not valid JSON: {exc.msg}") from exc
    return value


def normalize_init_mode(value: Any) -> str:
    mode = str(value or "auto").strip().lower()
    if mode not in {"auto", "manual", "renderer_gt"}:
        raise ValueError(f"unsupported tracking initialization mode: {mode}")
    return mode


def _point(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be an [x, y] pair")
    coords = tuple(float(part) for part in value)
    if not all(math.isfinite(part) and part >= 0 for part in coords):
        raise ValueError(f"{name} coordinates must be finite and nonnegative")
    return coords  # type: ignore[return-value]


def _point_list(value: Any, name: str, *, required: bool) -> tuple[tuple[float, float], ...]:
    if value is None:
        points: list[Any] = []
    elif isinstance(value, (list, tuple)):
        # A single [x, y] is accepted as a one-point list.
        points = [value] if len(value) == 2 and all(not isinstance(v, (list, tuple)) for v in value) else list(value)
    else:
        raise ValueError(f"{name} must be a list of [x, y] pairs")
    result = tuple(_point(item, name) for item in points)
    if required and not result:
        raise ValueError(f"{name} must contain at least one point")
    return result


def _box(value: Any) -> tuple[float, float, float, float] | None:
    if value is None or value == "":
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("box must be an [x1, y1, x2, y2] list")
    box = tuple(float(part) for part in value)
    if not all(math.isfinite(part) and part >= 0 for part in box):
        raise ValueError("box coordinates must be finite and nonnegative")
    x1, y1, x2, y2 = box
    if x2 < x1 or y2 < y1:
        raise ValueError("box must have x2 >= x1 and y2 >= y1")
    return box  # type: ignore[return-value]


def normalize_manual_prompts(raw: Any, expected_objects: int) -> list[PromptSpec]:
    data = parse_json_value(raw, "Tracking_Prompts_JSON")
    if not isinstance(data, list):
        raise ValueError("Tracking_Prompts_JSON must be a JSON list")
    if len(data) != int(expected_objects):
        raise ValueError(
            f"manual prompt count ({len(data)}) must match expected object count ({expected_objects})"
        )
    prompts: list[PromptSpec] = []
    seen_ids: set[int] = set()
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError("each manual prompt must be a JSON object")
        object_id = int(item.get("object_id", index))
        if object_id < 1 or object_id in seen_ids:
            raise ValueError("object_id values must be unique positive integers")
        seen_ids.add(object_id)
        frame_idx = int(item.get("frame_idx", item.get("frame", 0)))
        if frame_idx < 0:
            raise ValueError("frame_idx must be a nonnegative integer")
        positives = _point_list(
            item.get("positive_points", item.get("points", item.get("point"))),
            "positive_points",
            required=True,
        )
        negatives = _point_list(item.get("negative_points", []), "negative_points", required=False)
        box = _box(item.get("box"))
        if box is not None:
            x1, y1, x2, y2 = box
            if any(not (x1 <= x <= x2 and y1 <= y <= y2) for x, y in positives):
                raise ValueError("box must contain all positive points")
        prompts.append(
            PromptSpec(
                object_id=object_id,
                frame_idx=frame_idx,
                point=positives[0],
                positive_points=positives,
                negative_points=negatives,
                box=box,
                source=str(item.get("source") or "manual"),
            )
        )
    return prompts


def renderer_gt_eligibility(settings: Any = None) -> dict[str, int]:
    settings = settings if isinstance(settings, dict) else {}
    defaults = {
        "min_mask_area_px": 1,
        "min_bbox_side_px": 1,
        "min_border_margin_px": 0,
    }
    result: dict[str, int] = {}
    for key, default in defaults.items():
        value = settings.get(key, default)
        if isinstance(value, bool):
            raise ValueError(f"{key} must be a nonnegative integer")
        try:
            number = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be a nonnegative integer") from exc
        if number != value or number < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
        result[key] = number
    return result


def _mask_geometry(path: Path, rgba: Any) -> tuple[int, tuple[int, int, int, int], tuple[float, float], tuple[int, int]] | None:
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGBA"))
    color = np.asarray(rgba, dtype=np.uint8)
    if color.shape != (4,):
        raise ValueError("gt_rgba must contain four channel values")
    mask = np.all(pixels == color, axis=2)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    centroid = (float(xs.mean()), float(ys.mean()))
    return len(xs), (x1, y1, x2, y2), centroid, (pixels.shape[1], pixels.shape[0])


def renderer_gt_prompts(row: dict[str, Any], settings: dict[str, Any]) -> list[PromptSpec]:
    params = parse_json_value(row.get("Known_Parameters_JSON"), "Known_Parameters_JSON") or {}
    if not isinstance(params, dict):
        raise ValueError("Known_Parameters_JSON must be an object")
    pattern = params.get("gt_mask_pattern")
    rgba = params.get("gt_rgba")
    if not pattern or rgba is None:
        raise ValueError("renderer_gt requires gt_mask_pattern and gt_rgba in Known_Parameters_JSON")
    root = Path(params.get("workspace_root") or row.get("_Workspace_Root") or ".").expanduser().resolve()
    eligibility = renderer_gt_eligibility(settings)
    frames = params.get("gt_visible_frames")
    if frames is None:
        raw_frame_count = row.get("Expected_Frame_Count")
        frame_count = 0 if raw_frame_count is None or (isinstance(raw_frame_count, float) and math.isnan(raw_frame_count)) else int(raw_frame_count)
        frames = list(range(frame_count))
    if not isinstance(frames, list) or not frames:
        raise ValueError("renderer_gt requires a nonempty gt_visible_frames list or Expected_Frame_Count")
    padding = int(settings.get("box_padding_px", 0))
    if padding < 0:
        raise ValueError("box_padding_px must be nonnegative")
    for frame in frames:
        frame_idx = int(frame)
        path = (root / str(pattern).format(frame=frame_idx)).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("gt_mask_pattern must resolve inside workspace_root") from exc
        if not path.is_file():
            continue
        geometry = _mask_geometry(path, rgba)
        if geometry is None:
            continue
        area, (x1, y1, x2, y2), point, (width, height) = geometry
        box_width, box_height = x2 - x1 + 1, y2 - y1 + 1
        margin = min(x1, y1, width - 1 - x2, height - 1 - y2)
        if (
            area < eligibility["min_mask_area_px"]
            or min(box_width, box_height) < eligibility["min_bbox_side_px"]
            or margin < eligibility["min_border_margin_px"]
        ):
            continue
        box = (
            float(max(0, x1 - padding)),
            float(max(0, y1 - padding)),
            float(min(width - 1, x2 + padding)),
            float(min(height - 1, y2 + padding)),
        )
        return [
            PromptSpec(
                object_id=1,
                frame_idx=frame_idx,
                point=point,
                positive_points=(point,),
                negative_points=(),
                box=box,
                source=(
                    "renderer_gt_first_visible_exact_color"
                    if eligibility == renderer_gt_eligibility()
                    else "renderer_gt_first_eligible_exact_color"
                ),
            )
        ]
    raise ValueError("renderer_gt found no visible mask satisfying the predeclared eligibility gates")


def load_prompt_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read prompt manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("prompt manifest must be a JSON object")
    manifest["mode"] = normalize_init_mode(manifest.get("mode"))
    manifest["manifest_path"] = str(manifest_path)
    prompts = manifest.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("prompt manifest must contain a nonempty prompts list")
    expected = int(manifest.get("num_objects", len(prompts)))
    normalized = normalize_manual_prompts(prompts, expected)
    manifest["prompts"] = [item.to_dict() for item in normalized]
    return manifest


def select_automatic_prompts(yolo_detector, motion_detector, *, expected_objects: int):
    """Apply the fixed formal YOLO → motion initialization protocol.

    Detector callbacks return the existing candidate dictionaries. A partial
    YOLO detection is not sufficient for a multi-object scene. Fallback reasons
    are retained in each motion candidate for the existing result schema.
    """
    count = int(expected_objects)
    if count < 1:
        raise ValueError("expected_objects must be positive")
    reason = None
    try:
        candidates = yolo_detector()
        if len(candidates) == count:
            return candidates, "yolo"
        reason = f"YOLO returned {len(candidates)}/{count} target prompts"
    except Exception as exc:
        reason = str(exc)
    print(f"[WARN] YOLO auto-init failed, falling back to motion detector: {reason}")
    candidates = motion_detector()
    if len(candidates) != count:
        raise RuntimeError(
            f"YOLO and motion initialization failed: {reason}; "
            f"motion returned {len(candidates)}/{count} target prompts"
        )
    for item in candidates:
        item["yolo_fallback_reason"] = reason
    return candidates, "motion"


def manifest_detector_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Adapt explicit validated prompts to the existing tracking evidence schema."""
    records: list[dict[str, Any]] = []
    for prompt in manifest["prompts"]:
        positive_points = [
            [float(value) for value in point]
            for point in prompt.get("positive_points", [prompt["point"]])
        ]
        negative_points = [
            [float(value) for value in point]
            for point in prompt.get("negative_points", [])
        ]
        point = positive_points[0]
        box_raw = prompt.get("box")
        if box_raw is None:
            box = [point[0] - 2.0, point[1] - 2.0, point[0] + 2.0, point[1] + 2.0]
            prompt_box = None
        else:
            box = [float(value) for value in box_raw]
            prompt_box = box
        width = max(1.0, box[2] - box[0] + 1.0)
        height = max(1.0, box[3] - box[1] + 1.0)
        records.append({
            "frame_idx": int(prompt["frame_idx"]),
            "point": point,
            "positive_points": positive_points,
            "negative_points": negative_points,
            "bbox": box,
            "prompt_box": prompt_box,
            "motion_bbox": box,
            "area": int(round(width * height)),
            "bbox_area": int(round(width * height)),
            "fill_ratio": 1.0,
            "aspect_ratio": max(width / height, height / width),
            "motion_strength": 0.0,
            "intensity_delta": 0.0,
            "color_change": 0.0,
            "shadow_score": 0.0,
            "texture_strength": 0.0,
            "score": 1.0,
            "object_area": int(round(width * height)),
            "object_refined": False,
            "object_refine_source": prompt.get("source", manifest["mode"]),
            "search_bbox": box,
            "cluster_score": 1.0,
            "cluster_frames": [int(prompt["frame_idx"])],
            "cluster_n_frames": 1,
            "cluster_n_candidates": 1,
            "method": prompt.get("source", manifest["mode"]),
            "n_candidates": 1,
            "scan_frames": 1,
            "scan_mode": "metadata",
            "prompt_manifest": str(manifest.get("manifest_path") or ""),
        })
    return records


def _write_prompt_debug(frames_dir: Path, records: list[dict[str, Any]], debug_path: Path) -> None:
    import cv2

    first_frame = min(int(item["frame_idx"]) for item in records)
    image = cv2.imread(str(frames_dir / f"{first_frame:05d}.jpg"))
    if image is None:
        return
    for idx, item in enumerate(records, start=1):
        if int(item["frame_idx"]) != first_frame:
            continue
        x1, y1, x2, y2 = [int(round(value)) for value in item["bbox"]]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
        for point in item.get("positive_points", [item["point"]]):
            px, py = [int(round(value)) for value in point]
            cv2.circle(image, (px, py), 5, (0, 255, 0), -1)
        for point in item.get("negative_points", []):
            px, py = [int(round(value)) for value in point]
            cv2.circle(image, (px, py), 5, (0, 0, 255), -1)
        cv2.putText(image, f"obj {idx}", (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_path), image)


def build_tracking_prompts(
    args, *, frames_dir: Path, width: int, height: int,
    debug_path: Path | None = None,
    prompt_manifest: dict[str, Any] | None = None,
):
    """Create explicit SAM2 prompt dictionaries without runtime callback mutation.

    Manual/renderer_gt manifests bypass automatic detection and retain positive
    points, negative points, optional boxes, prompt frames, and provenance.
    """
    init_obj_id = int(args.init_obj_id)
    init_label = int(args.init_label)
    auto_results = []
    manifest_prompts = None
    if prompt_manifest is not None:
        mode = normalize_init_mode(prompt_manifest.get("mode"))
        if mode == "auto":
            raise ValueError("supplied prompt manifests must use manual or renderer_gt mode")
        manifest_prompts = normalize_manual_prompts(
            prompt_manifest.get("prompts"),
            int(prompt_manifest.get("num_objects", len(prompt_manifest.get("prompts", [])))),
        )
        normalized_manifest = dict(prompt_manifest)
        normalized_manifest["mode"] = mode
        normalized_manifest["prompts"] = [prompt.to_dict() for prompt in manifest_prompts]
        auto_results = manifest_detector_records(normalized_manifest)
        prompt_mode = f"metadata_{mode}"
        if debug_path is not None:
            _write_prompt_debug(frames_dir, auto_results, debug_path)
    elif bool(args.auto_init) or args.init_point_x is None or args.init_point_y is None:
        # Heavy detector dependencies are loaded only for automatic initialization.
        from .yolo import detect_yolo_prompts, draw_yolo_debug
        from .motion import detect_moving_object_prompts

        def yolo():
            return detect_yolo_prompts(
                frames_dir=frames_dir, width=width, height=height,
                num_objects=args.auto_init_num_objects,
                scan_frames=args.auto_init_scan_frames,
                scan_mode=args.auto_init_scan_mode,
                weights=args.yolo_weights, conf=args.yolo_conf, iou=args.yolo_iou,
                target_hint=args.auto_init_target_hint, classes=args.yolo_classes,
            )

        def motion():
            return detect_moving_object_prompts(
                frames_dir=frames_dir, width=width, height=height,
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

        auto_results, detector_used = select_automatic_prompts(
            yolo, motion, expected_objects=args.auto_init_num_objects,
        )
        prompt_mode = f"auto_{detector_used}"
        if detector_used == "yolo" and debug_path is not None:
            frame = frames_dir / f"{int(auto_results[0]['frame_idx']):05d}.jpg"
            draw_yolo_debug(frame, auto_results, debug_path)
    else:
        return [{
            "obj_id": init_obj_id,
            "frame_idx": int(args.init_frame_idx),
            "point": [float(args.init_point_x), float(args.init_point_y)],
            "positive_points": [[float(args.init_point_x), float(args.init_point_y)]],
            "label": init_label,
            "box": list(args.init_box) if args.init_box is not None else None,
            "negative_points": [],
            "auto_init": None,
        }], "manual", []

    from .motion import make_negative_points_around_box

    prompts = []
    for index, item in enumerate(auto_results):
        positives = [[float(value) for value in point] for point in item.get("positive_points", [item["point"]])]
        if "prompt_box" in item:
            box = [float(value) for value in item["prompt_box"]] if item["prompt_box"] is not None else None
        else:
            box = [float(value) for value in item["bbox"]] if not args.no_auto_init_box else None
        if "negative_points" in item:
            negatives = [[float(value) for value in point] for point in item["negative_points"]]
        elif args.no_auto_init_negative_points or not item.get("object_refined", False):
            negatives = []
        else:
            negatives = make_negative_points_around_box(box, width, height)
        # The former manifest adapter assigned consecutive tracker IDs; retain
        # that schema while preserving original IDs in batch_initialization.
        prompts.append({
            "obj_id": init_obj_id + index,
            "frame_idx": int(item["frame_idx"]),
            "point": positives[0],
            "positive_points": positives,
            "label": init_label,
            "box": box,
            "negative_points": negatives,
            "auto_init": item,
        })
    return prompts, prompt_mode, auto_results
