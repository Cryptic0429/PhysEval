"""Validated initialization prompts for metadata-driven SAM2 tracking."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PromptSpec:
    object_id: int
    frame_idx: int
    point: tuple[float, float]
    positive_points: tuple[tuple[float, float], ...]
    negative_points: tuple[tuple[float, float], ...]
    box: tuple[float, float, float, float] | None = None
    source: str = "manual"

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "frame_idx": self.frame_idx,
            "point": list(self.point),
            "positive_points": [list(point) for point in self.positive_points],
            "negative_points": [list(point) for point in self.negative_points],
            "box": list(self.box) if self.box is not None else None,
            "source": self.source,
        }


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
