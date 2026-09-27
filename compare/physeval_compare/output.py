from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config import ModeSpec
from .io_utils import dump_json, json_safe
from .metrics import mask_geometry, mask_quality, tracking_quality
from .types import MaskTracks, Prompt, VideoInfo


def write_tracking_result(
    *,
    output_dir: Path,
    mode: ModeSpec,
    row: dict[str, Any],
    info: VideoInfo,
    prompts: list[Prompt],
    prompt_provenance: dict[str, Any],
    masks: MaskTracks,
    min_coverage: float,
    min_motion_ratio: float,
    save_masks: bool,
    save_visualization: bool,
    qmask_protocol: str = "grouped_v2_candidate",
) -> dict[str, Any]:
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = sorted(info.frames_dir.glob("*.jpg"))
    records: list[dict[str, Any]] = []
    for object_id, object_masks in sorted(masks.items()):
        object_mask_dir = output_dir / "masks" / f"obj_{object_id:06d}"
        if save_masks:
            object_mask_dir.mkdir(parents=True, exist_ok=True)
        for frame_idx in range(info.total_frames):
            mask = object_masks.get(frame_idx)
            if mask is None:
                mask = np.zeros((info.height, info.width), dtype=bool)
            centroid, bbox, area = mask_geometry(mask)
            mask_path = None
            if save_masks and area > 0:
                path = object_mask_dir / f"{frame_idx:05d}.png"
                cv2.imwrite(str(path), mask.astype(np.uint8) * 255)
                mask_path = str(path.resolve())
            records.append({
                "object_id": object_id,
                "valid": centroid is not None,
                "x": centroid[0] if centroid else None,
                "y": centroid[1] if centroid else None,
                "centroid": centroid,
                "bbox": bbox,
                "area": area,
                "mask_path": mask_path,
                "frame_idx": frame_idx,
                "frame_name": frame_paths[frame_idx].name,
                "frame_path": str(frame_paths[frame_idx].resolve()),
                "time_sec": frame_idx / info.fps if info.fps > 0 else None,
            })

    expected = max(1, int(float(row.get("Num_Objects") or len(prompts) or 1)))
    tracking = tracking_quality(
        records,
        expected_objects=expected,
        total_frames=info.total_frames,
        min_coverage=min_coverage,
        min_motion_ratio=min_motion_ratio,
    )
    masks_qc = mask_quality(records, masks, total_frames=info.total_frames, protocol=qmask_protocol, required_object_ids=[p.obj_id for p in prompts])
    from physics_eval.quality.adapters import normalize_records, nested_frames
    canonical_records, _ = normalize_records({"frames": records, "total_frames": info.total_frames}, "exclusive")
    tracking_path = output_dir / "tracking_points.json"
    tracking_payload = {
        "video_info": info.to_dict() | {
            "scene_name": str(row.get("Prompt_ID") or info.path.stem),
            "comparison_mode": mode.name,
            "comparison_label": mode.label,
        },
        "object_id": prompts[0].obj_id if prompts else None,
        "object_ids": [prompt.obj_id for prompt in prompts],
        "objects": [
            {
                "obj_id": prompt.obj_id,
                "init_frame_idx": prompt.frame_idx,
                "init_point": prompt.point,
                "init_label": 1,
                "init_box": prompt.box,
                "negative_points": prompt.negative_points,
                "auto_init": prompt.to_dict(),
            }
            for prompt in prompts
        ],
        "prompt": {
            "mode": f"auto_{prompt_provenance.get('detector_used')}",
            "prompts": [prompt.to_dict() for prompt in prompts],
            "provenance": prompt_provenance,
        },
        "schema_version": 2,
        "bbox_convention": "inclusive",
        "total_frames": info.total_frames,
        "frames": nested_frames(canonical_records, info.total_frames),
        "quality_checks": {
            "tracking_quality": tracking,
            "mask_qc": masks_qc,
        },
    }
    dump_json(tracking_path, json_safe(tracking_payload))

    if save_visualization:
        _write_visualization(output_dir / "tracking_visualization.mp4", info, masks)

    result = {
        "status": "ok",
        "comparison_mode": mode.name,
        "comparison_label": mode.label,
        "metadata": json_safe(row),
        "prompt_provenance": prompt_provenance,
        "tracking_quality": tracking,
        "mask_qc": masks_qc,
        "artifacts": {
            "tracking_json": str(tracking_path.resolve()),
            "run_config": str((output_dir / "run_config.json").resolve()),
            "visualization": str((output_dir / "tracking_visualization.mp4").resolve()) if save_visualization else None,
        },
    }
    dump_json(output_dir / "result.json", json_safe(result))
    return result


def write_failed_result(
    output_dir: Path,
    *,
    mode: ModeSpec,
    row: dict[str, Any],
    error: Exception,
) -> None:
    dump_json(output_dir / "result.json", {
        "status": "failed",
        "comparison_mode": mode.name,
        "comparison_label": mode.label,
        "metadata": json_safe(row),
        "error_type": type(error).__name__,
        "error": str(error),
    })


def _write_visualization(path: Path, info: VideoInfo, masks: MaskTracks) -> None:
    import cv2

    frame_paths = sorted(info.frames_dir.glob("*.jpg"))
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), info.fps if info.fps > 0 else 15.0,
        (info.width, info.height),
    )
    colors = [(0, 255, 0), (255, 0, 255), (0, 255, 255), (255, 128, 0)]
    try:
        for frame_idx, frame_path in enumerate(frame_paths):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue
            for offset, (object_id, object_masks) in enumerate(sorted(masks.items())):
                mask = object_masks.get(frame_idx)
                if mask is None or not np.any(mask):
                    continue
                color = colors[offset % len(colors)]
                overlay = frame.copy()
                overlay[mask] = color
                frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
                centroid, bbox, _ = mask_geometry(mask)
                if bbox is not None and centroid is not None:
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    cv2.putText(
                        frame, f"obj {object_id}", (bbox[0], max(18, bbox[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                    )
            writer.write(frame)
    finally:
        writer.release()
