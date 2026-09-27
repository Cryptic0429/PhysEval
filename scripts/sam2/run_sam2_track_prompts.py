#!/usr/bin/env python3
"""Adapter that feeds deterministic metadata prompts to run_sam2_track.py.

The adapter replaces the automatic detector callback with a validated prompt
manifest, so one SAM2 inference state can receive one or more objects, each
with one or more positive points plus optional explicit negative points/box.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import cv2


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from physics_eval.utils.tracking_init import load_prompt_manifest


def parse_adapter_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAM2 with a validated metadata prompt manifest")
    parser.add_argument("--prompt-manifest", required=True)
    parser.add_argument("--base-script", default=str(Path(__file__).with_name("run_sam2_track.py")))
    parser.add_argument("tracker_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.tracker_args and args.tracker_args[0] == "--":
        args.tracker_args = args.tracker_args[1:]
    return args


def _load_tracker(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("physeval_sam2_tracker", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import SAM2 tracker: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _detector_records(manifest: dict[str, Any], debug_path: Path | None = None) -> list[dict[str, Any]]:
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
        records.append(
            {
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
            }
        )

    return records


def _write_debug(frames_dir: Path, records: list[dict[str, Any]], debug_path: Path) -> None:
    first_frame = min(int(item["frame_idx"]) for item in records)
    frame_path = frames_dir / f"{first_frame:05d}.jpg"
    image = cv2.imread(str(frame_path))
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


def _update_provenance(workspace_root: Path, scene_name: str, manifest: dict[str, Any]) -> None:
    scene_dir = workspace_root / "sam2_tracks" / scene_name
    for path in (scene_dir / "tracking_points.json", scene_dir / "run_config.json"):
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["batch_initialization"] = manifest
        if path.name == "run_config.json":
            payload["prompt_mode"] = f"metadata_{manifest['mode']}"
        else:
            prompt = payload.get("prompt")
            if isinstance(prompt, dict):
                prompt["mode"] = f"metadata_{manifest['mode']}"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    adapter_args = parse_adapter_args()
    manifest_path = Path(adapter_args.prompt_manifest).expanduser().resolve()
    manifest = load_prompt_manifest(manifest_path)
    manifest["manifest_path"] = str(manifest_path)
    base_script = Path(adapter_args.base_script).expanduser().resolve()
    if not base_script.is_file():
        raise FileNotFoundError(f"base SAM2 tracker not found: {base_script}")
    tracker = _load_tracker(base_script)

    records = _detector_records(manifest)

    def supplied_prompts(*, frames_dir, debug_path=None, **_kwargs):
        if debug_path is not None:
            _write_debug(Path(frames_dir), records, Path(debug_path))
        return records

    tracker.detect_moving_object_prompts = supplied_prompts
    argv = list(adapter_args.tracker_args)
    argv.extend(
        [
            "--auto-init",
            "--auto-init-detector",
            "motion",
            "--auto-init-num-objects",
            str(len(records)),
            "--init-obj-id",
            "1",
            "--no-auto-init-negative-points",
        ]
    )
    if all(prompt.get("box") is None for prompt in manifest["prompts"]):
        argv.append("--no-auto-init-box")
    sys.argv = [str(base_script), *argv]
    tracker.main()

    def option_value(flag: str) -> str:
        try:
            return argv[argv.index(flag) + 1]
        except (ValueError, IndexError) as exc:
            raise RuntimeError(f"adapter requires tracker argument {flag}") from exc

    workspace_root = Path(option_value("--workspace-root")).expanduser().resolve()
    scene_name = option_value("--scene-name")
    _update_provenance(workspace_root, scene_name, manifest)


if __name__ == "__main__":
    main()
