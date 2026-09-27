from __future__ import annotations

import argparse
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

from .config import DEFAULT_MODES, MODE_SPECS, ModeSpec
from .detectors import FasterRcnnDetector, MotionDetector, YoloDetector, detect_with_fallback
from .io_utils import (
    dump_json,
    expected_objects,
    extract_video_frames,
    filter_rows,
    infer_target_hint,
    load_json,
    load_metadata,
    resolve_video,
)
from .output import write_failed_result, write_tracking_result
from .trackers import Sam2Tracker, TamTracker
from .types import Prompt, VideoInfo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the paired Faster R-CNN/YOLO and SAM2/TAM comparison on PhysEval videos."
    )
    parser.add_argument(
        "--metadata",
        default=str(
            WORKSPACE_ROOT / "benchmark" / "metadata" / "csv"
            / "phys_t2v_bench_metadata__metadata.csv"
        ),
        help="PhysEval metadata .xlsx/.csv; pass 'none' to scan the video folder without metadata.",
    )
    parser.add_argument("--video-root", default=str(WORKSPACE_ROOT / "data" / "t2v_videos"))
    parser.add_argument("--qmask-protocol", default="grouped_v2_candidate", choices=["legacy_v1", "grouped_v2_candidate", "compare_legacy_v1"])
    parser.add_argument("--model-name", default=None, help="Default: basename of --video-root")
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--cache-root", default=str(PROJECT_ROOT / "cache"))
    parser.add_argument(
        "--modes", nargs="+", default=["all"],
        choices=["all", *MODE_SPECS],
        help="Default runs all three paired comparison modes.",
    )
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--prompt-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-prompts", action="store_true")
    parser.add_argument("--overwrite-frames", action="store_true")
    parser.add_argument(
        "--cleanup-frames",
        action="store_true",
        help="Delete each video's extracted JPEG cache after all selected modes finish.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--skip-missing-videos", action="store_true")
    parser.add_argument("--save-visualization", action="store_true")
    parser.add_argument("--no-save-masks", action="store_true")

    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--precision", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    parser.add_argument("--scan-frames", type=int, default=60)
    parser.add_argument("--scan-mode", default="uniform", choices=["uniform", "start", "middle"])
    parser.add_argument("--detector-conf", type=float, default=0.25)
    parser.add_argument("--yolo-iou", type=float, default=0.50)
    parser.add_argument("--motion-threshold", type=float, default=25.0)
    parser.add_argument("--motion-min-area", type=int, default=40)
    parser.add_argument("--motion-max-area-ratio", type=float, default=0.20)
    parser.add_argument("--min-object-coverage", type=float, default=0.50)
    parser.add_argument("--min-motion-extent-ratio", type=float, default=0.75)

    parser.add_argument("--yolo-weights", default=str(WORKSPACE_ROOT / "yolov8n.pt"))
    parser.add_argument("--sam2-repo", default=str(WORKSPACE_ROOT / "repo" / "sam2"))
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    parser.add_argument(
        "--sam2-checkpoint",
        default=str(WORKSPACE_ROOT / "repo" / "sam2" / "checkpoints" / "sam2.1_hiera_base_plus.pt"),
    )
    parser.add_argument("--offload-video-to-cpu", action="store_true")
    parser.add_argument("--offload-state-to-cpu", action="store_true")

    parser.add_argument(
        "--segment-anything-repo", default=str(WORKSPACE_ROOT / "repo" / "segment-anything")
    )
    parser.add_argument(
        "--sam1-checkpoint",
        default=str(WORKSPACE_ROOT / "repo" / "segment-anything" / "checkpoints" / "sam_vit_b_01ec64.pth"),
    )
    parser.add_argument("--sam1-model-type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument(
        "--track-anything-repo", default=str(WORKSPACE_ROOT / "repo" / "Track-Anything")
    )
    parser.add_argument(
        "--xmem-checkpoint",
        default=str(WORKSPACE_ROOT / "repo" / "Track-Anything" / "checkpoints" / "XMem.pth"),
    )
    return parser.parse_args(argv)


def resolve_modes(raw: list[str]) -> list[ModeSpec]:
    names = list(DEFAULT_MODES) if "all" in raw else raw
    return [MODE_SPECS[name] for name in dict.fromkeys(names)]


def resolve_device(raw: str) -> str:
    if raw != "auto":
        return raw
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def safe_name(value: Any, fallback: str) -> str:
    raw = str(value or fallback).strip()
    for character in '<>:"/\\|?*':
        raw = raw.replace(character, "_")
    return raw or fallback


def select_shard(
    rows: list[dict[str, Any]], num_shards: int, shard_index: int,
) -> list[dict[str, Any]]:
    if num_shards < 1:
        raise ValueError("--num-shards must be at least 1.")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards.")
    return rows[shard_index::num_shards]


class CompareRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = resolve_device(args.device)
        self.video_root = Path(args.video_root).expanduser().resolve()
        self.output_root = Path(args.output_root).expanduser().resolve()
        self.cache_root = Path(args.cache_root).expanduser().resolve()
        self.model_name = safe_name(args.model_name or self.video_root.name, "model")
        self.modes = resolve_modes(args.modes)
        self._detectors: dict[str, Any] = {}
        self._trackers: dict[str, Any] = {}
        self.motion = MotionDetector(
            threshold=args.motion_threshold,
            min_area=args.motion_min_area,
            max_area_ratio=args.motion_max_area_ratio,
        )

    def run(self) -> int:
        if not self.video_root.exists():
            raise FileNotFoundError(f"Video root not found: {self.video_root}")
        metadata = None if str(self.args.metadata).lower() == "none" else Path(self.args.metadata).expanduser().resolve()
        rows = filter_rows(
            load_metadata(metadata, self.video_root),
            index=self.args.index,
            prompt_id=self.args.prompt_id,
            limit=self.args.limit,
        )
        rows = select_shard(rows, self.args.num_shards, self.args.shard_index)
        if not rows:
            raise RuntimeError("No metadata/video rows matched the requested filters.")
        if self.args.dry_run:
            return self._preflight(rows)

        failures = 0
        for position, row in enumerate(rows, start=1):
            prompt_id = safe_name(row.get("Prompt_ID"), f"row_{position:04d}")
            try:
                video_path = resolve_video(self.video_root, row.get("Video_File"))
            except FileNotFoundError as exc:
                if self.args.skip_missing_videos:
                    print(f"[skip missing] {prompt_id}: {exc}")
                    continue
                raise
            print(f"[{position}/{len(rows)}] {prompt_id}: {video_path.name}")
            frames_dir = self.cache_root / "frames" / self.model_name / prompt_id
            try:
                info = extract_video_frames(video_path, frames_dir, overwrite=self.args.overwrite_frames)
                for mode in self.modes:
                    output_dir = self._mode_output_dir(mode, row, prompt_id)
                    if (output_dir / "result.json").exists() and not self.args.overwrite:
                        cached = load_json(output_dir / "result.json")
                        stored = (cached.get("mask_qc") or {}).get("protocol_id", "compare_legacy_v1")
                        if cached.get("status") == "ok" and stored != self.args.qmask_protocol:
                            raise ValueError("Existing result uses a different Qmask protocol; use a separate output root or --overwrite")
                        print(f"  [reuse] {mode.name}")
                        continue
                    try:
                        prompts, provenance = self._prompts(mode, row, info, prompt_id)
                        tracker = self._tracker(mode.tracker)
                        masks = tracker.track(info, prompts)
                        run_config = self._run_config(mode, row, info, prompts, provenance)
                        dump_json(output_dir / "run_config.json", run_config)
                        result = write_tracking_result(
                            output_dir=output_dir,
                            mode=mode,
                            row=row,
                            info=info,
                            prompts=prompts,
                            prompt_provenance=provenance,
                            masks=masks,
                            min_coverage=self.args.min_object_coverage,
                            min_motion_ratio=self.args.min_motion_extent_ratio,
                            save_masks=not self.args.no_save_masks,
                            save_visualization=self.args.save_visualization,
                            qmask_protocol=self.args.qmask_protocol,
                        )
                        print(
                            f"  [ok] {mode.name}: tracking={result['tracking_quality']['status']}, "
                            f"Q_mask={result['mask_qc']['score']}"
                        )
                    except Exception as exc:
                        failures += 1
                        write_failed_result(output_dir, mode=mode, row=row, error=exc)
                        print(f"  [failed] {mode.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
                        if self.args.fail_fast:
                            traceback.print_exc()
                            raise
            finally:
                if self.args.cleanup_frames:
                    self._cleanup_frames(frames_dir)
        print(f"Completed with {failures} failed mode-runs.")
        return 1 if failures else 0

    def _cleanup_frames(self, frames_dir: Path) -> None:
        cache_frames_root = (self.cache_root / "frames").resolve()
        target = frames_dir.resolve()
        if target == cache_frames_root or cache_frames_root not in target.parents:
            raise RuntimeError(f"Refusing to clean a path outside the frame cache: {target}")
        if target.exists():
            shutil.rmtree(target)
            print(f"  [cleanup] removed extracted frames: {target}")

    def _preflight(self, rows: list[dict[str, Any]]) -> int:
        present = 0
        missing = []
        for row in rows:
            try:
                resolve_video(self.video_root, row.get("Video_File"))
                present += 1
            except FileNotFoundError:
                missing.append(str(row.get("Video_File")))
        print(f"model={self.model_name}")
        print(f"modes={','.join(mode.name for mode in self.modes)}")
        print(f"shard={self.args.shard_index}/{self.args.num_shards}")
        print(f"rows={len(rows)}, videos_present={present}, videos_missing={len(missing)}")
        if missing:
            print("missing examples:")
            for name in missing[:20]:
                print(f"  {name}")
        return 1 if missing and not self.args.skip_missing_videos else 0

    def _mode_output_dir(self, mode: ModeSpec, row: dict[str, Any], prompt_id: str) -> Path:
        metric = safe_name(row.get("Metric"), "tracking_only")
        return self.output_root / self.model_name / mode.name / metric / prompt_id

    def _detector(self, name: str):
        if name in self._detectors:
            return self._detectors[name]
        if name == "yolo":
            detector = YoloDetector(
                self.args.yolo_weights, self.device, self.args.detector_conf, self.args.yolo_iou,
            )
        elif name == "fasterrcnn":
            detector = FasterRcnnDetector(self.device, self.args.detector_conf)
        else:
            raise ValueError(f"Unknown detector: {name}")
        self._detectors[name] = detector
        return detector

    def _tracker(self, name: str):
        if name in self._trackers:
            return self._trackers[name]
        if name == "sam2":
            tracker = Sam2Tracker(
                repo=Path(self.args.sam2_repo).expanduser().resolve(),
                model_config=self.args.sam2_config,
                checkpoint=Path(self.args.sam2_checkpoint).expanduser().resolve(),
                device=self.device,
                precision=self.args.precision,
                offload_video_to_cpu=self.args.offload_video_to_cpu,
                offload_state_to_cpu=self.args.offload_state_to_cpu,
            )
        elif name == "tam":
            tracker = TamTracker(
                segment_anything_repo=Path(self.args.segment_anything_repo).expanduser().resolve(),
                sam_checkpoint=Path(self.args.sam1_checkpoint).expanduser().resolve(),
                sam_model_type=self.args.sam1_model_type,
                track_anything_repo=Path(self.args.track_anything_repo).expanduser().resolve(),
                xmem_checkpoint=Path(self.args.xmem_checkpoint).expanduser().resolve(),
                device=self.device,
            )
        else:
            raise ValueError(f"Unknown tracker: {name}")
        self._trackers[name] = tracker
        return tracker

    def _prompts(
        self,
        mode: ModeSpec,
        row: dict[str, Any],
        info: VideoInfo,
        prompt_id: str,
    ) -> tuple[list[Prompt], dict[str, Any]]:
        target_hint = infer_target_hint(row)
        count = expected_objects(row)
        signature = {
            "detector": mode.detector,
            "video_path": str(info.path),
            "video_size": info.path.stat().st_size,
            "target_hint": target_hint,
            "expected_objects": count,
            "scan_frames": self.args.scan_frames,
            "scan_mode": self.args.scan_mode,
            "detector_conf": self.args.detector_conf,
            "yolo_iou": self.args.yolo_iou if mode.detector == "yolo" else None,
            "same_frame_multi_object": True,
        }
        cache_path = self.output_root / self.model_name / "_prompts" / mode.detector / f"{prompt_id}.json"
        if cache_path.exists() and not self.args.overwrite_prompts:
            cached = load_json(cache_path)
            if cached.get("signature") == signature:
                return [Prompt.from_dict(value) for value in cached["prompts"]], cached["provenance"]
        detector = self._detector(mode.detector)
        prompts, provenance = detect_with_fallback(
            detector,
            self.motion,
            info=info,
            target_hint=target_hint,
            count=count,
            scan_frames=self.args.scan_frames,
            scan_mode=self.args.scan_mode,
        )
        provenance = provenance | {
            "target_hint": target_hint,
            "expected_objects": count,
            "shared_prompt_cache": str(cache_path.resolve()),
        }
        dump_json(cache_path, {
            "signature": signature,
            "provenance": provenance,
            "prompts": [prompt.to_dict() for prompt in prompts],
        })
        return prompts, provenance

    def _run_config(
        self,
        mode: ModeSpec,
        row: dict[str, Any],
        info: VideoInfo,
        prompts: list[Prompt],
        provenance: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "project_root": str(PROJECT_ROOT),
            "qmask_protocol": self.args.qmask_protocol,
            "model_name": self.model_name,
            "comparison_mode": mode.name,
            "comparison_label": mode.label,
            "detector": mode.detector,
            "tracker": mode.tracker,
            "device": self.device,
            "precision": self.args.precision,
            "video": info.to_dict(),
            "metadata_index": row.get("Index"),
            "prompt_id": row.get("Prompt_ID"),
            "prompts": [prompt.to_dict() for prompt in prompts],
            "prompt_provenance": provenance,
            "thresholds": {
                "detector_conf": self.args.detector_conf,
                "yolo_iou": self.args.yolo_iou,
                "scan_frames": self.args.scan_frames,
                "scan_mode": self.args.scan_mode,
                "min_object_coverage": self.args.min_object_coverage,
                "min_motion_extent_ratio": self.args.min_motion_extent_ratio,
            },
            "assets": {
                "yolo_weights": self.args.yolo_weights,
                "sam2_repo": self.args.sam2_repo,
                "sam2_config": self.args.sam2_config,
                "sam2_checkpoint": self.args.sam2_checkpoint,
                "segment_anything_repo": self.args.segment_anything_repo,
                "sam1_checkpoint": self.args.sam1_checkpoint,
                "sam1_model_type": self.args.sam1_model_type,
                "track_anything_repo": self.args.track_anything_repo,
                "xmem_checkpoint": self.args.xmem_checkpoint,
            },
        }


def main(argv: list[str] | None = None) -> int:
    return CompareRunner(parse_args(argv)).run()
