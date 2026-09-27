#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simple physics batch entrypoint. Defaults match the workspace layout: "
            "data/metadata, data/t2v_videos, repo/sam2, and compact outputs under batch_eval_results/<metric>/<prompt_id>."
        )
    )
    parser.add_argument("--workspace-root", default=str(WORKSPACE_ROOT), help="Default: this workspace directory")
    parser.add_argument("--metadata", default=None, help="Metadata .xlsx. Default: the only .xlsx in data/metadata")
    parser.add_argument("--video-root", default=None, help="Default: <workspace-root>/data/t2v_videos")
    parser.add_argument("--output-dir", default=None, help="Default: <workspace-root>/batch_eval_results")
    parser.add_argument("--sam2-repo", default=None, help="Default: <workspace-root>/repo/sam2")
    parser.add_argument("--model-cfg", default=None, help="SAM2 Hydra config, e.g. configs/sam2.1/sam2.1_hiera_b+.yaml")
    parser.add_argument("--model-weights", default=None, help="SAM2 checkpoint path")
    parser.add_argument("--sam2-precision", default="auto", choices=["auto", "fp32", "fp16", "bf16"],
                        help="SAM2 inference precision. Use fp32 for maximum compatibility.")
    parser.add_argument("--python-executable", default=None, help="Python executable used by the SAM2 subprocess")
    parser.add_argument("--index", type=int, default=None, help="Run one metadata Index")
    parser.add_argument("--prompt-id", default=None, help="Run one Prompt_ID")
    parser.add_argument("--detector", default="yolo_then_motion", choices=["motion", "yolo", "yolo_then_motion"])
    parser.add_argument(
        "--yolo-weights",
        default=None,
        help="Optional YOLO weights. Default: <workspace-root>/yolov8n.pt if it exists.",
    )
    parser.add_argument("--reuse-tracking", action="store_true", help="Reuse existing tracking JSON if present")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate existing tracking JSON only")
    parser.add_argument("--tracking-only", action="store_true", help="Only track videos, skip physics evaluation")
    parser.add_argument("--vis", action="store_true", help="Save tracking visualization videos")
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Write extra CSV/plot diagnostics and save the auto-init debug image",
    )
    parser.add_argument("--no-plots", action="store_true", help="Do not save mask QC plots")
    parser.add_argument("--area-action", choices=["warn", "skip"], default="warn", help="What to do when mask QC flags a video")
    parser.add_argument("--min-motion-extent-ratio", type=float, default=None,
                        help="Tracking quality gate: minimum motion extent in object diameters")
    parser.add_argument("--min-object-coverage", type=float, default=None,
                        help="Tracking quality gate: minimum valid-frame coverage per counted object")
    parser.add_argument("--strict", action="store_true", help="Mark weak/invalid quality as invalid")
    parser.add_argument(
        "--allow-renderer-gt-init",
        action="store_true",
        help=(
            "Allow metadata rows to use GT-assisted renderer point/box initialization. "
            "The row may predeclare first-visible or first-eligible mask selection."
        ),
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def append_optional(cmd: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def main() -> None:
    args = parse_args()
    workspace_root = Path(args.workspace_root).expanduser().resolve()
    yolo_weights = args.yolo_weights
    if yolo_weights is None:
        default_yolo = workspace_root / "yolov8n.pt"
        if default_yolo.exists():
            yolo_weights = str(default_yolo)

    batch_argv = [
        "physics_eval.run_video_batch",
        "--workspace-root",
        str(workspace_root),
        "--detector",
        args.detector,
        "--save-mask-png",
        "--area-check",
        "--area-prefer-mask-area",
        "--area-stability-action",
        args.area_action,
        "--log-level",
        args.log_level,
        "--sam2-precision",
        args.sam2_precision,
    ]
    if not args.no_plots:
        batch_argv.append("--area-stability-save-plots")
    if args.vis:
        batch_argv.append("--save-vis-video")
    if args.diagnostics:
        batch_argv.append("--save-diagnostics")
        batch_argv.append("--save-auto-init-debug")
    if args.reuse_tracking:
        batch_argv.append("--skip-tracking-if-exists")
    if args.eval_only:
        batch_argv.append("--eval-only")
    if args.tracking_only:
        batch_argv.append("--tracking-only")
    if args.strict:
        batch_argv.append("--strict")
    if args.allow_renderer_gt_init:
        batch_argv.append("--allow-renderer-gt-init")

    append_optional(batch_argv, "--metadata", args.metadata)
    append_optional(batch_argv, "--video-root", args.video_root)
    append_optional(batch_argv, "--output-dir", args.output_dir)
    append_optional(batch_argv, "--sam2-repo", args.sam2_repo)
    append_optional(batch_argv, "--model-cfg", args.model_cfg)
    append_optional(batch_argv, "--model-weights", args.model_weights)
    append_optional(batch_argv, "--python-executable", args.python_executable)
    append_optional(batch_argv, "--index", args.index)
    append_optional(batch_argv, "--prompt-id", args.prompt_id)
    append_optional(batch_argv, "--yolo-weights", yolo_weights)
    append_optional(batch_argv, "--min-motion-extent-ratio", args.min_motion_extent_ratio)
    append_optional(batch_argv, "--min-object-coverage", args.min_object_coverage)

    sys.argv = batch_argv
    from physics_eval.run_video_batch import main as batch_main

    batch_main()


if __name__ == "__main__":
    main()
