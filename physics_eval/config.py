"""Shared defaults and path resolution for the formal evaluation pipeline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
INITIALIZATION_POLICY = "yolo_then_motion"
TRACKING_WORKER_MODULE = "physics_eval.tracking.worker"

def default_model_weights(sam2_repo: Path) -> Path:
    candidates = [
        sam2_repo / "checkpoints" / "sam2.1_hiera_base_plus.pt",
        sam2_repo / "checkpoints" / "sam2.1_hiera_b+.pt",
        sam2_repo / "checkpoints" / "sam2_hiera_base_plus.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def default_metadata_path(workspace_root: Path) -> Path | None:
    metadata_dir = workspace_root / "data" / "metadata"
    if not metadata_dir.exists():
        return None
    files = sorted(
        p for p in metadata_dir.glob("*.xlsx")
        if not p.name.startswith("~$")
    )
    return files[0] if len(files) == 1 else None


def resolve_cli_path(raw: str | Path, workspace_root: Path) -> Path:
    """Resolve user-facing relative paths against the repository, not the shell CWD."""
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (workspace_root / path).resolve()


def normalize_eval_args(args: argparse.Namespace) -> argparse.Namespace:
    workspace_root = Path(args.workspace_root).expanduser().resolve() if args.workspace_root else WORKSPACE_ROOT.resolve()
    sam2_repo = resolve_cli_path(args.sam2_repo, workspace_root) if args.sam2_repo else workspace_root / "repo" / "sam2"

    args.workspace_root = str(workspace_root)
    if not args.metadata:
        auto_metadata = default_metadata_path(workspace_root)
        if auto_metadata is not None:
            args.metadata = str(auto_metadata)
    elif args.metadata:
        args.metadata = str(resolve_cli_path(args.metadata, workspace_root))
    args.video_root = str(resolve_cli_path(args.video_root, workspace_root)) if args.video_root else str(workspace_root / "data" / "t2v_videos")
    args.output_dir = str(resolve_cli_path(args.output_dir, workspace_root)) if args.output_dir else str(workspace_root / "batch_eval_results")
    args.sam2_repo = str(sam2_repo)
    args.model_weights = (
        str(resolve_cli_path(args.model_weights, workspace_root))
        if args.model_weights
        else str(default_model_weights(sam2_repo))
    )
    if args.sam2_script:
        args.sam2_script = str(resolve_cli_path(args.sam2_script, workspace_root))
    if args.yolo_weights:
        args.yolo_weights = str(resolve_cli_path(args.yolo_weights, workspace_root))
    if args.area_stability_output_dir:
        args.area_stability_output_dir = str(resolve_cli_path(args.area_stability_output_dir, workspace_root))

    args.detector = INITIALIZATION_POLICY
    # Preserve the former public eval wrapper's evidence-generation defaults.
    args.save_mask_png = True
    args.area_stability_check = True
    args.area_prefer_mask_area = True
    args.area_stability_save_plots = not getattr(args, "no_plots", False)
    if args.save_diagnostics:
        args.save_auto_init_debug = True
    if not args.yolo_weights:
        local_weights = workspace_root / "yolov8n.pt"
        if local_weights.is_file():
            args.yolo_weights = str(local_weights)

    if args.debug:
        args.save_vis_video = True
        args.save_mask_png = True
        args.save_auto_init_debug = True
        args.save_diagnostics = True
        args.area_stability_check = True
        args.area_prefer_mask_area = True
        args.area_stability_save_plots = True

    return args


