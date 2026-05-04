from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import pandas as pd

from physics_eval.run_batch_eval import filter_metadata, process_row
from physics_eval.utils.io import failure_result, read_metadata, row_to_dict, write_results
from physics_eval.utils.tracking import normalize_tracking_json
from physics_eval.utils.tracking_quality import write_tracking_quality
from scripts.quality.check_mask_area_stability import (
    analyze_series,
    load_series_from_json,
    save_plot,
    write_csv,
)


LOGGER = logging.getLogger("physics_eval.video_batch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end batch pipeline: read metadata Excel, track each video with SAM2 auto-init, "
            "then run physics evaluators."
        )
    )
    parser.add_argument("--metadata", default=None,
                        help="Path to metadata .xlsx. If omitted, use the only .xlsx under <workspace-root>/data/metadata")
    parser.add_argument("--video-root", default=None, help="Default: <workspace-root>/data/t2v_videos")
    parser.add_argument("--output-dir", default=None, help="Default: <workspace-root>/batch_eval_results")
    parser.add_argument("--output-layout", choices=["organized", "flat"], default="organized",
                        help="organized writes <output-dir>/<metric>/<prompt_id>/result.json. flat keeps the older shared folders.")

    parser.add_argument("--workspace-root", default=None, help="Default: repository/workspace root inferred from this file")
    parser.add_argument("--sam2-script", default=None, help="Path to scripts/sam2/run_sam2_track.py")
    parser.add_argument("--sam2-repo", default=None, help="Default: <workspace-root>/repo/sam2")
    parser.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_b+.yaml",
                        help="SAM2 model cfg")
    parser.add_argument("--model-weights", default=None,
                        help="Default: <sam2-repo>/checkpoints/sam2.1_hiera_base_plus.pt")

    parser.add_argument("--index", type=int, default=None, help="Only process a single Index")
    parser.add_argument("--prompt-id", default=None, help="Only process a single Prompt_ID")
    parser.add_argument("--fps", type=float, default=None, help="Fallback fps for evaluation if tracking JSON has no fps/time")
    parser.add_argument("--strict", action="store_true", help="Mark weak/invalid quality as invalid")

    parser.add_argument("--skip-tracking-if-exists", action="store_true", help="Reuse output tracking JSON if already present")
    parser.add_argument("--tracking-only", action="store_true", help="Only run tracking, do not run physics evaluation")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation from existing output tracking JSON")
    parser.add_argument("--save-diagnostics", action="store_true", help="Save normalized tracks and diagnostic summaries")
    parser.add_argument("--save-vis-video", action="store_true", help="Save SAM2 tracking visualization video")
    parser.add_argument("--save-mask-png", action="store_true", help="Save SAM2 mask PNGs")
    parser.add_argument("--save-auto-init-debug", action="store_true", help="Save image showing the selected auto-init prompt")
    parser.add_argument("--copy-video-to-workspace", action="store_true")
    parser.add_argument("--debug", action="store_true",
                        help="Shortcut: save vis video, masks, auto-init debug image, diagnostics, and area plots")

    parser.add_argument("--auto-init-scan-frames", type=int, default=60)
    parser.add_argument("--auto-init-scan-mode", default="uniform", choices=["start", "middle", "uniform"],
                        help="Which part of each video to scan for auto-init prompts.")
    parser.add_argument("--detector", default="motion", choices=["motion", "yolo", "yolo_then_motion"],
                        help="Initial target detector before SAM2 tracking. Default: motion")
    parser.add_argument("--yolo-weights", default=None,
                        help="YOLO weights, e.g. yolo11n.pt or custom .pt")
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yolo-iou", type=float, default=0.50)
    parser.add_argument("--yolo-classes", default=None,
                        help="Comma-separated YOLO class names or ids to allow")
    parser.add_argument("--auto-init-threshold", type=float, default=25.0)
    parser.add_argument("--auto-init-min-area", type=int, default=40)
    parser.add_argument("--auto-init-max-area-ratio", type=float, default=0.20)
    parser.add_argument("--auto-init-min-fill-ratio", type=float, default=0.08)
    parser.add_argument("--auto-init-max-aspect-ratio", type=float, default=6.0)
    parser.add_argument("--auto-init-object-refine", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--auto-init-box-expand", type=float, default=2.8)
    parser.add_argument("--auto-init-target-hint", default="metadata",
                        help="Target hint passed to auto tracking. Use 'metadata' to read Calibration_Object.")
    parser.add_argument("--disable-auto-init-shadow-filter", action="store_true")
    parser.add_argument("--no-auto-init-negative-points", action="store_true")
    parser.add_argument("--no-auto-init-box", action="store_true")
    parser.add_argument("--no-bidirectional-propagation", action="store_true",
                        help="Only propagate forward from the selected init frame.")

    parser.add_argument("--area-stability-check", action="store_true",
                        help="After tracking, check middle-window mask area stability before evaluation")
    parser.add_argument("--area-check", dest="area_stability_check", action="store_true",
                        help="Short alias for --area-stability-check")
    parser.add_argument("--area-stability-action", choices=["warn", "skip"], default="warn",
                        help="warn keeps evaluating flagged videos; skip writes a failed result and skips evaluation. Default: warn")
    parser.add_argument("--area-stability-output-dir", default=None,
                        help="Area stability report dir. Default: <output-dir>/area_stability")
    parser.add_argument("--area-stability-save-plots", action="store_true",
                        help="Save area curve plots for each checked video/object")
    parser.add_argument("--area-prefer-mask-area", action="store_true",
                        help="If masks exist in <workspace-root>/sam2_tracks/<scene>/masks, count pixels from mask PNGs")
    parser.add_argument("--area-object-id", type=int, default=None,
                        help="Only check this object id for multi-object tracking JSONs")
    parser.add_argument("--area-middle-start", type=float, default=0.15,
                        help="Fraction trimmed from start for area check. Default: 0.15")
    parser.add_argument("--area-middle-end", type=float, default=0.85,
                        help="Fraction kept until this point for area check. Default: 0.85")
    parser.add_argument("--area-min-valid-frames", type=int, default=6)
    parser.add_argument("--area-min-coverage", type=float, default=0.70)
    parser.add_argument("--area-ratio-threshold", type=float, default=2.00,
                        help="Flag if middle p95(area)/p5(area) exceeds this value")
    parser.add_argument("--area-cv-threshold", type=float, default=0.25,
                        help="Flag if middle std(area)/mean(area) exceeds this value")
    parser.add_argument("--area-max-log-jump-threshold", type=float, default=0.45,
                        help="Flag if max adjacent abs diff(log(area)) exceeds this value")
    parser.add_argument("--area-smooth-scale-ratio-threshold", type=float, default=1.60,
                        help="Classify as possible depth motion if smooth trend changes by this ratio")
    parser.add_argument("--area-trend-r2-threshold", type=float, default=0.65,
                        help="Minimum log-area trend R2 for smooth scale-change classification")
    parser.add_argument("--area-bbox-size-cv-threshold", type=float, default=0.20,
                        help="Flag if middle-window bbox width/height CV exceeds this value")
    parser.add_argument("--area-bbox-aspect-cv-threshold", type=float, default=0.20,
                        help="Flag if middle-window bbox aspect-ratio CV exceeds this value")
    parser.add_argument("--area-aligned-iou-threshold", type=float, default=0.65,
                        help="Flag if median centroid-aligned mask IoU is below this value")
    parser.add_argument("--area-aligned-iou-sample-stride", type=int, default=5,
                        help="Frame stride for centroid-aligned mask IoU sampling")
    parser.add_argument("--area-max-centroid-step-ratio-threshold", type=float, default=2.50,
                        help="Flag if max centroid step divided by median equivalent diameter exceeds this value")
    parser.add_argument("--min-motion-extent-ratio", type=float, default=0.75,
                        help="Flag tracking as static if primary motion extent is below this many object diameters")
    parser.add_argument("--min-object-coverage", type=float, default=0.50,
                        help="Object must be valid in at least this fraction of frames to count")

    parser.add_argument("--python-executable", default=sys.executable, help="Python executable used to run SAM2 tracking")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


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


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    workspace_root = Path(args.workspace_root).expanduser().resolve() if args.workspace_root else WORKSPACE_ROOT.resolve()
    sam2_repo = Path(args.sam2_repo).expanduser().resolve() if args.sam2_repo else workspace_root / "repo" / "sam2"

    args.workspace_root = str(workspace_root)
    if not args.metadata:
        auto_metadata = default_metadata_path(workspace_root)
        if auto_metadata is not None:
            args.metadata = str(auto_metadata)
    args.video_root = str(Path(args.video_root).expanduser().resolve()) if args.video_root else str(workspace_root / "data" / "t2v_videos")
    args.output_dir = str(Path(args.output_dir).expanduser().resolve()) if args.output_dir else str(workspace_root / "batch_eval_results")
    args.sam2_repo = str(sam2_repo)
    args.model_weights = (
        str(Path(args.model_weights).expanduser().resolve())
        if args.model_weights
        else str(default_model_weights(sam2_repo))
    )

    if args.debug:
        args.save_vis_video = True
        args.save_mask_png = True
        args.save_auto_init_debug = True
        args.save_diagnostics = True
        args.area_stability_check = True
        args.area_prefer_mask_area = True
        args.area_stability_save_plots = True

    return args


def resolve_video_path(row: dict[str, Any], video_root: Path, metadata_path: Path) -> Path:
    raw = row.get("Video_File")
    if not raw:
        raise FileNotFoundError("Video_File is empty")
    candidate = Path(str(raw))
    candidates = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.append(video_root / candidate)
        candidates.append(metadata_path.parent / candidate)
        candidates.append(Path.cwd() / candidate)
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError("Video_File not found. searched: " + ", ".join(str(p) for p in candidates))


def scene_name_for_row(row: dict[str, Any]) -> str:
    prompt_id = str(row.get("Prompt_ID") or "").strip()
    if prompt_id:
        return prompt_id
    return f"index_{int(row.get('Index')):04d}"


def slug_value(value: Any, fallback: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        raw = fallback
    out = []
    for ch in raw:
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out).strip("_")
    return slug or fallback


def metric_dir_name(row: dict[str, Any]) -> str:
    return slug_value(row.get("Metric") or row.get("Evaluator"), "unknown_metric")


def prompt_dir_name(row: dict[str, Any]) -> str:
    return slug_value(row.get("Prompt_ID"), scene_name_for_row(row))


def per_video_output_dir(output_dir: Path, row: dict[str, Any], layout: str) -> Path:
    if layout == "flat":
        return output_dir
    return output_dir / metric_dir_name(row) / prompt_dir_name(row)


def num_objects_for_row(row: dict[str, Any]) -> int:
    try:
        return max(1, int(row.get("Num_Objects") or 1))
    except (TypeError, ValueError):
        return 1


def output_tracking_json(output_dir: Path, row: dict[str, Any], layout: str = "organized") -> Path:
    if layout == "flat":
        return output_dir / "tracking_json" / f"{scene_name_for_row(row)}.json"
    return per_video_output_dir(output_dir, row, layout) / "tracking_points.json"


def run_tracking_for_row(
    row: dict[str, Any],
    metadata_path: Path,
    video_root: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[Path | None, str | None]:
    out_json = output_tracking_json(output_dir, row, args.output_layout)
    if args.skip_tracking_if_exists and out_json.exists():
        return out_json, None

    video_path = resolve_video_path(row, video_root, metadata_path)
    workspace_root = Path(args.workspace_root).expanduser().resolve() if args.workspace_root else Path.cwd().resolve()
    sam2_script = Path(args.sam2_script).expanduser().resolve() if args.sam2_script else workspace_root / "scripts" / "sam2" / "run_sam2_track.py"
    if not sam2_script.exists():
        raise FileNotFoundError(f"SAM2 tracking script not found: {sam2_script}")

    scene = scene_name_for_row(row)
    num_objects = num_objects_for_row(row)
    cmd = [
        str(Path(args.python_executable).expanduser()),
        str(sam2_script),
        "--workspace-root",
        str(workspace_root),
        "--scene-name",
        scene,
        "--video-path",
        str(video_path),
        "--sam2-repo",
        str(Path(args.sam2_repo).expanduser()),
        "--model-cfg",
        args.model_cfg,
        "--model-weights",
        str(Path(args.model_weights).expanduser()),
        "--auto-init",
        "--auto-init-num-objects",
        str(num_objects),
        "--auto-init-scan-frames",
        str(args.auto_init_scan_frames),
        "--auto-init-scan-mode",
        args.auto_init_scan_mode,
        "--auto-init-detector",
        args.detector,
        "--yolo-conf",
        str(args.yolo_conf),
        "--yolo-iou",
        str(args.yolo_iou),
        "--auto-init-threshold",
        str(args.auto_init_threshold),
        "--auto-init-min-area",
        str(args.auto_init_min_area),
        "--auto-init-max-area-ratio",
        str(args.auto_init_max_area_ratio),
        "--auto-init-min-fill-ratio",
        str(args.auto_init_min_fill_ratio),
        "--auto-init-max-aspect-ratio",
        str(args.auto_init_max_aspect_ratio),
        "--auto-init-box-expand",
        str(args.auto_init_box_expand),
        "--auto-init-target-hint",
        str(row.get("Calibration_Object") if args.auto_init_target_hint == "metadata" else args.auto_init_target_hint),
        "--overwrite-frames",
        "--overwrite-outputs",
    ]
    if not args.auto_init_object_refine:
        cmd.append("--no-auto-init-object-refine")
    if args.yolo_weights:
        cmd.extend(["--yolo-weights", str(Path(args.yolo_weights).expanduser())])
    if args.yolo_classes:
        cmd.extend(["--yolo-classes", str(args.yolo_classes)])
    if args.no_auto_init_box:
        cmd.append("--no-auto-init-box")
    if args.disable_auto_init_shadow_filter:
        cmd.append("--disable-auto-init-shadow-filter")
    if args.no_auto_init_negative_points:
        cmd.append("--no-auto-init-negative-points")
    if args.no_bidirectional_propagation:
        cmd.append("--no-bidirectional-propagation")
    if args.save_vis_video:
        cmd.append("--save-vis-video")
    if args.save_mask_png:
        cmd.append("--save-mask-png")
    if args.save_auto_init_debug:
        cmd.append("--save-auto-init-debug")
    if args.copy_video_to_workspace:
        cmd.append("--copy-video-to-workspace")

    LOGGER.info("tracking Index=%s Prompt_ID=%s video=%s", row.get("Index"), row.get("Prompt_ID"), video_path)
    proc = subprocess.run(cmd, cwd=str(workspace_root), text=True, capture_output=True)
    if proc.returncode != 0:
        return None, proc.stderr.strip() or proc.stdout.strip() or f"SAM2 exited with code {proc.returncode}"

    source_json = workspace_root / "sam2_tracks" / scene / "tracking_points.json"
    if not source_json.exists():
        return None, f"SAM2 completed but tracking JSON was not found: {source_json}"

    out_json.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_json, out_json)
    write_tracking_quality(
        path=out_json,
        expected_objects=num_objects,
        min_motion_extent_ratio=args.min_motion_extent_ratio,
        min_object_coverage=args.min_object_coverage,
    )

    run_config = workspace_root / "sam2_tracks" / scene / "run_config.json"
    if run_config.exists():
        shutil.copy2(run_config, out_json.with_name("run_config.json"))
    return out_json.resolve(), None


def write_resolved_metadata(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    path = output_dir / "metadata_with_resolved_tracking.xlsx"
    pd.DataFrame(rows).to_excel(path, index=False)
    return path


def area_check_namespace(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        middle_start=args.area_middle_start,
        middle_end=args.area_middle_end,
        min_valid_frames=args.area_min_valid_frames,
        min_coverage=args.area_min_coverage,
        area_ratio_threshold=args.area_ratio_threshold,
        area_cv_threshold=args.area_cv_threshold,
        max_log_jump_threshold=args.area_max_log_jump_threshold,
        smooth_scale_ratio_threshold=args.area_smooth_scale_ratio_threshold,
        trend_r2_threshold=args.area_trend_r2_threshold,
        bbox_size_cv_threshold=args.area_bbox_size_cv_threshold,
        bbox_aspect_cv_threshold=args.area_bbox_aspect_cv_threshold,
        aligned_iou_threshold=args.area_aligned_iou_threshold,
        aligned_iou_sample_stride=args.area_aligned_iou_sample_stride,
        max_centroid_step_ratio_threshold=args.area_max_centroid_step_ratio_threshold,
    )


def check_area_stability_for_row(
    row: dict[str, Any],
    tracking_json: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[bool, list[dict[str, Any]], dict[str, Any]]:
    area_output_dir = (
        Path(args.area_stability_output_dir).expanduser().resolve()
        if args.area_stability_output_dir
        else output_dir / "area_stability"
    )
    area_output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_layout == "organized":
        per_scene_dir = per_video_output_dir(output_dir, row, args.output_layout) / "mask_qc"
    else:
        per_scene_dir = area_output_dir / scene_name_for_row(row)
    per_scene_dir.mkdir(parents=True, exist_ok=True)

    check_args = area_check_namespace(args)
    workspace_root = Path(args.workspace_root).expanduser().resolve() if args.workspace_root else Path.cwd().resolve()
    mask_dir = workspace_root / "sam2_tracks" / scene_name_for_row(row) / "masks"
    if not mask_dir.exists():
        mask_dir = None
    series_list = load_series_from_json(
        json_path=tracking_json,
        mask_dir=mask_dir,
        mask_pattern="{frame_idx:05d}.png",
        object_id=args.area_object_id,
        prefer_mask_area=bool(args.area_prefer_mask_area),
    )

    summaries: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    for series in series_list:
        summary, frame_rows = analyze_series(series, check_args)
        summaries.append(summary)
        detail_rows.extend(frame_rows)

        obj_suffix = "obj_unknown" if series.object_id is None else f"obj_{series.object_id:06d}"
        base_name = f"{series.scene}_{obj_suffix}"
        detail_path = None
        if args.output_layout == "flat":
            detail_path = per_scene_dir / f"{base_name}_area_series.csv"
            write_csv(detail_path, frame_rows, overwrite=True)

        plot_path = None
        if args.area_stability_save_plots:
            plot_dir = per_video_output_dir(output_dir, row, args.output_layout) / "plots" if args.output_layout == "organized" else per_scene_dir / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            plot_path = plot_dir / f"{obj_suffix}_mask_area.png" if args.output_layout == "organized" else plot_dir / f"{base_name}_area.png"
            save_plot(series, summary, frame_rows, plot_path)
            summary["plot"] = str(plot_path)

        outputs.append({
            "scene": series.scene,
            "object_id": series.object_id,
            "detail_csv": str(detail_path) if detail_path is not None else None,
            "plot": str(plot_path) if plot_path is not None else None,
        })

    report_csv = None if args.output_layout == "organized" else per_scene_dir / "mask_area_stability_report.csv"
    detail_csv = None if args.output_layout == "organized" else per_scene_dir / "mask_area_stability_frame_details.csv"
    report_json = per_scene_dir / ("mask_qc.json" if args.output_layout == "organized" else "mask_area_stability_report.json")
    if summaries and report_csv is not None:
        write_csv(report_csv, summaries, overwrite=True)
    if detail_rows and detail_csv is not None:
        write_csv(detail_csv, detail_rows, overwrite=True)
    report = {
        "tracking_json": str(tracking_json),
        "summaries": summaries,
        "series_outputs": outputs,
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        tracking_data = json.loads(tracking_json.read_text(encoding="utf-8"))
        quality_checks = tracking_data.get("quality_checks")
        if not isinstance(quality_checks, dict):
            quality_checks = {}
        quality_checks["mask_qc"] = {
            "status": "pass" if all(item.get("decision") == "keep" for item in summaries) else "flagged",
            "objects": summaries,
            "report_json": str(report_json),
        }
        tracking_data["quality_checks"] = quality_checks
        tracking_json.write_text(json.dumps(tracking_data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        LOGGER.warning("failed to embed mask QC into tracking JSON %s: %s", tracking_json, exc)

    passed = all(item.get("decision") == "keep" for item in summaries)
    return passed, summaries, {
        "area_report_json": str(report_json),
        "area_report_csv": str(report_csv) if report_csv is not None else None,
        "area_detail_csv": str(detail_csv) if detail_csv is not None else None,
    }


def json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def parse_extra_json(result: dict[str, Any]) -> dict[str, Any]:
    raw = result.get("Extra_JSON")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"raw_extra_json": raw}
    return parsed if isinstance(parsed, dict) else {"extra": parsed}


def save_trajectory_plot(row_dir: Path, tracking_json: Path, fps: float | None) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tracking_df = normalize_tracking_json(tracking_json, fps_override=fps)
        plot_dir = row_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_dir / "trajectory.png"

        fig, ax = plt.subplots(figsize=(6, 4))
        for obj_id, grp in tracking_df.groupby("object_id"):
            ax.plot(grp["cx_px"], grp["cy_px"], marker="o", markersize=2, linewidth=1, label=f"obj {obj_id}")
        ax.invert_yaxis()
        ax.set_xlabel("cx_px")
        ax.set_ylabel("cy_px")
        ax.set_title("tracked trajectory")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        return plot_path
    except Exception as exc:
        LOGGER.warning("compact trajectory plot failed for %s: %s", tracking_json, exc)
        return None


def write_per_video_result(
    output_dir: Path,
    row: dict[str, Any],
    result: dict[str, Any],
    tracking_json: Path | None,
    area_summaries: list[dict[str, Any]],
    area_paths: dict[str, Any],
    args: argparse.Namespace,
) -> Path | None:
    if args.output_layout != "organized":
        return None

    row_dir = per_video_output_dir(output_dir, row, args.output_layout)
    row_dir.mkdir(parents=True, exist_ok=True)
    trajectory_plot = save_trajectory_plot(row_dir, tracking_json, args.fps) if tracking_json is not None and tracking_json.exists() else None
    extra = parse_extra_json(result)
    tracking_quality = None
    if tracking_json is not None and tracking_json.exists():
        try:
            tracking_data = json.loads(tracking_json.read_text(encoding="utf-8"))
            tracking_quality = (tracking_data.get("quality_checks") or {}).get("tracking_quality")
        except Exception:
            tracking_quality = None

    result_compact = {k: v for k, v in result.items() if k != "Extra_JSON"}
    artifact_paths = {
        "tracking_json": str(tracking_json) if tracking_json is not None else None,
        "mask_qc_json": area_paths.get("area_report_json"),
        "trajectory_plot": str(trajectory_plot) if trajectory_plot is not None else None,
        "mask_qc_plots": [
            item.get("plot")
            for item in area_summaries
            if item.get("plot") is not None
        ],
    }

    payload = {
        "schema_version": 1,
        "index": row.get("Index"),
        "prompt_id": row.get("Prompt_ID"),
        "metric": row.get("Metric"),
        "video_file": row.get("Video_File"),
        "metadata": row,
        "physics_result": result_compact,
        "physics_extra": extra,
        "mask_qc": {
            "status": "pass" if all(item.get("decision") == "keep" for item in area_summaries) else "flagged",
            "objects": area_summaries,
        },
        "tracking_quality": tracking_quality,
        "artifacts": artifact_paths,
    }
    path = row_dir / "result.json"
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> None:
    args = normalize_args(parse_args())
    logging.basicConfig(level=getattr(logging, args.log_level), format="[%(levelname)s] %(message)s")

    if not args.metadata:
        metadata_dir = Path(args.workspace_root) / "data" / "metadata"
        raise SystemExit(
            "Metadata was not provided and could not be inferred. "
            f"Put exactly one .xlsx file in {metadata_dir}, or pass --metadata path/to/file.xlsx."
        )

    metadata_path = Path(args.metadata).expanduser().resolve()
    video_root = Path(args.video_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("workspace_root=%s", args.workspace_root)
    LOGGER.info("video_root=%s", video_root)
    LOGGER.info("output_dir=%s", output_dir)
    LOGGER.info("sam2_repo=%s", args.sam2_repo)
    LOGGER.info("model_weights=%s", args.model_weights)

    df = filter_metadata(read_metadata(metadata_path), args.index, args.prompt_id)
    LOGGER.info("processing %d row(s)", len(df))

    eval_results: list[dict[str, Any]] = []
    resolved_rows: list[dict[str, Any]] = []
    area_summary_rows: list[dict[str, Any]] = []

    for _, pd_row in df.iterrows():
        row = row_to_dict(pd_row)
        try:
            out_json = output_tracking_json(output_dir, row, args.output_layout)
            track_error = None
            if not args.eval_only:
                out_json, track_error = run_tracking_for_row(row, metadata_path, video_root, output_dir, args)

            if track_error:
                LOGGER.error("tracking failed Index=%s Prompt_ID=%s: %s", row.get("Index"), row.get("Prompt_ID"), track_error)
                row["Tracking_JSON"] = str(output_tracking_json(output_dir, row, args.output_layout))
                resolved_rows.append(dict(row))
                if not args.tracking_only:
                    from physics_eval.utils.io import failure_result

                    result = failure_result(row, "tracking_failed", {"tracking_error": track_error})
                    eval_results.append(result)
                    result_path = write_per_video_result(
                        output_dir=output_dir,
                        row=row,
                        result=result,
                        tracking_json=None,
                        area_summaries=[],
                        area_paths={},
                        args=args,
                    )
                    if result_path is not None:
                        LOGGER.info("wrote compact result: %s", result_path)
                continue

            row["Tracking_JSON"] = str(out_json)
            tracking_quality = write_tracking_quality(
                path=out_json,
                expected_objects=num_objects_for_row(row),
                min_motion_extent_ratio=args.min_motion_extent_ratio,
                min_object_coverage=args.min_object_coverage,
            )
            if tracking_quality.get("status") != "pass":
                LOGGER.warning(
                    "tracking quality flagged Index=%s Prompt_ID=%s: %s",
                    row.get("Index"),
                    row.get("Prompt_ID"),
                    tracking_quality,
                )
            resolved_rows.append(dict(row))
            if args.tracking_only:
                continue

            area_summaries: list[dict[str, Any]] = []
            area_paths: dict[str, Any] = {}
            if args.area_stability_check:
                area_ok, area_summaries, area_paths = check_area_stability_for_row(row, Path(out_json), output_dir, args)
                for item in area_summaries:
                    area_summary_rows.append({
                        **item,
                        "Index": row.get("Index"),
                        "Prompt_ID": row.get("Prompt_ID"),
                        "Video_File": row.get("Video_File"),
                        **area_paths,
                    })
                flagged = [item for item in area_summaries if item.get("decision") != "keep"]
                if flagged:
                    compact = [
                        {
                            "scene": item.get("scene"),
                            "object_id": item.get("object_id"),
                            "decision": item.get("decision"),
                            "reasons": item.get("reasons"),
                            "area_ratio_p95_p05": item.get("area_ratio_p95_p05"),
                            "area_cv": item.get("area_cv"),
                            "bbox_width_cv": item.get("bbox_width_cv"),
                            "bbox_height_cv": item.get("bbox_height_cv"),
                            "bbox_aspect_cv": item.get("bbox_aspect_cv"),
                            "translation_aligned_iou_median": item.get("translation_aligned_iou_median"),
                            "max_centroid_step_norm_by_diameter": item.get("max_centroid_step_norm_by_diameter"),
                            "max_log_jump": item.get("max_log_jump"),
                            "trend_r2": item.get("trend_r2"),
                        }
                        for item in flagged
                    ]
                    LOGGER.warning(
                        "area stability flagged Index=%s Prompt_ID=%s: %s",
                        row.get("Index"),
                        row.get("Prompt_ID"),
                        compact,
                    )
                    if args.area_stability_action == "skip":
                        result = failure_result(
                            row,
                            "mask_area_unstable",
                            {"area_stability": compact, **area_paths},
                        )
                        eval_results.append(result)
                        result_path = write_per_video_result(
                            output_dir=output_dir,
                            row=row,
                            result=result,
                            tracking_json=Path(out_json),
                            area_summaries=area_summaries,
                            area_paths=area_paths,
                            args=args,
                        )
                        if result_path is not None:
                            LOGGER.info("wrote compact result: %s", result_path)
                        continue
                elif area_ok:
                    LOGGER.info("area stability passed Index=%s Prompt_ID=%s", row.get("Index"), row.get("Prompt_ID"))

            result = process_row(
                row=row,
                metadata_path=metadata_path,
                json_root=None,
                output_dir=per_video_output_dir(output_dir, row, args.output_layout) if args.output_layout == "organized" else output_dir,
                fps=args.fps,
                save_diag=args.save_diagnostics,
                strict=args.strict,
            )
            eval_results.append(result)
            result_path = write_per_video_result(
                output_dir=output_dir,
                row=row,
                result=result,
                tracking_json=Path(out_json),
                area_summaries=area_summaries,
                area_paths=area_paths,
                args=args,
            )
            if result_path is not None:
                LOGGER.info("wrote compact result: %s", result_path)
        except Exception as exc:
            LOGGER.error("row failed Index=%s Prompt_ID=%s: %s", row.get("Index"), row.get("Prompt_ID"), exc)
            resolved_rows.append(dict(row))
            if not args.tracking_only:
                from physics_eval.utils.io import failure_result

                result = failure_result(row, "failed", {"exception": str(exc)})
                eval_results.append(result)
                result_path = write_per_video_result(
                    output_dir=output_dir,
                    row=row,
                    result=result,
                    tracking_json=None,
                    area_summaries=[],
                    area_paths={},
                    args=args,
                )
                if result_path is not None:
                    LOGGER.info("wrote compact result: %s", result_path)

    resolved_metadata = write_resolved_metadata(resolved_rows, output_dir)
    LOGGER.info("wrote resolved metadata: %s", resolved_metadata)

    manifest = {
        "metadata": str(metadata_path),
        "video_root": str(video_root),
        "output_dir": str(output_dir),
        "n_rows": len(resolved_rows),
        "tracking_only": bool(args.tracking_only),
        "eval_only": bool(args.eval_only),
    }
    (output_dir / "batch_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.area_stability_check and area_summary_rows:
        area_output_dir = (
            Path(args.area_stability_output_dir).expanduser().resolve()
            if args.area_stability_output_dir
            else output_dir / "area_stability"
        )
        area_output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(area_summary_rows).to_csv(
            area_output_dir / "mask_area_stability_report.csv",
            index=False,
            encoding="utf-8-sig",
        )
        (area_output_dir / "mask_area_stability_report.json").write_text(
            json.dumps(area_summary_rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        LOGGER.info("wrote area stability report: %s", area_output_dir / "mask_area_stability_report.csv")

    if not args.tracking_only:
        write_results(eval_results, output_dir)
        LOGGER.info("wrote evaluation results: %s", output_dir / "results.csv")


if __name__ == "__main__":
    main()
