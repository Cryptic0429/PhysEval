"""Inspect saved track/mask evidence and persist per-video quality reports."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .mask_evidence import analyze_series, load_series_from_json
from ..io.results import per_video_output_dir, scene_name_for_row
from ..reporting import save_plot, write_csv

LOGGER = logging.getLogger("physics_eval.video_batch")

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
            "status": (
                "not_evaluated"
                if not summaries
                else "pass" if all(item.get("decision") == "keep" for item in summaries) else "flagged"
            ),
            "objects": summaries,
            "report_json": str(report_json),
        }
        tracking_data["quality_checks"] = quality_checks
        tracking_json.write_text(json.dumps(tracking_data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        LOGGER.warning("failed to embed mask QC into tracking JSON %s: %s", tracking_json, exc)

    passed = bool(summaries) and all(item.get("decision") == "keep" for item in summaries)
    return passed, summaries, {
        "area_report_json": str(report_json),
        "area_report_csv": str(report_csv) if report_csv is not None else None,
        "area_detail_csv": str(detail_csv) if detail_csv is not None else None,
    }
