#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from physics_eval.quality.diagnostics import (
    AreaSeries,
    IMAGE_EXTS,
    ensure_dir,
    read_mask_area,
    read_mask_bool,
    mask_geometry,
    normalize_bbox,
    bbox_stats,
    bbox_area,
    centroid_from_record,
    parse_frame_idx,
    resolve_mask_path,
    area_from_record,
    load_series_from_masks,
    infer_scene_name,
    load_series_from_json,
    linear_trend_r2,
    safe_ratio,
    finite_positive,
    finite_values,
    coefficient_of_variation,
    shift_mask,
    mask_iou,
    aligned_iou_stats,
    centroid_step_stats,
    analyze_series
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether SAM2 mask area is stable in the middle part of a video. "
            "Large middle-window area changes are flagged for tracking review or rejection."
        )
    )
    parser.add_argument(
        "--workspace-root",
        type=str,
        default=".",
        help="Workspace root. Default: current directory.",
    )
    parser.add_argument(
        "--tracks-root",
        type=str,
        default=None,
        help="Root containing scene folders, default: <workspace-root>/sam2_tracks.",
    )
    parser.add_argument(
        "--scene-name",
        type=str,
        nargs="*",
        default=None,
        help="Scene names under tracks-root. If omitted in batch mode, all scenes with tracking_points.json or masks are checked.",
    )
    parser.add_argument(
        "--sam2-json",
        type=str,
        default=None,
        help="Single SAM2 tracking_points.json to check.",
    )
    parser.add_argument(
        "--mask-dir",
        type=str,
        default=None,
        help="Single mask directory to check. Used directly, or as an override for --sam2-json.",
    )
    parser.add_argument(
        "--mask-pattern",
        type=str,
        default="{frame_idx:05d}.png",
        help="Mask file pattern when mask-dir is given. Default: {frame_idx:05d}.png",
    )
    parser.add_argument(
        "--object-id",
        type=int,
        default=None,
        help="Only check this object id for multi-object tracking jsons.",
    )
    parser.add_argument(
        "--prefer-mask-area",
        action="store_true",
        help="Always count pixels from mask pngs when available, instead of trusting json area.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: <workspace-root>/runs/quality/mask_area_stability",
    )
    parser.add_argument(
        "--middle-start",
        type=float,
        default=0.15,
        help="Fraction of the series to trim from the beginning. Default: 0.15",
    )
    parser.add_argument(
        "--middle-end",
        type=float,
        default=0.85,
        help="Fraction of the series kept until this point. Default: 0.85",
    )
    parser.add_argument(
        "--min-valid-frames",
        type=int,
        default=6,
        help="Minimum valid middle-window frames required. Default: 6",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.70,
        help="Minimum valid-area coverage inside the middle window. Default: 0.70",
    )
    parser.add_argument(
        "--area-ratio-threshold",
        type=float,
        default=2.00,
        help="Flag if middle p95(area)/p5(area) exceeds this value. Default: 2.00",
    )
    parser.add_argument(
        "--area-cv-threshold",
        type=float,
        default=0.25,
        help="Flag if middle std(area)/mean(area) exceeds this value. Default: 0.25",
    )
    parser.add_argument(
        "--max-log-jump-threshold",
        type=float,
        default=0.45,
        help="Flag if max abs diff(log(area)) between adjacent middle frames exceeds this value. Default: 0.45",
    )
    parser.add_argument(
        "--smooth-scale-ratio-threshold",
        type=float,
        default=1.60,
        help="Classify as possible depth motion if smooth trend changes by this ratio. Default: 1.60",
    )
    parser.add_argument(
        "--trend-r2-threshold",
        type=float,
        default=0.65,
        help="Minimum linear R2 in log-area vs time to call the change smooth/monotonic. Default: 0.65",
    )
    parser.add_argument(
        "--bbox-size-cv-threshold",
        type=float,
        default=0.20,
        help="Flag if middle-window bbox width or height CV exceeds this value. Default: 0.20",
    )
    parser.add_argument(
        "--bbox-aspect-cv-threshold",
        type=float,
        default=0.20,
        help="Flag if middle-window bbox aspect-ratio CV exceeds this value. Default: 0.20",
    )
    parser.add_argument(
        "--aligned-iou-threshold",
        type=float,
        default=0.65,
        help=(
            "Flag if median centroid-aligned mask IoU to the reference mask is below this value. "
            "Only used when mask PNGs are available. Default: 0.65"
        ),
    )
    parser.add_argument(
        "--aligned-iou-sample-stride",
        type=int,
        default=5,
        help="Frame stride for centroid-aligned IoU sampling in the middle window. Default: 5",
    )
    parser.add_argument(
        "--max-centroid-step-ratio-threshold",
        type=float,
        default=2.50,
        help=(
            "Flag if max adjacent centroid step divided by median equivalent diameter exceeds this value. "
            "Default: 2.50"
        ),
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save area curve pngs for visual inspection.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing report files.",
    )
    return parser.parse_args()


def discover_inputs(args: argparse.Namespace) -> list[tuple[Path | None, Path | None, str]]:
    workspace_root = Path(args.workspace_root).expanduser().resolve()
    tracks_root = Path(args.tracks_root).expanduser().resolve() if args.tracks_root else workspace_root / "sam2_tracks"
    inputs: list[tuple[Path | None, Path | None, str]] = []

    if args.sam2_json:
        json_path = Path(args.sam2_json).expanduser().resolve()
        mask_dir = Path(args.mask_dir).expanduser().resolve() if args.mask_dir else None
        inputs.append((json_path, mask_dir, infer_scene_name(json_path)))
        return inputs

    if args.mask_dir:
        mask_dir = Path(args.mask_dir).expanduser().resolve()
        inputs.append((None, mask_dir, infer_scene_name(mask_dir)))
        return inputs

    scene_names = args.scene_name
    scene_dirs = []
    if scene_names:
        scene_dirs = [tracks_root / name for name in scene_names]
    elif tracks_root.exists():
        scene_dirs = sorted(p for p in tracks_root.iterdir() if p.is_dir())

    for scene_dir in scene_dirs:
        json_path = scene_dir / "tracking_points.json"
        masks_dir = scene_dir / "masks"
        if json_path.exists():
            inputs.append((json_path, None, scene_dir.name))
        elif masks_dir.exists():
            inputs.append((None, masks_dir, scene_dir.name))
    return inputs


def write_csv(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {path}. Use --overwrite to replace.")
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_plot(series: AreaSeries, summary: dict[str, Any], frame_rows: list[dict[str, Any]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    frames = [r["frame_idx"] for r in frame_rows]
    areas = [np.nan if r["area"] is None else float(r["area"]) for r in frame_rows]
    middle_frames = [r["frame_idx"] for r in frame_rows if r["in_middle_window"]]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(frames, areas, color="#2563eb", linewidth=1.8, label="mask area")
    if middle_frames:
        ax.axvspan(middle_frames[0], middle_frames[-1], color="#f59e0b", alpha=0.15, label="middle window")
    if math.isfinite(float(summary.get("median_area", float("nan")))):
        ax.axhline(float(summary["median_area"]), color="#111827", linestyle="--", linewidth=1.0, label="middle median")
    title_obj = "" if series.object_id is None else f" obj={series.object_id}"
    ax.set_title(f"{series.scene}{title_obj} area stability: {summary['status']} / {summary['decision']}")
    ax.set_xlabel("frame")
    ax.set_ylabel("mask area (px)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    workspace_root = Path(args.workspace_root).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else workspace_root / "runs" / "quality" / "mask_area_stability"
    )
    ensure_dir(output_dir)

    inputs = discover_inputs(args)
    if not inputs:
        raise RuntimeError("No inputs found. Provide --sam2-json, --mask-dir, or valid --tracks-root/--scene-name.")

    summaries: list[dict[str, Any]] = []
    detail_rows_all: list[dict[str, Any]] = []
    series_outputs: list[dict[str, Any]] = []

    for json_path, mask_dir, scene in inputs:
        if json_path is not None:
            series_list = load_series_from_json(
                json_path=json_path,
                mask_dir=mask_dir,
                mask_pattern=args.mask_pattern,
                object_id=args.object_id,
                prefer_mask_area=args.prefer_mask_area,
            )
        elif mask_dir is not None:
            series_list = [load_series_from_masks(mask_dir=mask_dir, scene=scene, mask_pattern=args.mask_pattern)]
        else:
            continue

        for series in series_list:
            summary, frame_rows = analyze_series(series, args)
            summaries.append(summary)
            detail_rows_all.extend(frame_rows)
            obj_suffix = "obj_unknown" if series.object_id is None else f"obj_{series.object_id:06d}"
            base_name = f"{series.scene}_{obj_suffix}"
            detail_path = output_dir / f"{base_name}_area_series.csv"
            write_csv(detail_path, frame_rows, overwrite=args.overwrite)
            plot_path = None
            if args.save_plots:
                plot_dir = output_dir / "plots"
                ensure_dir(plot_dir)
                plot_path = plot_dir / f"{base_name}_area.png"
                if plot_path.exists() and not args.overwrite:
                    raise FileExistsError(f"Output exists: {plot_path}. Use --overwrite to replace.")
                save_plot(series, summary, frame_rows, plot_path)

            series_outputs.append(
                {
                    "scene": series.scene,
                    "object_id": series.object_id,
                    "detail_csv": str(detail_path),
                    "plot": str(plot_path) if plot_path is not None else None,
                }
            )

    report_csv = output_dir / "mask_area_stability_report.csv"
    report_json = output_dir / "mask_area_stability_report.json"
    detail_csv = output_dir / "mask_area_stability_frame_details.csv"

    write_csv(report_csv, summaries, overwrite=args.overwrite)
    write_csv(detail_csv, detail_rows_all, overwrite=args.overwrite)

    if report_json.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {report_json}. Use --overwrite to replace.")
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "middle_start": args.middle_start,
                    "middle_end": args.middle_end,
                    "min_valid_frames": args.min_valid_frames,
                    "min_coverage": args.min_coverage,
                    "area_ratio_threshold": args.area_ratio_threshold,
                    "area_cv_threshold": args.area_cv_threshold,
                    "max_log_jump_threshold": args.max_log_jump_threshold,
                    "smooth_scale_ratio_threshold": args.smooth_scale_ratio_threshold,
                    "trend_r2_threshold": args.trend_r2_threshold,
                    "bbox_size_cv_threshold": args.bbox_size_cv_threshold,
                    "bbox_aspect_cv_threshold": args.bbox_aspect_cv_threshold,
                    "aligned_iou_threshold": args.aligned_iou_threshold,
                    "aligned_iou_sample_stride": args.aligned_iou_sample_stride,
                    "max_centroid_step_ratio_threshold": args.max_centroid_step_ratio_threshold,
                },
                "summaries": summaries,
                "series_outputs": series_outputs,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    counts: dict[str, int] = {}
    for item in summaries:
        decision = str(item["decision"])
        counts[decision] = counts.get(decision, 0) + 1

    print("=" * 72)
    print("Mask area stability check completed")
    print("=" * 72)
    print(f"Checked series: {len(summaries)}")
    for decision, count in sorted(counts.items()):
        print(f"{decision}: {count}")
    print(f"Report CSV: {report_csv}")
    print(f"Report JSON: {report_json}")
    print(f"Frame details CSV: {detail_csv}")

    warn_items = [s for s in summaries if s["decision"] != "keep"]
    if warn_items:
        print("\nFlagged:")
        for item in warn_items:
            obj = "" if item["object_id"] is None else f" obj={item['object_id']}"
            print(
                f"- {item['scene']}{obj}: {item['decision']} "
                f"(ratio={item['area_ratio_p95_p05']:.3f}, cv={item['area_cv']:.3f}, "
                f"bbox_w_cv={item['bbox_width_cv']:.3f}, bbox_h_cv={item['bbox_height_cv']:.3f}, "
                f"aligned_iou={item['translation_aligned_iou_median']:.3f}, "
                f"centroid_jump={item['max_centroid_step_norm_by_diameter']:.3f}, "
                f"reasons={item['reasons']})"
            )


if __name__ == "__main__":
    main()
