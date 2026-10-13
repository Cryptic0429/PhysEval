from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

from physics_eval.config import WORKSPACE_ROOT, TRACKING_WORKER_MODULE
import os

import pandas as pd

from physics_eval.estimation import filter_metadata, process_row
from physics_eval.io.metadata import read_metadata, row_to_dict
from physics_eval.io.results import (
    failure_result, write_results, scene_name_for_row, slug_value, metric_dir_name,
    prompt_dir_name, per_video_output_dir, output_tracking_json, write_resolved_metadata,
    json_safe, parse_extra_json, write_per_video_result, _embed_initialization_provenance,
)
from physics_eval.io.video import resolve_video_path
from physics_eval.io.tracks import normalize_tracking_json
# Reexport row-policy helpers for existing internal callers and test imports.
from physics_eval.tracking.policy import (
    num_objects_for_row,
    tracking_overrides_for_row,
    effective_tracking_settings,
    resolve_tracking_initialization,
    _TRACKING_OVERRIDE_ALIASES,
    _TRACKING_OVERRIDE_KEYS,
    _bool_setting,
    _finite_setting,
)
from physics_eval.quality.tracking_checks import write_tracking_quality
from physics_eval.quality.inspection import area_check_namespace, check_area_stability_for_row
from physics_eval.reporting import save_trajectory_plot


LOGGER = logging.getLogger("physics_eval.video_batch")


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
    worker_script = getattr(args, "sam2_script", None)
    if worker_script:
        worker_script = Path(worker_script).expanduser().resolve()
        if not worker_script.is_file():
            raise FileNotFoundError(f"Tracking worker not found: {worker_script}")
        worker_command = [str(Path(args.python_executable).expanduser()), str(worker_script)]
    else:
        worker_command = [str(Path(args.python_executable).expanduser()), "-m", TRACKING_WORKER_MODULE]

    scene = scene_name_for_row(row)
    settings, prompts = resolve_tracking_initialization(row, args)
    provenance = settings["initialization"]
    num_objects = int(settings["num_objects"])
    row["_Resolved_Tracking_Num_Objects"] = num_objects
    row["Resolved_Tracking_Initialization_JSON"] = json.dumps(
        provenance, ensure_ascii=False, sort_keys=True
    )

    tracker_args = [
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
        "--precision",
        args.sam2_precision,
        "--auto-init-scan-frames",
        str(settings["auto_init_scan_frames"]),
        "--auto-init-scan-mode",
        str(settings["auto_init_scan_mode"]),
        "--yolo-conf",
        str(settings["yolo_conf"]),
        "--yolo-iou",
        str(settings["yolo_iou"]),
        "--auto-init-threshold",
        str(settings["auto_init_threshold"]),
        "--auto-init-min-area",
        str(settings["auto_init_min_area"]),
        "--auto-init-max-area-ratio",
        str(settings["auto_init_max_area_ratio"]),
        "--auto-init-min-fill-ratio",
        str(settings["auto_init_min_fill_ratio"]),
        "--auto-init-max-aspect-ratio",
        str(settings["auto_init_max_aspect_ratio"]),
        "--auto-init-box-expand",
        str(settings["auto_init_box_expand"]),
        "--auto-init-target-hint",
        str(
            row.get("Calibration_Object")
            if settings["auto_init_target_hint"] == "metadata"
            else settings["auto_init_target_hint"]
        ),
        "--overwrite-frames",
        "--overwrite-outputs",
    ]
    if not settings["auto_init_object_refine"]:
        tracker_args.append("--no-auto-init-object-refine")
    if args.yolo_weights:
        tracker_args.extend(["--yolo-weights", str(Path(args.yolo_weights).expanduser())])
    if settings["yolo_classes"]:
        tracker_args.extend(["--yolo-classes", str(settings["yolo_classes"])])
    if settings["no_auto_init_box"]:
        tracker_args.append("--no-auto-init-box")
    if settings["disable_auto_init_shadow_filter"]:
        tracker_args.append("--disable-auto-init-shadow-filter")
    if settings["no_auto_init_negative_points"]:
        tracker_args.append("--no-auto-init-negative-points")
    if settings["no_bidirectional_propagation"]:
        tracker_args.append("--no-bidirectional-propagation")
    if settings["offload_video_to_cpu"]:
        tracker_args.append("--offload-video-to-cpu")
    if args.save_vis_video:
        tracker_args.append("--save-vis-video")
    if args.save_mask_png:
        tracker_args.append("--save-mask-png")
    if args.save_auto_init_debug:
        tracker_args.append("--save-auto-init-debug")
    if args.copy_video_to_workspace:
        tracker_args.append("--copy-video-to-workspace")

    source_dir = workspace_root / "sam2_tracks" / scene
    prompt_manifest: Path | None = None
    if prompts:
        source_dir.mkdir(parents=True, exist_ok=True)
        prompt_manifest = source_dir / "resolved_init_prompts.json"
        prompt_manifest.write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        cmd = [*worker_command, *tracker_args, "--prompt-manifest", str(prompt_manifest)]
    else:
        cmd = [
            *worker_command,
            *tracker_args,
            "--auto-init",
            "--auto-init-num-objects",
            str(num_objects),
        ]

    LOGGER.info("tracking Index=%s Prompt_ID=%s video=%s", row.get("Index"), row.get("Prompt_ID"), video_path)
    worker_env = dict(os.environ)
    worker_env["PYTHONPATH"] = str(WORKSPACE_ROOT) + os.pathsep + worker_env.get("PYTHONPATH", "")
    proc = subprocess.run(cmd, cwd=str(workspace_root), env=worker_env, text=True, capture_output=True)
    if proc.returncode != 0:
        return None, proc.stderr.strip() or proc.stdout.strip() or f"SAM2 exited with code {proc.returncode}"

    source_json = source_dir / "tracking_points.json"
    if not source_json.exists():
        return None, f"SAM2 completed but tracking JSON was not found: {source_json}"

    out_json.parent.mkdir(parents=True, exist_ok=True)
    _embed_initialization_provenance(source_json, provenance)
    shutil.copy2(source_json, out_json)
    write_tracking_quality(
        path=out_json,
        expected_objects=num_objects,
        min_motion_extent_ratio=args.min_motion_extent_ratio,
        min_object_coverage=args.min_object_coverage,
    )

    run_config = source_dir / "run_config.json"
    if run_config.exists():
        _embed_initialization_provenance(run_config, provenance)
        shutil.copy2(run_config, out_json.with_name("run_config.json"))
    # Keep the high-value review artifacts beside the organized result.  Masks
    # remain in sam2_tracks/<scene>/masks to avoid duplicating hundreds of PNGs.
    for source_name, output_name in (
        ("tracking_vis.mp4", "tracking_vis.mp4"),
        ("auto_init_debug.jpg", "auto_init_debug.jpg"),
    ):
        source_path = source_dir / source_name
        if source_path.exists():
            shutil.copy2(source_path, out_json.with_name(output_name))
    if prompt_manifest is not None and prompt_manifest.exists():
        shutil.copy2(prompt_manifest, out_json.with_name("resolved_init_prompts.json"))
    return out_json.resolve(), None


def run(args: argparse.Namespace) -> None:
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
        # Runtime-only context for evaluators that resolve repository-relative
        # oracle assets.  It is deliberately not required in the portable XLSX.
        row["_Workspace_Root"] = args.workspace_root
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
                    from physics_eval.io.results import failure_result

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
                from physics_eval.io.results import failure_result

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

