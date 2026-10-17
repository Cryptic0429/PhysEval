"""SAM2 inference for physical evaluation."""
import json
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .initialization import build_tracking_prompts, load_prompt_manifest
from physics_eval.reporting import alpha_blend_mask, draw_tracking_result
from .output import mask_to_bbox, mask_to_centroid, save_mask_png
from .video import ensure_dir, extract_video_to_frames

SAM2_PRECISION_CHOICES = ("auto", "fp32", "fp16", "bf16")


def resolve_precision_policy(
    requested: str,
    *,
    cuda_available: bool,
    bf16_supported: bool,
):
    """Resolve a requested SAM2 precision without touching CUDA.

    Keeping this decision function pure makes the precision behavior testable on
    CPU-only machines. Runtime CUDA checks are supplied by ``main``.
    """
    requested = str(requested).strip().lower()
    if requested not in SAM2_PRECISION_CHOICES:
        choices = ", ".join(SAM2_PRECISION_CHOICES)
        raise ValueError(f"Unsupported precision '{requested}'. Choose one of: {choices}.")

    if requested == "auto":
        if not cuda_available:
            resolved = "fp32"
        else:
            resolved = "bf16" if bf16_supported else "fp16"
    else:
        resolved = requested

    if resolved in ("fp16", "bf16") and not cuda_available:
        raise RuntimeError(
            f"--precision {resolved} requires CUDA. Use --precision fp32 on CPU."
        )
    if resolved == "bf16" and not bf16_supported:
        raise RuntimeError(
            "--precision bf16 was requested, but the active CUDA device does not "
            "report BF16 support. Use --precision fp16 or --precision fp32."
        )

    return {
        "requested": requested,
        "resolved": resolved,
        "autocast_enabled": resolved in ("fp16", "bf16"),
        "autocast_device_type": "cuda" if resolved in ("fp16", "bf16") else None,
        "autocast_dtype": resolved if resolved in ("fp16", "bf16") else None,
    }


@contextmanager
def sam2_inference_context(precision_policy):
    """Apply one inference/autocast policy to a complete SAM2 operation."""
    import torch

    with torch.inference_mode():
        if not precision_policy["autocast_enabled"]:
            yield
            return

        dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[precision_policy["resolved"]]
        with torch.autocast(
            device_type=precision_policy["autocast_device_type"],
            dtype=dtype,
        ):
            yield


def initialize_predictor_state(
    predictor,
    *,
    frames_dir: Path,
    offload_video_to_cpu: bool,
    offload_state_to_cpu: bool,
    prompts,
    inference_context_factory,
):
    """Initialize SAM2 and inject prompts inside the supplied inference context."""
    with inference_context_factory():
        try:
            inference_state = predictor.init_state(
                video_path=str(frames_dir),
                offload_video_to_cpu=offload_video_to_cpu,
                offload_state_to_cpu=offload_state_to_cpu,
            )
        except TypeError:
            inference_state = predictor.init_state(video_path=str(frames_dir))

        for prompt in prompts:
            positive_points = prompt.get("positive_points", prompt.get("points"))
            if positive_points is None:
                positive_points = [prompt["point"]]
            negative_points = list(prompt.get("negative_points", []))
            points_in = list(positive_points) + negative_points
            labels_in = [prompt["label"]] * len(positive_points) + [0] * len(negative_points)
            point = np.array(points_in, dtype=np.float32)
            label = np.array(labels_in, dtype=np.int32)

            prompt_kwargs = {
                "inference_state": inference_state,
                "frame_idx": int(prompt["frame_idx"]),
                "obj_id": int(prompt["obj_id"]),
                "points": point,
                "labels": label,
            }
            if prompt["box"] is not None:
                prompt_kwargs["box"] = np.array(prompt["box"], dtype=np.float32)

            predictor.add_new_points_or_box(**prompt_kwargs)

    return inference_state


def propagate_in_video_with_context(
    predictor,
    inference_state,
    *,
    start_frame_idx: int,
    reverse: bool,
    inference_context_factory,
    strict_reverse: bool = False,
):
    """Yield a complete lazy SAM2 propagation while its precision context is active."""
    with inference_context_factory():
        try:
            iterator = predictor.propagate_in_video(
                inference_state,
                start_frame_idx=int(start_frame_idx),
                reverse=bool(reverse),
            )
        except TypeError:
            if reverse:
                if strict_reverse:
                    raise RuntimeError("Installed SAM2 version does not support reverse propagation.")
                print("Current SAM2 predictor does not support reverse propagation; skipping reverse pass.")
                return
            iterator = predictor.propagate_in_video(inference_state)

        # SAM2 returns a lazy generator. The context must remain entered while
        # every item is produced, not only while the generator is constructed.
        yield from iterator


def load_sam2_predictor(sam2_repo: Path, model_cfg: str, model_weights: str, vos_optimized: bool):
    sam2_repo = sam2_repo.expanduser().resolve()
    if not sam2_repo.exists():
        raise RuntimeError(f"SAM2 repo not found: {sam2_repo}")

    if str(sam2_repo) not in sys.path:
        sys.path.insert(0, str(sam2_repo))

    from sam2.build_sam import build_sam2_video_predictor

    try:
        predictor = build_sam2_video_predictor(
            model_cfg,
            model_weights,
            vos_optimized=vos_optimized,
        )
    except TypeError:
        predictor = build_sam2_video_predictor(
            model_cfg,
            model_weights,
        )

    return predictor


def run_tracking(args, prompt_manifest=None):
    """Track one video using explicit options and optional validated prompts."""
    import torch

    if prompt_manifest is None and getattr(args, "prompt_manifest", None):
        prompt_manifest = load_prompt_manifest(args.prompt_manifest)

    workspace_root = Path(args.workspace_root).expanduser().resolve()
    scene_name = args.scene_name
    video_path = Path(args.video_path).expanduser().resolve()
    sam2_repo = Path(args.sam2_repo).expanduser().resolve()
    model_weights = Path(args.model_weights).expanduser().resolve()

    if not video_path.exists():
        raise RuntimeError(f"Input video not found: {video_path}")
    if not model_weights.exists():
        raise RuntimeError(f"Model weights not found: {model_weights}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for SAM2 video inference.")

    bf16_supported = bool(
        hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
    )
    precision_policy = resolve_precision_policy(
        args.precision,
        cuda_available=True,
        bf16_supported=bf16_supported,
    )
    inference_context_factory = lambda: sam2_inference_context(precision_policy)

    data_scene_dir = workspace_root / "data" / "custom" / scene_name
    raw_dir = data_scene_dir / "raw"
    frames_dir = data_scene_dir / "frames"

    sam2_out_dir = workspace_root / "sam2_tracks" / scene_name
    masks_dir = sam2_out_dir / "masks"
    json_path = sam2_out_dir / "tracking_points.json"
    vis_video_path = sam2_out_dir / "tracking_vis.mp4"
    run_config_path = sam2_out_dir / "run_config.json"

    ensure_dir(workspace_root)
    ensure_dir(data_scene_dir)
    ensure_dir(raw_dir)
    ensure_dir(frames_dir)
    ensure_dir(sam2_out_dir)
    if args.save_mask_png:
        ensure_dir(masks_dir)

    workspace_video_path = raw_dir / "input.mp4"
    if args.copy_video_to_workspace:
        if (not workspace_video_path.exists()) or args.overwrite_outputs:
            shutil.copy2(video_path, workspace_video_path)

    tf32_enabled = precision_policy["resolved"] != "fp32"
    torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
    torch.backends.cudnn.allow_tf32 = tf32_enabled
    torch.set_float32_matmul_precision("high" if tf32_enabled else "highest")

    print("=" * 70)
    print("SAM2 tracking inside workspace")
    print("=" * 70)
    print(
        "Precision policy: "
        f"requested={precision_policy['requested']}, "
        f"resolved={precision_policy['resolved']}, "
        f"autocast={precision_policy['autocast_enabled']}, "
        f"tf32={tf32_enabled}"
    )

    print("\n[1] Preparing shared frames ...")
    fps, total_frames, W, H, extracted_now = extract_video_to_frames(
        video_path=video_path,
        frames_dir=frames_dir,
        overwrite=args.overwrite_frames,
    )

    if extracted_now:
        print(f"Frames extracted to: {frames_dir}")
    else:
        print(f"Reusing existing frames in: {frames_dir}")

    print(f"total_frames={total_frames}, size={W}x{H}, fps={fps:.3f}")

    debug_path = sam2_out_dir / "auto_init_debug.jpg" if args.save_auto_init_debug else None
    prompts, prompt_mode, auto_init_results = build_tracking_prompts(
        args, frames_dir=frames_dir, width=W, height=H,
        debug_path=debug_path, prompt_manifest=prompt_manifest,
    )
    auto_init_result = auto_init_results[0] if auto_init_results else None
    init_obj_id = int(prompts[0]["obj_id"]) if prompts else int(args.init_obj_id)
    init_label = int(args.init_label)
    if not prompts:
        raise RuntimeError("No SAM2 prompts were created.")
    args.init_point_x = float(prompts[0]["point"][0])
    args.init_point_y = float(prompts[0]["point"][1])
    init_box = prompts[0]["box"]
    init_frame_idx = int(prompts[0]["frame_idx"])

    print("\n[2] Loading SAM2 predictor ...")
    predictor = load_sam2_predictor(
        sam2_repo=sam2_repo,
        model_cfg=args.model_cfg,
        model_weights=str(model_weights),
        vos_optimized=args.vos_optimized,
    )
    print("SAM2 loaded.")

    print("\n[3] Initializing inference state ...")
    inference_state = initialize_predictor_state(
        predictor,
        frames_dir=frames_dir,
        offload_video_to_cpu=args.offload_video_to_cpu,
        offload_state_to_cpu=args.offload_state_to_cpu,
        prompts=prompts,
        inference_context_factory=inference_context_factory,
    )

    for prompt in prompts:
        print(
            f"Added prompt: mode={prompt_mode}, frame={prompt['frame_idx']}, "
            f"obj_id={prompt['obj_id']}, point={prompt['point']}, box={prompt['box']}"
        )

    writer = None
    if args.save_vis_video:
        if vis_video_path.exists() and not args.overwrite_outputs:
            raise RuntimeError(f"Output exists: {vis_video_path}. Use --overwrite-outputs to replace.")
        writer = cv2.VideoWriter(
            str(vis_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (W, H),
        )

    if json_path.exists() and not args.overwrite_outputs:
        raise RuntimeError(f"Output exists: {json_path}. Use --overwrite-outputs to replace.")

    object_ids = [int(p["obj_id"]) for p in prompts]
    prompt_records = [
        {
            "obj_id": int(p["obj_id"]),
            "init_frame_idx": int(p["frame_idx"]),
            "init_point": [float(p["point"][0]), float(p["point"][1])],
            "positive_points": [
                [float(value) for value in point]
                for point in p.get("positive_points", [p["point"]])
            ],
            "init_label": int(p["label"]),
            "init_box": p["box"],
            "negative_points": p.get("negative_points", []),
            "auto_init": p["auto_init"],
        }
        for p in prompts
    ]

    tracking_data = {
        "video_info": {
            "scene_name": scene_name,
            "video_path_original": str(video_path),
            "video_path_workspace": str(workspace_video_path) if args.copy_video_to_workspace else None,
            "frames_dir": str(frames_dir),
            "fps": fps,
            "width": W,
            "height": H,
            "total_frames": total_frames,
            "sam2_repo": str(sam2_repo),
            "model_cfg": args.model_cfg,
            "model_weights": str(model_weights),
        },
        "object_id": object_ids[0],
        "object_ids": object_ids,
        "objects": prompt_records,
        "prompt": {
            "mode": prompt_mode,
            "init_frame_idx": init_frame_idx,
            "init_point": [args.init_point_x, args.init_point_y],
            "positive_points": prompt_records[0]["positive_points"],
            "negative_points": prompt_records[0]["negative_points"],
            "init_label": init_label,
            "init_box": init_box,
            "auto_init": auto_init_result,
            "auto_init_results": auto_init_results,
        },
        "frames": []
    }

    run_config = {
        "workspace_root": str(workspace_root),
        "scene_name": scene_name,
        "video_path": str(video_path),
        "frames_dir": str(frames_dir),
        "sam2_output_dir": str(sam2_out_dir),
        "save_vis_video": bool(args.save_vis_video),
        "save_mask_png": bool(args.save_mask_png),
        "prompt_mode": prompt_mode,
        "init_frame_idx": init_frame_idx,
        "init_obj_id": init_obj_id,
        "init_point_x": args.init_point_x,
        "init_point_y": args.init_point_y,
        "init_positive_points": prompt_records[0]["positive_points"],
        "init_negative_points": prompt_records[0]["negative_points"],
        "init_label": init_label,
        "init_box": init_box,
        "auto_init": auto_init_result,
        "auto_init_results": auto_init_results,
        "prompts": prompt_records,
        "auto_init_detector": "yolo_then_motion",
        "yolo_weights": args.yolo_weights,
        "yolo_conf": args.yolo_conf,
        "yolo_iou": args.yolo_iou,
        "yolo_classes": args.yolo_classes,
        "auto_init_num_objects": args.auto_init_num_objects,
        "auto_init_scan_frames": args.auto_init_scan_frames,
        "auto_init_scan_mode": args.auto_init_scan_mode,
        "bidirectional_propagation": not args.no_bidirectional_propagation,
        "auto_init_threshold": args.auto_init_threshold,
        "auto_init_min_area": args.auto_init_min_area,
        "auto_init_max_area_ratio": args.auto_init_max_area_ratio,
        "auto_init_min_fill_ratio": args.auto_init_min_fill_ratio,
        "auto_init_max_aspect_ratio": args.auto_init_max_aspect_ratio,
        "auto_init_min_separation_ratio": args.auto_init_min_separation_ratio,
        "auto_init_shadow_filter": not args.disable_auto_init_shadow_filter,
        "auto_init_negative_points": not args.no_auto_init_negative_points,
        "auto_init_object_refine": bool(args.auto_init_object_refine),
        "auto_init_box_expand": args.auto_init_box_expand,
        "auto_init_target_hint": args.auto_init_target_hint,
        "model_cfg": args.model_cfg,
        "model_weights": str(model_weights),
        "precision": precision_policy["resolved"],
        "precision_requested": precision_policy["requested"],
        "precision_policy": precision_policy,
        "tf32_enabled": tf32_enabled,
    }

    if prompt_manifest is not None:
        tracking_data["batch_initialization"] = prompt_manifest
        run_config["batch_initialization"] = prompt_manifest

    with open(run_config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    print("\n[4] Tracking ...")
    t0 = time.time()
    processed = 0
    tracking_records_by_frame = {}
    vis_frames_by_frame = {}
    prompt_frame_indices = [int(p["frame_idx"]) for p in prompts] or [int(init_frame_idx)]
    first_prompt_frame = min(prompt_frame_indices)
    last_prompt_frame = max(prompt_frame_indices)
    propagation_jobs = [("forward", False, first_prompt_frame)]
    if not args.no_bidirectional_propagation and last_prompt_frame > 0:
        propagation_jobs.append(("reverse", True, last_prompt_frame))

    for direction_name, reverse, start_frame_idx in propagation_jobs:
        print(f"Propagating {direction_name} from frame {start_frame_idx} ...")
        iterator = propagate_in_video_with_context(
            predictor,
            inference_state,
            start_frame_idx=int(start_frame_idx),
            reverse=bool(reverse),
            inference_context_factory=inference_context_factory,
        )

        for out_frame_idx, out_obj_ids, out_mask_logits in iterator:
            frame_idx = int(out_frame_idx)
            if frame_idx in tracking_records_by_frame:
                continue

            frame_name = f"{frame_idx:05d}.jpg"
            frame_path = frames_dir / frame_name
            masks_by_id = {int(oid): out_mask_logits[i] for i, oid in enumerate(out_obj_ids)}
            object_records = []
            vis_frame = cv2.imread(str(frame_path)) if writer is not None else None
            colors = [(0, 255, 0), (255, 0, 255), (255, 180, 0), (0, 180, 255)]

            for obj_pos, obj_id in enumerate(object_ids):
                target_mask = masks_by_id.get(int(obj_id))
                if target_mask is None:
                    object_records.append({
                        "object_id": int(obj_id),
                        "valid": False,
                        "x": None,
                        "y": None,
                        "centroid": None,
                        "bbox": None,
                        "area": 0,
                        "mask_path": None,
                    })
                    continue

                mask_bool = (target_mask > 0.0).detach().cpu().numpy().squeeze().astype(bool)
                area = int(mask_bool.sum())
                centroid = mask_to_centroid(mask_bool)
                bbox = mask_to_bbox(mask_bool)

                mask_path_str = None
                if args.save_mask_png:
                    if len(object_ids) == 1:
                        mask_path = masks_dir / f"{frame_idx:05d}.png"
                    else:
                        obj_mask_dir = masks_dir / f"obj_{int(obj_id):06d}"
                        ensure_dir(obj_mask_dir)
                        mask_path = obj_mask_dir / f"{frame_idx:05d}.png"
                    save_mask_png(mask_path, mask_bool)
                    mask_path_str = str(mask_path)

                object_record = {
                    "object_id": int(obj_id),
                    "valid": centroid is not None and area > 0,
                    "x": float(centroid[0]) if centroid is not None else None,
                    "y": float(centroid[1]) if centroid is not None else None,
                    "centroid": [float(centroid[0]), float(centroid[1])] if centroid is not None else None,
                    "bbox": bbox,
                    "area": area,
                    "mask_path": mask_path_str,
                    "propagation_direction": direction_name,
                }
                object_records.append(object_record)

                if vis_frame is not None:
                    color = colors[obj_pos % len(colors)]
                    if area > 0:
                        alpha_blend_mask(vis_frame, mask_bool, color_bgr=color, alpha=0.4)
                    if centroid is not None:
                        draw_tracking_result(vis_frame, centroid[0], centroid[1], obj_id, color=color)

            if len(object_ids) == 1:
                record = dict(object_records[0])
                record.update({
                    "frame_idx": frame_idx,
                    "frame_name": frame_name,
                    "frame_path": str(frame_path),
                    "time_sec": frame_idx / fps,
                })
            else:
                record = {
                    "frame_idx": frame_idx,
                    "frame_name": frame_name,
                    "frame_path": str(frame_path),
                    "time_sec": frame_idx / fps,
                    "objects": object_records,
                    "propagation_direction": direction_name,
                }
            tracking_records_by_frame[frame_idx] = record

            if writer is not None and vis_frame is not None:
                vis_frames_by_frame[frame_idx] = vis_frame

            processed += 1
            if processed % 10 == 0:
                avg_fps = processed / max(time.time() - t0, 1e-6)
                print(
                    f"Processed {processed}/{total_frames}, avg_fps={avg_fps:.2f}",
                    end="\r",
                )

    tracking_data["frames"] = [
        tracking_records_by_frame[i]
        for i in sorted(tracking_records_by_frame)
    ]

    if writer is not None:
        for frame_idx in sorted(vis_frames_by_frame):
            writer.write(vis_frames_by_frame[frame_idx])
        writer.release()

    recorded = {int(fr["frame_idx"]) for fr in tracking_data["frames"]}
    if len(recorded) < total_frames:
        for frame_idx in range(total_frames):
            if frame_idx in recorded:
                continue
            frame_name = f"{frame_idx:05d}.jpg"
            frame_path = frames_dir / frame_name
            reason = "not_tracked_before_init" if frame_idx < init_frame_idx else "not_tracked"
            if len(object_ids) == 1:
                tracking_data["frames"].append({
                    "frame_idx": frame_idx,
                    "frame_name": frame_name,
                    "frame_path": str(frame_path),
                    "time_sec": frame_idx / fps,
                    "object_id": int(object_ids[0]),
                    "valid": False,
                    "x": None,
                    "y": None,
                    "centroid": None,
                    "bbox": None,
                    "area": 0,
                    "mask_path": None,
                    "reason": reason,
                })
            else:
                tracking_data["frames"].append({
                    "frame_idx": frame_idx,
                    "frame_name": frame_name,
                    "frame_path": str(frame_path),
                    "time_sec": frame_idx / fps,
                    "objects": [
                        {
                            "object_id": int(obj_id),
                            "valid": False,
                            "x": None,
                            "y": None,
                            "centroid": None,
                            "bbox": None,
                            "area": 0,
                            "mask_path": None,
                            "reason": reason,
                        }
                        for obj_id in object_ids
                    ],
                })
        tracking_data["frames"].sort(key=lambda fr: int(fr["frame_idx"]))

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(tracking_data, f, ensure_ascii=False, indent=2)

    print("\nTracking completed.")
    print(f"Frames dir: {frames_dir}")
    print(f"Tracking JSON: {json_path}")
    if args.save_vis_video:
        print(f"Visualization video: {vis_video_path}")
    if args.save_mask_png:
        print(f"Masks dir: {masks_dir}")
    print(f"Run config: {run_config_path}")
    return tracking_data

