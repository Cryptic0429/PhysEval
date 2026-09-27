from __future__ import annotations

import os
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .types import MaskTracks, Prompt, VideoInfo


@contextmanager
def working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _precision_context(device: str, precision: str):
    import torch

    if not device.startswith("cuda") or precision == "fp32":
        return nullcontext()
    if precision == "auto":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif precision == "bf16":
        dtype = torch.bfloat16
    elif precision == "fp16":
        dtype = torch.float16
    else:
        raise ValueError(f"Unsupported precision: {precision}")
    return torch.autocast(device_type="cuda", dtype=dtype)


class Sam2Tracker:
    name = "sam2"

    def __init__(
        self,
        *,
        repo: Path,
        model_config: str,
        checkpoint: Path,
        device: str,
        precision: str = "auto",
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
    ) -> None:
        if not repo.exists():
            raise FileNotFoundError(f"SAM2 repository not found: {repo}")
        if not checkpoint.exists():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint}")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from sam2.build_sam import build_sam2_video_predictor

        try:
            self.predictor = build_sam2_video_predictor(model_config, str(checkpoint), device=device)
        except TypeError:
            self.predictor = build_sam2_video_predictor(model_config, str(checkpoint))
        self.device = device
        self.precision = precision
        self.offload_video_to_cpu = offload_video_to_cpu
        self.offload_state_to_cpu = offload_state_to_cpu

    def track(self, info: VideoInfo, prompts: list[Prompt]) -> MaskTracks:
        import torch

        if not prompts:
            raise ValueError("SAM2 received no prompts.")
        init_frames = {prompt.frame_idx for prompt in prompts}
        if len(init_frames) != 1:
            raise ValueError("Comparison protocol requires all objects to use one common initialization frame.")
        init_frame = next(iter(init_frames))
        context = lambda: _precision_context(self.device, self.precision)
        with torch.inference_mode(), context():
            try:
                state = self.predictor.init_state(
                    video_path=str(info.frames_dir),
                    offload_video_to_cpu=self.offload_video_to_cpu,
                    offload_state_to_cpu=self.offload_state_to_cpu,
                )
            except TypeError:
                state = self.predictor.init_state(video_path=str(info.frames_dir))
            for prompt in prompts:
                points = [prompt.point, *prompt.negative_points]
                labels = [1, *([0] * len(prompt.negative_points))]
                kwargs = {
                    "inference_state": state,
                    "frame_idx": prompt.frame_idx,
                    "obj_id": prompt.obj_id,
                    "points": np.asarray(points, dtype=np.float32),
                    "labels": np.asarray(labels, dtype=np.int32),
                    "box": np.asarray(prompt.box, dtype=np.float32),
                }
                self.predictor.add_new_points_or_box(**kwargs)

        tracks: MaskTracks = {prompt.obj_id: {} for prompt in prompts}
        self._propagate(state, tracks, init_frame, reverse=False, context=context)
        if init_frame > 0:
            self._propagate(state, tracks, init_frame, reverse=True, context=context)
        return tracks

    def _propagate(
        self,
        state: Any,
        tracks: MaskTracks,
        start_frame: int,
        *,
        reverse: bool,
        context: Any,
    ) -> None:
        import torch

        with torch.inference_mode(), context():
            try:
                iterator = self.predictor.propagate_in_video(
                    state, start_frame_idx=start_frame, reverse=reverse,
                )
            except TypeError:
                if reverse:
                    raise RuntimeError("Installed SAM2 version does not support reverse propagation.")
                iterator = self.predictor.propagate_in_video(state)
            for frame_idx, object_ids, mask_logits in iterator:
                for position, object_id in enumerate(object_ids):
                    mask = (mask_logits[position] > 0.0).detach().cpu().numpy().squeeze().astype(bool)
                    tracks.setdefault(int(object_id), {})[int(frame_idx)] = mask


class TamTracker:
    """YOLO-prompted SAM1 initialization followed by Track-Anything's XMem wrapper."""

    name = "tam"

    def __init__(
        self,
        *,
        segment_anything_repo: Path,
        sam_checkpoint: Path,
        sam_model_type: str,
        track_anything_repo: Path,
        xmem_checkpoint: Path,
        device: str,
    ) -> None:
        for path, label in (
            (segment_anything_repo, "Segment Anything repository"),
            (sam_checkpoint, "SAM1 checkpoint"),
            (track_anything_repo, "Track-Anything repository"),
            (xmem_checkpoint, "XMem checkpoint"),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")
        # Track-Anything mixes package-qualified imports (tracker.model.*) with
        # top-level imports (inference.*). Both the repository root and its
        # tracker directory therefore need to be importable.
        for repo in (
            segment_anything_repo,
            track_anything_repo,
            track_anything_repo / "tracker",
        ):
            if str(repo) not in sys.path:
                sys.path.insert(0, str(repo))

        import torch
        from segment_anything import SamPredictor, sam_model_registry

        sam = sam_model_registry[sam_model_type](checkpoint=str(sam_checkpoint)).to(device).eval()
        self.sam_predictor = SamPredictor(sam)
        with working_directory(track_anything_repo):
            from tracker.base_tracker import BaseTracker
            self.xmem = BaseTracker(str(xmem_checkpoint), device)
        self.device = device
        self.track_anything_repo = track_anything_repo
        self.torch = torch

    def track(self, info: VideoInfo, prompts: list[Prompt]) -> MaskTracks:
        from PIL import Image

        if not prompts:
            raise ValueError("TAM received no prompts.")
        init_frames = {prompt.frame_idx for prompt in prompts}
        if len(init_frames) != 1:
            raise ValueError("Comparison protocol requires all objects to use one common initialization frame.")
        init_frame = next(iter(init_frames))
        frame_paths = sorted(info.frames_dir.glob("*.jpg"))
        frames = [np.asarray(Image.open(path).convert("RGB")) for path in frame_paths]
        template = self._initial_label_mask(frames[init_frame], prompts)
        tracks: MaskTracks = {prompt.obj_id: {} for prompt in prompts}

        forward_indices = list(range(init_frame, len(frames)))
        self._propagate(frames, forward_indices, template, tracks)
        if init_frame > 0:
            backward_indices = list(range(init_frame, -1, -1))
            self._propagate(frames, backward_indices, template, tracks)
        return tracks

    def _initial_label_mask(self, image: np.ndarray, prompts: list[Prompt]) -> np.ndarray:
        self.sam_predictor.set_image(image)
        predictions: list[tuple[float, Prompt, np.ndarray]] = []
        for prompt in prompts:
            points = np.asarray([prompt.point, *prompt.negative_points], dtype=np.float32)
            labels = np.asarray([1, *([0] * len(prompt.negative_points))], dtype=np.int32)
            masks, scores, _ = self.sam_predictor.predict(
                point_coords=points,
                point_labels=labels,
                box=np.asarray(prompt.box, dtype=np.float32),
                multimask_output=True,
            )
            best = int(np.argmax(scores))
            predictions.append((float(scores[best]), prompt, masks[best].astype(bool)))
        label_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for _, prompt, mask in sorted(predictions, key=lambda item: item[0], reverse=True):
            label_mask[(label_mask == 0) & mask] = int(prompt.obj_id)
        missing = [prompt.obj_id for prompt in prompts if not np.any(label_mask == prompt.obj_id)]
        if missing:
            raise RuntimeError(f"SAM1 produced empty initialization masks for object ids: {missing}")
        return label_mask

    def _propagate(
        self,
        frames: list[np.ndarray],
        indices: list[int],
        template: np.ndarray,
        tracks: MaskTracks,
    ) -> None:
        self.xmem.clear_memory()
        with self.torch.inference_mode(), working_directory(self.track_anything_repo):
            for position, frame_idx in enumerate(indices):
                if position == 0:
                    label_mask, _, _ = self.xmem.track(frames[frame_idx], template)
                else:
                    label_mask, _, _ = self.xmem.track(frames[frame_idx])
                for object_id in tracks:
                    tracks[object_id][frame_idx] = label_mask == object_id
        self.xmem.clear_memory()
