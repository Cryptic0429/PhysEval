"""SAM1 + XMem tracking backend for component comparison experiments."""
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np

from physics_eval.schemas import MaskTracks, Prompt, VideoInfo


@contextmanager
def working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


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
