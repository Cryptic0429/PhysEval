"""Stable prompt, video, and mask-track structures shared by all workflows."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Prompt:
    obj_id: int
    frame_idx: int
    point: list[float]
    box: list[float]
    score: float
    detector: str
    class_id: int | None = None
    class_name: str | None = None
    confidence: float | None = None
    negative_points: list[list[float]] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Prompt":
        return cls(**value)


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    fps: float
    width: int
    height: int
    total_frames: int
    frames_dir: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_path_original": str(self.path),
            "frames_dir": str(self.frames_dir),
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "total_frames": self.total_frames,
        }


MaskTracks = dict[int, dict[int, Any]]


@dataclass(frozen=True)
class PromptSpec:
    object_id: int
    frame_idx: int
    point: tuple[float, float]
    positive_points: tuple[tuple[float, float], ...]
    negative_points: tuple[tuple[float, float], ...]
    box: tuple[float, float, float, float] | None = None
    source: str = "manual"

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "frame_idx": self.frame_idx,
            "point": list(self.point),
            "positive_points": [list(point) for point in self.positive_points],
            "negative_points": [list(point) for point in self.negative_points],
            "box": list(self.box) if self.box is not None else None,
            "source": self.source,
        }
