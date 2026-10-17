"""Stable prompt structure for tracking initialization."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
