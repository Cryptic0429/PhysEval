from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModeSpec:
    name: str
    detector: str
    tracker: str
    label: str


MODE_SPECS = {
    "fasterrcnn_sam2": ModeSpec(
        name="fasterrcnn_sam2",
        detector="fasterrcnn",
        tracker="sam2",
        label="Faster R-CNN + SAM2",
    ),
    "yolo_sam2": ModeSpec(
        name="yolo_sam2",
        detector="yolo",
        tracker="sam2",
        label="YOLOv8n + SAM2",
    ),
    "yolo_tam": ModeSpec(
        name="yolo_tam",
        detector="yolo",
        tracker="tam",
        label="YOLOv8n + TAM (SAM1 + XMem)",
    ),
}

DEFAULT_MODES = tuple(MODE_SPECS)
