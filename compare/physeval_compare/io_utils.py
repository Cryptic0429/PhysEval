from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .types import VideoInfo


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_metadata(path: Path | None, video_root: Path) -> list[dict[str, Any]]:
    if path is None:
        rows = []
        for index, video_path in enumerate(discover_videos(video_root), start=1):
            rows.append({
                "Index": index,
                "Prompt_ID": video_path.stem,
                "Video_File": video_path.name,
                "Metric": "tracking_only",
                "Num_Objects": 1,
                "Tracking_Objects": "object_center",
                "Calibration_Object": "none",
                "Known_Parameters_JSON": "{}",
            })
        return rows

    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    if path.suffix.lower() == ".csv":
        table = pd.read_csv(path)
    else:
        workbook = pd.ExcelFile(path)
        sheet = "metadata" if "metadata" in workbook.sheet_names else workbook.sheet_names[0]
        table = pd.read_excel(path, sheet_name=sheet)
    table = table.where(pd.notna(table), None)
    return [json_safe(row) for row in table.to_dict(orient="records")]


def discover_videos(video_root: Path) -> list[Path]:
    return sorted(
        path for path in video_root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )


def resolve_video(video_root: Path, raw_name: Any) -> Path:
    name = str(raw_name or "").strip()
    if not name:
        raise FileNotFoundError("Metadata row has an empty Video_File value.")
    direct = (video_root / name).resolve()
    if direct.exists():
        return direct
    requested = Path(name)
    matches = [path.resolve() for path in video_root.rglob(requested.name)]
    if not matches:
        prompt_stem = requested.stem.lower()
        matches = [
            path.resolve()
            for path in video_root.rglob(f"*{requested.suffix}")
            if path.stem.lower().endswith((f"_{prompt_stem}", f"-{prompt_stem}"))
        ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Video not found below {video_root}: {name}")
    raise RuntimeError(f"Ambiguous video filename below {video_root}: {name} -> {matches}")


def extract_video_frames(video_path: Path, frames_dir: Path, overwrite: bool = False) -> VideoInfo:
    import cv2

    frames_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(frames_dir.glob("*.jpg"))
    if overwrite:
        for path in existing:
            path.unlink()
        existing = []

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    declared_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))

    if existing and (declared_frames <= 0 or len(existing) == declared_frames):
        capture.release()
        return VideoInfo(video_path, fps, width, height, len(existing), frames_dir)

    for path in existing:
        path.unlink()
    frame_idx = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        out_path = frames_dir / f"{frame_idx:05d}.jpg"
        if not cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            capture.release()
            raise RuntimeError(f"Failed to write extracted frame: {out_path}")
        frame_idx += 1
    capture.release()
    if frame_idx == 0:
        raise RuntimeError(f"Video contains no readable frames: {video_path}")
    return VideoInfo(video_path, fps, width, height, frame_idx, frames_dir)


def filter_rows(
    rows: Iterable[dict[str, Any]],
    *,
    index: int | None,
    prompt_id: str | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if index is not None and int(row.get("Index") or -1) != index:
            continue
        if prompt_id is not None and str(row.get("Prompt_ID")) != prompt_id:
            continue
        selected.append(row)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def infer_target_hint(row: dict[str, Any]) -> str:
    known_raw = row.get("Known_Parameters_JSON")
    known: dict[str, Any] = {}
    if isinstance(known_raw, dict):
        known = known_raw
    elif known_raw:
        try:
            known = json.loads(str(known_raw))
        except json.JSONDecodeError:
            known = {}
    for key in ("object", "tracking_target", "target_object"):
        value = known.get(key)
        if value and str(value).lower() != "none":
            return str(value)
    calibration = row.get("Calibration_Object")
    if calibration and str(calibration).lower() != "none":
        return str(calibration)
    tracking = str(row.get("Tracking_Objects") or "object")
    return tracking.replace("_center", "")


def expected_objects(row: dict[str, Any]) -> int:
    try:
        return max(1, int(float(row.get("Num_Objects") or 1)))
    except (TypeError, ValueError):
        return 1
