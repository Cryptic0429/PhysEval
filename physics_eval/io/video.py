"""Video discovery and metadata filename resolution (no decoder dependencies)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

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


def resolve_video_path(row: dict[str, Any], video_root: Path, metadata_path: Path) -> Path:
    raw = row.get("Video_File")
    if not raw:
        raise FileNotFoundError("Video_File is empty")
    candidate = Path(str(raw))
    candidates = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.append(video_root / candidate)
        candidates.append(metadata_path.parent / candidate)
        candidates.append(Path.cwd() / candidate)
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError("Video_File not found. searched: " + ", ".join(str(p) for p in candidates))
