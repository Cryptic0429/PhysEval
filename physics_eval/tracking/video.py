"""Video metadata and full-sequence JPEG extraction for formal tracking."""
from pathlib import Path

import cv2



def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def list_jpg_frames(frames_dir: Path):
    return sorted(frames_dir.glob("*.jpg"))


def read_video_meta(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return float(fps), frame_count, width, height


def extract_video_to_frames(video_path: Path, frames_dir: Path, overwrite: bool = False):
    ensure_dir(frames_dir)

    existing = list_jpg_frames(frames_dir)
    if existing and not overwrite:
        fps, _, width, height = read_video_meta(video_path)
        return fps, len(existing), width, height, False

    if overwrite:
        for p in frames_dir.glob("*.jpg"):
            p.unlink()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_name = f"{idx:05d}.jpg"
        frame_path = frames_dir / frame_name
        cv2.imwrite(str(frame_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        idx += 1

    cap.release()
    return float(fps), idx, width, height, True

