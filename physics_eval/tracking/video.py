"""Video metadata and JPEG extraction with explicit cache policies.

Formal tracking reuses an existing frame directory as before. Comparison
extraction additionally checks the declared frame count before reuse.
"""
from pathlib import Path

import cv2

from physics_eval.schemas import VideoInfo


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
