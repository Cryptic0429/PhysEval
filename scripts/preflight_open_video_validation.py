#!/usr/bin/env python3
"""Validate the open-video package before launching the GPU pipeline.

The default mode checks data, Python dependencies, CUDA, and SAM2 assets.  Use
``--data-only`` on a laptop without the inference environment to verify the
portable paths, workbook schema, checksums-sized files, and renderer oracles.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import re
import shutil
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from physics_eval.evaluators import EVALUATOR_REGISTRY
from physics_eval.utils.io import parse_known_params, read_metadata, row_to_dict


SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
RUNTIME_MODULES = (
    "numpy",
    "pandas",
    "scipy",
    "cv2",
    "torch",
    "torchvision",
    "ultralytics",
    "hydra",
    "iopath",
)


@dataclass
class Report:
    ok: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def add_ok(self, message: str) -> None:
        self.ok.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def emit(self) -> None:
        for message in self.ok:
            print(f"[OK]    {message}")
        for message in self.warnings:
            print(f"[WARN]  {message}")
        for message in self.errors:
            print(f"[ERROR] {message}")
        print(
            f"\nPreflight summary: {len(self.ok)} OK, "
            f"{len(self.warnings)} warning(s), {len(self.errors)} blocker(s)."
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "warnings": self.warnings,
            "errors": self.errors,
            "ready": not self.errors,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight the curated open-video validation batch.")
    parser.add_argument(
        "--workspace-root",
        default=str(WORKSPACE_ROOT),
        help="Repository root. Relative CLI paths are resolved against this directory.",
    )
    parser.add_argument(
        "--metadata",
        default="validation/open_video_sources/metadata/validation_metadata.xlsx",
    )
    parser.add_argument("--video-root", default="validation/open_video_sources")
    parser.add_argument("--sam2-repo", default="repo/sam2")
    parser.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    parser.add_argument("--model-weights", default="repo/sam2/checkpoints/sam2.1_hiera_base_plus.pt")
    parser.add_argument("--yolo-weights", default="yolov8n.pt")
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Skip runtime dependency, CUDA, and model-asset blockers.",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional path for a machine-readable report (resolved against workspace root).",
    )
    return parser.parse_args()


def resolve_from_root(raw: str | Path, workspace_root: Path) -> Path:
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (workspace_root / path).resolve()


def portable_relative_path(raw: Any) -> tuple[bool, str]:
    text = str(raw or "").strip()
    if not text:
        return False, "path is empty"
    if "\\" in text:
        return False, "must use forward slashes, not backslashes"
    path = Path(text)
    if path.is_absolute() or re.match(r"^[A-Za-z]:", text):
        return False, "must be relative, not absolute"
    if any(part == ".." for part in text.split("/")):
        return False, "must not contain '..' traversal"
    return True, text


def slug(value: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    return out.strip("_")


def check_mp4_header(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            header = stream.read(64)
        return len(header) >= 12 and b"ftyp" in header[:32]
    except OSError:
        return False


def mp4_stsz_sample_counts(path: Path) -> list[int]:
    """Read sample counts from MP4 stsz boxes without a codec dependency."""
    try:
        payload = path.read_bytes()
    except OSError:
        return []
    counts: list[int] = []
    offset = 0
    while True:
        marker = payload.find(b"stsz", offset)
        if marker < 0:
            break
        if marker + 16 <= len(payload):
            count = struct.unpack(">I", payload[marker + 12 : marker + 16])[0]
            if count > 0:
                counts.append(int(count))
        offset = marker + 4
    return counts


def check_decodable(path: Path) -> tuple[bool, str]:
    try:
        import cv2
    except Exception:
        return False, "cv2 unavailable; MP4 container header checked only"
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return False, "OpenCV could not open the video"
        frame_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        ok, frame = cap.read()
        if not ok or frame is None:
            return False, "OpenCV could not decode the first frame"
        return True, f"decoded; frames={frame_count}, fps={fps:.3f}, size={frame.shape[1]}x{frame.shape[0]}"
    finally:
        cap.release()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def validate_video_manifest(metadata_path: Path, video_root: Path, report: Report) -> None:
    manifest_path = metadata_path.parent / "video_manifest.csv"
    if not manifest_path.is_file():
        report.error(f"video integrity manifest missing: {manifest_path}")
        return
    try:
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, csv.Error) as exc:
        report.error(f"video integrity manifest could not be read: {exc}")
        return
    failures = 0
    for item in rows:
        prompt_id = item.get("Prompt_ID") or "unknown"
        raw_path = item.get("Video_File") or ""
        path_ok, path_or_reason = portable_relative_path(raw_path)
        if not path_ok:
            report.error(f"manifest {prompt_id}: Video_File {path_or_reason}")
            failures += 1
            continue
        path = (video_root / path_or_reason).resolve()
        if not path.is_file():
            report.error(f"manifest {prompt_id}: video missing: {path}")
            failures += 1
            continue
        try:
            expected_bytes = int(item.get("Bytes") or -1)
        except ValueError:
            expected_bytes = -1
        expected_hash = str(item.get("SHA256") or "").strip().upper()
        if path.stat().st_size != expected_bytes:
            report.error(
                f"manifest {prompt_id}: byte-size mismatch, expected {expected_bytes}, got {path.stat().st_size}"
            )
            failures += 1
            continue
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            report.error(f"manifest {prompt_id}: SHA-256 mismatch for {path}")
            failures += 1
    if failures == 0:
        report.add_ok(f"video manifest: {len(rows)} file size/SHA-256 checks passed")


def validate_oracle(row: dict[str, Any], params: dict[str, Any], workspace_root: Path, report: Report) -> None:
    prompt_id = str(row.get("Prompt_ID"))
    pattern = params.get("gt_mask_pattern")
    rgba = params.get("gt_rgba")
    if not isinstance(pattern, str) or "{frame" not in pattern:
        report.error(f"{prompt_id}: eval_oracle_tracking requires gt_mask_pattern with a {{frame...}} placeholder")
        return
    if not isinstance(rgba, list) or len(rgba) not in {3, 4}:
        report.error(f"{prompt_id}: eval_oracle_tracking requires gt_rgba as a 3- or 4-element list")
        return
    if any(not isinstance(value, int) or not 0 <= value <= 255 for value in rgba):
        report.error(f"{prompt_id}: gt_rgba values must be integers in [0,255]")
        return

    first_visible = int(params.get("first_visible_frame", 0) or 0)
    try:
        sample_text = pattern.format(frame=first_visible)
    except (KeyError, ValueError) as exc:
        report.error(f"{prompt_id}: invalid gt_mask_pattern: {exc}")
        return
    sample_path = resolve_from_root(sample_text, workspace_root)
    if not sample_path.exists():
        report.error(f"{prompt_id}: GT sample mask missing: {sample_path}")
        return
    try:
        from PIL import Image

        with Image.open(sample_path) as image:
            colors = image.convert("RGBA").getcolors(maxcolors=4096)
        target_rgba = tuple(rgba) if len(rgba) == 4 else (*rgba, 255)
        if colors is None or not any(color == target_rgba and count > 0 for count, color in colors):
            report.error(
                f"{prompt_id}: configured GT color {list(target_rgba)} is absent from first visible mask {sample_path}"
            )
            return
    except Exception as exc:
        report.error(f"{prompt_id}: could not inspect GT sample mask color: {type(exc).__name__}: {exc}")
        return
    parent = sample_path.parent
    stem_prefix = sample_path.name.split(f"{first_visible:04d}", 1)[0]
    mask_count = len(list(parent.glob(f"{stem_prefix}*.png")))
    expected = int(row.get("Expected_Frame_Count") or params.get("frame_count") or 0)
    if expected and mask_count < expected:
        report.error(f"{prompt_id}: found {mask_count} GT PNGs, expected at least {expected}")
    else:
        report.add_ok(f"{prompt_id}: renderer oracle present ({mask_count} PNGs; color={rgba})")


def validate_metadata(
    metadata_path: Path,
    video_root: Path,
    workspace_root: Path,
    report: Report,
    decode_videos: bool,
) -> None:
    if not metadata_path.is_file():
        report.error(f"metadata workbook not found: {metadata_path}")
        return
    if not video_root.is_dir():
        report.error(f"video root not found: {video_root}")
        return

    try:
        df = read_metadata(metadata_path)
    except Exception as exc:
        report.error(f"metadata could not be read: {type(exc).__name__}: {exc}")
        return
    if df.empty:
        report.error("metadata contains no rows")
        return
    report.add_ok(f"metadata schema loaded: {len(df)} row(s), {len(df.columns)} column(s)")

    if df["Index"].isna().any() or df["Index"].duplicated().any():
        report.error("Index must contain unique non-empty values")
    if df["Prompt_ID"].isna().any() or df["Prompt_ID"].astype(str).duplicated().any():
        report.error("Prompt_ID must contain unique non-empty values")

    prompt_ids: list[str] = []
    slugs: dict[str, str] = {}
    resolved_videos: dict[Path, str] = {}
    decoded = 0
    container_counts_verified = 0
    for _, pd_row in df.iterrows():
        row = row_to_dict(pd_row)
        prompt_id = str(row.get("Prompt_ID") or "").strip()
        prompt_ids.append(prompt_id)
        if not SAFE_ID.fullmatch(prompt_id):
            report.error(f"Prompt_ID is not path-safe: {prompt_id!r}")
        normalized = slug(prompt_id)
        if normalized in slugs and slugs[normalized] != prompt_id:
            report.error(f"Prompt_ID slug collision: {slugs[normalized]!r} and {prompt_id!r}")
        slugs[normalized] = prompt_id

        evaluator = str(row.get("Evaluator") or "")
        if evaluator not in EVALUATOR_REGISTRY:
            report.error(f"{prompt_id}: evaluator is not registered: {evaluator!r}")

        try:
            params = parse_known_params(row)
        except ValueError as exc:
            report.error(f"{prompt_id}: {exc}")
            params = {}

        try:
            n_objects = int(row.get("Num_Objects") or 1)
            if n_objects < 1:
                raise ValueError
        except (TypeError, ValueError):
            report.error(f"{prompt_id}: Num_Objects must be a positive integer")

        path_ok, path_or_reason = portable_relative_path(row.get("Video_File"))
        if not path_ok:
            report.error(f"{prompt_id}: Video_File {path_or_reason}")
            continue
        video_path = (video_root / Path(path_or_reason)).resolve()
        try:
            video_path.relative_to(video_root)
        except ValueError:
            report.error(f"{prompt_id}: resolved video escapes --video-root: {video_path}")
            continue
        if video_path in resolved_videos:
            report.warn(f"{prompt_id}: duplicates video used by {resolved_videos[video_path]}")
        resolved_videos[video_path] = prompt_id
        if not video_path.is_file():
            report.error(f"{prompt_id}: video missing: {video_path}")
            continue
        if video_path.stat().st_size < 1024 or not check_mp4_header(video_path):
            report.error(f"{prompt_id}: file is not a plausible MP4: {video_path}")
            continue

        expected_frames = row.get("Expected_Frame_Count")
        try:
            expected_frames_int = int(expected_frames) if expected_frames is not None else None
        except (TypeError, ValueError):
            expected_frames_int = None
        sample_counts = mp4_stsz_sample_counts(video_path)
        if expected_frames_int is not None:
            if expected_frames_int in sample_counts:
                container_counts_verified += 1
            elif sample_counts:
                report.error(
                    f"{prompt_id}: Expected_Frame_Count={expected_frames_int}, MP4 stsz counts={sample_counts}"
                )
            else:
                report.warn(f"{prompt_id}: no stsz sample count found; decoder must verify the frame count")

        if decode_videos:
            is_decodable, detail = check_decodable(video_path)
            if is_decodable:
                decoded += 1
                if expected_frames and f"frames={int(expected_frames)}" not in detail:
                    report.warn(f"{prompt_id}: workbook frame count differs or decoder did not report it ({detail})")
            else:
                report.error(f"{prompt_id}: {detail}: {video_path}")

        if evaluator == "eval_oracle_tracking":
            validate_oracle(row, params, workspace_root, report)

    report.add_ok(f"video paths and MP4 headers verified: {len(resolved_videos)}/{len(df)}")
    report.add_ok(f"MP4 container frame counts verified: {container_counts_verified}/{len(df)}")
    if decode_videos:
        report.add_ok(f"OpenCV first-frame decode verified: {decoded}/{len(df)}")

    category_col = "Source_Category"
    if category_col in df.columns:
        categories = df[category_col].fillna("missing").astype(str).value_counts().to_dict()
        report.add_ok(f"source categories: {categories}")
    if "Repeated_Measure_Group" in df.columns:
        nvidia = df[df.get("Source_Category", "").astype(str) == "synthetic_renderer"]
        groups = nvidia["Repeated_Measure_Group"].dropna().astype(str).nunique()
        report.add_ok(f"NVIDIA repeated-measure design: {len(nvidia)} view rows in {groups} independent run group(s)")


def check_runtime_and_assets(args: argparse.Namespace, workspace_root: Path, report: Report) -> None:
    imported: list[str] = []
    for module_name in RUNTIME_MODULES:
        try:
            importlib.import_module(module_name)
            imported.append(module_name)
        except Exception as exc:
            report.error(f"Python dependency import failed: {module_name}: {type(exc).__name__}: {exc}")
    if imported:
        report.add_ok(f"runtime imports: {', '.join(imported)}")

    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            report.add_ok(f"CUDA available: {device_name}")
        else:
            report.error("CUDA is not available to PyTorch; SAM2 video inference requires a CUDA GPU")
    except Exception as exc:
        report.error(f"PyTorch CUDA check failed: {type(exc).__name__}: {exc}")

    sam2_repo = resolve_from_root(args.sam2_repo, workspace_root)
    model_weights = resolve_from_root(args.model_weights, workspace_root)
    yolo_weights = resolve_from_root(args.yolo_weights, workspace_root)
    sam2_script = workspace_root / "scripts" / "sam2" / "run_sam2_track.py"
    if not sam2_script.is_file():
        report.error(f"SAM2 runner missing: {sam2_script}")
    else:
        report.add_ok(f"SAM2 runner: {sam2_script}")
    if not sam2_repo.is_dir():
        report.error(f"SAM2 repository missing: {sam2_repo}")
    else:
        report.add_ok(f"SAM2 repository: {sam2_repo}")
        cfg_candidates = [sam2_repo / args.model_cfg, sam2_repo / "sam2" / args.model_cfg]
        if not any(path.is_file() for path in cfg_candidates):
            report.error(
                "SAM2 model config missing; searched: " + ", ".join(str(path) for path in cfg_candidates)
            )
        else:
            report.add_ok(f"SAM2 config: {next(path for path in cfg_candidates if path.is_file())}")
    if not model_weights.is_file() or model_weights.stat().st_size < 1_000_000:
        report.error(f"SAM2 checkpoint missing or implausibly small: {model_weights}")
    else:
        report.add_ok(f"SAM2 checkpoint: {model_weights} ({model_weights.stat().st_size / 2**20:.1f} MiB)")
    if not yolo_weights.is_file():
        report.warn(
            f"YOLO weights not found at {yolo_weights}; yolo_then_motion can fall back to motion, "
            "but Ultralytics may first attempt a network download"
        )
    else:
        report.add_ok(f"YOLO weights: {yolo_weights}")

    usage = shutil.disk_usage(workspace_root)
    free_gib = usage.free / 2**30
    if free_gib < 5:
        report.warn(f"only {free_gib:.1f} GiB free; masks, frames, and visualizations can be large")
    else:
        report.add_ok(f"free disk space: {free_gib:.1f} GiB")


def main() -> int:
    args = parse_args()
    workspace_root = Path(args.workspace_root).expanduser().resolve()
    metadata_path = resolve_from_root(args.metadata, workspace_root)
    video_root = resolve_from_root(args.video_root, workspace_root)
    report = Report()

    print(f"workspace_root={workspace_root}")
    print(f"metadata={metadata_path}")
    print(f"video_root={video_root}")
    print(f"mode={'data-only' if args.data_only else 'full'}\n")

    cv2_available = importlib.util.find_spec("cv2") is not None
    validate_metadata(
        metadata_path=metadata_path,
        video_root=video_root,
        workspace_root=workspace_root,
        report=report,
        decode_videos=cv2_available,
    )
    validate_video_manifest(metadata_path, video_root, report)
    if not cv2_available:
        message = "cv2 is unavailable locally; container headers passed but frames were not decoded"
        if args.data_only:
            report.warn(message)
        else:
            report.error(message)

    if not args.data_only:
        check_runtime_and_assets(args, workspace_root, report)

    report.emit()
    if args.json_output:
        json_path = resolve_from_root(args.json_output, workspace_root)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Report JSON: {json_path}")
    return 0 if not report.errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
