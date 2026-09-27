#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


MODULES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "opencv-python": "cv2",
    "openpyxl": "openpyxl",
    "ultralytics": "ultralytics",
    "torch": "torch",
    "torchvision": "torchvision",
    "tqdm": "tqdm",
    "hydra-core": "hydra",
    "iopath": "iopath",
    "pillow": "PIL",
    "PyYAML": "yaml",
    "progressbar2": "progressbar",
    "gdown": "gdown",
    "GitPython": "git",
    "hickle": "hickle",
    "psutil": "psutil",
    "tensorboard": "tensorboard",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check PhysEval comparison dependencies and local assets.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    return parser.parse_args()


def package_version(distribution: str, module: object) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return str(getattr(module, "__version__", "unknown"))


def main() -> int:
    args = parse_args()
    root = Path(args.project_root).expanduser().resolve()
    missing = []
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    if sys.version_info[:2] not in {(3, 10), (3, 11)}:
        print("[warn] Python 3.10 or 3.11 is recommended for SAM2 + Track-Anything compatibility.")
    for distribution, module_name in MODULES.items():
        try:
            module = importlib.import_module(module_name)
            print(f"[ok] {distribution}: {package_version(distribution, module)}")
        except Exception as exc:
            missing.append(distribution)
            print(f"[missing] {distribution}: {type(exc).__name__}: {exc}")

    try:
        import torch

        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA runtime: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except Exception:
        pass

    assets = {
        "YOLOv8n": root / "yolov8n.pt",
        "SAM2 repo": root / "repo" / "sam2",
        "SAM2 checkpoint": root / "repo" / "sam2" / "checkpoints" / "sam2.1_hiera_base_plus.pt",
        "Segment Anything repo": root / "repo" / "segment-anything",
        "SAM1 checkpoint": root / "repo" / "segment-anything" / "checkpoints" / "sam_vit_b_01ec64.pth",
        "Track-Anything repo": root / "repo" / "Track-Anything",
        "XMem checkpoint": root / "repo" / "Track-Anything" / "checkpoints" / "XMem.pth",
    }
    missing_assets = []
    for label, path in assets.items():
        if path.exists():
            print(f"[ok] {label}: {path}")
        else:
            missing_assets.append(label)
            print(f"[missing] {label}: {path}")

    if missing:
        print(f"Missing Python dependencies: {', '.join(missing)}")
    if missing_assets:
        print(f"Missing assets: {', '.join(missing_assets)}")
        print("Run from the PhysEval project root: bash compare/scripts/setup_compare_assets.sh --install-editable")
    return 1 if missing or missing_assets else 0


if __name__ == "__main__":
    raise SystemExit(main())
