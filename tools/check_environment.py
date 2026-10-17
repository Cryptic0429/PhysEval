#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check PhysEval evaluation dependencies and local assets.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    return parser.parse_args(argv)


def package_version(distribution: str, module: object) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return str(getattr(module, "__version__", "unknown"))


def run(args: argparse.Namespace) -> int:
    root = Path(args.project_root).expanduser().resolve()
    missing = []
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    if sys.version_info < (3, 10):
        print("[warn] Python 3.10 or newer is required.")
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
        print("Run from the PhysEval project root: bash tools/setup_assets.sh --install-sam2")
    return 1 if missing or missing_assets else 0


def main(argv: list[str] | None = None):
    return run(parse_args(argv))

if __name__ == "__main__":
    raise SystemExit(main())
