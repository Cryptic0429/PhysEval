#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from physeval_compare.io_utils import load_metadata
from physeval_compare.runner import safe_name, select_shard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safely remove selected shard outputs before a clean rerun."
    )
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--detector", default=None)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--shard-index", type=int, nargs="+", required=True)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def ensure_child(path: Path, root: Path) -> Path:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    if resolved_path == resolved_root or resolved_root not in resolved_path.parents:
        raise RuntimeError(f"Refusing path outside expected root: {resolved_path}")
    return resolved_path


def main() -> int:
    args = parse_args()
    metadata_path = Path(args.metadata).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    model_root = output_root / safe_name(args.model_name, "model")
    mode_root = model_root / args.mode
    prompt_root = model_root / "_prompts" / args.detector if args.detector else None
    rows = load_metadata(metadata_path, Path("."))

    prompt_ids: set[str] = set()
    result_dirs: set[Path] = set()
    for shard_index in args.shard_index:
        for row in select_shard(rows, args.num_shards, shard_index):
            prompt_id = safe_name(row.get("Prompt_ID"), "unknown")
            metric = safe_name(row.get("Metric"), "tracking_only")
            prompt_ids.add(prompt_id)
            candidate = mode_root / metric / prompt_id
            if (candidate / "result.json").exists():
                result_dirs.add(ensure_child(candidate, mode_root))

    prompt_files: set[Path] = set()
    if prompt_root is not None:
        for prompt_id in prompt_ids:
            candidate = prompt_root / f"{prompt_id}.json"
            if candidate.exists():
                prompt_files.add(ensure_child(candidate, prompt_root))

    action = "remove" if args.apply else "would remove"
    print(f"{action}: result_dirs={len(result_dirs)}, prompt_files={len(prompt_files)}")
    if not args.apply:
        print("Dry-run only. Re-run with --apply after reviewing the counts.")
        return 0

    for path in sorted(result_dirs):
        shutil.rmtree(path)
    for path in sorted(prompt_files):
        path.unlink()
    print("Cleanup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
