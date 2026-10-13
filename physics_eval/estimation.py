"""Physical estimation dispatch for normalized or saved tracking data."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .evaluators import EVALUATOR_REGISTRY
from .io.metadata import read_metadata, row_to_dict
from .io.results import failure_result, write_results
from .io.tracks import normalize_tracking_json, resolve_tracking_json
from .reporting import save_diagnostics

LOGGER = logging.getLogger("physics_eval")

def filter_metadata(df: pd.DataFrame, index: int | None, prompt_id: str | None) -> pd.DataFrame:
    if index is not None:
        df = df[df["Index"] == index]
    if prompt_id is not None:
        df = df[df["Prompt_ID"].astype(str) == str(prompt_id)]
    return df.reset_index(drop=True)


def process_row(
    row: dict[str, Any],
    metadata_path: Path,
    json_root: Path | None,
    output_dir: Path,
    fps: float | None,
    save_diag: bool,
    strict: bool,
) -> dict[str, Any]:
    evaluator_name = str(row.get("Evaluator") or "")
    evaluator = EVALUATOR_REGISTRY.get(evaluator_name)
    if evaluator is None:
        return failure_result(row, f"unknown_evaluator:{evaluator_name}")

    try:
        json_path = resolve_tracking_json(row, metadata_path, json_root)
        tracking_df = normalize_tracking_json(json_path, fps_override=fps)
        result = evaluator(tracking_df, row)
        result["Tracking_JSON"] = str(json_path)
        if strict and result.get("Status") in {"weak_valid", "invalid"}:
            result["Status"] = "invalid"
            if not result.get("Failure_Reason"):
                result["Failure_Reason"] = "strict_quality_check"
        if save_diag:
            save_diagnostics(output_dir, row, tracking_df, result)
        return result
    except Exception as exc:
        LOGGER.error("row failed: Index=%s Prompt_ID=%s reason=%s", row.get("Index"), row.get("Prompt_ID"), exc)
        return failure_result(row, "tracking_missing" if isinstance(exc, FileNotFoundError) else "failed", {"exception": str(exc)})


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=getattr(logging, args.log_level), format="[%(levelname)s] %(message)s")

    metadata_path = Path(args.metadata).expanduser().resolve()
    json_root = Path(args.json_root).expanduser().resolve() if args.json_root else None
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = filter_metadata(read_metadata(metadata_path), args.index, args.prompt_id)
    LOGGER.info("processing %d metadata row(s)", len(df))

    results: list[dict[str, Any]] = []
    for _, pd_row in df.iterrows():
        row = row_to_dict(pd_row)
        LOGGER.info("Index=%s Prompt_ID=%s Evaluator=%s", row.get("Index"), row.get("Prompt_ID"), row.get("Evaluator"))
        result = process_row(
            row=row,
            metadata_path=metadata_path,
            json_root=json_root,
            output_dir=output_dir,
            fps=args.fps,
            save_diag=args.save_diagnostics,
            strict=args.strict,
        )
        results.append(result)

    write_results(results, output_dir)
    LOGGER.info("wrote %s and %s", output_dir / "results.csv", output_dir / "results.json")
