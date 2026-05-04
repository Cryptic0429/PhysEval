from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from physics_eval.evaluators import EVALUATOR_REGISTRY
from physics_eval.utils.io import (
    failure_result,
    read_metadata,
    resolve_tracking_json,
    row_to_dict,
    write_results,
)
from physics_eval.utils.tracking import normalize_tracking_json


LOGGER = logging.getLogger("physics_eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch physics evaluation from metadata Excel and SAM2 tracking JSON.")
    parser.add_argument("--metadata", required=True, help="Path to metadata .xlsx")
    parser.add_argument("--json-root", default=None, help="Root directory for Tracking_JSON paths")
    parser.add_argument("--output-dir", required=True, help="Directory for results.csv/results.json")
    parser.add_argument("--index", type=int, default=None, help="Only process a single Index")
    parser.add_argument("--prompt-id", default=None, help="Only process a single Prompt_ID")
    parser.add_argument("--save-diagnostics", action="store_true", help="Save normalized tracks and plots")
    parser.add_argument("--fps", type=float, default=None, help="Fallback fps if tracking JSON has no fps/time")
    parser.add_argument("--strict", action="store_true", help="Mark weak/invalid quality as invalid")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def filter_metadata(df: pd.DataFrame, index: int | None, prompt_id: str | None) -> pd.DataFrame:
    if index is not None:
        df = df[df["Index"] == index]
    if prompt_id is not None:
        df = df[df["Prompt_ID"].astype(str) == str(prompt_id)]
    return df.reset_index(drop=True)


def save_diagnostics(output_dir: Path, row: dict[str, Any], tracking_df: pd.DataFrame, result: dict[str, Any]) -> None:
    diag_dir = output_dir / "diagnostics" / f"{row.get('Index')}_{row.get('Prompt_ID')}"
    diag_dir.mkdir(parents=True, exist_ok=True)
    tracking_df.to_csv(diag_dir / "normalized_tracking.csv", index=False, encoding="utf-8-sig")

    extra = {}
    try:
        extra = json.loads(result.get("Extra_JSON") or "{}")
    except json.JSONDecodeError:
        extra = {"raw_extra": result.get("Extra_JSON")}
    fit_summary = {**result, "Extra": extra}
    (diag_dir / "fit_summary.json").write_text(json.dumps(fit_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        for obj_id, grp in tracking_df.groupby("object_id"):
            ax.plot(grp["cx_px"], grp["cy_px"], marker="o", markersize=2, linewidth=1, label=f"obj {obj_id}")
        ax.invert_yaxis()
        ax.set_xlabel("cx_px")
        ax.set_ylabel("cy_px")
        ax.set_title(f"{row.get('Index')} {row.get('Prompt_ID')}")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(diag_dir / "trajectory_plot.png", dpi=150)
        fig.savefig(diag_dir / "fit_plot.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        LOGGER.warning("diagnostic plot failed for %s: %s", row.get("Prompt_ID"), exc)


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


def main() -> None:
    args = parse_args()
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


if __name__ == "__main__":
    main()
