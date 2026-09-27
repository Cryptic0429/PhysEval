#!/usr/bin/env python3
"""Combine per-model comparison reports with paired, model-stratified inference."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path


BASELINE = "yolo_sam2"
ALTERNATIVES = ("fasterrcnn_sam2", "yolo_tam")
LABELS = {
    "yolo_sam2": "YOLOv8n + SAM2",
    "fasterrcnn_sam2": "Faster R-CNN + SAM2",
    "yolo_tam": "YOLOv8n + TAM (SAM1 + XMem)",
}


def as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def as_float(value: str) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * p
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def exact_two_sided_binomial(k: int, n: int) -> float:
    """Exact two-sided p-value for Binomial(n, .5), used by McNemar's test."""
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def load_runs(results_root: Path, models: list[str]):
    records: dict[str, dict[tuple[str, str], dict[str, dict]]] = {}
    for model in models:
        path = results_root / model / "per_run_results.csv"
        model_records: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (row["metric"], row["prompt_id"])
                model_records[key][row["mode"]] = row
        records[model] = model_records
    return records


def build_pairs(records, models: list[str], alternative: str):
    strata = {}
    for model in models:
        rows = []
        for key, modes in records[model].items():
            if BASELINE not in modes or alternative not in modes:
                continue
            baseline = modes[BASELINE]
            alt = modes[alternative]
            baseline_pass = as_bool(baseline["tracking_pass"]) if as_bool(baseline["success"]) else False
            alt_pass = as_bool(alt["tracking_pass"]) if as_bool(alt["success"]) else False
            baseline_q = as_float(baseline["mask_quality_score"]) if as_bool(baseline["success"]) else None
            alt_q = as_float(alt["mask_quality_score"]) if as_bool(alt["success"]) else None
            rows.append(
                {
                    "pass_diff": float(alt_pass) - float(baseline_pass),
                    "mask_diff": None if baseline_q is None or alt_q is None else alt_q - baseline_q,
                    "alt_pass": alt_pass,
                    "baseline_pass": baseline_pass,
                }
            )
        strata[model] = rows
    return strata


def stratified_bootstrap(strata, field: str, iterations: int, seed: int):
    rng = random.Random(seed)
    observed_by_model = []
    counts = {}
    for model, rows in strata.items():
        values = [row[field] for row in rows if row[field] is not None]
        counts[model] = len(values)
        observed_by_model.append(sum(values) / len(values))
    observed = sum(observed_by_model) / len(observed_by_model)
    samples = []
    for _ in range(iterations):
        model_means = []
        for rows in strata.values():
            values = [row[field] for row in rows if row[field] is not None]
            model_means.append(sum(rng.choice(values) for _ in values) / len(values))
        samples.append(sum(model_means) / len(model_means))
    return observed, percentile(samples, 0.025), percentile(samples, 0.975), counts


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()

    records = load_runs(args.results_root, args.models)
    output_rows = []
    p_values = []
    for index, alternative in enumerate(ALTERNATIVES):
        strata = build_pairs(records, args.models, alternative)
        pass_est, pass_low, pass_high, pass_n = stratified_bootstrap(
            strata, "pass_diff", args.bootstrap, args.seed + index
        )
        mask_est, mask_low, mask_high, mask_n = stratified_bootstrap(
            strata, "mask_diff", args.bootstrap, args.seed + 100 + index
        )
        alt_only = sum(
            row["alt_pass"] and not row["baseline_pass"]
            for rows in strata.values() for row in rows
        )
        baseline_only = sum(
            row["baseline_pass"] and not row["alt_pass"]
            for rows in strata.values() for row in rows
        )
        p_value = exact_two_sided_binomial(alt_only, alt_only + baseline_only)
        p_values.append(p_value)
        output_rows.append(
            {
                "alternative_mode": alternative,
                "baseline_mode": BASELINE,
                "n_paired_total": sum(pass_n.values()),
                "paired_n_by_model": json.dumps(pass_n, ensure_ascii=False, sort_keys=True),
                "mean_tracking_pass_difference": pass_est,
                "tracking_pass_bootstrap_low": pass_low,
                "tracking_pass_bootstrap_high": pass_high,
                "discordant_alt_only": alt_only,
                "discordant_baseline_only": baseline_only,
                "mcnemar_exact_p": p_value,
                "mask_quality_common_success_n": sum(mask_n.values()),
                "mask_quality_n_by_model": json.dumps(mask_n, ensure_ascii=False, sort_keys=True),
                "mean_mask_quality_difference": mask_est,
                "mask_quality_bootstrap_low": mask_low,
                "mask_quality_bootstrap_high": mask_high,
            }
        )

    adjusted = holm_adjust(p_values)
    for row, value in zip(output_rows, adjusted):
        row["mcnemar_holm_p"] = value

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "cross_model_pairwise.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    report = [
        "# Cross-model paired comparison",
        "",
        f"Models: {', '.join(args.models)}; paired videos: {output_rows[0]['n_paired_total']}; "
        f"stratified bootstrap: {args.bootstrap} resamples (seed {args.seed}).",
        "",
        "Primary outcome: tracking-pass difference (alternative minus YOLOv8n+SAM2), with failed runs retained as failures. "
        "Each bootstrap resamples videos within each generation model and gives all generation models equal weight. "
        "McNemar exact p-values are Holm-corrected across the two planned comparisons.",
        "",
        "Secondary outcome: mask-quality difference among paired runs where both methods succeeded; it is not used to hide failures.",
        "",
        "| Alternative | Tracking-pass difference (95% CI) | Holm p | Mask-quality difference (95% CI) |",
        "|---|---:|---:|---:|",
    ]
    for row in output_rows:
        report.append(
            f"| {LABELS[row['alternative_mode']]} | "
            f"{row['mean_tracking_pass_difference']:+.3f} "
            f"[{row['tracking_pass_bootstrap_low']:+.3f}, {row['tracking_pass_bootstrap_high']:+.3f}] | "
            f"{row['mcnemar_holm_p']:.4g} | "
            f"{row['mean_mask_quality_difference']:+.3f} "
            f"[{row['mask_quality_bootstrap_low']:+.3f}, {row['mask_quality_bootstrap_high']:+.3f}] |"
        )
    report.extend(
        [
            "",
            "Interpretation is based on effect sizes and confidence intervals. A confidence interval crossing zero is treated as "
            "insufficient evidence of improvement, not proof of exact equivalence.",
        ]
    )
    (args.output_dir / "cross_model_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
