"""Diagnostic and score report presentation, separate from numeric algorithms."""
from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .quality.mask_evidence import AreaSeries
from .io.tracks import normalize_tracking_json

LOGGER = logging.getLogger("physics_eval")

def write_score_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_reason_csv(path: Path, name: str, counts: dict[str, int]) -> None:
    rows = [{"reason_type": name, "reason": reason, "count": count}
            for reason, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    write_score_csv(path, rows)


def plot_bar(path: Path, labels: list[str], values: list[float], title: str, ylabel: str,
             color: str = "#64748b", ylim: tuple[float, float] | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(5, 0.75 * len(labels)), 4))
    ax.bar(labels, values, color=color)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.tick_params(axis="x", rotation=35)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_discard_overlap(plots_dir: Path, summary: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rates = summary.get("discard_overlap_rates") or {}
    counts = summary.get("discard_overlap_counts") or {}
    keys = ["tracking_only", "physics_only", "both_tracking_and_physics"]
    labels = ["Tracking only", "Physics only", "Both"]
    values = [float(rates.get(key, 0.0)) * 100.0 for key in keys]

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#3b82f6", "#f59e0b", "#ef4444"]
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0, max(100.0, max(values, default=0.0) * 1.15))
    ax.set_ylabel("rate (%)")
    ax.set_title("Discard diagnostic overlap")
    for bar, key, value in zip(bars, keys, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}%\n(n={int(counts.get(key, 0))})",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(plots_dir / "discard_overlap_overall.png", dpi=150)
    plt.close(fig)


def plot_reports(out_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for old_plot in plots_dir.glob("*.png"):
        old_plot.unlink()

    metrics = [item["metric"] for item in summary["metrics"]]
    e2e = [item["end_to_end_score"] or 0.0 for item in summary["metrics"]]
    effective = [item["effective_video_score"] or 0.0 for item in summary["metrics"]]
    if metrics:
        fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(metrics)), 4))
        x = list(range(len(metrics)))
        ax.bar([i - 0.18 for i in x], effective, width=0.36, label="effective-only")
        ax.bar([i + 0.18 for i in x], e2e, width=0.36, label="end-to-end")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=35, ha="right")
        ax.set_ylim(0, 100)
        ax.set_ylabel("score")
        ax.set_title("Scores by metric")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(plots_dir / "metric_scores.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(metrics)), 4))
        x = list(range(len(metrics)))
        for offset, key, label in [
            (-0.30, "physics_status_pass_rate", "physics status pass"),
            (-0.10, "tracking_quality_pass_rate", "tracking pass"),
            (0.10, "both_gates_pass_rate", "both gates pass"),
            (0.30, "effective_video_rate", "effective"),
        ]:
            ax.bar([i + offset for i in x], [m[key] * 100 for m in summary["metrics"]],
                   width=0.18, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=35, ha="right")
        ax.set_ylim(0, 100)
        ax.set_ylabel("rate (%)")
        ax.set_title("Gate pass and effective-video rates by metric")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(plots_dir / "metric_rates.png", dpi=150)
        plt.close(fig)

    plot_discard_overlap(plots_dir, summary)


def write_csv(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {path}. Use --overwrite to replace.")
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_plot(series: AreaSeries, summary: dict[str, Any], frame_rows: list[dict[str, Any]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    frames = [r["frame_idx"] for r in frame_rows]
    areas = [np.nan if r["area"] is None else float(r["area"]) for r in frame_rows]
    middle_frames = [r["frame_idx"] for r in frame_rows if r["in_middle_window"]]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(frames, areas, color="#2563eb", linewidth=1.8, label="mask area")
    if middle_frames:
        ax.axvspan(middle_frames[0], middle_frames[-1], color="#f59e0b", alpha=0.15, label="middle window")
    if math.isfinite(float(summary.get("median_area", float("nan")))):
        ax.axhline(float(summary["median_area"]), color="#111827", linestyle="--", linewidth=1.0, label="middle median")
    title_obj = "" if series.object_id is None else f" obj={series.object_id}"
    ax.set_title(f"{series.scene}{title_obj} area stability: {summary['status']} / {summary['decision']}")
    ax.set_xlabel("frame")
    ax.set_ylabel("mask area (px)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


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


# Tracking overlays retain the existing OpenCV raster rendering.


def draw_tracking_result(frame, cx, cy, obj_id, color=(0, 255, 0)):
    import cv2
    cx_i, cy_i = int(round(cx)), int(round(cy))
    cv2.circle(frame, (cx_i, cy_i), 5, color, -1)
    cv2.putText(
        frame,
        f"ID:{obj_id} ({cx_i},{cy_i})",
        (cx_i + 8, cy_i - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def alpha_blend_mask(frame_bgr: np.ndarray, mask_bool: np.ndarray, color_bgr=(0, 255, 0), alpha=0.4):
    import cv2
    if mask_bool.ndim != 2:
        return
    if mask_bool.shape[:2] != frame_bgr.shape[:2]:
        return

    frame_f = frame_bgr.astype(np.float32)
    color = np.array(color_bgr, dtype=np.float32).reshape(1, 1, 3)
    frame_f[mask_bool] = frame_f[mask_bool] * (1.0 - alpha) + color * alpha
    frame_bgr[:] = np.clip(frame_f, 0, 255).astype(np.uint8)


def save_trajectory_plot(row_dir: Path, tracking_json: Path, fps: float | None) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tracking_df = normalize_tracking_json(tracking_json, fps_override=fps)
        plot_dir = row_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_dir / "trajectory.png"

        fig, ax = plt.subplots(figsize=(6, 4))
        for obj_id, grp in tracking_df.groupby("object_id"):
            ax.plot(grp["cx_px"], grp["cy_px"], marker="o", markersize=2, linewidth=1, label=f"obj {obj_id}")
        ax.invert_yaxis()
        ax.set_xlabel("cx_px")
        ax.set_ylabel("cy_px")
        ax.set_title("tracked trajectory")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        return plot_path
    except Exception as exc:
        LOGGER.warning("compact trajectory plot failed for %s: %s", tracking_json, exc)
        return None
