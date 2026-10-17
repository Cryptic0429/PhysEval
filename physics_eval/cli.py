"""Public command-line boundary. Core components receive explicit configuration."""
from __future__ import annotations

import argparse
import importlib
import sys
from physics_eval.config import WORKSPACE_ROOT, normalize_eval_args

def build_eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end batch pipeline: read metadata Excel, track each video with SAM2 auto-init, "
            "then run physics evaluators."
        )
    )

    parser.add_argument("--metadata", default=None,
                        help="Path to metadata .xlsx. If omitted, use the only .xlsx under <workspace-root>/data/metadata")

    parser.add_argument("--video-root", default=None, help="Default: <workspace-root>/data/t2v_videos")

    parser.add_argument("--output-dir", default=None, help="Default: <workspace-root>/batch_eval_results")

    parser.add_argument("--output-layout", choices=["organized", "flat"], default="organized",
                        help="organized writes <output-dir>/<metric>/<prompt_id>/result.json. flat keeps the older shared folders.")

    parser.add_argument("--workspace-root", default=None, help="Default: repository/workspace root inferred from this file")

    parser.add_argument("--tracking-worker", dest="sam2_script", default=None, help="Optional custom tracking worker script; default uses the bundled module")

    parser.add_argument("--sam2-repo", default=None, help="Default: <workspace-root>/repo/sam2")

    parser.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_b+.yaml",
                        help="SAM2 model cfg")

    parser.add_argument("--model-weights", default=None,
                        help="Default: <sam2-repo>/checkpoints/sam2.1_hiera_base_plus.pt")

    parser.add_argument("--sam2-precision", default="auto", choices=["auto", "fp32", "fp16", "bf16"],
                        help="SAM2 inference precision passed to the tracker. Default: auto")

    parser.add_argument("--index", type=int, default=None, help="Only process a single Index")

    parser.add_argument("--prompt-id", default=None, help="Only process a single Prompt_ID")

    parser.add_argument("--fps", type=float, default=None, help="Fallback fps for evaluation if tracking JSON has no fps/time")

    parser.add_argument("--strict", action="store_true", help="Mark weak/invalid quality as invalid")

    parser.add_argument("--reuse-tracking", "--skip-tracking-if-exists", dest="skip_tracking_if_exists", action="store_true", help="Reuse output tracking JSON if already present")

    parser.add_argument("--tracking-only", action="store_true", help="Only run tracking, do not run physics evaluation")

    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation from existing output tracking JSON")

    parser.add_argument("--diagnostics", "--save-diagnostics", dest="save_diagnostics", action="store_true", help="Save normalized tracks and diagnostic summaries")

    parser.add_argument("--vis", "--save-vis-video", dest="save_vis_video", action="store_true", help="Save SAM2 tracking visualization video")

    parser.add_argument("--save-mask-png", action="store_true", help="Save SAM2 mask PNGs")

    parser.add_argument("--save-auto-init-debug", action="store_true", help="Save image showing the selected auto-init prompt")

    parser.add_argument("--copy-video-to-workspace", action="store_true")

    parser.add_argument(
        "--offload-video-to-cpu",
        action="store_true",
        help="Keep decoded SAM2 video frames on CPU; useful for very long clips.",
    )

    parser.add_argument("--debug", action="store_true",
                        help="Shortcut: save vis video, masks, auto-init debug image, diagnostics, and area plots")

    parser.add_argument("--auto-init-scan-frames", type=int, default=60)

    parser.add_argument("--auto-init-scan-mode", default="uniform", choices=["start", "middle", "uniform"],
                        help="Which part of each video to scan for auto-init prompts.")

    parser.add_argument("--yolo-weights", default=None,
                        help="YOLO weights, e.g. yolo11n.pt or custom .pt")

    parser.add_argument("--yolo-conf", type=float, default=0.25)

    parser.add_argument("--yolo-iou", type=float, default=0.50)

    parser.add_argument("--yolo-classes", default=None,
                        help="Comma-separated YOLO class names or ids to allow")

    parser.add_argument("--auto-init-threshold", type=float, default=25.0)

    parser.add_argument("--auto-init-min-area", type=int, default=40)

    parser.add_argument("--auto-init-max-area-ratio", type=float, default=0.20)

    parser.add_argument("--auto-init-min-fill-ratio", type=float, default=0.08)

    parser.add_argument("--auto-init-max-aspect-ratio", type=float, default=6.0)

    parser.add_argument("--auto-init-object-refine", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--auto-init-box-expand", type=float, default=2.8)

    parser.add_argument("--auto-init-target-hint", default="metadata",
                        help="Target hint passed to auto tracking. Use 'metadata' to read Calibration_Object.")

    parser.add_argument("--disable-auto-init-shadow-filter", action="store_true")

    parser.add_argument("--no-auto-init-negative-points", action="store_true")

    parser.add_argument("--no-auto-init-box", action="store_true")

    parser.add_argument("--no-bidirectional-propagation", action="store_true",
                        help="Only propagate forward from the selected init frame.")

    parser.add_argument(
        "--allow-renderer-gt-init",
        action="store_true",
        help=(
            "Allow rows with Tracking_Init_Mode=renderer_gt to derive one point/box from a "
            "predeclared first-visible or first-eligible renderer mask. Required explicitly to "
            "prevent accidental GT-assisted evaluation."
        ),
    )

    parser.add_argument("--area-stability-check", action="store_true",
                        help="After tracking, check middle-window mask area stability before evaluation")

    parser.add_argument("--area-check", dest="area_stability_check", action="store_true",
                        help="Short alias for --area-stability-check")

    parser.add_argument("--area-action", "--area-stability-action", dest="area_stability_action", choices=["warn", "skip"], default="warn",
                        help="warn keeps evaluating flagged videos; skip writes a failed result and skips evaluation. Default: warn")

    parser.add_argument("--area-stability-output-dir", default=None,
                        help="Area stability report dir. Default: <output-dir>/area_stability")

    parser.add_argument("--area-stability-save-plots", action="store_true",
                        help="Save area curve plots for each checked video/object")

    parser.add_argument("--area-prefer-mask-area", action="store_true",
                        help="If masks exist in <workspace-root>/sam2_tracks/<scene>/masks, count pixels from mask PNGs")

    parser.add_argument("--area-object-id", type=int, default=None,
                        help="Only check this object id for multi-object tracking JSONs")

    parser.add_argument("--area-middle-start", type=float, default=0.15,
                        help="Fraction trimmed from start for area check. Default: 0.15")

    parser.add_argument("--area-middle-end", type=float, default=0.85,
                        help="Fraction kept until this point for area check. Default: 0.85")

    parser.add_argument("--area-min-valid-frames", type=int, default=6)

    parser.add_argument("--area-min-coverage", type=float, default=0.70)

    parser.add_argument("--area-ratio-threshold", type=float, default=2.00,
                        help="Flag if middle p95(area)/p5(area) exceeds this value")

    parser.add_argument("--area-cv-threshold", type=float, default=0.25,
                        help="Flag if middle std(area)/mean(area) exceeds this value")

    parser.add_argument("--area-max-log-jump-threshold", type=float, default=0.45,
                        help="Flag if max adjacent abs diff(log(area)) exceeds this value")

    parser.add_argument("--area-smooth-scale-ratio-threshold", type=float, default=1.60,
                        help="Classify as possible depth motion if smooth trend changes by this ratio")

    parser.add_argument("--area-trend-r2-threshold", type=float, default=0.65,
                        help="Minimum log-area trend R2 for smooth scale-change classification")

    parser.add_argument("--area-bbox-size-cv-threshold", type=float, default=0.20,
                        help="Flag if middle-window bbox width/height CV exceeds this value")

    parser.add_argument("--area-bbox-aspect-cv-threshold", type=float, default=0.20,
                        help="Flag if middle-window bbox aspect-ratio CV exceeds this value")

    parser.add_argument("--area-aligned-iou-threshold", type=float, default=0.65,
                        help="Flag if median centroid-aligned mask IoU is below this value")

    parser.add_argument("--area-aligned-iou-sample-stride", type=int, default=5,
                        help="Frame stride for centroid-aligned mask IoU sampling")

    parser.add_argument("--area-max-centroid-step-ratio-threshold", type=float, default=2.50,
                        help="Flag if max centroid step divided by median equivalent diameter exceeds this value")

    parser.add_argument("--min-motion-extent-ratio", type=float, default=0.75,
                        help="Flag tracking as static if primary motion extent is below this many object diameters")

    parser.add_argument("--min-object-coverage", type=float, default=0.50,
                        help="Object must be valid in at least this fraction of frames to count")

    parser.add_argument("--python-executable", default=sys.executable, help="Python executable used to run SAM2 tracking")

    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    parser.add_argument("--no-plots", action="store_true", help="Skip mask diagnostic plots")
    return parser


def build_score_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate compact per-video result.json files into metric/model scores."
    )

    parser.add_argument("--result-root", default="batch_eval_results",
                        help="Root containing <metric>/<prompt_id>/result.json")

    parser.add_argument("--scoring-protocol", default="grouped_v2_candidate", help="Built-in Qmask protocol name or JSON path")

    parser.add_argument("--metadata", default=str(WORKSPACE_ROOT / "benchmark/metadata/csv/phys_t2v_bench_metadata__metadata.csv"), help="Expected benchmark CSV/XLSX; use none for explicitly partial/custom scoring")

    parser.add_argument("--metric", default=None,
                        help="Only score one metric directory, e.g. friction_coefficient")

    parser.add_argument("--model-name", default=None,
                        help="Name written into the score summary")

    parser.add_argument("--output-dir", default=None,
                        help="Default: <result-root>/score_reports[/<metric>]")

    parser.add_argument("--tau", type=float, default=None,
                        help="Override tolerance for all metrics")

    parser.add_argument("--tolerance-json", default=None,
                        help="Optional JSON object mapping metric names to tau values")

    parser.add_argument("--penalty-multiplier", type=float, default=None,
                        help="Convenience multiplier for weak_valid; also used by legacy --mask-policy penalty")

    parser.add_argument("--weak-valid-multiplier", type=float, default=0.8,
                        help="Multiplier for physics status weak_valid. Default: 0.8 (paper protocol)")

    parser.add_argument("--mask-policy", choices=["continuous", "penalty", "hard_fail", "ignore"], default="continuous",
                        help="How to handle mask-QC. Default: continuous 0-1 mask quality score")

    parser.add_argument("--qc-flag-multiplier", type=float, default=0.7,
                        help="Legacy multiplier for mask-QC flagged videos when --mask-policy penalty. Default: 0.7")

    parser.add_argument("--strict-qc", action="store_true",
                        help="Alias for --mask-policy hard_fail")

    parser.add_argument("--tracking-policy", choices=["hard_fail", "penalty", "ignore"], default="hard_fail",
                        help="How to handle tracking-quality flags. Default: hard_fail")

    parser.add_argument("--quality-flag-multiplier", type=float, default=0.0,
                        help="Multiplier for tracking-quality flags when --tracking-policy penalty. Default: 0.0")

    parser.add_argument("--min-motion-extent-ratio", type=float, default=0.75,
                        help="Flag if primary object motion extent is below this many object diameters. Default: 0.75")

    parser.add_argument("--min-object-coverage", type=float, default=0.50,
                        help="Object must be valid in at least this fraction of frames to count. Default: 0.50")

    parser.add_argument("--recompute-tracking-quality", action="store_true",
                        help=("Recompute tracking-quality status from tracking_points.json with the requested "
                              "thresholds instead of reusing an embedded status. Required for threshold sensitivity."))

    parser.add_argument("--allow-contract-errors", action="store_true",
                        help="Collect evaluator_contract_error rows into an incomplete report instead of failing at the first row; no formal aggregate is published")

    parser.add_argument("--no-plots", action="store_true",
                        help="Do not generate plots")

    return parser


def parse_eval_args(argv: list[str] | None = None) -> argparse.Namespace:
    return normalize_eval_args(build_eval_parser().parse_args(argv))


def parse_score_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_score_parser()
    args = parser.parse_args(argv)
    from physics_eval.scoring import prepare_args
    try:
        return prepare_args(args)
    except ValueError as exc:
        parser.error(str(exc))


EXTRA_COMMANDS = {
    "analyze-sensitivity": ("experiments.scoring_sensitivity", "main"),
    "analyze-mask-protocols": ("experiments.historical.qmask_protocol_comparison", "main"),
    "validate-masks": ("validation.synthetic_masks", "main"),
    "validate-open-video": ("validation.open_video_preflight", "main"),
}


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="PhysEval: evaluation, scoring, saved-evidence analysis and validation.")
    parser.add_argument("command", choices=["eval", "score", *EXTRA_COMMANDS])
    parser.add_argument("args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    command = parser.parse_args(argv)
    if command.command == "eval":
        args = parse_eval_args(command.args)
        from physics_eval.pipeline import run
        return run(args)
    if command.command == "score":
        args = parse_score_args(command.args)
        from physics_eval.scoring import run
        return run(args)
    module, function = EXTRA_COMMANDS[command.command]
    return getattr(importlib.import_module(module), function)(command.args)
