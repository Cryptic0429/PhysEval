# Batch Evaluation Usage

[English](BATCH_USAGE.md) | [简体中文](BATCH_USAGE_zh-CN.md)

This guide covers the executable pipeline behind the evaluation protocol described in the PhysEval paper: object initialization, SAM2 mask propagation, tracking-quality gates, task-specific inverse-physics estimation, and compact per-video records.

The batch stage decides whether a video is measurable and estimates its physical quantity; model-level scoring is a separate step. Under the paper's default protocol, expected-object-count and meaningful-motion checks are hard tracking gates, evaluator status is the second hard gate, and mask reliability is retained as a soft score multiplier.

The recommended entrypoint is:

```bash
cd /path/to/workspace
python scripts/run_batch_simple.py
```

By default it uses:

```text
metadata    data/metadata/*.xlsx, if there is exactly one file
videos      data/t2v_videos
sam2 repo   repo/sam2
outputs     batch_eval_results
detector    yolo_then_motion
yolo model  yolov8n.pt, if it exists under workspace root
```

The automatic metadata default is retained for local workspaces. For the released 500-prompt benchmark, pass the repository workbook explicitly:

```text
benchmark/metadata/phys_t2v_bench_metadata.xlsx
```

Default output layout:

```text
batch_eval_results/
  <metric>/
    <prompt_id>/
      result.json
      tracking_points.json
      run_config.json
      mask_qc/
        mask_qc.json
      plots/
        obj_000001_mask_area.png
        trajectory.png
```

The file to inspect first is always:

```text
batch_eval_results/<metric>/<prompt_id>/result.json
```

The copied tracking file also contains quality markers:

```text
batch_eval_results/<metric>/<prompt_id>/tracking_points.json
  quality_checks.tracking_quality
  quality_checks.mask_qc
```

These markers are written during batch processing and are later used by scoring.

Each compact record preserves the benchmark specification, physics output, tracking status, mask diagnostics, target and measured values, relative error when available, and reason tokens. This is the auditable unit that is later aggregated into effective-video scores and discard diagnostics.

## Running One Model on the Released 500-Prompt Benchmark

The released benchmark contains 10 physical metrics with 50 prompts per metric. Generate one video per prompt, use the exact filename in the workbook's `Video_File` column, and organize one model as follows:

```text
benchmark/metadata/
  phys_t2v_bench_metadata.xlsx
data/t2v_videos/<model_name>/
  V001.mp4
  V002.mp4
  ...
  P050.mp4
```

Video order in the folder does not matter. The batch script looks up each file by the exact value in the Excel `Video_File` column. On Linux servers, names are case-sensitive, so `V001.mp4` and `v001.mp4` are different files.

If you are comparing multiple text-to-video models, keep one output directory per model:

```bash
python scripts/run_batch_simple.py \
  --metadata benchmark/metadata/phys_t2v_bench_metadata.xlsx \
  --video-root data/t2v_videos/model_a \
  --output-dir batch_eval_results/model_a
```

After the batch run, score that model with:

```bash
python scripts/score_results.py \
  --result-root batch_eval_results/model_a \
  --model-name model_a \
  --weak-valid-multiplier 0.8
```

The explicit `0.8` matches the paper protocol; see `SCORING_USAGE.md` for the current CLI-default compatibility note.

## Common Commands

Run all rows in the metadata:

```bash
python scripts/run_batch_simple.py
```

Run one row by `Index`:

```bash
python scripts/run_batch_simple.py --index 1
```

Run one row by `Prompt_ID`:

```bash
python scripts/run_batch_simple.py --prompt-id B3_1_friction_001
```

Reuse existing tracking outputs when possible:

```bash
python scripts/run_batch_simple.py --reuse-tracking
```

Evaluate existing tracking only, without running SAM2:

```bash
python scripts/run_batch_simple.py --eval-only
```

Track only, without physics evaluation:

```bash
python scripts/run_batch_simple.py --tracking-only
```

Save SAM2 visualization videos:

```bash
python scripts/run_batch_simple.py --vis
```

Write extra debug CSV files and diagnostic plots:

```bash
python scripts/run_batch_simple.py --diagnostics
```

Adjust the static-object quality gate:

```bash
python scripts/run_batch_simple.py --min-motion-extent-ratio 1.0
```

Adjust how many frames an object must appear in to count:

```bash
python scripts/run_batch_simple.py --min-object-coverage 0.6
```

## Detector Options

Default detector:

```bash
python scripts/run_batch_simple.py --detector yolo_then_motion
```

This tries YOLO first and falls back to motion-based initialization if YOLO fails.

Use motion only:

```bash
python scripts/run_batch_simple.py --detector motion
```

Use YOLO only:

```bash
python scripts/run_batch_simple.py --detector yolo
```

Use a specific YOLO weights file:

```bash
python scripts/run_batch_simple.py --yolo-weights yolov8n.pt
```

## Mask QC Behavior

The simple entrypoint enables mask QC by default. It saves:

```text
batch_eval_results/<metric>/<prompt_id>/mask_qc/mask_qc.json
batch_eval_results/<metric>/<prompt_id>/plots/obj_000001_mask_area.png
```

If a mask QC issue is found, the default action is warning only. The physics evaluator still runs:

```bash
python scripts/run_batch_simple.py --area-action warn
```

To skip physics evaluation for videos flagged by mask QC:

```bash
python scripts/run_batch_simple.py --area-action skip
```

Disable mask QC plots:

```bash
python scripts/run_batch_simple.py --no-plots
```

## Metadata Requirements

The metadata Excel must contain the required columns checked by `physics_eval/utils/io.py`, including:

```text
Index
Prompt_ID
Video_File
Tracking_JSON
Module
Metric
Evaluator
Measurement_Method
Scale_Mode
Target_Type
Target_Value
Target_Unit
Known_Parameters_JSON
```

For videos under `data/t2v_videos`, `Video_File` can be a relative path such as:

```text
B3_1_friction_001.mp4
```

For two-object tasks, set:

```text
Num_Objects = 2
```

The current two-object evaluator is mainly `eval_momentum_1d`.

## Optional Check Commands

These commands are not required for every run. Use them after copying code to the server or after editing code.

Syntax check:

```bash
python -m py_compile physics_eval/run_video_batch.py scripts/run_batch_simple.py scripts/quality/check_mask_area_stability.py
```

Show available options:

```bash
python scripts/run_batch_simple.py --help
```

## Result Files Used by Model-Level Scoring

The model-level scoring script reads:

```text
batch_eval_results/<metric>/*/result.json
```

Each `result.json` contains:

```text
metadata
physics_result
physics_extra
mask_qc
tracking_quality
artifacts
```

This keeps per-video data compact while preserving enough information for later aggregation.

After a batch run finishes, score the compact results with:

```bash
python scripts/score_results.py \
  --result-root batch_eval_results \
  --model-name my_model \
  --weak-valid-multiplier 0.8
```

Detailed scoring rules are documented in:

```text
SCORING_USAGE.md
```
