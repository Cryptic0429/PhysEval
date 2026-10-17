# Batch Evaluation Usage

[Project README](README.md) | [中文项目说明](README_zh-CN.md) | [Architecture](docs/ARCHITECTURE.md)

This guide covers the executable pipeline behind the evaluation protocol described in the PhysEval paper: object initialization, SAM2 mask propagation, tracking-quality gates, task-specific inverse-physics estimation, and compact per-video records.

The batch stage records measurability checks and estimates each video's physical quantity; model-level scoring applies the gates and is a separate step. Under the paper's default protocol, expected-object-count and meaningful-motion checks are hard tracking gates, evaluator status is the second hard gate, and mask reliability enters the score as a soft multiplier.

Use the unified public entry point from the repository root:

```bash
cd /path/to/PhysEval
python physeval.py eval
```

By default it uses:

```text
metadata    data/metadata/*.xlsx, if there is exactly one file
videos      data/t2v_videos
sam2 repo   repo/sam2
outputs     batch_eval_results
init policy YOLO first, motion fallback (fixed)
yolo model  yolov8n.pt, if it exists under workspace root
```

Metadata auto-discovery is convenient for local workspaces. For the released 500-prompt benchmark, pass the repository workbook explicitly:

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

These markers are written during batch processing and are later used by scoring. Mask PNGs remain under `sam2_tracks/<scene>/masks/` and are referenced from the tracking evidence; extracted frames are under `data/custom/<scene>/frames/`. Preserve these referenced assets when moving results or recomputing diagnostics.

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
python physeval.py eval \
  --metadata benchmark/metadata/phys_t2v_bench_metadata.xlsx \
  --video-root data/t2v_videos/model_a \
  --output-dir batch_eval_results/model_a
```

After the batch run, score that model with:

```bash
python physeval.py score \
  --result-root batch_eval_results/model_a \
  --model-name model_a \
  --weak-valid-multiplier 0.8
```

The explicit `0.8` matches the paper protocol and is also the CLI default; see [SCORING_USAGE.md](SCORING_USAGE.md).

## Common Commands

Run all rows in the metadata:

```bash
python physeval.py eval
```

Run one row by `Index`:

```bash
python physeval.py eval --index 1
```

Run one row by `Prompt_ID`:

```bash
python physeval.py eval --prompt-id F001
```

Reuse existing tracking outputs when possible:

```bash
python physeval.py eval --reuse-tracking
```

Evaluate existing tracking only, without running SAM2:

```bash
python physeval.py eval --eval-only
```

Track only, without physics evaluation:

```bash
python physeval.py eval --tracking-only
```

Save SAM2 visualization videos:

```bash
python physeval.py eval --vis
```

Write extra debug CSV files and diagnostic plots:

```bash
python physeval.py eval --diagnostics
```

Adjust the static-object quality gate:

```bash
python physeval.py eval --min-motion-extent-ratio 1.0
```

Adjust how many frames an object must appear in to count:

```bash
python physeval.py eval --min-object-coverage 0.6
```

## Fixed Initialization Protocol

Formal `eval` follows one automatic protocol:

```text
YOLO target candidates
  → accept when the usable prompt count matches Num_Objects
  → otherwise motion detection and object refinement
  → accept when motion provides the required count
  → SAM2 mask propagation
```

YOLO candidate generation applies the target-class, confidence, and geometry filters. A detector exception, empty usable candidates, or an incomplete object count triggers motion fallback. For example, a two-object task cannot proceed with only one YOLO prompt. If motion also fails to provide the required count, the sample is recorded as an initialization failure.

The scan defaults to up to 60 uniformly sampled frames, so initialization need not start at frame zero. YOLO provides initial prompts; it is not rerun to recover lost tracks. SAM2 performs video propagation. Formal evaluation always follows this fixed protocol.

The adopted source is saved as `auto_yolo` or `auto_motion`; motion candidates retain `yolo_fallback_reason`. Inspect `tracking_points.json` and `run_config.json` to distinguish attempted YOLO detection from adopted YOLO initialization. Keep these records when reporting how often each path was used.

Use a specific YOLO weights file when required by a declared experimental setup:

```bash
python physeval.py eval --yolo-weights yolov8n.pt
```

Manual prompt metadata is also supported. Renderer ground-truth initialization requires the explicit `--allow-renderer-gt-init` flag, records its supervision provenance, and must be reported separately from automatic evaluation. Reusing tracks or estimating from existing tracks does not rerun initialization.

## Mask QC Behavior

The public `eval` command enables saved masks and mask QC by default. It saves:

```text
batch_eval_results/<metric>/<prompt_id>/mask_qc/mask_qc.json
batch_eval_results/<metric>/<prompt_id>/plots/obj_000001_mask_area.png
```

If a mask QC issue is found, the default action is warning only. The physics evaluator still runs:

```bash
python physeval.py eval --area-action warn
```

To skip physics evaluation for videos flagged by mask QC:

```bash
python physeval.py eval --area-action skip
```

Disable mask QC plots:

```bash
python physeval.py eval --no-plots
```

## Metadata Requirements

The metadata Excel must contain the required columns checked by `physics_eval/io/metadata.py`, including:

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
F001.mp4
```

For two-object tasks, set:

```text
Num_Objects = 2
```

The current two-object evaluator is `eval_momentum_1d` for one-dimensional momentum conservation.

## Optional Check Commands

These commands are not required for every run. Use them after copying code to the server or after editing code.

Syntax check:

```bash
python -m compileall -q physics_eval experiments validation tools tests
```

Show available options:

```bash
python physeval.py eval --help
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
python physeval.py score \
  --result-root batch_eval_results \
  --model-name my_model \
  --weak-valid-multiplier 0.8
```

Detailed scoring rules are documented in:

```text
SCORING_USAGE.md
```
