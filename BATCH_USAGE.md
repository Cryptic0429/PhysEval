# Batch Evaluation Usage

This workspace is intended to run video tracking, mask quality control, and physics evaluation in batch.

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

## Running One Model With the 500-Video Metadata

For `t2v_physics_50_per_indicator_metadata_v2_earth_b2.xlsx`, put the metadata and videos under the workspace like this:

```text
data/
  metadata/
    t2v_physics_50_per_indicator_metadata_v2_earth_b2.xlsx
  t2v_videos/
    V001.mp4
    V002.mp4
    ...
    P050.mp4
```

Video order in the folder does not matter. The batch script looks up each file by the exact value in the Excel `Video_File` column. On Linux servers, names are case-sensitive, so `V001.mp4` and `v001.mp4` are different files.

If you are comparing multiple text-to-video models, keep one output directory per model:

```bash
python scripts/run_batch_simple.py \
  --metadata data/metadata/t2v_physics_50_per_indicator_metadata_v2_earth_b2.xlsx \
  --video-root data/t2v_videos/model_a \
  --output-dir batch_eval_results/model_a
```

After the batch run, score that model with:

```bash
python scripts/score_results.py --result-root batch_eval_results/model_a --model-name model_a
```

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

## Result Files for Future Scoring

Future model-level scoring scripts should read:

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
python scripts/score_results.py --result-root batch_eval_results --model-name my_model
```

Detailed scoring rules are documented in:

```text
SCORING_USAGE.md
```
