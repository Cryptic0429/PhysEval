# Scoring Compact Batch Results

This document explains how to score compact per-video results produced by:

```bash
python scripts/run_batch_simple.py
```

The scoring script reads:

```text
batch_eval_results/<metric>/<prompt_id>/result.json
```

and writes model-level / metric-level score reports.

## Basic Command

Score all metrics under `batch_eval_results`:

```bash
cd /path/to/workspace
python scripts/score_results.py --result-root batch_eval_results --model-name my_model
```

Run with a more relaxed score scale:

```bash
python scripts/score_results.py \
  --result-root batch_eval_results \
  --model-name my_model_relaxed \
  --tau 0.40 \
  --weak-valid-multiplier 0.8
```

Here `--tau` relaxes the physical-error curve, and `--weak-valid-multiplier` changes the evaluator-validity penalty for `weak_valid`.

Score one metric only:

```bash
python scripts/score_results.py \
  --result-root batch_eval_results \
  --metric friction \
  --model-name my_model
```

Default outputs:

```text
batch_eval_results/score_reports/all_metrics/
  score_summary.json
  metric_scores.csv
  per_video_scores.csv
  score_exclusion_reasons.csv
  plots/
    metric_scores.png
    metric_rates.png
```

For one metric, outputs go to:

```text
batch_eval_results/score_reports/<metric>/
```

## Score Formula

Each video has a target physical value `y` and a measured value `y_hat`.

Relative error:

```text
relative_error = abs(y_hat - y) / max(abs(y), epsilon)
```

The base physical score, also written as `physical_accuracy_score`, is:

```text
physical_accuracy_score = 100 * exp(-relative_error / tau)
```

`tau` is the tolerance for that metric. Smaller `tau` makes the score stricter.

Default tolerances are intentionally relaxed from the earlier draft. Low original values are increased by `0.10`; higher original values are increased by `0.05`:

```text
velocity                       0.30
acceleration                   0.35
gravity                        0.30
friction                       0.35
friction_coefficient           0.35
restitution_coefficient        0.30
density_acceleration           0.35
density_from_initial_acceleration 0.35
viscosity                      0.40
fluid_viscosity                0.40
spring_constant                0.35
energy                         0.35
mechanical_energy_conservation 0.35
momentum_1d                    0.35
momentum_conservation_1d       0.35
```

You can override all tolerances with one value:

```bash
python scripts/score_results.py --tau 0.35
```

Or pass a JSON file:

```bash
python scripts/score_results.py --tolerance-json tolerances.json
```

Example `tolerances.json`:

```json
{
  "friction_coefficient": 0.35,
  "velocity": 0.30,
  "spring_constant": 0.35
}
```

## Three-Part Score

The script now uses a three-part score:

```text
final_score
= physical_accuracy_score
  * measurement_quality_multiplier
  * estimation_validity_multiplier
```

Where:

```text
physical_accuracy_score = 100 * exp(-relative_error / tau)
measurement_quality_multiplier = tracking_quality_multiplier * mask_quality_score
estimation_validity_multiplier = status multiplier from the evaluator
```

## Measurement Quality

Tracking and mask checks are grouped as Measurement Quality.

Tracking is a hard gate by default:

```text
tracking_quality_status = pass    -> tracking_quality_multiplier = 1.0
tracking_quality_status = flagged -> tracking_quality_multiplier = 0.0
```

Mask is continuous by default:

```text
mask_quality_score in [0, 1]
1 = stable and reliable mask
0 = severe mask instability
```

The default mapping is intentionally lenient. Mild mask variation stays close to `1`, while only severe instability is pushed close to `0`. This makes the score behave more like a reliability gate: most normal videos keep almost full credit, and a small number of clearly problematic videos receive strong penalties.

The mask score is computed from the worst mask-stability component among:

```text
area_ratio_p95_p05
area_cv
bbox_width_cv
bbox_height_cv
bbox_aspect_cv
max_log_jump
trend_scale_ratio
max_centroid_step_norm_by_diameter
translation_aligned_iou_median
middle_coverage
```

For example, high `area_cv`, high `area_ratio_p95_p05`, large log-area jumps, or low aligned IoU reduce the mask score gradually rather than using a single `pass/flagged` penalty.

Current relaxed mask thresholds:

```text
area_ratio_p95_p05              good <= 2.00, bad >= 4.00
area_cv                         good <= 0.25, bad >= 0.70
bbox_width_cv / bbox_height_cv  good <= 0.20, bad >= 0.60
bbox_aspect_cv                  good <= 0.20, bad >= 0.60
max_log_jump                    good <= 0.45, bad >= 1.20
trend_scale_ratio               good <= 1.60, bad >= 3.00
max_centroid_step_norm          good <= 2.50, bad >= 5.00
translation_aligned_iou_median  good >= 0.65, bad <= 0.30
middle_coverage                 good >= 0.70, bad <= 0.50
```

To use the old fixed mask penalty:

```bash
python scripts/score_results.py --mask-policy penalty --qc-flag-multiplier 0.7
```

To make mask flags hard failures:

```bash
python scripts/score_results.py --strict-qc
```

## Estimation Validity

The evaluator status adjusts the score:

```text
valid       multiplier 1.0
weak_valid  multiplier 0.7
invalid     multiplier 0.0
failed      multiplier 0.0
```

This layer answers whether the physical evaluator successfully produced a trustworthy measured value.
Tracking completeness, object count, and meaningful-motion checks are handled by the Tracking Quality layer, not by the physics evaluator status. Physics status is reserved for fit quality, event detection, calibration/scale availability, and metric-specific measurement validity.

## Effective Videos

The main effective-video rule is:

```text
effective_for_score = true
only if:
  tracking passes under the selected tracking policy
  physics status is valid or weak_valid
```

The scorer applies this in order. Tracking quality is the first gate. If tracking fails, the video is excluded and physics scoring, mask-quality scoring, and physical-accuracy scoring are skipped for that video.

Only after tracking passes does the scorer evaluate physics status. For `valid` or `weak_valid` physics results, the evaluator is expected to provide either `Relative_Error` or enough measured/target values for scoring. If that contract is broken, the scorer stops by default instead of treating it as an ordinary invalid video.

So by default:

```text
tracking flagged      -> excluded / score 0
mask mildly unstable  -> still effective, but multiplied by mask_quality_score
mask severely unstable -> still effective, but multiplied by a low or zero mask_quality_score
physics failed        -> excluded / score 0
```

There should not be a normal exclusion category for "score unavailable".
If a video is physically fit as `valid` or `weak_valid`, the evaluator is required to provide a computable score:

```text
physics status is valid/weak_valid
=> Relative_Error exists, or Measured_Value/Target_Value can produce one
```

If this contract is violated, the scoring script stops by default because the evaluator output is malformed.
For debugging old outputs only, `--allow-contract-errors` converts this to `evaluator_contract_error`.
Normal `invalid` or `failed` physics outputs are counted as `physics_status_not_valid`.
For conservation metrics, the scorer also accepts evaluator outputs that directly use `relative_error` as the measured unit. Mechanical-energy conservation can fall back to a default conservation-ratio target of `1.0` when needed.

Final adjusted score:

```text
adjusted_score = physical_accuracy_score * estimation_validity_multiplier * measurement_quality_multiplier
```

To change only the `weak_valid` multiplier:

```bash
python scripts/score_results.py --weak-valid-multiplier 0.8
```

To make tracking flags a soft penalty instead:

```bash
python scripts/score_results.py --tracking-policy penalty --quality-flag-multiplier 0.5
```

## Tracking Quality Gates

The scoring script also reads each video's `tracking_points.json` and applies two quality gates.
For new batch outputs, these gates are already written by `run_video_batch.py` into:

```text
tracking_points.json -> quality_checks.tracking_quality
result.json          -> tracking_quality
```

The scoring script reads those saved markers first. For older outputs without the markers, it recomputes the same checks from `tracking_points.json`.

### Object Count Gate

The expected number of tracked objects is read from metadata:

```text
Num_Objects
```

If it is missing, the default expected count is `1`.

The script counts object IDs that are valid in at least `--min-object-coverage` fraction of frames. Default:

```text
min_object_coverage = 0.50
```

If the actual count differs from the expected count, the video is flagged:

```text
object_count_mismatch
```

This catches cases where a one-object task tracks multiple objects, or a two-object task loses one object.

### Meaningful Motion Gate

The script computes the primary object's motion extent:

```text
motion_extent_px = diagonal of the centroid trajectory bounding box
motion_extent_norm_by_diameter = motion_extent_px / median_object_diameter_px
```

If the normalized motion extent is too small, the video is flagged:

```text
no_meaningful_motion
```

Default threshold:

```text
min_motion_extent_ratio = 0.75
```

This catches videos where the ball is almost static and only jitters slightly.

Change the threshold:

```bash
python scripts/score_results.py --min-motion-extent-ratio 1.0
```

By default, these tracking-quality flags are hard failures:

```text
tracking_quality_multiplier = 0.0
```

So the adjusted score becomes zero even if a numerical fit was produced.

## Valid-Only vs End-to-End

The script reports two score families.

`valid_only_score` / `effective_video_score`:

```text
Mean adjusted score over effective videos.
```

This answers:

```text
When the video is valid and trackable under the protocol, how physically accurate is it?
```

`end_to_end_score`:

```text
Mean adjusted score over all videos, with invalid/failed videos counted as zero.
```

This answers:

```text
How reliably does the model produce videos that are both measurable and physically correct?
```

For model comparison, report both.

## Simple Examples

Assume velocity has:

```text
tau = 0.30
target = 1.0
measured = 1.1
relative_error = 0.10
physical_accuracy_score = 100 * exp(-0.10 / 0.30) = 71.65
```

Example 1: clean valid video:

```text
status = valid                  -> 1.0
tracking_quality_status = pass  -> 1.0
mask_quality_score = 1.0
adjusted_score = 71.65 * 1.0 * 1.0 * 1.0 = 71.65
effective_for_score = true
```

Example 2: weak valid video with moderately unstable mask:

```text
status = weak_valid             -> 0.7
tracking_quality_status = pass  -> 1.0
mask_quality_score = 0.6
adjusted_score = 71.65 * 0.7 * 1.0 * 0.6 = 30.09
effective_for_score = true
```

Example 3: physics valid but object almost static:

```text
status = valid                       -> 1.0
tracking_quality_status = flagged    -> 0.0
tracking_quality_reasons = no_meaningful_motion
adjusted_score = 0
effective_for_score = false
```

Example 4: tracking pass but physics fit failed:

```text
status = invalid                -> 0.0
tracking_quality_status = pass  -> 1.0
adjusted_score = 0
effective_for_score = false
```

## Suggested Tables and Figures

### Table 1: Overall Model Score Table

Columns:

```text
Model
Overall valid-only score
Overall end-to-end score
Valid rate
QC pass rate
Mean mask quality
Median relative error
```

Meaning:

```text
This is the main leaderboard. It separates physical accuracy on measurable videos from end-to-end robustness.
```

How it is computed:

```text
Overall valid-only score = average of per-metric valid_only_score
Overall end-to-end score = average of per-metric end_to_end_score
Valid rate = valid_or_weak / total
QC pass rate = strict mask pass / total
Mean mask quality = average continuous mask_quality_score
Median relative error = median over valid/weak_valid videos
```

### Table 2: Metric-Level Score Table

Columns:

```text
Metric
N total
Physics valid rate
Tracking pass rate
Mask-QC pass rate
Mean mask quality
Effective video rate
Effective video score
End-to-end score
Median relative error
Status counts
```

Meaning:

```text
Shows which physical tasks are easy or hard for the model.
```

The script writes this table as:

```text
metric_scores.csv
```

### Figure 1: Metric Scores Bar Chart

File:

```text
plots/metric_scores.png
```

Meaning:

```text
Compares effective-only and end-to-end score for each metric.
```

Interpretation:

```text
Large gap between valid-only and end-to-end means the model may be physically accurate when measurable, but often fails tracking/QC/generation validity.
```

### Figure 2: Metric Rates Bar Chart

File:

```text
plots/metric_rates.png
```

Meaning:

```text
Compares physics-valid rate, tracking-pass rate, and final effective-video rate for each metric.
```

### Optional Figure 5: Per-Video Failure Case Grid

This is not implemented in the first scoring script, but is recommended later.

Inputs:

```text
batch_eval_results/<metric>/<prompt_id>/plots/obj_000001_mask_area.png
batch_eval_results/<metric>/<prompt_id>/plots/trajectory.png
batch_eval_results/<metric>/<prompt_id>/result.json
```

Meaning:

```text
Shows representative failures: wrong tracking, scale drift, unstable mask, poor physical fit, or large relative error.
```

This is useful for paper qualitative analysis.

## Output Files

`score_summary.json` contains:

```text
model_name
result_root
scoring settings
overall scores
metric summaries
per-video rows
```

`metric_scores.csv` contains one row per metric.

`per_video_scores.csv` contains one row per video:

```text
metric
prompt_id
target_value
measured_value
relative_error
tau
base_score
raw_score
status
physics_valid
status_multiplier
estimation_validity_multiplier
mask_qc_status
mask_qc_pass
mask_quality_score
mask_qc_multiplier  # alias of mask_quality_score
tracking_quality_status
tracking_quality_pass
tracking_quality_multiplier
measurement_quality_status
measurement_quality_multiplier
effective_for_score
expected_objects
actual_objects
motion_extent_norm_by_diameter
adjusted_score
failure_reason
score_exclusion_reasons
mask_qc_reasons
mask_quality_components_json
tracking_quality_reasons
```

## Future Multi-Model Scoring

For multiple models, keep one result root per model, for example:

```text
outputs/
  model_a/
    batch_eval_results/
  model_b/
    batch_eval_results/
```

Then run:

```bash
python scripts/score_results.py --result-root outputs/model_a/batch_eval_results --model-name model_a
python scripts/score_results.py --result-root outputs/model_b/batch_eval_results --model-name model_b
```

A future script can merge multiple `score_summary.json` files into:

```text
overall leaderboard table
model-by-metric heatmap
failure-mode comparison table
```
