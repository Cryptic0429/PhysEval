# Scoring Batch Results

This guide explains how PhysEval turns per-video `result.json` files into model and metric scores. Run the commands from the repository root. For the complete evaluation workflow, see [BATCH_USAGE.md](BATCH_USAGE.md).

## Run the scorer

The scorer reads `<result-root>/<metric>/<prompt_id>/result.json`. For example:

```bash
python physeval.py score \
  --result-root batch_eval_results/my_model \
  --model-name my_model
```

The default `weak_valid` weight is **0.8**, matching the paper. It does not need to be passed explicitly for ordinary scoring.

Useful options:

| Option | Purpose |
| --- | --- |
| `--metric friction_coefficient` | Score one metric instead of the full result root. |
| `--metadata path/to/metadata.csv` | Check result IDs against another CSV or XLSX benchmark file. |
| `--metadata none` | Allow an explicitly partial or exploratory result set. |
| `--output-dir path/to/report` | Choose a report directory. |
| `--no-plots` | Skip plot generation. |
| `--tau 0.35` or `--tolerance-json tolerances.json` | Override physical-error tolerances for a sensitivity analysis. |
| `--weak-valid-multiplier 0.6` | Change the weak-validity weight for a sensitivity analysis. |

The PhysEval scoring protocol defined in the paper uses continuous mask quality and treats failed tracking checks as hard exclusions. Threshold and policy options are listed by `python physeval.py score --help`; use supported overrides only for explicitly reported sensitivity analyses. The default grouped protocol requires continuous mask weighting and hard-fail tracking. Alternative legacy mask/tracking policies require an explicitly selected historical protocol and change the paper's scoring settings.

## Which videos receive a score?

The scorer applies checks in order:

1. **Tracking quality:** the expected object count, object coverage and meaningful motion must pass. The paper requires an object to be tracked in at least half the frames, and the primary object's motion to span at least 0.75 of its median diameter. A failed tracking check excludes the video.
2. **Physics validity:** the evaluator must return `valid` or `weak_valid`. `invalid` and `failed` videos are excluded.
3. **Mask quality:** after both hard checks pass, mask evidence supplies a continuous multiplier in `[0, 1]`. A low measured value reduces the score but does not itself exclude the video.

The paper defines the mask-quality score, **Q_mask**, by averaging component scores within three groups—shape and scale, segmentation consistency, and local completeness—then weighting the groups equally and averaging across required objects. Q_mask supplies the `mask_quality_score` term in the formula below. Centroid jumps are recorded for diagnosis but do not enter this multiplier. The mask score is a heuristic for two-dimensional track reliability, not a probability that tracking is correct.

Missing evidence for a required mask component or object makes the score unavailable; it is never replaced with a perfect score. A measured mask score of zero remains a valid measurement and does not change whether the video passed the two hard checks.

## Score formula

For a target value `y` and measured value `ŷ`, the scorer uses an evaluator-provided relative error when available; otherwise it computes:

```text
relative_error = |ŷ - y| / max(|y|, epsilon)
physical_accuracy_score = 100 × exp(-relative_error / tau)
adjusted_score = physical_accuracy_score
                 × tracking_quality_multiplier
                 × mask_quality_score
                 × estimation_validity_multiplier
```

`tau` is a metric-specific tolerance: a smaller value penalizes the same error more strongly. The default tracking multiplier is `1` for a pass and `0` for a flagged track. Estimation validity uses `1` for `valid`, `0.8` for `weak_valid`, and `0` for `invalid` or `failed`.

For example, with a physical-accuracy score of `70`, a passing track, mask quality `0.6`, and `weak_valid` status, the adjusted score is `70 × 1 × 0.6 × 0.8 = 33.6`.

## Read the results

The report separates physical accuracy from measurability:

| Quantity | Meaning |
| --- | --- |
| **Effective-video score** | Mean adjusted score over videos that pass both hard checks; the primary physical-accuracy result. |
| **Discard rate** | `1 − effective videos / total videos`; the primary measurability result. |
| **End-to-end score** | Mean adjusted score over all videos, counting excluded videos as zero; a supplementary result combining accuracy and measurability. |

Discard diagnostics distinguish tracking-only failures, physics-only failures, and failures of both checks. Report effective-video score together with discard rate when comparing models.
Each metric's score is averaged over its videos; the overall model score then averages the metric-level scores equally.

By default, reports are written to:

```text
<result-root>/score_reports/<protocol_id>/all_metrics/
```

With `--metric`, the final directory is the metric name instead of `all_metrics`. The main files are:

| File | Contents |
| --- | --- |
| `score_summary.json` | Model summary, scoring settings, protocol fingerprint, metric summaries and per-video records. |
| `metric_scores.csv` | One summary row per metric. |
| `per_video_scores.csv` | Per-video status, scores, quality factors and failure reasons. |
| `score_exclusion_reasons.csv` | Counts of exclusion reasons. |
| `plots/` | Score and discard charts, unless `--no-plots` is used. |

The scorer checks result IDs against benchmark metadata by default. If required scoring evidence is missing, or result IDs are missing, unexpected or duplicated, it writes `incomplete_report.json` and refuses to publish a formal aggregate. A `valid` or `weak_valid` result without a computable physical error also violates the evaluator contract and stops scoring by default. Fix the inputs before comparing model scores.

## Saved-evidence research analyses

The formal scorer is implemented in `physics_eval/scoring.py`; experiment modules call its explicit data interface without retracking or replacing its loaders. Use a fresh output directory for each analysis:

```bash
# One-factor sensitivity: fixed original cohort of six models / 3,000 results.
python physeval.py analyze-sensitivity \
  --results-root ../results \
  --output-dir runs/scoring_sensitivity

# Historical and candidate Q_mask protocol comparisons.
python physeval.py analyze-mask-protocols \
  --results-root ../results \
  --output-dir runs/historical_mask_protocols
```

Sensitivity preserves saved trajectories and physical measurements, varies Q_mask gamma, weak-valid weights, or the four applicable R² consumers, and retains the spring period-confidence rule. Its original cohort assertions are intentional; it is not a general arbitrary-size analysis command. Historical comparisons preserve protocol identity/hash and the optional `--baseline-scorer path/to/frozen_score.py` CLI regression check. Candidate parameter changes are not evidence of improved segmentation and have not been independently calibrated against manual labels or ground truth.

See [Architecture](docs/ARCHITECTURE.md) for the separation between formal scoring, experimental analyses, and validation. Formal `eval` uses the fixed YOLO-then-motion initialization protocol described in [BATCH_USAGE.md](BATCH_USAGE.md).
