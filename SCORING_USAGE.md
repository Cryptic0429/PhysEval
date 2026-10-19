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

PhysEval uses one mask-quality scoring rule, configured in `configs/scoring/mask_quality.json`. The defaults are continuous mask weighting and hard-fail tracking. Threshold and policy overrides are listed by `python physeval.py score --help`; changing them is a sensitivity analysis and must be reported explicitly.

## Which videos receive a score?

The scorer applies checks in order:

1. **Tracking quality:** the expected object count, object coverage and meaningful motion must pass. The paper requires an object to be tracked in at least half the frames, and the primary object's motion to span at least 0.75 of its median diameter. A failed tracking check excludes the video.
2. **Physics validity:** the evaluator must return `valid` or `weak_valid`. `invalid` and `failed` videos are excluded.
3. **Mask quality:** after both hard checks pass, mask evidence supplies a continuous multiplier in `[0, 1]`. A low measured value reduces the score but does not itself exclude the video.

The mask-quality score, **Q_mask**, uses ten diagnostics: area P95/P05 ratio, area CV, box width CV, box height CV, box aspect CV, maximum log-area jump, trend scale ratio, maximum centroid step divided by diameter, translation-aligned IoU, and middle-window coverage. Each diagnostic maps normalized degradation `p` to `1-p^0.35`. Take the minimum available component score within each object, then the minimum object score across objects. Q_mask is a heuristic reliability multiplier, not a probability that tracking is correct.

For an object whose decision is neither empty nor `keep`, a positive score is raised to at least **0.85**. An exact zero stays zero. The reasons `insufficient_valid_middle_frames` and, when its numerical component is missing, `low_middle_coverage` contribute a zero component before this floor is applied.

Missing/nonfinite component values are ignored. If an object has no usable components, its score is `1` for an empty/`keep` decision and `0.85` for a flagged decision. If there are no object records, Q_mask is `1` unless the mask-QC status is `flagged`, in which case it is `0`. Only saved `mask_qc.objects` are used; no extra object-ID gate is introduced at scoring. The object-count tracking gate still applies. Reports record configuration hashes, missing components and fallback warnings in per-video Q_mask details.

A Q_mask of zero reduces contribution to zero without changing eligibility. A missing diagnostic does not establish that the track is reliable; these fallbacks reproduce the defined scoring behavior.

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
<result-root>/score_reports/all_metrics/
```

With `--metric`, the final directory is the metric name instead of `all_metrics`. The main files are:

| File | Contents |
| --- | --- |
| `score_summary.json` | Model summary, scoring settings, mask-configuration fingerprint, metric summaries and per-video records. |
| `metric_scores.csv` | One summary row per metric. |
| `per_video_scores.csv` | Per-video status, scores, quality factors and failure reasons. |
| `score_exclusion_reasons.csv` | Counts of exclusion reasons. |
| `plots/` | Score and discard charts, unless `--no-plots` is used. |

The scorer checks result IDs against benchmark metadata by default. If result IDs are missing, unexpected or duplicated, it writes `incomplete_report.json` and refuses to publish a formal aggregate. Missing mask diagnostics follow the fallback rules above. A `valid` or `weak_valid` result without a computable physical error also violates the evaluator contract and stops scoring by default.

## Saved-evidence research analyses

The formal scorer is implemented in `physics_eval/scoring.py`; experiment modules call its explicit data interface without retracking or replacing its loaders. Use a fresh output directory for each analysis:

```bash
# One-factor sensitivity over the saved six-model / 3,000-result cohort.
python physeval.py analyze-sensitivity \
  --results-root ../results \
  --output-dir runs/scoring_sensitivity
```

Sensitivity analysis uses the same scorer and saved six-model / 3,000-result cohort. Its baseline is gamma `0.35`, weak-valid weight `0.8`, and the four applicable R² cutoffs `0.65/0.85`. It varies gamma, the weak-valid weight or those cutoffs one factor at a time, retaining minimum aggregation and the flagged-score floor. Spring retains its own period-confidence rule. Outputs and ranks are computed anew from the supplied evidence; previously published values from other scoring rules must be recomputed.

See [Architecture](docs/ARCHITECTURE.md) for the separation between formal scoring, experimental analyses, and validation. Formal `eval` uses the fixed YOLO-then-motion initialization protocol described in [BATCH_USAGE.md](BATCH_USAGE.md).
