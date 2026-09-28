# PhysEval

[English](README.md) | [简体中文](README_zh-CN.md)

**Quantifying the gap between video generation and physical laws.**

PhysEval is a benchmark dataset and automatic evaluation protocol for testing whether text-to-video (T2V) generations obey **quantitative** physical constraints. It asks a stricter question than visual plausibility: when a prompt specifies a physical quantity, does the generated video contain measurable motion whose estimated value matches that target?

Each benchmark instance pairs a natural-language prompt with auditable evaluation metadata: the physical metric, target value and unit, evaluator, expected object count, calibration setting, and known physical parameters. Given generated videos, the pipeline initializes and tracks objects, checks whether each video is measurable, estimates the requested quantity with a task-specific evaluator, and reports normalized scores together with failure diagnostics.

## Benchmark at a Glance

- **500 prompt-metadata pairs**: 50 prompts for each of 10 physical metrics.
- **Four task families**: kinematics, physical constants, material parameters, and conservation-style tasks.
- **Automatic and auditable**: every score can be traced to a prompt, target, tracked trajectory, measured value, validity status, mask-quality score, and discard reason.
- **Designed for T2V output**: prompts request a fixed camera, approximately planar motion, visible targets, and simple scenes suitable for automatic measurement.

| Family | Metric | Quantity estimated from the video | Expected objects |
|---|---|---|---:|
| Kinematics | Velocity | Linear trajectory slope | 1 |
| Kinematics | Acceleration | Quadratic trajectory coefficient | 1 |
| Physical constant | Gravity | Free-fall acceleration | 1 |
| Material parameter | Friction coefficient | Deceleration ratio | 1 |
| Material parameter | Restitution coefficient | Post-/pre-impact speed ratio | 1 |
| Material parameter | Density | Early falling acceleration in fluid | 1 |
| Material parameter | Fluid viscosity | Terminal velocity | 1 |
| Material parameter | Spring constant | Oscillation period | 1 |
| Conservation | Mechanical energy conservation | Energy ratio | 1 |
| Conservation | Momentum conservation 1D | Relative momentum error | 2 |

The released prompt and metadata workbooks are documented in [`benchmark/metadata/README.md`](benchmark/metadata/README.md).

## How Evaluation Works

PhysEval deliberately separates **measurability** from **physical accuracy**.

1. **Object tracking**: YOLO-assisted initialization and SAM2 mask propagation recover visible object trajectories.
2. **Tracking-quality gate**: the expected object count, valid-frame coverage, and meaningful motion are checked. A failed gate makes the video ineffective for scoring.
3. **Physics-validity gate**: a task-specific evaluator returns `valid`, `weak_valid`, `invalid`, or `failed`. Only `valid` and `weak_valid` videos enter the effective set.
4. **Mask reliability**: after both hard gates pass, a continuous mask-quality score in `[0, 1]` acts as a soft proxy for 2D trajectory reliability. It reduces the score but does not remove a video under the paper's default policy.
5. **Physical scoring**: relative error is converted to a metric-normalized physical score and multiplied by tracking, mask-quality, and estimation-validity terms.

The paper reports three complementary views:

- **Effective-video score (primary physical-accuracy result)**: mean adjusted score over videos that pass both hard gates.
- **Discard rate (measurability result)**: fraction of videos that fail tracking quality, physics validity, or both.
- **End-to-end score (supplementary)**: mean adjusted score over all videos, assigning zero to discarded samples. This mixes measurability with physical accuracy and should not replace the two reports above.

See [`SCORING_USAGE.md`](SCORING_USAGE.md) for equations, thresholds, status rules, and output fields.

## Scope and Assumptions

The current evaluators use simple, inspectable inverse-physics models. They assume short monocular videos, a fixed camera, approximately planar motion parallel to the image plane, a known or inferable time base, stable 2D tracks, and benchmark-provided scale when metric units are required.

This release does not establish full 3D physical correctness. Depth drift, camera motion, perspective change, occlusion, deformation, complex fluids, rotation, or long-horizon interactions may invalidate the 2D estimates. Tracking gates and mask reliability expose some failures, but they are not proof of true 3D planarity. Benchmark scores should therefore be treated as diagnostic measurements, not as certification for safety-critical simulation.

## Repository Contents

- `benchmark/metadata/`: released prompt workbook, evaluation metadata, and UTF-8 CSV exports.
- `physics_eval/`: task-specific inverse-physics evaluators and batch evaluation logic.
- `scripts/`: SAM2 tracking, YOLO initialization, mask checks, batch execution, and score aggregation.
- `physeval.py`: unified entry point for `eval`, `score`, and optional `compare` / `compare-summary` commands.
- [`compare/`](compare/README.md): optional detector/tracker comparison, including saved comparison reports and its own outputs.
- `requirements-compare.txt`: opt-in comparison dependencies, including the base requirements.
- [`BATCH_USAGE.md`](BATCH_USAGE.md): complete batch-running and troubleshooting guide.
- [`SCORING_USAGE.md`](SCORING_USAGE.md): paper-aligned scoring and report interpretation.
- `configs/scoring/`: versioned mask-quality scoring protocols used by the scorer.

Generated videos, SAM2 source/checkpoints, YOLO weights, and batch outputs are
not included in this code repository. Supply videos locally when running an
evaluation.

## Installation

Install Python dependencies first. On a GPU server, ensure that `torch` and `torchvision` match the machine's CUDA runtime. If the environment already has a working PyTorch installation, keep it and install the remaining packages around it.

```bash
pip install -r requirements.txt
```

Prepare the external SAM2 and YOLO assets:

```bash
bash scripts/setup_sam2_assets.sh
```

By default, the setup script:

- clones `https://github.com/facebookresearch/sam2.git` into `repo/sam2/`;
- downloads `sam2.1_hiera_base_plus.pt` into `repo/sam2/checkpoints/`;
- downloads `yolov8n.pt` into the repository root.

Useful alternatives:

```bash
# Use the larger SAM2 checkpoint.
bash scripts/setup_sam2_assets.sh --sam2-model large

# Clone/download assets and install SAM2 in editable mode.
bash scripts/setup_sam2_assets.sh --install-sam2

# Skip YOLO when using motion-only initialization.
bash scripts/setup_sam2_assets.sh --skip-yolo
```

If `--install-sam2` was not used, install SAM2 after cloning:

```bash
pip install -e repo/sam2
```

## Quick Start

Run the commands below from the repository root.

Generate one video for every prompt in `benchmark/metadata/phys_t2v_bench_prompts.xlsx`. Name each video according to the `Video_File` column in the metadata workbook and place all videos for one model under:

```text
data/t2v_videos/<model_name>/
```

Run tracking and physical evaluation for that model:

```bash
python physeval.py eval \
  --metadata benchmark/metadata/phys_t2v_bench_metadata.xlsx \
  --video-root data/t2v_videos/<model_name> \
  --output-dir batch_eval_results/<model_name> \
  --detector yolo_then_motion
```

Score the compact results using the protocol reported in the paper:

```bash
python physeval.py score \
  --result-root batch_eval_results/<model_name> \
  --model-name <model_name> \
  --weak-valid-multiplier 0.8
```

> **Protocol compatibility note:** the CLI default for `weak_valid` is `0.8`, matching the paper protocol. The option remains available for explicit sensitivity analyses.

The principal outputs are:

```text
batch_eval_results/<model_name>/score_reports/grouped_v2_candidate/all_metrics/
  score_summary.json
  metric_scores.csv
  per_video_scores.csv
  score_exclusion_reasons.csv
  plots/
```

For detailed commands, reuse modes, output layouts, and diagnostics, continue with [`BATCH_USAGE.md`](BATCH_USAGE.md).

### Scoring protocol and outputs

The default scoring protocol is `grouped_v2_candidate`, defined in
[`configs/scoring/grouped_v2_candidate.json`](configs/scoring/grouped_v2_candidate.json).
It combines mask-quality evidence in three equally weighted groups and averages
across required objects. Missing evidence makes a score unavailable; it is not
filled in with a perfect score. `legacy_v1.json` remains available for explicitly
reproducing older scoring behavior and is not the default.

The per-video score combines physical accuracy, measurement quality, and
estimation validity:

```text
final_score = physical_accuracy_score
              × measurement_quality_multiplier
              × estimation_validity_multiplier
```

The physical-accuracy term converts relative error to a metric-specific score.
Tracking and physics-validity gates determine whether a video enters the
effective set; a low but measured Qmask does not by itself remove it. The default
`weak_valid` multiplier is `0.8`, matching the paper protocol. Reports are written
under `score_reports/<protocol_id>/<scope>/` and include model and metric summaries,
per-video scores, and exclusion reasons. See [`SCORING_USAGE.md`](SCORING_USAGE.md)
for formula details, status rules, and report fields.

## Optional: detector/tracker comparison

Qmask scoring is shared and versioned. The default is `grouped_v2_candidate`,
with three equally weighted quality groups. See [scoring usage](SCORING_USAGE.md)
for the formula, completeness checks, and offline sensitivity-analysis command.

The core `eval` and `score` commands do not import the comparison package or require
the TAM/XMem dependencies. Enable comparison explicitly when studying detector
or tracker choices:

```bash
python -m pip install -r requirements-compare.txt
bash compare/scripts/setup_compare_assets.sh --install-editable
python physeval.py compare --video-root data/t2v_videos/model_name --model-name model_name --dry-run
# Remove --dry-run to run the selected comparison modes.
python physeval.py compare-summary --model-name model_name
```

Comparison shares `benchmark/metadata/`, `repo/`, and `yolov8n.pt` with the core
project. Its generated outputs and frame cache stay in `compare/outputs/` and
`compare/cache/`; existing reports are preserved in `compare/results/`.
These reports measure tracking and mask quality, not physical accuracy scores.
See [the comparison guide](compare/README.md) for modes and full commands.

The original `scripts/run_batch_simple.py`, `scripts/score_results.py`, and
`compare/scripts/` entry points remain available. All script locations and
default asset paths are resolved from the checkout, so the outer
directory may be renamed after cloning.
