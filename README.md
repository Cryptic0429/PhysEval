# PhysEval

**Quantitative Physical Evaluation for Text-to-Video Generation**

[English](README.md) | [简体中文](README_zh-CN.md)

PhysEval measures how closely generated videos follow quantitative physical constraints. It pairs **500 prompts across 10 physical metrics** with an automatic pipeline that tracks objects, estimates physical quantities, and compares them with the targets specified in the benchmark.

[Benchmark](#-benchmark-data) · [Setup](#-environment-setup) · [Quick Start](#-running-evaluation) · [Scoring Guide](SCORING_USAGE.md) · [Batch Guide](BATCH_USAGE.md) · [Architecture](docs/ARCHITECTURE.md)

---

## ✨ Key Features

- **Quantitative targets:** 50 prompt-metadata pairs per metric, covering kinematics, gravity, material parameters, and conservation tasks.
- **Automatic measurement:** a fixed YOLO-first initialization protocol with motion fallback, SAM2 tracking, and task-specific inverse-physics estimation.
- **Quality-aware evaluation:** tracking and physics-validity gates determine measurability; continuous mask quality adjusts scores.
- **Traceable results:** each video retains its target, measurement, trajectory, evaluator status, quality evidence, and failure reasons.
- **Unified workflow:** evaluate and score through `physeval.py`, with an optional detector/tracker comparison module.

## 🛠️ Environment Setup

Use **Python 3.10+**; Python 3.11 is a convenient choice for a new environment. The core requirements include **PyTorch ≥ 2.5.1** and **Torchvision ≥ 0.20.1**, consistent with [SAM2's installation requirements](https://github.com/facebookresearch/sam2#installation). A CUDA-capable GPU is recommended for video tracking.

```bash
# 1. Clone the repository.
git clone https://github.com/Cryptic0429/PhysEval.git
cd PhysEval

# 2. Create an environment.
conda create -n physeval python=3.11 -y
conda activate physeval

# 3. Install the core dependencies.
python -m pip install -r requirements.txt
```

On GPU servers, install a compatible CUDA build of `torch` and `torchvision` before the final step when needed.

> Run all commands below from the repository root, the directory containing `physeval.py`.

## 🤗 Model Assets

Prepare the SAM2 source, checkpoint, and YOLO weights, and install SAM2:

```bash
bash tools/setup_assets.sh --install-sam2
```

The default setup uses the following paths:

| Asset | Local path |
| --- | --- |
| SAM2 source | `repo/sam2/` |
| SAM2.1 Base+ checkpoint | `repo/sam2/checkpoints/sam2.1_hiera_base_plus.pt` |
| YOLOv8n weights | `yolov8n.pt` |

Formal evaluation always tries YOLO first and falls back to motion initialization when YOLO fails or does not provide the required number of usable target prompts. Prepare both YOLO and SAM2 assets for ordinary evaluation. For other SAM2 sizes or custom asset locations, run `bash tools/setup_assets.sh --help` and pass the matching `--model-cfg` and `--model-weights` to evaluation.

## 📦 Benchmark Data

The prompt and metadata files are included in [`benchmark/metadata/`](benchmark/metadata/README.md); no separate benchmark download is required.

| File | Contents |
| --- | --- |
| [`phys_t2v_bench_prompts.xlsx`](benchmark/metadata/phys_t2v_bench_prompts.xlsx) | Generation prompts in the `All_Prompts` sheet |
| [`phys_t2v_bench_metadata.xlsx`](benchmark/metadata/phys_t2v_bench_metadata.xlsx) | Targets, units, evaluator settings, object counts, and calibration in the `metadata` sheet |
| [`csv/`](benchmark/metadata/csv/) | UTF-8 exports of all workbook sheets |

Each metric contains **50 samples**, for **500 samples** in total:

| Family | Metric | Measurement basis | Objects |
| --- | --- | --- | ---: |
| Kinematics | Velocity | Linear trajectory slope | 1 |
| Kinematics | Acceleration | Quadratic trajectory coefficient | 1 |
| Physical constant | Gravity | Free-fall acceleration | 1 |
| Material parameter | Friction coefficient | Deceleration relative to gravity | 1 |
| Material parameter | Restitution coefficient | Post-/pre-impact speed ratio | 1 |
| Material parameter | Density | Initial falling acceleration in fluid | 1 |
| Material parameter | Fluid viscosity | Terminal velocity | 1 |
| Material parameter | Spring constant | Oscillation period with known mass | 1 |
| Conservation | Mechanical energy conservation | Energy ratio | 1 |
| Conservation | Momentum conservation 1D | Relative momentum error | 2 |

Generate one video per prompt and name it exactly as specified by the metadata's `Video_File` column. Keep each model's videos in a separate folder; the examples below use `my_model`:

```text
data/t2v_videos/my_model/
├── V001.mp4
├── V002.mp4
├── ...
└── P050.mp4
```

Generated videos and external model assets are supplied locally. Match videos by metadata keys and filenames; keep failed generations in the evaluation accounting.

## 🚀 Running Evaluation

```text
Generated videos → YOLO initialization → motion fallback when needed → SAM2 tracking
                 → Quality diagnostics and physics estimation (eval)
                 → Quality-aware scoring and model reports (score)
```

YOLO is used only to initialize the target prompts; SAM2 propagates the masks through the video. An exception, no usable detection, or an incomplete target count triggers motion fallback. Motion must also supply the required count or the sample is recorded as an initialization failure. `run_config.json` and `tracking_points.json` record the fixed policy, the adopted initialization source (`auto_yolo` or `auto_motion`), and the fallback reason. There is no detector-selection option in formal `eval`.

### 1. Try one video

Evaluate `V001` and save a tracking visualization to inspect initialization and tracking:

```bash
python physeval.py eval \
  --metadata benchmark/metadata/phys_t2v_bench_metadata.xlsx \
  --video-root data/t2v_videos/my_model \
  --output-dir batch_eval_results/my_model \
  --prompt-id V001 \
  --vis
```

### 2. Evaluate the full benchmark

Run tracking and physics estimation for all metadata rows:

```bash
python physeval.py eval \
  --metadata benchmark/metadata/phys_t2v_bench_metadata.xlsx \
  --video-root data/t2v_videos/my_model \
  --output-dir batch_eval_results/my_model
```

Inspect `batch_eval_results/my_model/<metric>/<prompt_id>/result.json` first. Each sample also saves `tracking_points.json`, `run_config.json`, and mask-quality diagnostics.

### 3. Score the results

After the full evaluation, aggregate results using the PhysEval scoring protocol defined in the paper:

```bash
python physeval.py score \
  --result-root batch_eval_results/my_model \
  --model-name my_model \
  --weak-valid-multiplier 0.8
```

The explicit `0.8` is the paper-specified weight for `weak_valid` estimates and is also the CLI default. The scorer checks benchmark completeness by default. For an explicitly partial smoke test, add `--metadata none`; treat that report as exploratory.

## 📊 Scoring and Reports

PhysEval separates **measurability** from **physical accuracy**. A video enters the effective set only if its tracking passes the expected-object-count, frame-coverage, and meaningful-motion checks, and its physics evaluator returns `valid` or `weak_valid`.

For videos in that set, PhysEval defines the score as:

```text
physical_accuracy_score = 100 × exp(-relative_error / tau)
adjusted_score = physical_accuracy_score
                 × tracking_quality_multiplier
                 × mask_quality_score
                 × estimation_validity_multiplier
```

`tau` is a metric-specific error tolerance. Passing tracks have a tracking multiplier of `1`; estimation validity contributes `1` for `valid` or `0.8` for `weak_valid`.

The paper defines the **mask-quality score, Q_mask**, using three groups: **shape and scale**, **segmentation consistency**, and **local completeness**. Component scores are averaged within each group, the three groups receive equal weight, and the resulting object scores are averaged across required objects. Q_mask supplies the `mask_quality_score` term above. A low measured Q_mask reduces the adjusted score without excluding the video; missing required evidence makes scoring unavailable.

| Reported quantity | Interpretation |
| --- | --- |
| **Effective-video score** | Mean adjusted score over effective videos; report this with discard rate. |
| **Discard rate** | Fraction of videos that fail tracking quality, physics validity, or both. |
| **End-to-end score** | Supplementary mean over all videos, assigning zero to discarded videos. |

Scores are aggregated per metric, and the overall model score gives each metric equal weight. Formal aggregation stops if benchmark IDs or required scoring evidence are incomplete.

Reports are saved under `batch_eval_results/my_model/score_reports/<protocol_id>/all_metrics/`:

| Output | Contents |
| --- | --- |
| `score_summary.json` | Model and metric summaries, scoring settings, and protocol fingerprint |
| `metric_scores.csv` | Per-metric scores and measurability statistics |
| `per_video_scores.csv` | Per-video measurements, quality factors, statuses, and scores |
| `score_exclusion_reasons.csv` | Counts of exclusion reasons |
| `plots/` | Score and discard-rate charts |

See the [scoring guide](SCORING_USAGE.md) for relative-error definitions, thresholds, completeness checks, and sensitivity-analysis options.

## ⚙️ Key Evaluation Options

| Option | Default | Purpose |
| --- | --- | --- |
| `--metadata` | Auto-discovery in `data/metadata/` | Pass the released workbook explicitly, as shown above. |
| `--prompt-id` | All rows | Evaluate one prompt for a smoke test. |
| `--reuse-tracking` | Off | Reuse existing tracking JSON when available. |
| `--eval-only` | Off | Estimate physics from existing tracking JSON. |
| `--vis` / `--diagnostics` | Off | Save tracking videos / additional debugging outputs. |
| `--sam2-precision` | `auto` | Select `auto`, `fp32`, `fp16`, or `bf16` inference. |
| `--area-action` | `warn` | Keep mask-QC flags as warnings; `skip` changes the evaluation policy. |

Run `python physeval.py eval --help` or `python physeval.py score --help` for all options. See the [batch guide](BATCH_USAGE.md) for tracking reuse, custom checkpoints, and troubleshooting.

## 🔬 Optional: Detector and Tracker Comparison

The comparison module supports `fasterrcnn_sam2`, `yolo_sam2`, and `yolo_tam`. Install its additional dependencies only when using this module:

```bash
python -m pip install -r requirements-compare.txt
bash tools/setup_assets.sh --mode compare --install-editable

# Check video/metadata matching before inference.
python physeval.py compare \
  --video-root data/t2v_videos/my_model \
  --model-name my_model \
  --dry-run
```

Remove `--dry-run` to run the comparison, then summarize it with:

```bash
python physeval.py compare-summary --model-name my_model
```

Comparison source lives in `experiments/tracking_comparison/` and shares the core model assets and metadata. For compatibility, its runtime outputs remain in `compare/outputs/`, frame caches in `compare/cache/`, and preserved reports in `compare/results/`; these are data directories. These reports describe tracking and mask quality; physical-accuracy results come from `eval` and `score`. See the [comparison guide](experiments/tracking_comparison/README.md) for paired modes and full commands.

## 🧱 Project Structure

```text
PhysEval/
├── physeval.py                    # Public command-line entry point
├── physics_eval/
│   ├── cli.py / config.py         # Commands and shared defaults
│   ├── pipeline.py / estimation.py # Evaluation orchestration
│   ├── schemas.py                # Shared prompts and video interfaces
│   ├── tracking/                 # Initialization and shared SAM2/TAM backends
│   ├── evaluators/               # Ten task-specific physics evaluators
│   ├── calibration.py / math/    # Scale calibration, fitting and geometry
│   ├── quality/                  # Tracking gates and mask evidence/Q_mask
│   ├── io/                       # Metadata, tracks, results and reports
│   └── scoring.py / reporting.py # Scores, summaries and plots
├── experiments/                  # Tracking comparisons and saved-evidence analyses
├── validation/                   # Open-video preflight and controlled masks
├── tools/                        # Asset setup, environment checks and maintenance
├── tests/                        # Unit, synthetic and integration checks
├── benchmark/metadata/           # Released workbooks and CSVs
├── configs/scoring/              # Versioned Q_mask protocols
├── compare/                      # Runtime outputs/cache and preserved reports
├── requirements*.txt             # Core and optional comparison dependencies
└── docs/ARCHITECTURE.md           # Responsibilities and migration coverage
```

The [architecture guide](docs/ARCHITECTURE.md) maps the complete workflow and the migrated source files. Experimental and validation commands are explicit; `eval` does not run them automatically.

## 🧪 Research, Validation and Maintenance

```bash
# Analyze saved results without retracking (the original six-model/3,000-video cohort).
python physeval.py analyze-sensitivity --results-root ../results --output-dir runs/sensitivity

# Controlled mask checks, including a stable wrong-target negative control.
python physeval.py validate-masks --output runs/validation/mask_controls.csv

# Inspect a curated open-video package without requiring GPU inference.
python physeval.py validate-open-video --data-only

# Optional comparison environment/GPU checks and saved-mask inspection.
python tools/check_environment.py
python tools/check_gpu.py
python tools/inspect_masks.py --sam2-json path/to/tracking_points.json

# CPU regression tests.
python -m pip install -r requirements-test.txt
python -m pytest tests
```

`analyze-mask-protocols` reproduces historical Q_mask comparisons. `compare-cross-model` combines model-stratified paired reports; see the [comparison guide](experiments/tracking_comparison/README.md). These analyses retain their declared cohorts and scientific limitations.

## 🔍 Scope and Assumptions

The evaluators use inspectable inverse-physics models for short monocular videos. They assume a fixed camera, approximately planar motion parallel to the image plane, a known or inferable time base, stable 2D tracks, and scale calibration when metric units are required.

Camera motion, depth changes, perspective variation, occlusion, deformation, rotation, or complex interactions can compromise these estimates. Tracking gates and mask quality reveal some failures, but do not establish full 3D physical correctness. Interpret scores within these measurement assumptions.

## 🙏 Acknowledgements

The tracking pipeline builds on [SAM2](https://github.com/facebookresearch/sam2) and [Ultralytics YOLO](https://github.com/ultralytics/ultralytics). The optional comparison module also uses Torchvision's Faster R-CNN and Track-Anything / XMem; see its [guide](experiments/tracking_comparison/README.md) for setup and dependencies.
