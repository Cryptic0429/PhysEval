# PhysT2V-Bench Prompt and Metadata Release

[English](README.md) | [简体中文](README_zh-CN.md)

This directory contains the prompt table and evaluation metadata used by the
PhysEval / PhysT2V-Bench pipeline.

The dataset is structured so that each generated video is evaluated against one primary physical quantity estimated from tracked motion, pairing each instance with a single target value and dedicated evaluator.

## Files

- `phys_t2v_bench_prompts.xlsx`: original Excel prompt workbook.
- `phys_t2v_bench_metadata.xlsx`: original Excel evaluation metadata workbook.
- `csv/`: UTF-8 CSV exports of every worksheet in the two workbooks, provided for
  easier inspection, versioning, and non-Excel workflows.

## Dataset Size

The release contains 500 prompt-metadata pairs: 10 physical metrics with 50
instances per metric.

| Module | Metric | Prompt ID prefix | Count | Note |
|---|---|---:|---:|---|
| B1.1 | velocity | V | 50 |  |
| B1.2 | acceleration | A | 50 |  |
| B2 | gravity | G | 50 | Earth gravity only |
| B3.1 | friction_coefficient | F | 50 |  |
| B3.2 | restitution_coefficient | R | 50 |  |
| B3.3 | density_from_initial_acceleration | D | 50 |  |
| B3.4 | fluid_viscosity | ETA | 50 |  |
| B3.5 | spring_constant | K | 50 |  |
| B4.1 | mechanical_energy_conservation | E | 50 |  |
| B4.2 | momentum_conservation_1d | P | 50 |  |

## Task Taxonomy and Measurement

| Family | Metric | Expected objects | Measured quantity | Spatial calibration |
|---|---|---:|---|---|
| Kinematics | `velocity` | 1 | Linear trajectory slope | Required for metric units |
| Kinematics | `acceleration` | 1 | Quadratic trajectory coefficient | Required for metric units |
| Physical constant | `gravity` | 1 | Free-fall acceleration | Required for metric units |
| Material | `friction_coefficient` | 1 | Deceleration divided by gravity | Required |
| Material | `restitution_coefficient` | 1 | Post-/pre-impact speed ratio | Not required; scale cancels |
| Material | `density_from_initial_acceleration` | 1 | Early falling acceleration in fluid | Required |
| Material | `fluid_viscosity` | 1 | Terminal velocity with known radius/densities | Required |
| Material | `spring_constant` | 1 | Oscillation period with known mass | Not required; period-only |
| Conservation | `mechanical_energy_conservation` | 1 | Potential-to-kinetic energy ratio | Required |
| Conservation | `momentum_conservation_1d` | 2 | Relative pre/post momentum error | Not required when scale is shared |

The prompts are written for short monocular videos with a fixed camera, approximately planar motion parallel to the image plane, visible target boundaries, and minimal clutter or occlusion. The evaluation pipeline applies tracking-quality gates, physics-validity checks, and continuous mask reliability to account for deviations from these assumptions in generated videos.

## Prompt Workbook

`phys_t2v_bench_prompts.xlsx` contains:

- `All_Prompts`: 500 text-to-video generation prompts.
- `Summary`: per-module prompt counts and ID ranges.
- `README`: source and matching-key notes.

Main columns in `All_Prompts`:

- `Global_Index`: global row number.
- `Module`: benchmark module.
- `Index`: index used to match the metadata workbook.
- `Prompt_ID`: prompt identifier and recommended filename prefix.
- `English_Prompt`: prompt text used for generation.

## Metadata Workbook

`phys_t2v_bench_metadata.xlsx` contains:

- `metadata`: 500 evaluation rows consumed by the batch evaluation scripts.
- `summary`: per-module metric counts.
- `field_guide`: field descriptions.

Main columns in `metadata`:

- `Index`, `Prompt_ID`: keys that match the prompt workbook.
- `Video_File`: expected generated video filename.
- `Tracking_JSON`: expected tracking JSON path relative to a tracking root.
- `Module`, `Metric`, `Evaluator`, `Measurement_Method`: evaluator selection.
- `Tracking_Objects`, `Num_Objects`, `View`, `Motion_Axis`: tracking and view assumptions.
- `Scale_Mode`, `Calibration_Object`, `Calibration_Dimension`, `Calibration_Value_m`: scale calibration.
- `Target_Type`, `Target_Value`, `Target_Unit`: physical target specification.
- `Known_Parameters_JSON`: JSON-encoded auxiliary physical parameters.

The schema separates natural-language generation from evaluation instructions. In particular:

- `Num_Objects` is part of the hard tracking-quality gate; momentum requires two persistent tracks while the other current tasks expect one.
- `Scale_Mode` indicates whether a known-size object provides meters-per-pixel calibration or whether the evaluator uses a ratio-only / period-only quantity for which spatial scale cancels.
- `Known_Parameters_JSON` provides auxiliary physical constants and object parameters (such as mass, fluid density, or reference dimensions) required by specific inverse evaluators.
- `Target_Value` and `Target_Unit` define the auditable numerical quantity used to compute relative error.

## How to Use

Generate videos using the prompts in `phys_t2v_bench_prompts.xlsx`, and name each video according to the corresponding `Video_File` column. Evaluation matches generated videos strictly by filename against the `Video_File` column in the metadata. Place the generated videos under a model-specific directory:

```text
data/t2v_videos/<model_name>/
```

For instructions on running batch evaluation, tracking, and scoring, see the [Batch Guide](../../BATCH_USAGE.md) and [Scoring Guide](../../SCORING_USAGE.md).

## Validation

Before release, the files were checked for:

- 500 prompt rows and 500 metadata rows.
- unique `Index` and `Prompt_ID` keys.
- valid JSON in `Known_Parameters_JSON`.
- no local absolute path strings in prompt or metadata cells.
