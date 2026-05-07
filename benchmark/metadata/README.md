# PhysT2V-Bench Prompt and Metadata Release

This directory contains the prompt table and evaluation metadata used by the
PhysEval / PhysT2V-Bench pipeline.

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

## How to Use

Generate videos using the prompts in `phys_t2v_bench_prompts.xlsx`, name each
video according to `Video_File`, and place the files under a model-specific
video root such as:

```text
data/t2v_videos/<model_name>/
```

Then run:

```bash
python scripts/run_batch_simple.py \
  --metadata benchmark/metadata/phys_t2v_bench_metadata.xlsx \
  --video-root data/t2v_videos/<model_name> \
  --output-dir batch_eval_results/<model_name>
```

Score the outputs with:

```bash
python scripts/score_results.py \
  --result-root batch_eval_results/<model_name> \
  --model-name <model_name>
```

## Validation

Before release, the files were checked for:

- 500 prompt rows and 500 metadata rows.
- unique `Index` and `Prompt_ID` keys.
- valid JSON in `Known_Parameters_JSON`.
- no local absolute path strings in prompt or metadata cells.

