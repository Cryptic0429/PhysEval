# PhysEval

Physics-oriented evaluation tools for text-to-video generation.

This repository contains the reusable parts of the PhysT2V-Bench workspace:

- `physics_eval/`: task-specific physics evaluators and batch evaluation logic.
- `scripts/`: entrypoints for SAM2-based tracking, mask quality control, batch runs, and score aggregation.
- `BATCH_USAGE.md`: how to run video tracking and physics evaluation in batch.
- `SCORING_USAGE.md`: scoring formula, validity rules, and output files.
- `paper_neurips2026/`: current paper source and figure assets.

Large or machine-specific assets are intentionally not included:

- generated videos under `data/t2v_videos/`
- metadata spreadsheets under `data/metadata/`
- SAM2 source checkout and checkpoints under `repo/sam2/`
- batch outputs, temporary renders, and cache files

## Typical Workflow

Put one metadata spreadsheet under `data/metadata/`, put generated videos under
`data/t2v_videos/<model_name>/`, and make sure SAM2 is available under
`repo/sam2/` with the required checkpoints.

Run one model:

```bash
python scripts/run_batch_simple.py \
  --metadata data/metadata/metadata.xlsx \
  --video-root data/t2v_videos/model_name \
  --output-dir batch_eval_results/model_name
```

Score one model:

```bash
python scripts/score_results.py \
  --result-root batch_eval_results/model_name \
  --model-name model_name
```

See `BATCH_USAGE.md` and `SCORING_USAGE.md` for detailed options.

