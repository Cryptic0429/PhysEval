# PhysEval

Physics-oriented evaluation tools and benchmark videos for text-to-video generation.

This repository contains the reusable parts of the PhysT2V-Bench workspace:

- `benchmark/metadata/`: released prompt table and evaluation metadata.
- `physics_eval/`: task-specific physics evaluators and batch evaluation logic.
- `scripts/`: entrypoints for SAM2-based tracking, mask quality control, batch runs, and score aggregation.
- `BATCH_USAGE.md`: how to run video tracking and physics evaluation in batch.
- `SCORING_USAGE.md`: scoring formula, validity rules, and output files.

The released benchmark videos are available under `video/<model_name>/` and are
tracked with Git LFS. Clone with LFS enabled before running an evaluation:

```bash
git lfs install
git clone <repository-url>
cd PhysEval
git lfs pull
```

Large or machine-specific assets other than the released videos are intentionally not included:

- metadata spreadsheets under `data/metadata/`
- SAM2 source checkout and checkpoints under `repo/sam2/`
- batch outputs, temporary renders, and cache files

## Prepare SAM2 and YOLO Assets

The repository includes our SAM2 and YOLO integration code, but it does not
commit third-party repositories, model checkpoints, or detector weights.

Install the Python dependencies first. On a GPU server, make sure
`torch`/`torchvision` match the CUDA runtime of that machine; if you already have
a working PyTorch environment, keep it and install the remaining packages into
that environment.

```bash
pip install -r requirements.txt
```

Then prepare the external assets locally with:

```bash
bash scripts/setup_sam2_assets.sh
```

For SAM2, the pipeline needs both the SAM2 source checkout and a model
checkpoint. The model config is not a separate download: it is provided by the
SAM2 repository under `configs/sam2.1/`. By default this script:

- clones `https://github.com/facebookresearch/sam2.git` into `repo/sam2/`;
- downloads `sam2.1_hiera_base_plus.pt` into `repo/sam2/checkpoints/`;
- downloads `yolov8n.pt` into the repository root.

Optional examples:

```bash
# Use the larger SAM2 checkpoint.
bash scripts/setup_sam2_assets.sh --sam2-model large

# Clone/download assets and also run pip install -e repo/sam2.
bash scripts/setup_sam2_assets.sh --install-sam2

# Skip YOLO if you only want motion-based initialization.
bash scripts/setup_sam2_assets.sh --skip-yolo
```

If you do not use `--install-sam2`, install SAM2 after cloning:

```bash
pip install -e repo/sam2
```

The default batch detector is `yolo_then_motion`: YOLO is tried first and the
pipeline falls back to motion-based initialization when YOLO is unavailable or
does not find a suitable object. If `yolov8n.pt` exists in the repository root,
`scripts/run_batch_simple.py` uses it automatically; otherwise you can pass an
explicit detector weight path with `--yolo-weights`.

If you use a non-default SAM2 size, pass the matching config and checkpoint to
the batch command. For example, `--sam2-model large` corresponds to:

```bash
python scripts/run_batch_simple.py \
  --metadata benchmark/metadata/phys_t2v_bench_metadata.xlsx \
  --video-root data/t2v_videos/model_name \
  --output-dir batch_eval_results/model_name \
  --detector yolo_then_motion \
  --model-cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --model-weights repo/sam2/checkpoints/sam2.1_hiera_large.pt
```

## Typical Workflow

Use the released metadata under `benchmark/metadata/`, put generated videos under
`data/t2v_videos/<model_name>/`, and make sure SAM2 is available under
`repo/sam2/` with the required checkpoints.

Run one model:

```bash
python scripts/run_batch_simple.py \
  --metadata benchmark/metadata/phys_t2v_bench_metadata.xlsx \
  --video-root data/t2v_videos/model_name \
  --output-dir batch_eval_results/model_name \
  --detector yolo_then_motion
```

Score one model:

```bash
python scripts/score_results.py \
  --result-root batch_eval_results/model_name \
  --model-name model_name \
  --weak-valid-multiplier 0.8
```

The default `weak_valid` multiplier is **0.8**, matching the paper protocol.
See `BATCH_USAGE.md` and `SCORING_USAGE.md` for detailed options.
