# PhysEval detector/tracker comparison

## Shared quality protocols

New runs use the main project's shared diagnostics and `grouped_v2_candidate`
Qmask scoring by default. Historical compare outputs retain their recorded
protocol and tracking-gate decisions; the old compare formula is not used for new runs.

Results save protocol identity/hash, raw `evidence_objects`, and scored objects.
Tracking JSON uses schema 2: one entry per actual video frame, nested objects, and
inclusive bbox endpoints. Pairwise comparison reads both old flat and new nested
files. Summary refuses mixed quality protocols or successful runs with unavailable
Qmask evidence. Use separate output roots for different protocols; cached successful
results from a different protocol are not silently reused.

Comparison Qmask describes tracking runs without the main physics-validity gate;
its population differs from the main effective-video subset. Higher candidate
scores are a protocol effect and do not establish more accurate segmentation.

This optional module implements paired detector/tracker comparisons for the question
"Why YOLO + SAM2?" while keeping the PhysEval data and output conventions.

The three predeclared modes are:

| Mode | Initial proposal | Video mask propagation |
| --- | --- | --- |
| `fasterrcnn_sam2` | Faster R-CNN ResNet50-FPN, then motion fallback | SAM2.1 |
| `yolo_sam2` | YOLOv8n, then motion fallback | SAM2.1 |
| `yolo_tam` | The exact cached YOLO/motion prompts used above | SAM1 + XMem (TAM) |

`yolo_sam2` and `yolo_tam` always share the same cached initialization frame,
point, box, and negative points. For two-object samples the detector must find
both objects in one common frame. This makes the tracker comparison paired and
prevents detector variation from being attributed to SAM2 versus TAM.

## Directory layout

```text
PhysEval-main/
  physeval.py             unified entry point (compare is opt-in)
  benchmark/metadata/    shared benchmark metadata
  data/t2v_videos/        shared local video inputs (ignored)
  repo/                  shared third-party repositories/checkpoints (ignored)
  requirements.txt       core dependencies
  requirements-compare.txt  optional comparison dependencies
  compare/
    physeval_compare/    comparison package
    scripts/            comparison tools and asset setup
    tests/              comparison unit tests
    results/            preserved comparison reports
    cache/              extracted-frame cache (ignored)
    outputs/            generated comparison outputs (ignored)
```

The released PhysEval metadata is read by default from the main project
(the following path is relative to this guide):

```text
../benchmark/metadata/csv/phys_t2v_bench_metadata__metadata.csv
```

The version-controlled CSV avoids making `openpyxl` a preflight requirement;
the equivalent workbook is also accepted. You may override it with
`--metadata`. Passing `--metadata none` scans every
video under `--video-root` and treats each as a one-object tracking-only sample.

## Environment

All commands below run from `PhysEval-main/`. Standard evaluation only needs
`requirements.txt`; neither `eval` nor `score` imports this module. Install the
extra dependencies below only when using the explicit `compare` entry point.
YOLO and SAM2 assets are shared with the core pipeline under the main project
root. Existing assets at these paths are reused by the setup script.

Python 3.10 or 3.11 is recommended. The requirements retain the main PhysEval
numerical stack and its documented PyTorch/Torchvision lower bounds. On a GPU
server, install a compatible PyTorch wheel that matches the CUDA driver first,
then install the remaining packages:

```bash
cd PhysEval-main
python -m venv .venv
source .venv/bin/activate

# Install the correct CUDA build first when the default PyPI wheel is unsuitable.
python -m pip install -r requirements-compare.txt
bash compare/scripts/setup_compare_assets.sh --install-editable
```

If Git reports a refused proxy such as `127.0.0.1:7890`, and the server can
reach GitHub directly, disable proxy settings for this setup invocation only:

```bash
bash compare/scripts/setup_compare_assets.sh --install-editable --no-proxy
```

This option clears proxy environment variables in the child script and
overrides Git/curl/wget proxy use without modifying the user's global Git
configuration. If the server cannot reach GitHub directly, configure a proxy
address reachable from the server instead of using `127.0.0.1` from another
machine.

Check the complete environment before a GPU run:

```bash
python compare/scripts/check_environment.py
```

The setup script clones/downloads only third-party assets into ignored paths.
Faster R-CNN COCO weights are managed by Torchvision on first use.

TAM uses the Track-Anything XMem wrapper, but not its video-inpainting module.
This avoids the E2FGVI dependency and keeps the runtime close to the main SAM2
environment. If an old Track-Anything checkout is incompatible with a newer
PyTorch release, start from the minimum versions recorded in this directory
instead of upgrading Torch independently.

## Video preparation

Put one model's videos in a matching folder:

```text
data/t2v_videos/wan2.2/
  V001.mp4
  V002.mp4
  ...
  P050.mp4
```

Filenames, not directory order, are matched against metadata. Exact metadata
names such as `V001.mp4` are preferred; existing prefixed names such as
`kling_V001.mp4` or `Wan2.2-T2V-A14B_V001.mp4` are also accepted. Do not delete
or rename failed generations before running the comparison.

## Preflight

Verify metadata/video matching without loading any neural network:

```bash
python physeval.py compare \
  --video-root data/t2v_videos/wan2.2 \
  --model-name wan2.2 \
  --dry-run
```

Run one sample first:

```bash
python physeval.py compare \
  --video-root data/t2v_videos/wan2.2 \
  --model-name wan2.2 \
  --prompt-id V001 \
  --save-visualization
```

Run all three modes over every available metadata row:

```bash
python physeval.py compare \
  --video-root data/t2v_videos/wan2.2 \
  --model-name wan2.2 \
  --skip-missing-videos \
  --cleanup-frames
```

`--cleanup-frames` removes a video's temporary extracted JPEGs only after all
selected modes have finished. Use it for full runs on storage-constrained
servers; mode outputs, masks, prompts, and reports are retained.

Split a mode into four deterministic, non-overlapping GPU workers with
`--num-shards 4 --shard-index 0`, changing the shard index through `3` and
assigning a different `--device cuda:N` to each worker.

For a predeclared video subset, create a CSV containing the chosen
rows before looking at results and pass it through `--metadata`. The runner also
supports `--index`, `--prompt-id`, and `--limit` for smoke tests; `--limit` should
not be used as an undocumented scientific sampling rule.

## Useful options

Run selected modes:

```bash
python physeval.py compare \
  --video-root data/t2v_videos/wan2.2 \
  --modes yolo_sam2 yolo_tam
```

Recreate detector prompts after intentionally changing a detector threshold:

```bash
python physeval.py compare \
  --video-root data/t2v_videos/wan2.2 \
  --overwrite-prompts --overwrite
```

The prompt cache signature includes the video path/size, target hint, expected
object count, scan policy, and detector thresholds. A stale signature is
regenerated automatically.

Use CPU only for plumbing tests:

```bash
python physeval.py compare --video-root data/t2v_videos/demo --metadata none --device cpu --limit 1
```

Neural inference on CPU is supported where upstream models allow it but is not
practical for the full benchmark.

## Output format

Outputs follow the main per-video organization, with an additional mode level:

```text
compare/outputs/<model_name>/
  _prompts/
    yolo/<prompt_id>.json
    fasterrcnn/<prompt_id>.json
  yolo_sam2/<metric>/<prompt_id>/
    result.json
    run_config.json
    tracking_points.json
    masks/obj_000001/*.png
    tracking_visualization.mp4       # only with --save-visualization
  yolo_tam/<metric>/<prompt_id>/...
  fasterrcnn_sam2/<metric>/<prompt_id>/...
```

`tracking_points.json` retains the main-style `video_info`, `objects`, `prompt`,
`frames`, and `quality_checks` fields. `result.json` is the compact first file to
inspect. A failed mode-run writes a compact failed `result.json` and does not
silently disappear from the denominator.

## Summarization

After tracking one model, run:

```bash
python physeval.py compare-summary \
  --result-root compare/outputs \
  --model-name wan2.2
```

This writes:

```text
compare/outputs/wan2.2/comparison_reports/
  per_run_results.csv
  mode_summary.csv
  per_video_pairwise.csv
  pairwise_summary.csv
  comparison_summary.json
```

The mode report includes failures, tracking pass rate with Wilson intervals,
motion-fallback rate, coverage, motion extent, and mask quality. Pairwise reports
compare the same videos with centroid nRMSE (normalized by object diameter),
bbox IoU, mask IoU, tracking-pass difference, mask-quality difference, and
fixed-seed bootstrap confidence intervals.

These are tracking/mask comparison outputs. Physical evaluation and scoring are
provided by the main `physics_eval/` package via `physeval.py eval` and
`physeval.py score`. Comparison does not automatically run those stages.
The generated `tracking_points.json` uses the main tracking schema; downstream
evaluation requires matching the evaluator's input layout and metadata.

## Migration and development

The former workspace-level `physeval-compare/` directory now lives here. Its
saved reports are preserved in `results/`. Runtime paths and historical report
contents are not rewritten; reports may retain paths from their original runs.
Use the shared `data/t2v_videos/` and `repo/` directories for new inputs/assets,
or override paths with CLI flags if reusing assets stored elsewhere.

The standalone `compare/scripts/run_compare.py` and
`compare/scripts/summarize_compare.py` wrappers remain supported.
For development, install `pytest` separately and run the existing tests from
this directory with `python -m pytest tests`.

## Fairness constraints

- Keep detector confidence, scan frames, fallback rules, tracking gates, and
  mask scoring thresholds identical across modes.
- Do not tune one model after inspecting its result.
- Report detector-only success and motion-fallback rates separately.
- Include failed runs in the denominator.
- Use the paired all-video result as primary; use the common-success subset only
  for trajectory agreement diagnostics.
- The main method is accurately described as `YOLO -> motion fallback -> SAM2`,
  not as a pure YOLO-only pipeline.
