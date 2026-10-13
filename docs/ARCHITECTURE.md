# Architecture

PhysEval separates reusable measurement code, workflow orchestration, research experiments, validation, and operational tools. The public entry point is [physeval.py](../physeval.py); it calls [physics_eval.cli](../physics_eval/cli.py) directly. The former script-forwarding layers have been removed.

## Workflows

```mermaid
flowchart TD
    B["Benchmark metadata + supplied videos"] --> E["eval: formal pipeline"]
    A["Dependencies, model assets, input checks"] --> E
    E --> T["Explicit initialization → SAM2"]
    T --> M["Tracks, masks, diagnostics, physical estimates"]
    M --> R["Per-video result.json"]
    R --> S["score: gates, Q_mask, physical scores"]
    S --> O["Metric and model reports"]
    B --> C["compare: paired component experiments"]
    C --> CS["compare-summary"]
    CS --> CM["compare-cross-model"]
    R --> SA["analyze-sensitivity"]
    R --> HA["analyze-mask-protocols: historical replay"]
    V["Synthetic and real-video validation"] -.exercise shared implementations.-> T
    V -.exercise.-> S
    U["Unit and integration tests"] -.verify.-> E
    U -.verify.-> C
    MA["Asset, GPU, cache and shard tools"] -.support.-> A
    MA -.support.-> C
```

Video generation remains external. Validation, analysis and tests are separate workflows; eval does not automatically run them or invoke score.

## Module responsibilities

| Layer | Modules | Responsibility |
| --- | --- | --- |
| Public boundary | physics_eval/cli.py, config.py | Command parsing, validated defaults and paths; fixed formal initialization policy |
| Formal workflow | physics_eval/pipeline.py | Iterate samples, run the tracking worker, collect diagnostics and estimates, preserve failure records |
| Data interfaces | physics_eval/schemas.py, io/ | Prompt/video records, metadata, normalized tracks, result artifacts and reports |
| Initialization | physics_eval/tracking/initialization.py, policy.py, detectors.py, motion.py, yolo.py | Automatic/manual/renderer_gt prompts, detection candidates, fallback and provenance |
| Tracking | physics_eval/tracking/sam2.py, tam.py, video.py, output.py | Shared inference, frame extraction and per-frame track/mask records |
| Internal execution | physics_eval/tracking/worker.py | Isolated inference subprocess boundary; explicit prompt manifest rather than callback replacement |
| Estimation | physics_eval/estimation.py, calibration.py, evaluators/, math/ | Ten registered physical tasks, scale recovery, fits, geometry and time alignment |
| Reliability | physics_eval/quality/ | Tracking gate, estimation checks, mask evidence, versioned Q_mask and saved-mask inspection |
| Scoring | physics_eval/scoring.py | Eligibility, physical error scoring, quality weighting, completeness and aggregate statistics |
| Presentation | physics_eval/reporting.py | Diagnostic/trajectory plots, score plots and report writers |
| Research | experiments/ | Paired component comparison, model-stratified inference, sensitivity and historical replay |
| Validation | validation/ | Curated open-video package preflight and controlled synthetic-mask experiments |
| Maintenance | tools/ | Asset preparation, environment/GPU inspection and guarded output cleanup |
| Regression | tests/ | Unit, analytical-control, mocked inference and integration checks |

The ten evaluator filenames and metadata registry keys remain unchanged. The Q_mask protocol names and fingerprints remain unchanged.

## Formal initialization policy

Automatic initialization is fixed to YOLO first, with motion fallback when YOLO raises, has no usable candidates, or cannot provide the required object count. Formal eval and its worker have no detector-selection argument. Per-row overrides cannot select pure YOLO or pure motion.

Manual and explicitly enabled renderer_gt prompts retain their prior semantics and bypass automatic detection. Renderer_gt eligibility and provenance remain explicit.

Comparison experiments select named detector/tracker combinations as part of their experiment design. They share prediction, prompt injection, propagation, IO and quality implementations. Explicit formal/comparison candidate and reverse-propagation policies preserve the existing scientific behaviors instead of silently substituting one experiment's policy for another.

## Data flow and preserved outputs

1. Metadata supplies target values, units, evaluator keys, object counts, calibration and known parameters.
2. Initial prompts specify object ID, frame, foreground points and optional negative points/box.
3. The tracker writes centers, boxes, areas, masks and initialization evidence.
4. Formal evaluation records tracking gates, mask evidence, physical measurements and validity.
5. result.json organizes those records and artifact paths.
6. score applies tracking and physical-validity gates, then the continuous mask multiplier and physical-error score; reports aggregate within metrics and across metrics.

eval may still compute physical estimates for a flagged track, as before. score determines the effective scoring population. Component comparison has no physical-validity gate and does not produce complete physical scores.

Existing runtime paths are retained for compatibility:

| Content | Location |
| --- | --- |
| External model assets | repo/ and yolov8n.pt |
| User videos | data/t2v_videos/ |
| Formal frame cache | data/custom/<scene>/frames/ |
| Formal masks and intermediate tracks | sam2_tracks/<scene>/ |
| Organized formal results | batch_eval_results/<model>/<metric>/<prompt_id>/ |
| Score reports | <result-root>/score_reports/<protocol>/<metric-or-all_metrics>/ |
| Comparison outputs and frame cache | compare/outputs/ and compare/cache/ |

compare is now an output location; its source lives in experiments/tracking_comparison. Formal masks remain referenced from sam2_tracks to avoid duplicating every PNG.

## Dependency rules

- Core functions receive explicit arguments and do not parse sys.argv.
- The public CLI calls functions directly; it does not run scripts via runpy.
- The formal pipeline launches the internal tracking worker with an explicit PYTHONPATH, so a separate workspace root remains supported.
- Research calls scorer.score_data and stable configuration APIs; it does not replace scorer.load_json.
- Optional frozen-scorer historical regression uses an isolated subprocess.
- SAM2/TAM/PyTorch imports are lazy. CPU estimation, scoring, mask controls and tests do not require loading a GPU model.
- Tests and experiments depend on the core. Core modules do not import experiment runners, maintenance scripts or tests.
- The internal worker parses --help before loading any third-party package.

## Compatibility and validation

Numeric estimators, fits, score formulas, validity statuses, rule fingerprints, output schemas, output directories, paired statistics and maintenance protections are preserved. The intentional policy changes are removal of the formal detector selector and fallback when YOLO returns too few required targets. Old standalone wrapper commands are replaced by the documented public commands.

Regression checks compare frozen original code against the new implementations, including all ten estimators, scoring gates, mask evidence, protocol hashes, score-report bytes, paired cross-model reports and organized/flat eval-only artifacts. SAM2 orchestration is tested using deterministic mocked inference; these checks do not establish GPU model accuracy or performance.

[Migration coverage](MIGRATION.md) maps every original project file to its current owner.

[Validation results](VALIDATION.md) records regression checks and the GPU verification limit.
