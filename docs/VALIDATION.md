# Refactor validation

Date: 2026-10-13.

## Scope and intentional changes

The architecture migration preserves numeric algorithms, ten evaluator registry keys, quality protocol identities/hashes, result schemas, runtime directories, comparison statistics and guarded cleanup behavior.

Formal automatic initialization is now fixed to YOLO followed by motion fallback. The formal detector selector and legacy wrapper commands are removed. YOLO returning fewer than the required targets now triggers fallback. These are the agreed policy changes.

## Results

| Check | Result |
| --- | --- |
| Frozen original test baseline | 54 passed; 8 subtests passed |
| Refactored suite with frozen historical references enabled | 95 passed; 8 subtests passed |
| All ten evaluator controls, short-track failures, scoring gates, mask evidence and protocol fingerprints | Exact equality of a 29,073-byte JSON result |
| Full score report | All 7 JSON/CSV/PNG artifacts byte-identical |
| Original public eval vs new public eval, using existing tracks | All 15 artifacts identical: 14 byte-identical, XLSX cell-identical |
| Organized and flat existing-track evaluation | Valid, missing-track and static-track scenarios pass regression checks |
| Paired cross-model report | CSV and Markdown byte-identical to frozen original on fixed paired fixtures |
| Formal SAM2 orchestration with deterministic mocked inference | Tracking JSON, run configuration and mask PNG bytes equal |
| Formal/comparison candidate generation | Frozen motion and deterministic YOLO controls equal |
| Public command help | All 9 commands pass without loading torch |
| Minimal help environment | Formal eval/score and internal worker help pass without third-party packages |
| Documentation | 52 public CLI examples parse; all Markdown fences and local links resolve |
| Shell asset tool | Syntax and help checks pass |
| Migration coverage | 93/93 original project files have verified current owners |

The comparisons use a frozen copy of the workspace taken before code migration, including the user's then-current documentation edits. Tests ran on Python 3.12 in a CPU environment with NumPy, pandas, SciPy, Matplotlib, OpenCV, Pillow, openpyxl and pytest.

## Run the portable suite

Install the documented core environment, then:

~~~bash
python -m pip install -r requirements-test.txt
python -m pytest tests
python -m tests.integration.synthetic_estimation
~~~

The optional frozen-original historical regression is skipped when no reference executable is provided. The normal suite therefore has 94 passing tests plus one optional skip; the recorded 95-pass run supplied frozen original references through the environment variables described in the experimental regression tests.

## Limitations

No CUDA GPU model inference or live YOLO/SAM2/TAM model download was run in this environment. Mocked tracking and artifact equivalence verify orchestration and preserved behavior, not model accuracy or GPU throughput. Existing model-dependent behavior remains subject to the original observation assumptions and installed model versions.

[Architecture](ARCHITECTURE.md) · [Migration coverage](MIGRATION.md)
