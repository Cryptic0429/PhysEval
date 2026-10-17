# Refactor validation

Date: 2026-10-13.

## Scope and intentional changes

The architecture migration preserves numeric algorithms, ten evaluator registry keys, quality protocol identities/hashes, result schemas, formal runtime directories.

Formal automatic initialization is now fixed to YOLO followed by motion fallback. The formal detector selector and legacy wrapper commands are removed. YOLO returning fewer than the required targets now triggers fallback. These are the agreed policy changes.

## Historical migration results (2026-10-13)

These figures describe the pre-removal suite, including the subsequently removed component experiments. They are not the current test count.

| Check | Result |
| --- | --- |
| Frozen original test baseline | 54 passed; 8 subtests passed |
| Refactored suite with frozen historical references enabled | 95 passed; 8 subtests passed |
| All ten evaluator controls, short-track failures, scoring gates, mask evidence and protocol fingerprints | Exact equality of a 29,073-byte JSON result |
| Full score report | All 7 JSON/CSV/PNG artifacts byte-identical |
| Original public eval vs new public eval, using existing tracks | All 15 artifacts identical: 14 byte-identical, XLSX cell-identical |
| Organized and flat existing-track evaluation | Valid, missing-track and static-track scenarios pass regression checks |
| Formal SAM2 orchestration with deterministic mocked inference | Tracking JSON, run configuration and mask PNG bytes equal |
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

The historical 95-pass run used external frozen reference scripts. The optional test requiring that external scorer was removed on 2026-10-17; the current suite does not require historical executables.

## Limitations

No CUDA GPU model inference or live YOLO/SAM2 model download was run in this environment. Mocked tracking and artifact equivalence verify orchestration and preserved behavior, not model accuracy or GPU throughput. Existing model-dependent behavior remains subject to the original observation assumptions and installed model versions.

[Architecture](ARCHITECTURE.md) · [Migration coverage](MIGRATION.md)

## Component experiment removal (2026-10-17)

The optional detector/tracker component experiment, its CLI commands, dedicated backends, dependencies, tools, and tests have been removed. Formal YOLO initialization, motion fallback, SAM2 tracking, scoring and saved-evidence analyses are retained. The remaining CPU suite passes with 79 tests, no skips, and 8 subtests. Shell syntax/help and public CLI help were checked. No GPU inference was run.
