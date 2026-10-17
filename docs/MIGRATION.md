# Migration coverage

This table records the retained evaluation files from the architecture migration. The optional detector/tracker component experiment and its dedicated dependencies, tools, and tests were removed on 2026-10-17.

This table tracks capability ownership, not one-to-one renames. Wrapper commands were merged. Numeric algorithms, registry keys and output contracts are protected by regression tests. Formal initialization intentionally fixes YOLO then motion and adds fallback for insufficient required targets.

| Original path | Current owner | Treatment |
| --- | --- | --- |
| .gitattributes | [.gitattributes](../.gitattributes) | Preserved in place |
| .gitignore | [.gitignore](../.gitignore) | Preserved in place |
| BATCH_USAGE.md | [BATCH_USAGE.md](../BATCH_USAGE.md) | Preserved in place |
| README.md | [README.md](../README.md) | Preserved in place |
| README_zh-CN.md | [README_zh-CN.md](../README_zh-CN.md) | Preserved in place |
| SCORING_USAGE.md | [SCORING_USAGE.md](../SCORING_USAGE.md) | Preserved in place |
| benchmark/metadata/README.md | [benchmark/metadata/README.md](../benchmark/metadata/README.md) | Preserved in place |
| benchmark/metadata/README_zh-CN.md | [benchmark/metadata/README_zh-CN.md](../benchmark/metadata/README_zh-CN.md) | Preserved in place |
| benchmark/metadata/csv/phys_t2v_bench_metadata__field_guide.csv | [benchmark/metadata/csv/phys_t2v_bench_metadata__field_guide.csv](../benchmark/metadata/csv/phys_t2v_bench_metadata__field_guide.csv) | Preserved in place |
| benchmark/metadata/csv/phys_t2v_bench_metadata__metadata.csv | [benchmark/metadata/csv/phys_t2v_bench_metadata__metadata.csv](../benchmark/metadata/csv/phys_t2v_bench_metadata__metadata.csv) | Preserved in place |
| benchmark/metadata/csv/phys_t2v_bench_metadata__summary.csv | [benchmark/metadata/csv/phys_t2v_bench_metadata__summary.csv](../benchmark/metadata/csv/phys_t2v_bench_metadata__summary.csv) | Preserved in place |
| benchmark/metadata/csv/phys_t2v_bench_prompts__All_Prompts.csv | [benchmark/metadata/csv/phys_t2v_bench_prompts__All_Prompts.csv](../benchmark/metadata/csv/phys_t2v_bench_prompts__All_Prompts.csv) | Preserved in place |
| benchmark/metadata/csv/phys_t2v_bench_prompts__README.csv | [benchmark/metadata/csv/phys_t2v_bench_prompts__README.csv](../benchmark/metadata/csv/phys_t2v_bench_prompts__README.csv) | Preserved in place |
| benchmark/metadata/csv/phys_t2v_bench_prompts__Summary.csv | [benchmark/metadata/csv/phys_t2v_bench_prompts__Summary.csv](../benchmark/metadata/csv/phys_t2v_bench_prompts__Summary.csv) | Preserved in place |
| benchmark/metadata/phys_t2v_bench_metadata.xlsx | [benchmark/metadata/phys_t2v_bench_metadata.xlsx](../benchmark/metadata/phys_t2v_bench_metadata.xlsx) | Preserved in place |
| benchmark/metadata/phys_t2v_bench_prompts.xlsx | [benchmark/metadata/phys_t2v_bench_prompts.xlsx](../benchmark/metadata/phys_t2v_bench_prompts.xlsx) | Preserved in place |
| configs/scoring/grouped_v2_candidate.json | [configs/scoring/grouped_v2_candidate.json](../configs/scoring/grouped_v2_candidate.json) | Preserved in place |
| configs/scoring/legacy_v1.json | [configs/scoring/legacy_v1.json](../configs/scoring/legacy_v1.json) | Preserved in place |
| physeval.py | [physeval.py](../physeval.py), [physics_eval/cli.py](../physics_eval/cli.py) | Migrated / merged |
| physics_eval/__init__.py | [physics_eval/__init__.py](../physics_eval/__init__.py) | Preserved in place |
| physics_eval/evaluators/__init__.py | [physics_eval/evaluators/__init__.py](../physics_eval/evaluators/__init__.py) | Preserved in place |
| physics_eval/evaluators/acceleration.py | [physics_eval/evaluators/acceleration.py](../physics_eval/evaluators/acceleration.py) | Preserved in place |
| physics_eval/evaluators/common.py | [physics_eval/evaluators/common.py](../physics_eval/evaluators/common.py) | Preserved in place |
| physics_eval/evaluators/density_acceleration.py | [physics_eval/evaluators/density_acceleration.py](../physics_eval/evaluators/density_acceleration.py) | Preserved in place |
| physics_eval/evaluators/energy.py | [physics_eval/evaluators/energy.py](../physics_eval/evaluators/energy.py) | Preserved in place |
| physics_eval/evaluators/friction.py | [physics_eval/evaluators/friction.py](../physics_eval/evaluators/friction.py) | Preserved in place |
| physics_eval/evaluators/gravity.py | [physics_eval/evaluators/gravity.py](../physics_eval/evaluators/gravity.py) | Preserved in place |
| physics_eval/evaluators/momentum_1d.py | [physics_eval/evaluators/momentum_1d.py](../physics_eval/evaluators/momentum_1d.py) | Preserved in place |
| physics_eval/evaluators/restitution.py | [physics_eval/evaluators/restitution.py](../physics_eval/evaluators/restitution.py) | Preserved in place |
| physics_eval/evaluators/spring.py | [physics_eval/evaluators/spring.py](../physics_eval/evaluators/spring.py) | Preserved in place |
| physics_eval/evaluators/velocity.py | [physics_eval/evaluators/velocity.py](../physics_eval/evaluators/velocity.py) | Preserved in place |
| physics_eval/evaluators/viscosity.py | [physics_eval/evaluators/viscosity.py](../physics_eval/evaluators/viscosity.py) | Preserved in place |
| physics_eval/quality/__init__.py | [physics_eval/quality/__init__.py](../physics_eval/quality/__init__.py) | Preserved in place |
| physics_eval/quality/adapters.py | [physics_eval/io/tracks.py](../physics_eval/io/tracks.py) | Migrated / merged |
| physics_eval/quality/diagnostics.py | [physics_eval/quality/mask_evidence.py](../physics_eval/quality/mask_evidence.py) | Migrated / merged |
| physics_eval/quality/protocol.py | [physics_eval/quality/protocol.py](../physics_eval/quality/protocol.py) | Preserved in place |
| physics_eval/quality/scoring.py | [physics_eval/quality/mask_score.py](../physics_eval/quality/mask_score.py) | Migrated / merged |
| physics_eval/run_batch_eval.py | [physics_eval/estimation.py](../physics_eval/estimation.py) | Migrated / merged |
| physics_eval/run_video_batch.py | [physics_eval/pipeline.py](../physics_eval/pipeline.py), [physics_eval/config.py](../physics_eval/config.py), [physics_eval/tracking/policy.py](../physics_eval/tracking/policy.py), [physics_eval/io/results.py](../physics_eval/io/results.py), [physics_eval/quality/inspection.py](../physics_eval/quality/inspection.py) | Migrated / merged |
| physics_eval/tests/run_synthetic_smoke.py | [tests/integration/synthetic_estimation.py](../tests/integration/synthetic_estimation.py) | Migrated / merged |
| physics_eval/tests/test_batch_initialization.py | [tests/test_initialization.py](../tests/test_initialization.py) | Migrated / merged |
| physics_eval/tests/test_qmask_protocols.py | [tests/test_mask_quality.py](../tests/test_mask_quality.py) | Migrated / merged |
| physics_eval/tests/test_spring.py | [tests/test_spring.py](../tests/test_spring.py) | Migrated / merged |
| physics_eval/tests/test_viscosity_direction.py | [tests/test_viscosity.py](../tests/test_viscosity.py) | Migrated / merged |
| physics_eval/utils/__init__.py | [physics_eval/io/__init__.py](../physics_eval/io/__init__.py), [physics_eval/math/__init__.py](../physics_eval/math/__init__.py) | Migrated / merged |
| physics_eval/utils/calibration.py | [physics_eval/calibration.py](../physics_eval/calibration.py) | Migrated / merged |
| physics_eval/utils/fitting.py | [physics_eval/math/fitting.py](../physics_eval/math/fitting.py) | Migrated / merged |
| physics_eval/utils/geometry.py | [physics_eval/math/geometry.py](../physics_eval/math/geometry.py) | Migrated / merged |
| physics_eval/utils/io.py | [physics_eval/io/metadata.py](../physics_eval/io/metadata.py), [physics_eval/io/results.py](../physics_eval/io/results.py), [physics_eval/io/tracks.py](../physics_eval/io/tracks.py) | Migrated / merged |
| physics_eval/utils/quality.py | [physics_eval/quality/estimation_checks.py](../physics_eval/quality/estimation_checks.py) | Migrated / merged |
| physics_eval/utils/tracking.py | [physics_eval/io/tracks.py](../physics_eval/io/tracks.py) | Migrated / merged |
| physics_eval/utils/tracking_init.py | [physics_eval/tracking/initialization.py](../physics_eval/tracking/initialization.py), [physics_eval/schemas.py](../physics_eval/schemas.py) | Migrated / merged |
| physics_eval/utils/tracking_quality.py | [physics_eval/quality/tracking_checks.py](../physics_eval/quality/tracking_checks.py) | Migrated / merged |
| requirements.txt | [requirements.txt](../requirements.txt) | Preserved in place |
| scripts/analyze_qmask_protocols.py | [experiments/historical/qmask_protocol_comparison.py](../experiments/historical/qmask_protocol_comparison.py) | Migrated / merged |
| scripts/analyze_scoring_sensitivity.py | [experiments/scoring_sensitivity.py](../experiments/scoring_sensitivity.py) | Migrated / merged |
| scripts/detection/yolo_init.py | [physics_eval/tracking/yolo.py](../physics_eval/tracking/yolo.py) | Migrated / merged |
| scripts/preflight_open_video_validation.py | [validation/open_video_preflight.py](../validation/open_video_preflight.py) | Migrated / merged |
| scripts/quality/check_mask_area_stability.py | [tools/inspect_masks.py](../tools/inspect_masks.py), [physics_eval/quality/inspection.py](../physics_eval/quality/inspection.py), [physics_eval/reporting.py](../physics_eval/reporting.py) | Migrated / merged |
| scripts/report_io.py | [physics_eval/io/reports.py](../physics_eval/io/reports.py) | Migrated / merged |
| scripts/run_batch_simple.py | [physics_eval/cli.py](../physics_eval/cli.py), [physics_eval/config.py](../physics_eval/config.py) | Migrated / merged |
| scripts/sam2/run_sam2_track.py | [physics_eval/tracking/sam2.py](../physics_eval/tracking/sam2.py), [physics_eval/tracking/motion.py](../physics_eval/tracking/motion.py), [physics_eval/tracking/video.py](../physics_eval/tracking/video.py), [physics_eval/tracking/output.py](../physics_eval/tracking/output.py), [physics_eval/tracking/worker.py](../physics_eval/tracking/worker.py) | Migrated / merged |
| scripts/sam2/run_sam2_track_prompts.py | [physics_eval/tracking/initialization.py](../physics_eval/tracking/initialization.py), [physics_eval/tracking/worker.py](../physics_eval/tracking/worker.py) | Migrated / merged |
| scripts/sam2/run_sam2_track_two_objects.py | [physics_eval/config.py](../physics_eval/config.py), [physics_eval/tracking/worker.py](../physics_eval/tracking/worker.py) | Migrated / merged |
| scripts/score_results.py | [physics_eval/scoring.py](../physics_eval/scoring.py), [physics_eval/reporting.py](../physics_eval/reporting.py), [physics_eval/cli.py](../physics_eval/cli.py) | Migrated / merged |
| scripts/setup_sam2_assets.sh | [tools/setup_assets.sh](../tools/setup_assets.sh) | Migrated / merged |
| scripts/validate_qmask_synthetic.py | [validation/synthetic_masks.py](../validation/synthetic_masks.py) | Migrated / merged |

## Added boundaries and regression checks

New cli/config/schema modules make the public contract explicit. tracking.worker isolates inference and tracking.policy owns row configuration and provenance. quality.inspection owns saved-mask inspection. New public-CLI, tracking-backend, numeric-core, pipeline and experimental regression tests verify the reorganized interfaces.

The planning document outside the repository records pre-migration paths; [ARCHITECTURE.md](ARCHITECTURE.md) describes the implemented structure.

Coverage: **93/93 original project files; no unmapped source files.**
