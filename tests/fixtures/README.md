# Test fixtures

`reference_score.py` provides a reference scoring implementation for regression comparisons. Its tracking-quality import path is adapted to the current package layout.

For per-video, aggregate, and CLI comparisons, the tests explicitly set the reference object's score aggregation to the minimum available component score and use inputs with computable object diagnostics. Component calculations, tracking and validity gates, and model aggregation otherwise come from the reference implementation. Separate contract tests cover empty object evidence and objects with no computable components.

`legacy_metadata.csv` contains one explicit 20-column metadata row for each of the ten benchmark tasks. Compatibility tests use these records to check omitted-field restoration, numerical estimation, scoring, and explicit overrides of `Evaluator` and `Measurement_Method`.
