`reference_score.py` freezes the scorer downloaded from the supplied anonymous
PhysEval repository. Its original SHA-256 is
`c458b99a97a335e1504b4f599ac10ee89928a8268d7b0e79c97bffe8155ec7a8`.
Only the tracking-quality import path is adapted to the current package layout;
the tracking-quality implementation itself is unchanged. The regression suite
checks this provenance hash and compares actual per-video and aggregate outputs,
including the flagged-score floor and missing-evidence fallbacks. This reference
is a test fixture, not a selectable production scoring mode.
