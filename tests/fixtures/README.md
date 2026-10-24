`reference_score.py` freezes the scorer downloaded from the supplied anonymous
PhysEval repository. Its original SHA-256 is
`c458b99a97a335e1504b4f599ac10ee89928a8268d7b0e79c97bffe8155ec7a8`.
Only the tracking-quality import path is adapted to the current package layout;
the tracking-quality implementation itself is unchanged. The regression suite
checks this provenance hash. This historical source retains its original
flagged-score floor; production scoring no longer applies that floor.

For current per-video, aggregate and CLI comparisons, the tests load an isolated
copy of the historical module and explicitly replace only its object-score
aggregation: when components are available, use their minimum without an uplift.
The independent component calculations, missing-evidence fallbacks, gates and
model aggregation remain supplied by the frozen source. The fixture itself is
not rewritten and is not a selectable production scoring mode.
