"""CPU regression checks for saved-evidence experiments and their subprocess boundary."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import pytest

from experiments.historical.qmask_protocol_comparison import baseline_rows, protocols
from physics_eval import scoring
from physics_eval.cli import parse_score_args
from validation import synthetic_masks


ROOT = Path(__file__).resolve().parents[1]


def _write_results(root: Path) -> list[Path]:
    paths = []
    for index, (status, error, tracking) in enumerate(
        [("valid", 0.0, "pass"), ("weak_valid", 0.15, "pass"), ("valid", 0.0, "flagged")],
        start=1,
    ):
        path = root / "velocity" / f"V{index:03d}" / "result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "metric": "velocity", "prompt_id": path.parent.name,
            "metadata": {"Num_Objects": 1},
            "tracking_quality": {"status": tracking, "counted_object_ids": [1]},
            "physics_result": {
                "Status": status, "Relative_Error": error, "Measured_Value": 1.0,
                "Failure_Reason": "low_fit_r2" if status == "weak_valid" else "",
            },
            "mask_qc": {},
        }), encoding="utf-8")
        paths.append(path)
    return paths


def test_historical_baseline_subprocess_matches_explicit_legacy_scoring(tmp_path):
    result_root = tmp_path / "results"
    paths = _write_results(result_root)
    # The frozen scorer remains an executable input, not an imported module.
    # This local executable exercises the same stable command contract as a
    # pre-refactor scorer and makes the regression portable across checkouts.
    executable = tmp_path / "standalone_scorer.py"
    executable.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from physics_eval.cli import parse_score_args\n"
        "from physics_eval.scoring import run\n"
        "run(parse_score_args(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    rows = baseline_rows(executable, result_root, "model", tmp_path / "baseline", "none")
    args = parse_score_args([
        "--scoring-protocol", "legacy_v1", "--metadata", "none",
        "--result-root", str(result_root), "--model-name", "model", "--no-plots",
    ])
    for path in paths:
        current = scoring.score_data(json.loads(path.read_text()), path, scoring.DEFAULT_TOLERANCES, args)
        baseline = rows[(current["metric"], current["prompt_id"])]
        assert baseline == current
    assert rows[("velocity", "V002")]["status_multiplier"] == 0.8
    assert rows[("velocity", "V003")]["effective_for_score"] is False


def test_historical_protocol_order_and_synthetic_negative_control_are_preserved(tmp_path):
    values = protocols()
    assert values[0]["protocol_id"] == "legacy_v1"
    assert len(values) == 13
    output = tmp_path / "synthetic.csv"
    synthetic_masks.main(["--output", str(output)])
    with output.open(newline="", encoding="utf-8-sig") as handle:
        rows = {row["case"]: row for row in csv.DictReader(handle)}
    assert float(rows["correct_mask"]["ground_truth_mask_iou"]) == 1.0
    wrong = rows["stable_wrong_target_30px_offset"]
    assert float(wrong["centroid_rmse_px"]) == 30.0
    assert float(wrong["ground_truth_mask_iou"]) == 0.0
    assert float(wrong["grouped_v2_candidate_score"]) == 1.0
    assert float(rows["alternating_width"]["grouped_v2_candidate_score"]) == pytest.approx(0.6700680272108843)
