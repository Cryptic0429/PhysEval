"""CPU regression checks for saved-evidence experiments and their subprocess boundary."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.historical.qmask_protocol_comparison import baseline_rows, protocols
from experiments.tracking_comparison import cross_model_summary
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


def test_optional_frozen_scorer_regression(tmp_path):
    """Use the actual frozen original when provided by the migration audit."""
    raw = os.environ.get("PHYSEVAL_BASELINE_SCORER")
    if not raw:
        pytest.skip("Set PHYSEVAL_BASELINE_SCORER to audit the frozen original scorer")
    paths = _write_results(tmp_path / "results")
    rows = baseline_rows(Path(raw), tmp_path / "results", "model", tmp_path / "baseline", "none")
    args = parse_score_args([
        "--scoring-protocol", "legacy_v1", "--metadata", "none",
        "--result-root", str(tmp_path / "results"), "--model-name", "model", "--no-plots",
    ])
    for path in paths:
        current = scoring.score_data(json.loads(path.read_text()), path, scoring.DEFAULT_TOLERANCES, args)
        assert rows[(current["metric"], current["prompt_id"])] == current


def _comparison_fixture(root: Path):
    for model, baseline_pass, faster_pass, tam_pass in [
        ("model_a", [True, False], [True, True], [False, False]),
        ("model_b", [True, True], [False, True], [True, False]),
    ]:
        directory = root / model
        directory.mkdir(parents=True)
        with (directory / "per_run_results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "metric", "prompt_id", "mode", "success", "tracking_pass", "mask_quality_score",
            ])
            writer.writeheader()
            for index in range(2):
                for mode, passed, quality in [
                    ("yolo_sam2", baseline_pass[index], 0.7),
                    ("fasterrcnn_sam2", faster_pass[index], 0.8),
                    ("yolo_tam", tam_pass[index], 0.5),
                ]:
                    writer.writerow(dict(metric="velocity", prompt_id=f"V{index+1:03d}",
                                         mode=mode, success=True, tracking_pass=passed,
                                         mask_quality_score=quality))


def test_cross_model_report_matches_fixed_paired_fixture(tmp_path):
    _comparison_fixture(tmp_path / "results")
    args = cross_model_summary.parse_args([
        "--results-root", str(tmp_path / "results"), "--models", "model_a", "model_b",
        "--output-dir", str(tmp_path / "report"), "--bootstrap", "100", "--seed", "20260727",
    ])
    assert cross_model_summary.run(args) == 0
    with (tmp_path / "report/cross_model_pairwise.csv").open(newline="", encoding="utf-8") as handle:
        faster, tam = list(csv.DictReader(handle))
    assert faster["alternative_mode"] == "fasterrcnn_sam2"
    assert float(faster["mean_tracking_pass_difference"]) == 0.0
    assert float(tam["mean_tracking_pass_difference"]) == -0.5
    assert float(faster["mean_mask_quality_difference"]) == pytest.approx(0.1)
    assert float(tam["mean_mask_quality_difference"]) == pytest.approx(-0.2)
    assert faster["paired_n_by_model"] == '{"model_a": 2, "model_b": 2}'
    assert faster["discordant_alt_only"] == "1"
    assert faster["discordant_baseline_only"] == "1"
    assert tam["discordant_alt_only"] == "0"
    assert tam["discordant_baseline_only"] == "2"
    assert float(faster["mcnemar_exact_p"]) == 1.0
    assert float(tam["mcnemar_exact_p"]) == 0.5
    assert float(tam["mcnemar_holm_p"]) == 1.0
    report = (tmp_path / "report/cross_model_report.md").read_text()
    assert "paired videos: 4" in report
    assert "failed runs retained as failures" in report
    # Optional complete byte parity against the original cross-model script.
    frozen = os.environ.get("PHYSEVAL_BASELINE_CROSS_MODEL")
    if frozen:
        command = [sys.executable, frozen, "--results-root", str(tmp_path / "results"),
                   "--models", "model_a", "model_b", "--output-dir", str(tmp_path / "original"),
                   "--bootstrap", "100", "--seed", "20260727"]
        subprocess.run(command, check=True)
        for name in ["cross_model_pairwise.csv", "cross_model_report.md"]:
            assert (tmp_path / "original" / name).read_bytes() == (tmp_path / "report" / name).read_bytes()


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
