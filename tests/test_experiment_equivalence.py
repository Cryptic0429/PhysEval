"""Independent reference replay, saved-evidence sensitivity and mask controls."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import pytest

from experiments.scoring_sensitivity import scenarios
from physics_eval import scoring
from physics_eval.cli import parse_score_args
from physics_eval.quality.config import load_mask_config
from validation import synthetic_masks

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "tests/fixtures/reference_score.py"


def remove_positive_floor(reference):
    """Adapt only the intentional rule change in an isolated historical module.

    Replay inputs provide computable object diagnostics. The frozen scorer still
    supplies independent component calculations, gates and aggregation. Its
    source and provenance hash stay intact; missing-input errors are tested separately.
    """
    original = reference.object_mask_quality_score

    def object_score(item):
        historical_score, components = original(item)
        return (min(components.values()) if components else historical_score), components

    reference.object_mask_quality_score = object_score
    return reference


def reference_scorer():
    spec = importlib.util.spec_from_file_location("reference_score", REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return remove_positive_floor(module)


def assert_reference_fields(actual, expected):
    """Compare shared data recursively; reporting adds separate gate diagnostics."""
    if isinstance(actual, dict):
        for key in actual.keys() & expected.keys():
            assert_reference_fields(actual[key], expected[key])
    elif isinstance(actual, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_reference_fields(a, b)
    else:
        assert actual == expected


def test_reference_is_original_source_with_only_import_path_adapted():
    source = REFERENCE.read_bytes().replace(
        b"from physics_eval.quality.tracking_checks import analyze_tracking_quality_file",
        b"from physics_eval.utils.tracking_quality import analyze_tracking_quality_file",
    )
    assert hashlib.sha256(source).hexdigest() == "c458b99a97a335e1504b4f599ac10ee89928a8268d7b0e79c97bffe8155ec7a8"


def _write_results(root: Path, count=256) -> list[Path]:
    rng = random.Random(1742)
    config = load_mask_config()
    paths = []
    for index in range(count):
        metric = ("velocity", "acceleration", "mechanical_energy_conservation", "momentum_conservation_1d")[index % 4]
        physics = {
            "Status": ("valid", "weak_valid", "invalid", "failed")[index % 4],
            "Relative_Error": rng.choice([0., .01, .2, 2., 10.]),
            "Measured_Value": .8, "Target_Value": 1., "Failure_Reason": "",
        }
        if index % 7 == 0:
            physics.pop("Relative_Error")
        objects = []
        for oid in range(1, 2 + index % 3):
            obj = {"object_id": oid, "decision": rng.choice(["", "keep", "review", None]),
                   "reasons": rng.choice(["", "large_area_ratio", "low_middle_coverage", "insufficient_valid_middle_frames"])}
            for name, spec in config["components"].items():
                progress = rng.choice([0., .25, .5, .999, 1., None])
                if progress is not None:
                    obj[name] = spec["good_value"] + progress*(spec["bad_value"]-spec["good_value"])
            obj.setdefault("middle_coverage", 1.0)
            objects.append(obj)
        path = root / metric / f"P{index:03d}" / "result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "metric": metric, "prompt_id": path.parent.name, "metadata": {"Num_Objects": 2},
            "tracking_quality": {"status": "flagged" if index % 5 == 0 else "pass"},
            "physics_result": physics,
            "mask_qc": {"status": "flagged" if index % 11 == 0 else "pass", "objects": objects},
        }), encoding="utf-8")
        paths.append(path)
    return paths


@pytest.mark.parametrize("mask_policy", ["continuous", "penalty", "hard_fail", "ignore"])
@pytest.mark.parametrize("tracking_policy", ["hard_fail", "penalty", "ignore"])
def test_scores_and_aggregates_match_reference_without_positive_floor(tmp_path, mask_policy, tracking_policy):
    paths = _write_results(tmp_path)
    args = parse_score_args([
        "--metadata", "none", "--result-root", str(tmp_path), "--no-plots",
        "--mask-policy", mask_policy, "--tracking-policy", tracking_policy,
        "--quality-flag-multiplier", "0.4",
    ])
    reference = reference_scorer()
    current_rows, reference_rows = [], []
    for path in paths:
        current = scoring.score_one(path, scoring.DEFAULT_TOLERANCES, args)
        expected = reference.score_one(path, reference.DEFAULT_TOLERANCES, args)
        for key in current.keys() & expected.keys():
            assert current[key] == expected[key], (path, key)
        current_rows.append(current)
        reference_rows.append(expected)
    actual, expected = scoring.summarize(current_rows), reference.summarize(reference_rows)
    assert_reference_fields(actual, expected)
    assert actual["n_both_gates_pass"] == expected["n_physics_valid_or_weak"]
    for a, b in zip(actual["metrics"], expected["metrics"]):
        assert a["n_both_gates_pass"] == b["n_physics_valid_or_weak"]


def test_cli_reports_match_reference_and_have_no_version_directory(tmp_path):
    result_root = tmp_path / "results"
    _write_results(result_root, 24)
    reference_cli = tmp_path / "reference_without_floor.py"
    reference_cli.write_text(
        "import importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('reference_score', {str(REFERENCE)!r})\n"
        "reference = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(reference)\n\n"
        + inspect.getsource(remove_positive_floor)
        + "\nremove_positive_floor(reference).main()\n",
        encoding="utf-8",
    )
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    for executable, extra in (
        ([str(reference_cli)], ["--output-dir", str(tmp_path / "reference")]),
        ([str(ROOT / "physeval.py"), "score"], ["--metadata", "none"]),
    ):
        subprocess.run([sys.executable, *executable, "--result-root", str(result_root),
                        "--model-name", "fixture", "--no-plots", *extra],
                       check=True, capture_output=True, text=True, env=env)
    output = result_root / "score_reports/all_metrics/score_summary.json"
    current = json.loads(output.read_text())
    expected = json.loads((tmp_path / "reference/score_summary.json").read_text())
    assert_reference_fields(current["summary"], expected["summary"])
    settings = current["scoring"]
    assert settings["mask_score_gamma"] == .35
    assert "mask_flag_floor" not in settings
    assert "mask_flagged_missing_score" not in settings
    assert "otherwise stop scoring" in settings["mask_missing_evidence"]
    assert "mask_quality_config_hash" in settings
    assert not any("protocol" in key for key in settings)


def test_sensitivity_uses_production_baseline_without_positive_floor():
    values = scenarios()
    assert len(values) == 25
    assert values[0]["gamma"] == load_mask_config()["gamma"] == .35
    assert "flag_floor" not in load_mask_config()
    assert all(s["gamma"] == .35 for s in values if s["family"] != "gamma")


def test_synthetic_negative_control_reports_the_single_mask_score(tmp_path):
    output = tmp_path / "synthetic.csv"
    synthetic_masks.main(["--output", str(output)])
    with output.open(newline="", encoding="utf-8-sig") as handle:
        rows = {row["case"]: row for row in csv.DictReader(handle)}
    assert float(rows["correct_mask"]["ground_truth_mask_iou"]) == 1.0
    wrong = rows["stable_wrong_target_30px_offset"]
    assert float(wrong["centroid_rmse_px"]) == 30.0
    assert float(wrong["ground_truth_mask_iou"]) == 0.0
    assert float(wrong["mask_quality_score"]) == 1.0
    assert float(rows["alternating_width"]["mask_quality_score"]) == 0.0
