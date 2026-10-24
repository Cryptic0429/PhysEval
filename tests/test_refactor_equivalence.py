"""Frozen pre-refactor numeric outcomes across all physical estimators and gates.

Expected values were captured from the original source before the architecture
migration; these fixtures exercise the public data APIs without model weights.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physics_eval import scoring
from physics_eval.evaluators import EVALUATOR_REGISTRY
from physics_eval.quality.config import load_mask_config, mask_config_hash


def _track(position):
    rows = []
    for frame in range(120):
        t = frame / 30
        x, y = position(t)
        rows.append(dict(
            frame=frame, t=t, object_id="1", cx_px=x, cy_px=y,
            area_px2=100 * math.pi, bbox_x=x-10, bbox_y=y-10,
            bbox_w=20., bbox_h=20., equivalent_diameter_px=20.,
        ))
    return pd.DataFrame(rows)


def _fixtures():
    fixtures = {
        "eval_velocity": _track(lambda t: (100+100*t, 80)),
        "eval_acceleration": _track(lambda t: (100+40*t*t, 80)),
        "eval_gravity": _track(lambda t: (100, 80+980*t*t)),
        "eval_friction": _track(lambda t: (100+200*t-20*t*t, 80)),
        "eval_restitution": _track(lambda t: (100+60*min(t, 1)-30*max(0, t-1), 80)),
        "eval_density_acceleration": _track(lambda t: (100, 80+490*t*t)),
        "eval_viscosity": _track(lambda t: (100, 80+100*t)),
        "eval_spring": _track(lambda t: (100+30*math.sin(2*math.pi*t), 80)),
        "eval_energy": _track(lambda t: (100, 80+980*t*t)),
    }
    first = _track(lambda t: (100+60*min(t, 2), 80))
    second = _track(lambda t: (220+60*max(0, t-2), 80))
    second["object_id"] = "2"
    fixtures["eval_momentum_1d"] = pd.concat([first, second], ignore_index=True)
    return fixtures


@pytest.mark.parametrize("name, expected, status", [
    ("eval_velocity", .5, "valid"),
    ("eval_acceleration", .4, "valid"),
    ("eval_gravity", 9.8, "valid"),
    ("eval_friction", .020408163265306152, "valid"),
    ("eval_restitution", .5, "valid"),
    ("eval_density_acceleration", 2000., "valid"),
    ("eval_viscosity", 10.888888888888898, "valid"),
    ("eval_spring", 7.895683520871486, "weak_valid"),
    ("eval_energy", .991614292775941, "valid"),
    ("eval_momentum_1d", 0., "valid"),
])
def test_estimator_baseline_measurements(name, expected, status):
    row = dict(
        Evaluator=name, Target_Value=1., Motion_Axis="x",
        Scale_Mode="object_diameter", Calibration_Value_m=.1,
        Known_Parameters_JSON=json.dumps(dict(
            g=9.8, liquid_density_kg_m3=1000, solid_density_kg_m3=2000,
            mass_kg=.2, m1_kg=1., m2_kg=1.,
        )),
    )
    evaluator = EVALUATOR_REGISTRY[name]
    track = _fixtures()[name]
    result = evaluator(track, row)
    assert result["Measured_Value"] == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert result["Status"] == status
    assert evaluator(track.iloc[:1], row)["Status"] == "failed"


def test_scoring_rule_fingerprint():
    assert mask_config_hash(load_mask_config()) == "984bc4e9488887ead9f83eeccecdfe9b0b7f2903c685436bccbb323cc4401cec"


def _score_config():
    return SimpleNamespace(
        penalty_multiplier=None, weak_valid_multiplier=.8,
        mask_policy="continuous", qc_flag_multiplier=.7, strict_qc=False,
        tracking_policy="hard_fail", quality_flag_multiplier=0.,
        min_motion_extent_ratio=.75, min_object_coverage=.5,
        recompute_tracking_quality=False, allow_contract_errors=False,
        no_plots=True, tau=None, tolerance_json=None,
        model_name="fixture",
        result_root="/fixture", metadata="none", metric=None, output_dir=None,
        mask_config=load_mask_config(),
    )


def _score_evidence():
    config = load_mask_config()
    obj = {key: spec["good_value"] for key, spec in config["components"].items()}
    obj.update(object_id=1, decision="keep", reasons="")
    return {
        "metric": "velocity", "prompt_id": "V001", "metadata": {"Num_Objects": 1},
        "tracking_quality": {"status": "pass", "counted_object_ids": [1]},
        "physics_result": {"Status": "valid", "Relative_Error": .2,
                           "Measured_Value": .8, "Target_Value": 1.},
        "mask_qc": {"objects": [obj]},
    }


def test_in_memory_scoring_matches_reference_gates_and_missing_evidence_behavior(tmp_path):
    args = _score_config()
    data = _score_evidence()
    path = tmp_path / "result.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    score = lambda value: scoring.score_data(value, path, scoring.DEFAULT_TOLERANCES, args)
    original = score(data)
    assert scoring.score_one(path, scoring.DEFAULT_TOLERANCES, args) == original
    assert original["adjusted_score"] == pytest.approx(51.3417119032592)

    weak = copy.deepcopy(data)
    weak["physics_result"]["Status"] = "weak_valid"
    assert score(weak)["adjusted_score"] == pytest.approx(41.073369522607365)

    missing = copy.deepcopy(data)
    missing["mask_qc"] = {}
    unavailable = score(missing)
    assert unavailable["effective_for_score"]
    assert unavailable["adjusted_score"] == pytest.approx(original["adjusted_score"])
    assert unavailable["score_available"]

    for section, status, reason in [
        ("tracking_quality", "flagged", "tracking_quality_failed"),
        ("physics_result", "failed", "physics_status_not_valid"),
    ]:
        rejected = copy.deepcopy(data)
        rejected[section]["status" if section == "tracking_quality" else "Status"] = status
        result = score(rejected)
        assert result["adjusted_score"] == 0.
        assert not result["effective_for_score"]
        assert result["score_exclusion_reasons"] == reason

    invalid_contract = copy.deepcopy(data)
    invalid_contract["physics_result"] = {"Status": "valid"}
    with pytest.raises(ValueError, match="no score can be computed"):
        score(invalid_contract)
