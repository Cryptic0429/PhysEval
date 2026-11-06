"""Pixel-fit validity is independent of calibration; outputs retain SI units."""
from __future__ import annotations

import json

import numpy as np
import pytest

from physics_eval.evaluators import EVALUATOR_REGISTRY
from tests.test_refactor_equivalence import _fixtures


NAMES = ("velocity", "acceleration", "gravity", "friction", "density_acceleration")


def metadata(length):
    return dict(
        Target_Value=1., Motion_Axis="x", Scale_Mode="object_diameter",
        Calibration_Value_m=length,
        Known_Parameters_JSON=json.dumps({"g": 9.8, "liquid_density_kg_m3": 1000}),
    )


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("expected_r2, expected_status", [(0.50, "invalid"), (0.75, "weak_valid")])
def test_pixel_fit_quality_does_not_change_with_calibration(name, expected_r2, expected_status):
    track = _fixtures()["eval_" + name].copy()
    count = len(track) // 3 if name == "density_acceleration" else len(track)
    t = track["t"].to_numpy()[:count]
    vertical = name in {"gravity", "density_acceleration"}
    column = "cy_px" if vertical else "cx_px"
    fitted = track[column].to_numpy()[:count]
    degree = 1 if name == "velocity" else 2

    # Orthogonal noise gives a known R² without invoking the production fitter.
    basis = np.vander(t, degree + 1)
    noise = np.sin(7 * t)
    noise -= basis @ np.linalg.lstsq(basis, noise, rcond=None)[0]
    noise *= np.sqrt(
        np.sum((fitted - fitted.mean()) ** 2) * (1 / expected_r2 - 1)
        / np.sum(noise ** 2)
    )
    track.loc[track.index[:count], column] = fitted + noise

    # Include a very small scale: an absolute metric-domain R² floor must not
    # turn a poor pixel fit into a valid constant trajectory.
    for length in (0.1, 2e-10):
        result = EVALUATOR_REGISTRY["eval_" + name](track, metadata(length))
        assert result["Fit_R2"] == pytest.approx(expected_r2, abs=1e-12)
        assert result["Status"] == expected_status


@pytest.mark.parametrize("name, expected", [
    ("velocity", {"intercept": 0.5}),
    ("acceleration", {"quadratic_A": 0.2, "linear_B": 0., "constant_C": 0.5}),
    ("gravity", {"quadratic_A": 4.9, "linear_B": 0., "constant_C": 0.4}),
])
def test_fit_diagnostics_keep_their_original_metric_units(name, expected):
    result = EVALUATOR_REGISTRY["eval_" + name](_fixtures()["eval_" + name], metadata(0.1))
    extra = json.loads(result["Extra_JSON"])
    for key, value in expected.items():
        assert extra[key] == pytest.approx(value, rel=1e-12, abs=1e-12)
