"""Analytical controls for the signed downward terminal-velocity contract."""
from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

from physics_eval.evaluators.viscosity import eval_viscosity
from physics_eval.utils.fitting import terminal_velocity


class ViscosityDirectionTests(unittest.TestCase):
    def setUp(self):
        self.t = np.arange(90, dtype=float) / 30.0
        self.diameter_m = 0.003
        self.diameter_px = 20.0
        self.scale = self.diameter_m / self.diameter_px
        self.expected_eta = 0.4
        radius = self.diameter_m / 2.0
        self.downward_speed = 2.0 * radius**2 * (7800.0 - 1000.0) * 9.8 / (9.0 * self.expected_eta)
        self.metadata = {
            "Scale_Mode": "object_diameter",
            "Calibration_Value_m": self.diameter_m,
            "Target_Value": self.expected_eta,
            "Known_Parameters_JSON": json.dumps({
                "solid_density_kg_m3": 7800.0,
                "liquid_density_kg_m3": 1000.0,
                "g": 9.8,
            }),
        }

    def track(self, direction):
        return pd.DataFrame({
            "frame": np.arange(len(self.t)),
            "t": self.t,
            "object_id": 1,
            "cx_px": np.full(len(self.t), 100.0),
            "cy_px": 100.0 + direction * self.downward_speed * self.t / self.scale,
            "equivalent_diameter_px": np.full(len(self.t), self.diameter_px),
        })

    def test_terminal_velocity_preserves_direction(self):
        for direction in (1, -1):
            with self.subTest(direction=direction):
                displacement = direction * self.downward_speed * self.t
                velocity, _, _ = terminal_velocity(self.t, displacement)
                self.assertAlmostEqual(velocity, direction * self.downward_speed)

    def test_downward_settling_recovers_known_viscosity(self):
        result = eval_viscosity(self.track(1), self.metadata)
        self.assertEqual(result["Status"], "valid")
        self.assertAlmostEqual(result["Measured_Value"], self.expected_eta)
        self.assertAlmostEqual(json.loads(result["Extra_JSON"])["terminal_velocity_mps"], self.downward_speed)

    def test_upward_motion_is_not_a_settling_measurement(self):
        result = eval_viscosity(self.track(-1), self.metadata)
        self.assertEqual(result["Status"], "failed")
        self.assertEqual(result["Failure_Reason"], "terminal_velocity_not_found")
        self.assertIsNone(result["Measured_Value"])

    def test_stationary_object_has_no_terminal_settling_measurement(self):
        result = eval_viscosity(self.track(0), self.metadata)
        self.assertEqual(result["Status"], "failed")
        self.assertIsNone(result["Measured_Value"])


if __name__ == "__main__":
    unittest.main()
