"""Analytical spring controls for period recovery and confidence weighting."""
from __future__ import annotations

import json
import math
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from physics_eval.evaluators.spring import eval_spring


class SpringTests(unittest.TestCase):
    def setUp(self):
        t = np.arange(120, dtype=float) / 30.0
        self.track = pd.DataFrame({
            "frame": np.arange(len(t)),
            "t": t,
            "object_id": 1,
            "cx_px": 100.0 + 20.0 * np.sin(2.0 * np.pi * t),
            "cy_px": 100.0,
        })
        self.expected_k = math.pi**2
        self.metadata = {
            "Motion_Axis": "x",
            "Target_Value": self.expected_k,
            "Known_Parameters_JSON": json.dumps({"mass_kg": 0.25}),
        }

    def test_peak_confidence_uses_peak_count_per_sample(self):
        signal = types.ModuleType("scipy.signal")
        signal.find_peaks = Mock(return_value=(np.array([9, 39, 69, 99]), {}))
        with patch.dict("sys.modules", {"scipy.signal": signal}):
            result = eval_spring(self.track, self.metadata)
        self.assertEqual(result["Status"], "weak_valid")
        self.assertEqual(result["Failure_Reason"], "period_detection_weak")
        self.assertAlmostEqual(result["Fit_R2"], 4.0 / 120.0)
        self.assertAlmostEqual(result["Measured_Value"], self.expected_k)
        extra = json.loads(result["Extra_JSON"])
        self.assertAlmostEqual(extra["period_s"], 1.0)
        self.assertEqual(extra["period_method"], "peaks")

    def test_confidence_threshold_keeps_period_and_stiffness(self):
        for method in ("peaks", "autocorrelation"):
            for confidence, status in ((0.199, "weak_valid"), (0.20, "valid")):
                with self.subTest(method=method, confidence=confidence):
                    with patch("physics_eval.evaluators.spring.estimate_period",
                               return_value=(1.0, confidence, method)):
                        result = eval_spring(self.track, self.metadata)
                    self.assertEqual(result["Status"], status)
                    self.assertAlmostEqual(result["Measured_Value"], self.expected_k)
                    self.assertAlmostEqual(result["Relative_Error"], 0.0)

    def test_autocorrelation_recovers_analytical_period(self):
        with patch.dict("sys.modules", {"scipy.signal": None}):
            result = eval_spring(self.track, self.metadata)
        self.assertEqual(result["Status"], "valid")
        self.assertAlmostEqual(result["Measured_Value"], self.expected_k)
        extra = json.loads(result["Extra_JSON"])
        self.assertAlmostEqual(extra["period_s"], 1.0)
        self.assertEqual(extra["period_method"], "autocorrelation")

    def test_nonpositive_mass_fails(self):
        for mass in (0.0, -0.25):
            with self.subTest(mass=mass):
                metadata = {**self.metadata, "Known_Parameters_JSON": json.dumps({"mass_kg": mass})}
                result = eval_spring(self.track, metadata)
                self.assertEqual(result["Status"], "failed")
                self.assertEqual(result["Failure_Reason"], "invalid_mass_kg")

    def test_short_track_fails(self):
        result = eval_spring(self.track.iloc[:11], self.metadata)
        self.assertEqual(result["Status"], "failed")
        self.assertIsNone(result["Measured_Value"])


if __name__ == "__main__":
    unittest.main()
