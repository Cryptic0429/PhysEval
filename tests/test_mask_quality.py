"""CPU-only regression tests for mask reliability and shared input adapters."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
from physics_eval.io.tracks import normalize_records, nested_frames
from physics_eval.quality.mask_evidence import extract_quality_evidence, load_series_from_json, analyze_series, DEFAULT_DIAGNOSTICS
from physics_eval.quality.config import load_mask_config, mask_config_hash, validate_mask_config
from physics_eval.quality.mask_score import compute_qmask, component_score
from physics_eval.quality.tracking_checks import analyze_tracking_quality_data
from physics_eval import scoring as scorer
from physics_eval.cli import parse_score_args
from types import SimpleNamespace


class QmaskTests(unittest.TestCase):
    def setUp(self):
        self.config = load_mask_config()
        self.obj = {name: spec["good_value"] for name, spec in self.config["components"].items()}
        self.obj.update(object_id=1, decision="keep", reasons="")

    def test_single_scoring_rule_is_default(self):
        self.assertEqual(self.config["gamma"], .35)
        self.assertNotIn("flag_floor", self.config)
        self.assertNotIn("flagged_missing_score", self.config)
        self.assertEqual(len(self.config["components"]), 10)
        args = parse_score_args([])
        self.assertEqual(args.mask_config, self.config)
        self.assertNotIn("protocol_id", args.mask_config)

    def test_mapping_boundaries_and_monotonicity(self):
        for spec in self.config["components"].values():
            vals = [component_score(spec["good_value"] + r*(spec["bad_value"]-spec["good_value"]), spec, .35) for r in (0, .25, .5, 1)]
            for value, r in zip(vals, (0, .25, .5, 1)):
                self.assertAlmostEqual(value, 1-r**.35)

    def test_minimum_component_and_centroid_penalties(self):
        for name in ("area_cv", "max_centroid_step_norm_by_diameter"):
            obj = dict(self.obj, **{name: self.config["components"][name]["bad_value"]})
            self.assertEqual(compute_qmask({"objects": [obj]}, self.config)["score"], 0)

    def test_missing_object_evidence_is_an_error(self):
        for evidence in ({}, {"objects": []}, {"status": "flagged"}):
            with self.subTest(evidence=evidence):
                with self.assertRaisesRegex(ValueError, "mask_qc.objects"):
                    compute_qmask(evidence, self.config)

    def test_every_object_requires_a_computable_component(self):
        for decision in ("", "keep", "review"):
            obj = {"object_id": 2, "decision": decision, "area_cv": float("nan")}
            with self.subTest(decision=decision):
                with self.assertRaisesRegex(ValueError, "object 2.*no computable"):
                    compute_qmask({"objects": [self.obj, obj]}, self.config)

    def test_nonfinite_components_are_ignored(self):
        bad = dict(self.obj, area_cv=float("nan"))
        self.assertEqual(compute_qmask({"objects": [bad]}, self.config)["score"], 1)

    def test_minimum_object_score_and_order_invariance(self):
        other = dict(self.obj, object_id=2, area_cv=.7)
        a = compute_qmask({"objects": [self.obj, other]}, self.config)
        b = compute_qmask({"objects": [other, self.obj]}, self.config)
        self.assertEqual(a["score"], 0)
        self.assertEqual(a["score"], b["score"])

    def test_flags_preserve_measured_component_scores_and_zeros(self):
        for decision in ("keep", "review", "rerun_tracking"):
            obj = dict(self.obj, decision=decision, area_cv=.475)
            result = compute_qmask({"objects": [obj]}, self.config)
            self.assertAlmostEqual(result["score"], 1 - .5**.35)
            self.assertEqual(result["objects"][0]["decision"], decision)
            obj["area_cv"] = .7
            self.assertEqual(compute_qmask({"objects": [obj]}, self.config)["score"], 0)

    def test_flagged_quality_decreases_to_zero_without_a_floor(self):
        scores = []
        for cv in np.linspace(.25, .7, 20):
            obj = dict(self.obj, decision="keep" if cv <= .25 else "review", area_cv=cv)
            scores.append(compute_qmask({"objects": [obj]}, self.config)["score"])
        self.assertEqual(scores[0], 1)
        self.assertEqual(scores[-1], 0)
        self.assertTrue(all(a > b for a, b in zip(scores, scores[1:])))

    def test_missing_coverage_reason_preserves_zero(self):
        obj = {"object_id": 1, "decision": "review", "reasons": "low_middle_coverage"}
        self.assertEqual(compute_qmask({"objects": [obj]}, self.config)["score"], 0)

    def test_partial_evidence_is_measured_and_reports_missing_components(self):
        result = compute_qmask({"objects": [{"object_id": 1, "middle_coverage": 1.0}]}, self.config)
        self.assertEqual(result["score"], 1)
        self.assertEqual(len(result["missing_components"]), 9)
        self.assertIn("missing_components_ignored", result["warnings"])

    def test_config_validation_and_hash(self):
        config = copy.deepcopy(self.config)
        config["gamma"] = 2
        self.assertNotEqual(mask_config_hash(config), mask_config_hash(self.config))
        config["gamma"] = float("nan")
        with self.assertRaises(ValueError):
            validate_mask_config(config)
        config = copy.deepcopy(self.config)
        config["flag_floor"] = .85
        with self.assertRaisesRegex(ValueError, "flag_floor is no longer supported"):
            validate_mask_config(config)
        config = copy.deepcopy(self.config)
        config["flagged_missing_score"] = .85
        with self.assertRaisesRegex(ValueError, "flagged_missing_score is no longer supported"):
            validate_mask_config(config)

    @staticmethod
    def tracks(objects=1):
        records, masks = [], {}
        for oid in range(1, objects+1):
            masks[oid] = {}
            for i in range(20):
                mask = np.zeros((40, 60), dtype=bool)
                mask[oid*10:oid*10+5, 5+i:10+i] = True
                masks[oid][i] = mask
                records.append(dict(object_id=oid, frame_idx=i, valid=True, centroid=[7+i, oid*10+2],
                                    bbox=[5+i, oid*10, 10+i, oid*10+5], area=25, time_sec=i/20))
        return records, masks

    def test_double_object_frame_count(self):
        records, _ = self.tracks(2)
        canonical, n = normalize_records({"frames": records, "total_frames": 20}, "exclusive")
        self.assertEqual(n, 20)
        self.assertEqual(canonical[0]["bbox"], [5., 10., 9., 14.])
        gate = analyze_tracking_quality_data({"frames": nested_frames(canonical, n)}, expected_objects=2, min_object_coverage=.5, min_motion_extent_ratio=.75)
        self.assertEqual(gate["total_frames"], 20)
        self.assertEqual(gate["actual_objects"], 2)
        self.assertEqual(gate["coverage"], 1)
        self.assertEqual(gate["status"], "pass")
        nested, _ = normalize_records({"frames": nested_frames(canonical, n), "total_frames": n})
        self.assertEqual(nested, canonical)

    def test_two_object_mask_evidence_scores(self):
        records, masks = self.tracks(2)
        evidence = extract_quality_evidence(records, masks, total_frames=20, bbox_convention="exclusive")
        core = compute_qmask(evidence, self.config)
        self.assertEqual(core["score"], 1)
        self.assertEqual(len(core["objects"]), 2)

    def test_disk_and_memory_diagnostics_match(self):
        records, masks = self.tracks()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for rec in records:
                path = root / f"{rec['frame_idx']:05d}.png"
                Image.fromarray(masks[1][rec["frame_idx"]].astype(np.uint8)*255).save(path)
                rec["mask_path"] = str(path)
            canonical, n = normalize_records({"frames": records}, "exclusive")
            track = root / "tracking_points.json"
            track.write_text(json.dumps({"frames": nested_frames(canonical,n)}), encoding="utf-8")
            series = load_series_from_json(track, None, "{frame_idx:05d}.png", None, False)
            disk, _ = analyze_series(series[0], SimpleNamespace(**DEFAULT_DIAGNOSTICS))
            memory = extract_quality_evidence(records, masks, total_frames=n, bbox_convention="exclusive")["objects"][0]
            for key in self.config["components"]:
                self.assertAlmostEqual(disk[key], memory[key], msg=key)

    def test_scorer_gates_zero_and_missing_evidence(self):
        args = parse_score_args([])
        data = {"metric": "velocity", "metadata": {"Num_Objects": 1},
                "tracking_quality": {"status": "pass", "counted_object_ids": [1]},
                "physics_result": {"Status": "valid", "Relative_Error": 0, "Measured_Value": 1},
                "mask_qc": {"objects": [self.obj]}}
        with patch.object(scorer, "load_json", return_value=data):
            a = scorer.score_one(Path("velocity/V001/result.json"), scorer.DEFAULT_TOLERANCES, args)
            self.assertEqual(a["adjusted_score"],100)
            data["mask_qc"] = {"objects": [dict(self.obj, decision="review", area_cv=.475)]}
            a = scorer.score_one(Path("velocity/V001/result.json"), scorer.DEFAULT_TOLERANCES, args)
            self.assertTrue(a["effective_for_score"])
            self.assertAlmostEqual(a["adjusted_score"], 100 * (1 - .5**.35))
            data["mask_qc"] = {}
            for policy in ("continuous", "penalty", "hard_fail"):
                args.mask_policy = policy
                with self.assertRaisesRegex(ValueError, "V001/result.json: Mask diagnostics missing"):
                    scorer.score_one(Path("velocity/V001/result.json"), scorer.DEFAULT_TOLERANCES, args)
            args.mask_policy = "ignore"
            self.assertEqual(scorer.score_one(Path("velocity/V001/result.json"), scorer.DEFAULT_TOLERANCES, args)["adjusted_score"], 100)
            args.mask_policy = "continuous"
            data["mask_qc"] = {"objects": [{**{k:s["bad_value"] for k,s in self.config["components"].items()}, "object_id":1}]}
            a = scorer.score_one(Path("velocity/V001/result.json"), scorer.DEFAULT_TOLERANCES, args)
            self.assertTrue(a["effective_for_score"])
            self.assertEqual(a["adjusted_score"],0)
            data["tracking_quality"]["status"] = "flagged"
            with patch.object(scorer, "mask_quality", side_effect=AssertionError("should not compute")):
                a = scorer.score_one(Path("velocity/V001/result.json"), scorer.DEFAULT_TOLERANCES, args)
                self.assertFalse(a["effective_for_score"])

    def test_missing_diagnostics_stop_formal_report_generation(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            path = root / "velocity/V001/result.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({
                "metric": "velocity", "prompt_id": "V001",
                "tracking_quality": {"status": "pass"},
                "physics_result": {"Status": "valid", "Relative_Error": 0},
            }), encoding="utf-8")
            args = parse_score_args(["--result-root", str(root), "--metadata", "none", "--no-plots"])
            with self.assertRaisesRegex(ValueError, "V001/result.json: Mask diagnostics missing"):
                scorer.run(args)
            self.assertFalse((root / "score_reports/all_metrics/score_summary.json").exists())

    def test_metadata_keeps_eta_separate_from_acceleration(self):
        with tempfile.TemporaryDirectory() as raw:
            p = Path(raw)/"metadata.csv"
            p.write_text("Prompt_ID,Metric\nA001,acceleration\nETA001,fluid_viscosity\n",encoding="utf-8")
            paths=[Path("A001"),Path("ETA001")]
            with patch.object(scorer,"load_json",side_effect=lambda path:{"metadata":{"Prompt_ID":"model_"+path.name}}):
                report=scorer.check_completeness(paths,str(p))
                self.assertEqual(report["n_expected"],2)
                self.assertEqual(report["missing_ids"],[])


if __name__ == "__main__":
    unittest.main()
