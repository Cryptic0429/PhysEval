"""CPU-only regression tests for versioned Qmask and shared input adapters."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "compare"))

import numpy as np
from PIL import Image
from physics_eval.quality.adapters import normalize_records, nested_frames
from physics_eval.quality.diagnostics import extract_quality_evidence, load_series_from_json, analyze_series, DEFAULT_DIAGNOSTICS
from physics_eval.quality.protocol import load_protocol, protocol_hash, validate_protocol
from physics_eval.quality.scoring import compute_qmask, component_score
from physeval_compare.metrics import mask_quality, tracking_quality, compare_tracking_files
from scripts import score_results as scorer
from types import SimpleNamespace


class QmaskTests(unittest.TestCase):
    def setUp(self):
        self.protocol = load_protocol("grouped_v2_candidate")
        self.obj = {name: spec["good_value"] for name, spec in self.protocol["components"].items()}
        self.obj.update(object_id=1, decision="keep", reasons="")

    def test_grouped_protocol_is_default(self):
        self.assertEqual(load_protocol()["protocol_id"], "grouped_v2_candidate")
        with patch.object(sys, "argv", ["score"]):
            args = scorer.parse_args()
        self.assertEqual(args.qmask_protocol["protocol_id"], "grouped_v2_candidate")

    def test_mapping_boundaries_and_monotonicity(self):
        for spec in self.protocol["components"].values():
            vals = [component_score(spec["good_value"] + r*(spec["bad_value"]-spec["good_value"]), spec, 1) for r in (0, .25, .5, 1)]
            for value, expected in zip(vals, [1, .75, .5, 0]):
                self.assertAlmostEqual(value, expected)

    def test_group_weights(self):
        self.obj["area_cv"] = 0.7
        result = compute_qmask({"objects": [self.obj]}, self.protocol, [1])
        self.assertAlmostEqual(result["score"], 20/21)

    def test_missing_evidence_never_full_credit(self):
        for evidence in ({}, {"objects": []}, {"objects": [{"object_id": 1, "decision": "keep"}]}):
            self.assertEqual(compute_qmask(evidence, self.protocol, [1])["status"], "score_unavailable")

    def test_nonfinite_and_missing_object(self):
        bad = dict(self.obj, area_cv=float("nan"))
        self.assertIsNone(compute_qmask({"objects": [bad]}, self.protocol, [1])["score"])
        self.assertIsNone(compute_qmask({"objects": [self.obj]}, self.protocol, [1, 2])["score"])
        self.assertIsNone(compute_qmask({"objects": [self.obj, self.obj]}, self.protocol, [1])["score"])

    def test_order_invariance(self):
        other = dict(self.obj, object_id=2, area_cv=.7)
        a = compute_qmask({"objects": [self.obj, other]}, self.protocol, [1, 2])
        b = compute_qmask({"objects": [other, self.obj]}, self.protocol, [2, 1])
        self.assertEqual(a, b)

    def test_legacy_missing_fallback_is_explicit(self):
        result = compute_qmask({}, load_protocol("legacy_v1"), [1])
        self.assertEqual(result["score"], 1)
        self.assertIn("legacy_missing_evidence_fallback", result["warnings"])

    def test_config_validation_and_hash(self):
        config = copy.deepcopy(self.protocol)
        config["gamma"] = 2
        self.assertNotEqual(protocol_hash(config), protocol_hash(self.protocol))
        config["gamma"] = float("nan")
        with self.assertRaises(ValueError):
            validate_protocol(config)

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
        gate = tracking_quality(records, expected_objects=2, total_frames=20, min_coverage=.5, min_motion_ratio=.75)
        self.assertEqual(gate["total_frames"], 20)
        self.assertEqual(gate["actual_objects"], 2)
        self.assertEqual(gate["coverage"], 1)
        self.assertEqual(gate["status"], "pass")
        nested, _ = normalize_records({"frames": nested_frames(canonical, n), "total_frames": n})
        self.assertEqual(nested, canonical)

    def test_compare_and_core_scores_match(self):
        records, masks = self.tracks(2)
        evidence = extract_quality_evidence(records, masks, total_frames=20, bbox_convention="exclusive")
        core = compute_qmask(evidence, self.protocol, [1, 2])
        compare = mask_quality(records, masks, total_frames=20, protocol=self.protocol, required_object_ids=[1, 2])
        self.assertEqual(core["score"], compare["score"])
        self.assertEqual(core["objects"], compare["objects"])
        self.assertEqual(core["score"], 1)
        self.assertEqual(compute_qmask(compare, self.protocol, [1, 2])["score"], core["score"])

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
            for key in self.protocol["components"]:
                self.assertAlmostEqual(disk[key], memory[key], msg=key)
            self.assertEqual(compare_tracking_files(track, track)["bbox_iou_mean"], 1)

    def test_scorer_gates_zero_and_unavailable(self):
        with patch.object(sys, "argv", ["score", "--scoring-protocol", "grouped_v2_candidate"]):
            args = scorer.parse_args()
        data = {"metric": "velocity", "metadata": {"Num_Objects": 1},
                "tracking_quality": {"status": "pass", "counted_object_ids": [1]},
                "physics_result": {"Status": "valid", "Relative_Error": 0, "Measured_Value": 1},
                "mask_qc": {"objects": [self.obj]}}
        with patch.object(scorer, "load_json", return_value=data):
            a = scorer.score_one(Path("velocity/V001/result.json"), scorer.DEFAULT_TOLERANCES, args)
            self.assertEqual(a["adjusted_score"],100)
            data["mask_qc"] = {}
            a = scorer.score_one(Path("velocity/V001/result.json"), scorer.DEFAULT_TOLERANCES, args)
            self.assertTrue(a["effective_for_score"])
            self.assertIsNone(a["adjusted_score"])
            self.assertFalse(a["score_available"])
            data["mask_qc"] = {"objects": [{**{k:s["bad_value"] for k,s in self.protocol["components"].items()}, "object_id":1}]}
            a = scorer.score_one(Path("velocity/V001/result.json"), scorer.DEFAULT_TOLERANCES, args)
            self.assertTrue(a["effective_for_score"])
            self.assertEqual(a["adjusted_score"],0)
            data["tracking_quality"]["status"] = "flagged"
            with patch.object(scorer, "mask_quality", side_effect=AssertionError("should not compute")):
                a = scorer.score_one(Path("velocity/V001/result.json"), scorer.DEFAULT_TOLERANCES, args)
                self.assertFalse(a["effective_for_score"])

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
