"""Public-pipeline regressions captured against the original wrapper chain."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physics_eval import pipeline
from physics_eval.cli import parse_eval_args


def test_public_eval_defaults_preserve_evidence_generation(tmp_path):
    (tmp_path / "yolov8n.pt").touch()
    metadata = tmp_path / "data" / "metadata" / "fixture.xlsx"
    metadata.parent.mkdir(parents=True)
    metadata.touch()
    args = parse_eval_args(["--workspace-root", str(tmp_path)])
    expected = {
        "metadata": str(metadata.resolve()),
        "video_root": str(tmp_path / "data" / "t2v_videos"),
        "output_dir": str(tmp_path / "batch_eval_results"),
        "sam2_repo": str(tmp_path / "repo" / "sam2"),
        "model_weights": str(tmp_path / "repo" / "sam2" / "checkpoints" / "sam2.1_hiera_base_plus.pt"),
        "model_cfg": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "yolo_weights": str(tmp_path / "yolov8n.pt"),
        "detector": "yolo_then_motion",
        "save_mask_png": True,
        "area_stability_check": True,
        "area_prefer_mask_area": True,
        "area_stability_save_plots": True,
        "save_auto_init_debug": False,
        "save_diagnostics": False,
        "save_vis_video": False,
        "skip_tracking_if_exists": False,
        "tracking_only": False,
        "eval_only": False,
        "sam2_precision": "auto",
        "min_motion_extent_ratio": .75,
        "min_object_coverage": .5,
        "area_stability_action": "warn",
    }
    for name, value in expected.items():
        assert getattr(args, name) == value


def test_public_eval_short_options_keep_original_semantics(tmp_path):
    args = parse_eval_args([
        "--workspace-root", str(tmp_path), "--metadata", "metadata.xlsx",
        "--diagnostics", "--vis", "--reuse-tracking", "--no-plots",
        "--area-action", "skip", "--strict", "--index", "7",
        "--allow-renderer-gt-init", "--min-object-coverage", ".7",
    ])
    assert args.metadata == str(tmp_path / "metadata.xlsx")
    assert args.save_diagnostics and args.save_auto_init_debug and args.save_vis_video
    assert args.skip_tracking_if_exists and args.strict and args.allow_renderer_gt_init
    assert args.index == 7 and args.min_object_coverage == .7
    assert args.area_stability_action == "skip"
    assert args.area_stability_check and args.save_mask_png
    assert not args.area_stability_save_plots


def _write_saved_tracking(workspace, output, prompt, *, static, flat):
    path = output / "tracking_json" / f"{prompt}.json" if flat else output / "velocity" / prompt / "tracking_points.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    masks = workspace / "sam2_tracks" / prompt / "masks"
    masks.mkdir(parents=True, exist_ok=True)
    frames = []
    for frame in range(30):
        x = 20 if static else 20+frame
        pixels = np.zeros((80, 80), dtype=np.uint8)
        pixels[20:30, x:x+10] = 255
        mask = masks / f"{frame:05d}.png"
        Image.fromarray(pixels).save(mask)
        frames.append(dict(
            frame_idx=frame, time_sec=frame/30, object_id=1, valid=True,
            x=x+4.5, y=24.5, bbox=[x, 20, x+9, 29], area=100,
            mask_path=str(mask),
        ))
    path.write_text(json.dumps({
        "video_info": {"fps": 30, "total_frames": 30},
        "object_id": 1, "frames": frames,
    }), encoding="utf-8")


@pytest.mark.parametrize("flat", [False, True])
def test_eval_only_preserves_measurements_failure_rows_and_reports(tmp_path, monkeypatch, flat):
    metadata = tmp_path / "metadata.xlsx"
    output = tmp_path / "batch_eval_results"
    rows = []
    for index, (prompt, case) in enumerate([("V001", "valid"), ("V002", "missing"), ("V003", "static")], 1):
        rows.append(dict(
            Index=index, Prompt_ID=prompt, Video_File=f"{case}.mp4", Tracking_JSON="",
            Module="B1.1", Metric="velocity", Evaluator="eval_velocity",
            Measurement_Method="linear_fit", Scale_Mode="object_diameter",
            Calibration_Value_m=.1, Target_Type="target_speed", Target_Value=.5,
            Target_Unit="m/s", Known_Parameters_JSON="{}", Tracking_Objects="object_center",
            Num_Objects=1, Motion_Axis="x",
        ))
        if case != "missing":
            _write_saved_tracking(tmp_path, output, prompt, static=case == "static", flat=flat)
    pd.DataFrame(rows).to_excel(metadata, index=False)
    args = parse_eval_args([
        "--workspace-root", str(tmp_path), "--metadata", str(metadata),
        "--eval-only", "--no-plots", "--output-layout", "flat" if flat else "organized",
    ])
    def unexpected_worker(*unused_args, **unused_kwargs):
        pytest.fail("eval-only must not launch a tracking subprocess")
    monkeypatch.setattr(pipeline.subprocess, "run", unexpected_worker)
    pipeline.run(args)

    results = json.loads((output / "results.json").read_text())
    assert [row["Prompt_ID"] for row in results] == ["V001", "V002", "V003"]
    assert [row["Status"] for row in results] == ["valid", "failed", "valid"]
    assert results[0]["Measured_Value"] == pytest.approx(.26586807763582726)
    assert pd.isna(results[1]["Measured_Value"])
    assert results[1]["Failure_Reason"] == "failed"
    assert results[2]["Measured_Value"] == pytest.approx(0., abs=1e-12)
    csv = pd.read_csv(output / "results.csv")
    assert list(csv["Status"]) == ["valid", "failed", "valid"]
    assert len(pd.read_excel(output / "metadata_with_resolved_tracking.xlsx")) == 3
    manifest = json.loads((output / "batch_manifest.json").read_text())
    assert manifest["eval_only"] and not manifest["tracking_only"] and manifest["n_rows"] == 3

    if flat:
        assert not list(output.glob("*/*/result.json"))
        return
    expected_fields = {
        "schema_version", "index", "prompt_id", "metric", "video_file", "metadata",
        "physics_result", "physics_extra", "mask_qc", "tracking_quality",
        "tracking_initialization", "semantic_verification", "artifacts",
    }
    compact = [json.loads((output / "velocity" / prompt / "result.json").read_text()) for prompt in ["V001", "V002", "V003"]]
    assert all(set(row) == expected_fields and row["schema_version"] == 1 for row in compact)
    assert compact[0]["tracking_quality"]["status"] == "pass"
    assert compact[0]["mask_qc"]["status"] == "pass"
    assert compact[1]["tracking_quality"] is None
    assert compact[1]["mask_qc"]["status"] == "not_evaluated"
    assert compact[2]["tracking_quality"]["reasons"] == ["no_meaningful_motion"]
    assert compact[2]["physics_result"]["Status"] == "valid"
    assert (output / "velocity" / "V001" / "plots" / "trajectory.png").is_file()
