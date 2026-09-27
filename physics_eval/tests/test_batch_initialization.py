from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from physics_eval.run_video_batch import (
    resolve_tracking_initialization,
    run_tracking_for_row,
    write_per_video_result,
)
from physics_eval.utils.tracking_init import normalize_manual_prompts
from physics_eval.utils.tracking_init import renderer_gt_eligibility
import physics_eval.utils.tracking_init as tracking_init_utils


RGBA = (12, 34, 56, 255)


def _args(workspace: Path, **overrides) -> SimpleNamespace:
    values = {
        "allow_renderer_gt_init": False,
        "auto_init_box_expand": 2.8,
        "auto_init_max_area_ratio": 0.20,
        "auto_init_max_aspect_ratio": 6.0,
        "auto_init_min_area": 40,
        "auto_init_min_fill_ratio": 0.08,
        "auto_init_object_refine": True,
        "auto_init_scan_frames": 60,
        "auto_init_scan_mode": "uniform",
        "auto_init_target_hint": "metadata",
        "auto_init_threshold": 25.0,
        "copy_video_to_workspace": False,
        "detector": "motion",
        "disable_auto_init_shadow_filter": False,
        "fps": None,
        "min_motion_extent_ratio": 0.75,
        "min_object_coverage": 0.50,
        "model_cfg": "config.yaml",
        "model_weights": str(workspace / "model.pt"),
        "no_auto_init_box": False,
        "no_auto_init_negative_points": False,
        "no_bidirectional_propagation": False,
        "output_layout": "organized",
        "python_executable": sys.executable,
        "sam2_precision": "auto",
        "sam2_repo": str(workspace / "sam2"),
        "sam2_script": str(workspace / "scripts" / "sam2" / "run_sam2_track.py"),
        "save_auto_init_debug": False,
        "save_mask_png": False,
        "save_vis_video": False,
        "skip_tracking_if_exists": False,
        "workspace_root": str(workspace),
        "yolo_classes": None,
        "yolo_conf": 0.25,
        "yolo_iou": 0.50,
        "yolo_weights": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _row(**overrides) -> dict:
    values = {
        "Index": 1,
        "Prompt_ID": "sample",
        "Video_File": "video.mp4",
        "Metric": "tracking_coverage",
        "Num_Objects": 1,
        "Tracking_Objects": "ball",
        "Calibration_Object": "ball",
        "Known_Parameters_JSON": "{}",
    }
    values.update(overrides)
    return values


def _write_rgba_mask(path: Path, *, visible: bool) -> None:
    pixels = np.zeros((10, 12, 4), dtype=np.uint8)
    pixels[:, :, 3] = 255
    if visible:
        pixels[3:6, 4:8] = np.asarray(RGBA, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="RGBA").save(path)


def _write_rgba_region(path: Path, region: tuple[slice, slice] | None) -> None:
    pixels = np.zeros((20, 24, 4), dtype=np.uint8)
    pixels[:, :, 3] = 255
    if region is not None:
        pixels[region] = np.asarray(RGBA, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="RGBA").save(path)


def test_legacy_row_defaults_to_video_only_auto(tmp_path: Path):
    settings, prompts = resolve_tracking_initialization(_row(), _args(tmp_path))

    assert prompts == []
    assert settings["num_objects"] == 1
    assert settings["detector"] == "motion"
    assert settings["initialization"]["protocol"] == "video_only_auto"
    assert settings["initialization"]["uses_renderer_ground_truth"] is False


def test_row_can_enable_video_cpu_offload(tmp_path: Path):
    row = _row(Tracking_Overrides_JSON=json.dumps({"offload_video_to_cpu": True}))
    settings, prompts = resolve_tracking_initialization(row, _args(tmp_path))

    assert prompts == []
    assert settings["offload_video_to_cpu"] is True


def test_manual_two_object_prompts_and_per_row_overrides(tmp_path: Path):
    row = _row(
        Num_Objects=2,
        Tracking_Init_Mode="manual",
        Tracking_Prompts_JSON=json.dumps(
            [
                {"object_id": 1, "frame_idx": 7, "point": [20, 30], "box": [10, 20, 30, 40]},
                {"object_id": 2, "frame_idx": 7, "point": [80, 30], "box": [70, 20, 90, 40]},
            ]
        ),
        Tracking_Overrides_JSON=json.dumps(
            {"detector": "yolo_then_motion", "scan_frames": 24, "min_area": 12}
        ),
    )

    settings, prompts = resolve_tracking_initialization(row, _args(tmp_path))

    assert [item.object_id for item in prompts] == [1, 2]
    assert settings["detector"] == "yolo_then_motion"
    assert settings["auto_init_scan_frames"] == 24
    assert settings["auto_init_min_area"] == 12
    assert settings["initialization"]["semantic_verification"]["prompt_count_matches_expected"] is True


def test_manual_prompt_supports_multiple_positive_and_explicit_negative_points():
    prompts = normalize_manual_prompts(
        json.dumps(
            [
                {
                    "object_id": 1,
                    "frame_idx": 4,
                    "positive_points": [[20, 30], [24.5, 31]],
                    "negative_points": [[8, 9], [45, 42]],
                    "box": [10, 20, 40, 40],
                }
            ]
        ),
        expected_objects=1,
    )

    assert prompts[0].point == (20.0, 30.0)
    assert prompts[0].positive_points == ((20.0, 30.0), (24.5, 31.0))
    assert prompts[0].negative_points == ((8.0, 9.0), (45.0, 42.0))
    serialized = prompts[0].to_dict()
    assert serialized["point"] == [20.0, 30.0]
    assert serialized["positive_points"] == [[20.0, 30.0], [24.5, 31.0]]
    assert serialized["negative_points"] == [[8.0, 9.0], [45.0, 42.0]]


def test_manual_prompt_points_alias_and_legacy_point_are_backward_compatible():
    legacy = normalize_manual_prompts('[{"point":[3,4]}]', expected_objects=1)
    alias = normalize_manual_prompts(
        '[{"point":[3,4],"points":[[3,4],[5,6]]}]', expected_objects=1
    )

    assert legacy[0].positive_points == ((3.0, 4.0),)
    assert alias[0].point == (3.0, 4.0)
    assert alias[0].positive_points == ((3.0, 4.0), (5.0, 6.0))


def test_manual_prompt_box_must_contain_every_positive_point():
    with pytest.raises(ValueError, match="all positive points"):
        normalize_manual_prompts(
            '[{"positive_points":[[3,4],[50,60]],"box":[0,0,10,10]}]',
            expected_objects=1,
        )


@pytest.mark.parametrize(
    "payload",
    [
        '[{"positive_points":[]}]',
        '[{"positive_points":[[3,-1]]}]',
        '[{"positive_points":[[3,4]],"negative_points":[[1,"NaN"]]}]',
    ],
)
def test_manual_prompt_point_lists_require_nonnegative_finite_coordinates(payload: str):
    with pytest.raises(ValueError):
        normalize_manual_prompts(payload, expected_objects=1)


def test_renderer_gt_uses_first_visible_exact_color_frame_only(tmp_path: Path):
    masks = tmp_path / "masks"
    _write_rgba_mask(masks / "mask_0000.png", visible=False)
    _write_rgba_mask(masks / "mask_0001.png", visible=True)
    # A later mask is deliberately different.  Initialization must stop at frame 1.
    _write_rgba_mask(masks / "mask_0002.png", visible=True)
    params = {
        "workspace_root": str(tmp_path),
        "gt_mask_pattern": "masks/mask_{frame:04d}.png",
        "gt_rgba": list(RGBA),
        "gt_visible_frames": [0, 1, 2],
        "tracking_init": {"mode": "renderer_gt", "box_padding_px": 1},
    }
    row = _row(Expected_Frame_Count=3, Known_Parameters_JSON=json.dumps(params))

    settings, prompts = resolve_tracking_initialization(
        row, _args(tmp_path, allow_renderer_gt_init=True)
    )

    assert len(prompts) == 1
    assert prompts[0].frame_idx == 1
    assert prompts[0].point == pytest.approx((5.5, 4.0))
    assert prompts[0].box == pytest.approx((3.0, 2.0, 8.0, 6.0))
    provenance = settings["initialization"]
    assert provenance["uses_renderer_ground_truth"] is True
    assert provenance["uses_ground_truth_after_initialization"] is False
    assert provenance["protocol"] == "renderer_first_visible_gt_point_box"
    assert provenance["renderer_gt_eligibility"] == {
        "min_mask_area_px": 1,
        "min_bbox_side_px": 1,
        "min_border_margin_px": 0,
    }
    assert provenance["semantic_verification"]["future_frame_gt_used_for_initialization"] is False


def test_renderer_gt_uses_first_sequential_frame_meeting_predeclared_size_gate(tmp_path: Path):
    masks = tmp_path / "masks"
    _write_rgba_region(masks / "mask_0000.png", (slice(2, 4), slice(2, 4)))
    _write_rgba_region(masks / "mask_0001.png", (slice(5, 10), slice(8, 14)))
    _write_rgba_region(masks / "mask_0002.png", (slice(3, 15), slice(3, 20)))
    params = {
        "workspace_root": str(tmp_path),
        "gt_mask_pattern": "masks/mask_{frame:04d}.png",
        "gt_rgba": list(RGBA),
        "gt_visible_frames": [0, 1, 2],
        "tracking_init": {
            "mode": "renderer_gt",
            "box_padding_px": 1,
            "min_mask_area_px": 20,
            "min_bbox_side_px": 5,
        },
    }
    row = _row(Expected_Frame_Count=3, Known_Parameters_JSON=json.dumps(params))

    settings, prompts = resolve_tracking_initialization(
        row, _args(tmp_path, allow_renderer_gt_init=True)
    )

    assert prompts[0].frame_idx == 1
    assert prompts[0].source == "renderer_gt_first_eligible_exact_color"
    assert settings["initialization"]["protocol"] == "renderer_first_eligible_gt_point_box"
    assert settings["initialization"]["renderer_gt_eligibility"] == {
        "min_mask_area_px": 20,
        "min_bbox_side_px": 5,
        "min_border_margin_px": 0,
    }


def test_renderer_gt_uses_first_sequential_frame_meeting_border_margin_gate(
    tmp_path: Path,
    monkeypatch,
):
    masks = tmp_path / "masks"
    # Large enough, but touching the left border: frame 0 must be rejected.
    _write_rgba_region(masks / "mask_0000.png", (slice(4, 10), slice(0, 6)))
    # The first mask whose bbox is at least 3 px from all four borders.
    _write_rgba_region(masks / "mask_0001.png", (slice(5, 10), slice(6, 12)))
    _write_rgba_region(masks / "mask_0002.png", (slice(6, 12), slice(7, 14)))

    original_mask_geometry = tracking_init_utils._mask_geometry
    inspected: list[str] = []

    def guarded_mask_geometry(path, rgba):
        inspected.append(path.name)
        if path.name == "mask_0002.png":
            raise AssertionError("initializer inspected a frame after the first eligible frame")
        return original_mask_geometry(path, rgba)

    monkeypatch.setattr(tracking_init_utils, "_mask_geometry", guarded_mask_geometry)
    params = {
        "workspace_root": str(tmp_path),
        "gt_mask_pattern": "masks/mask_{frame:04d}.png",
        "gt_rgba": list(RGBA),
        "gt_visible_frames": [0, 1, 2],
        "tracking_init": {
            "mode": "renderer_gt",
            "min_border_margin_px": 3,
        },
    }
    row = _row(Expected_Frame_Count=3, Known_Parameters_JSON=json.dumps(params))

    settings, prompts = resolve_tracking_initialization(
        row, _args(tmp_path, allow_renderer_gt_init=True)
    )

    assert prompts[0].frame_idx == 1
    assert prompts[0].source == "renderer_gt_first_eligible_exact_color"
    assert inspected == ["mask_0000.png", "mask_0001.png"]
    assert settings["initialization"]["protocol"] == "renderer_first_eligible_gt_point_box"
    assert settings["initialization"]["renderer_gt_eligibility"] == {
        "min_mask_area_px": 1,
        "min_bbox_side_px": 1,
        "min_border_margin_px": 3,
    }


@pytest.mark.parametrize("value", [-1, 1.5, "NaN"])
def test_renderer_gt_border_margin_requires_nonnegative_integer(value):
    with pytest.raises(ValueError, match="min_border_margin_px"):
        renderer_gt_eligibility({"min_border_margin_px": value})


def test_renderer_gt_requires_explicit_opt_in(tmp_path: Path):
    row = _row(
        Tracking_Init_Mode="renderer_gt",
        Known_Parameters_JSON=json.dumps(
            {"gt_mask_pattern": "masks/{frame:04d}.png", "gt_rgba": list(RGBA)}
        ),
    )
    with pytest.raises(ValueError, match="allow-renderer-gt-init"):
        resolve_tracking_initialization(row, _args(tmp_path))


def test_unknown_tracking_override_is_rejected(tmp_path: Path):
    row = _row(Tracking_Overrides_JSON='{"shell_command":"unsafe"}')
    with pytest.raises(ValueError, match="unsupported Tracking_Overrides_JSON key"):
        resolve_tracking_initialization(row, _args(tmp_path))


def test_batch_command_threads_precision_and_writes_provenance(tmp_path: Path, monkeypatch):
    video = tmp_path / "videos" / "video.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    tracker = tmp_path / "scripts" / "sam2" / "run_sam2_track.py"
    adapter = tracker.with_name("run_sam2_track_prompts.py")
    tracker.parent.mkdir(parents=True)
    tracker.write_text("# fake", encoding="utf-8")
    adapter.write_text("# fake", encoding="utf-8")

    row = _row(
        Num_Objects=2,
        Tracking_Init_Mode="manual",
        Tracking_Prompts_JSON=json.dumps(
            [
                {"object_id": 1, "frame_idx": 0, "point": [10, 10]},
                {"object_id": 2, "frame_idx": 0, "point": [20, 20]},
            ]
        ),
    )
    args = _args(tmp_path, sam2_precision="fp32")
    captured: dict[str, list[str]] = {}

    def fake_run(command, **_kwargs):
        captured["command"] = command
        source_dir = tmp_path / "sam2_tracks" / "sample"
        source_dir.mkdir(parents=True, exist_ok=True)
        (source_dir / "tracking_points.json").write_text(
            json.dumps({"video_info": {"fps": 30}, "frames": []}), encoding="utf-8"
        )
        (source_dir / "run_config.json").write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("physics_eval.run_video_batch.subprocess.run", fake_run)
    monkeypatch.setattr("physics_eval.run_video_batch.write_tracking_quality", lambda **_kwargs: {})
    out_json, error = run_tracking_for_row(
        row=row,
        metadata_path=tmp_path / "metadata.xlsx",
        video_root=video.parent,
        output_dir=tmp_path / "outputs",
        args=args,
    )

    assert error is None
    assert out_json is not None and out_json.is_file()
    command = captured["command"]
    assert str(adapter) in command
    assert command[command.index("--precision") + 1] == "fp32"
    manifest = json.loads(out_json.with_name("resolved_init_prompts.json").read_text(encoding="utf-8"))
    assert manifest["num_objects"] == 2
    config = json.loads(out_json.with_name("run_config.json").read_text(encoding="utf-8"))
    assert config["batch_initialization"]["protocol"] == "one_frame_manual_point"
    assert config["semantic_verification"]["prompt_count_matches_expected"] is True


def test_empty_mask_qc_is_not_evaluated(tmp_path: Path):
    args = _args(tmp_path)
    args.output_layout = "organized"
    result_path = write_per_video_result(
        output_dir=tmp_path / "outputs",
        row=_row(),
        result={"Status": "failed", "Extra_JSON": "{}"},
        tracking_json=None,
        area_summaries=[],
        area_paths={},
        args=args,
    )

    assert result_path is not None
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["mask_qc"]["status"] == "not_evaluated"
