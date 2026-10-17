"""CPU coverage of the fixed initialization policy and shared inference boundary."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import numpy as np
import pytest

from physics_eval.tracking.initialization import (
    build_tracking_prompts,
    load_prompt_manifest,
    select_automatic_prompts,
)
from physics_eval.tracking.sam2 import (
    initialize_predictor_state,
    propagate_in_video_with_context,
    resolve_precision_policy,
    run_tracking,
)
from physics_eval.tracking.worker import parse_args


def test_yolo_success_does_not_run_motion():
    candidates = [{"point": [10, 20]}, {"point": [30, 20]}]
    def no_motion():
        pytest.fail("accepted YOLO targets must not invoke motion")
    actual, source = select_automatic_prompts(
        lambda: candidates, no_motion, expected_objects=2,
    )
    assert actual is candidates
    assert source == "yolo"


@pytest.mark.parametrize("yolo_result", [[], [{"point": [10, 20]}]])
def test_partial_yolo_falls_back_for_required_target_count(yolo_result):
    motion = [{"point": [10, 20]}, {"point": [30, 20]}]
    actual, source = select_automatic_prompts(
        lambda: yolo_result, lambda: motion, expected_objects=2,
    )
    assert source == "motion"
    assert actual is motion
    assert all(item["yolo_fallback_reason"] == f"YOLO returned {len(yolo_result)}/2 target prompts" for item in actual)


def test_yolo_exception_is_recorded_and_partial_motion_rejected():
    def unavailable():
        raise RuntimeError("missing detector weights")
    actual, source = select_automatic_prompts(
        unavailable, lambda: [{}], expected_objects=1,
    )
    assert source == "motion"
    assert actual[0]["yolo_fallback_reason"] == "missing detector weights"
    with pytest.raises(RuntimeError, match="motion returned 1/2"):
        select_automatic_prompts(unavailable, lambda: [{}], expected_objects=2)


def _args(tmp_path):
    return parse_args([
        "--workspace-root", str(tmp_path),
        "--scene-name", "scene", "--video-path", str(tmp_path / "input.mp4"),
        "--model-cfg", "config.yaml", "--model-weights", str(tmp_path / "weights.pt"),
        "--precision", "fp32", "--save-mask-png", "--overwrite-outputs",
    ])


def test_manifest_prompts_bypass_detector_and_keep_all_points(tmp_path, monkeypatch):
    from physics_eval.tracking import yolo, motion
    def no_detection(*args, **kwargs):
        pytest.fail("metadata prompts must bypass both automatic detectors")
    monkeypatch.setattr(yolo, "detect_yolo_prompts", no_detection)
    monkeypatch.setattr(motion, "detect_moving_object_prompts", no_detection)
    manifest_path = tmp_path / "prompts.json"
    manifest_path.write_text(json.dumps({
        "mode": "manual", "num_objects": 2,
        "prompts": [
            {"object_id": 7, "frame_idx": 1, "positive_points": [[2, 3], [4, 3]], "negative_points": [[8, 9]]},
            {"object_id": 9, "frame_idx": 2, "point": [12, 13], "box": [10, 10, 15, 15]},
        ],
    }))
    manifest = load_prompt_manifest(manifest_path)
    prompts, mode, evidence = build_tracking_prompts(
        _args(tmp_path), frames_dir=tmp_path, width=30, height=30,
        prompt_manifest=manifest,
    )
    assert mode == "metadata_manual"
    assert [prompt["obj_id"] for prompt in prompts] == [1, 2]
    assert [prompt["frame_idx"] for prompt in prompts] == [1, 2]
    assert prompts[0]["positive_points"] == [[2.0, 3.0], [4.0, 3.0]]
    assert prompts[0]["negative_points"] == [[8.0, 9.0]]
    assert prompts[0]["box"] is None
    assert prompts[1]["box"] == [10.0, 10.0, 15.0, 15.0]
    assert evidence[0]["prompt_manifest"] == str(manifest_path)
    assert [prompt["object_id"] for prompt in manifest["prompts"]] == [7, 9]


def test_shared_predictor_context_covers_lazy_propagation(tmp_path):
    active = []
    @contextmanager
    def context():
        active.append(True)
        try:
            yield
        finally:
            active.pop()
    class Predictor:
        def init_state(self, **kwargs):
            assert active
            return "state"
        def add_new_points_or_box(self, **kwargs):
            assert active
            np.testing.assert_array_equal(kwargs["labels"], [1, 1, 0])
            assert "box" not in kwargs
        def propagate_in_video(self, state, **kwargs):
            assert active
            yield 0, [1], "mask"
            assert active
    predictor = Predictor()
    state = initialize_predictor_state(
        predictor, frames_dir=tmp_path, offload_video_to_cpu=False,
        offload_state_to_cpu=False,
        prompts=[{"obj_id": 1, "frame_idx": 0, "positive_points": [[1, 2], [2, 3]],
                  "negative_points": [[9, 9]], "label": 1, "box": None}],
        inference_context_factory=context,
    )
    assert not active
    iterator = propagate_in_video_with_context(
        predictor, state, start_frame_idx=0, reverse=False,
        inference_context_factory=context,
    )
    assert list(iterator) == [(0, [1], "mask")]
    assert not active


def test_legacy_predictor_reverse_policy_remains_explicit():
    from contextlib import nullcontext
    class Predictor:
        def propagate_in_video(self, state):
            yield 0, [1], "mask"
    args = dict(start_frame_idx=1, reverse=True, inference_context_factory=nullcontext)
    assert list(propagate_in_video_with_context(Predictor(), None, **args)) == []
    with pytest.raises(RuntimeError, match="does not support reverse propagation"):
        list(propagate_in_video_with_context(Predictor(), None, strict_reverse=True, **args))


def test_formal_worker_has_no_detector_selector(tmp_path):
    with pytest.raises(SystemExit):
        parse_args([
            "--scene-name", "scene", "--video-path", "video.mp4",
            "--model-cfg", "config.yaml", "--model-weights", "weights.pt",
            "--auto-init-detector", "motion",
        ])
    assert resolve_precision_policy("auto", cuda_available=False, bf16_supported=False)["resolved"] == "fp32"
    assert resolve_precision_policy("auto", cuda_available=True, bf16_supported=True)["resolved"] == "bf16"
    with pytest.raises(RuntimeError, match="requires CUDA"):
        resolve_precision_policy("fp16", cuda_available=False, bf16_supported=False)


def test_mocked_sam2_preserves_multiobject_outputs_and_provenance(tmp_path, monkeypatch):
    from physics_eval.tracking import sam2
    (tmp_path / "input.mp4").touch()
    (tmp_path / "weights.pt").touch()
    active = []
    @contextmanager
    def inference_mode():
        active.append(True)
        try:
            yield
        finally:
            active.pop()
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: False),
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace()), cudnn=SimpleNamespace()),
        inference_mode=inference_mode, set_float32_matmul_precision=lambda _: None,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    class Tensor:
        def __init__(self, value):
            self.value = value
        def __gt__(self, other):
            return Tensor(self.value > other)
        def detach(self):
            return self
        def cpu(self):
            return self
        def numpy(self):
            return self.value
    calls = []
    class Predictor:
        def init_state(self, **kwargs):
            assert active
            return {}
        def add_new_points_or_box(self, **kwargs):
            assert active
            calls.append(kwargs)
        def propagate_in_video(self, state, start_frame_idx, reverse):
            for idx in ([1, 0] if reverse else [1, 2]):
                assert active
                mask = np.zeros((8, 8), dtype=float)
                mask[2:4, 2:4] = 1
                yield idx, [1, 2], [Tensor(mask), Tensor(np.roll(mask, 2, axis=1))]
    monkeypatch.setattr(sam2, "load_sam2_predictor", lambda **_: Predictor())
    monkeypatch.setattr(sam2, "extract_video_to_frames", lambda **_: (30.0, 3, 8, 8, True))
    manifest = {
        "mode": "manual", "num_objects": 2, "manifest_path": str(tmp_path / "manifest.json"),
        "prompts": [
            {"object_id": 1, "frame_idx": 1, "positive_points": [[2, 2], [3, 3]], "negative_points": [[7, 7]]},
            {"object_id": 2, "frame_idx": 1, "point": [4, 2], "box": [4, 2, 5, 3]},
        ],
    }
    result = run_tracking(_args(tmp_path), prompt_manifest=manifest)
    assert not active
    assert result["prompt"]["mode"] == "metadata_manual"
    assert result["object_ids"] == [1, 2]
    assert [frame["frame_idx"] for frame in result["frames"]] == [0, 1, 2]
    assert all(len(frame["objects"]) == 2 for frame in result["frames"])
    assert result["frames"][1]["objects"][0]["propagation_direction"] == "forward"
    assert all(Path(obj["mask_path"]).is_file() for frame in result["frames"] for obj in frame["objects"])
    assert result["batch_initialization"] == manifest
    np.testing.assert_array_equal(calls[0]["labels"], [1, 1, 0])
    saved = json.loads((tmp_path / "sam2_tracks/scene/tracking_points.json").read_text())
    config = json.loads((tmp_path / "sam2_tracks/scene/run_config.json").read_text())
    assert saved == result
    assert config["auto_init_detector"] == "yolo_then_motion"
    assert config["batch_initialization"] == manifest
    assert config["precision_policy"]["resolved"] == "fp32"


def test_direct_manual_prompt_keeps_frame_object_label_and_box(tmp_path):
    args = _args(tmp_path)
    args.init_frame_idx, args.init_obj_id, args.init_label = 17, 4, 0
    args.init_point_x, args.init_point_y = 3.25, 7.5
    args.init_box = [1.0, 2.0, 8.0, 9.0]
    prompts, mode, evidence = build_tracking_prompts(
        args, frames_dir=tmp_path, width=30, height=30,
    )
    assert mode == "manual"
    assert evidence == []
    assert prompts == [{
        "obj_id": 4, "frame_idx": 17, "point": [3.25, 7.5],
        "positive_points": [[3.25, 7.5]], "label": 0,
        "box": [1.0, 2.0, 8.0, 9.0], "negative_points": [], "auto_init": None,
    }]


def test_worker_help_does_not_load_third_party_runtime():
    import subprocess
    result = subprocess.run(
        [sys.executable, "-S", "-m", "physics_eval.tracking.worker", "--help"],
        cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--prompt-manifest" in result.stdout
    assert "--init-frame-idx" in result.stdout
    assert "--auto-init-detector" not in result.stdout
