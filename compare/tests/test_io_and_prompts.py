from __future__ import annotations

from pathlib import Path

from physeval_compare.io_utils import expected_objects, infer_target_hint
from physeval_compare.types import Prompt


def test_target_hint_prefers_known_object() -> None:
    row = {
        "Known_Parameters_JSON": '{"object": "standard_hockey_puck"}',
        "Calibration_Object": "none",
        "Tracking_Objects": "object_center",
    }
    assert infer_target_hint(row) == "standard_hockey_puck"


def test_expected_objects_accepts_excel_number() -> None:
    assert expected_objects({"Num_Objects": 2.0}) == 2


def test_prompt_json_round_trip() -> None:
    prompt = Prompt(
        obj_id=1,
        frame_idx=3,
        point=[12.5, 9.0],
        box=[2.0, 3.0, 20.0, 18.0],
        score=0.8,
        detector="yolo",
        negative_points=[[1.0, 1.0]],
        details={"scan_indices": [0, 3, 6]},
    )
    restored = Prompt.from_dict(prompt.to_dict())
    assert restored == prompt
