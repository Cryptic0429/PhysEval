from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


def write_tracking(path: Path, frames: list[dict], fps: float = 30.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"video_info": {"fps": fps}, "object_id": 1, "frames": frames}, indent=2),
        encoding="utf-8",
    )


def bbox(cx: float, cy: float, d: float = 20.0) -> list[float]:
    r = d / 2.0
    return [cx - r, cy - r, cx + r, cy + r]


def make_single_object_json(path: Path, func, fps: float = 30.0, n: int = 90) -> None:
    frames = []
    for frame in range(n):
        t = frame / fps
        x, y = func(t)
        frames.append({
            "frame_idx": frame,
            "time_sec": t,
            "valid": True,
            "x": x,
            "y": y,
            "bbox": bbox(x, y),
            "area": math.pi * 10.0 * 10.0,
        })
    write_tracking(path, frames, fps)


def make_restitution_json(path: Path, fps: float = 30.0) -> None:
    frames = []
    collision_t = 1.0
    for frame in range(75):
        t = frame / fps
        if t <= collision_t:
            x = 100.0 + 60.0 * t
        else:
            x = 100.0 + 60.0 * collision_t - 30.0 * (t - collision_t)
        y = 100.0
        frames.append({"frame_idx": frame, "time_sec": t, "valid": True, "x": x, "y": y, "bbox": bbox(x, y), "area": 314.0})
    write_tracking(path, frames, fps)


def build_fixture(root: Path) -> Path:
    if root.exists():
        shutil.rmtree(root)
    json_dir = root / "tracking_json"
    json_dir.mkdir(parents=True)

    # 20 px object diameter represents 0.1 m, so scale = 0.005 m/px.
    scale = 0.005
    make_single_object_json(
        json_dir / "velocity.json",
        lambda t: (100.0 + (0.5 / scale) * t, 80.0),
    )
    make_single_object_json(
        json_dir / "acceleration.json",
        lambda t: (100.0 + 0.5 * (0.4 / scale) * t * t, 80.0),
    )
    make_single_object_json(
        json_dir / "spring.json",
        lambda t: (100.0 + 30.0 * math.sin(2.0 * math.pi * t / 1.0), 80.0),
        n=120,
    )
    make_restitution_json(json_dir / "restitution.json")

    rows = [
        {
            "Index": 1,
            "Prompt_ID": "synthetic_velocity",
            "Video_File": "velocity.mp4",
            "Tracking_JSON": "tracking_json/velocity.json",
            "Module": "B1.1",
            "Metric": "velocity",
            "Evaluator": "eval_velocity",
            "Measurement_Method": "linear_fit",
            "Tracking_Objects": "object_center",
            "Num_Objects": 1,
            "View": "fixed_side",
            "Motion_Axis": "x",
            "Scale_Mode": "object_diameter",
            "Calibration_Object": "ball",
            "Calibration_Dimension": "diameter",
            "Calibration_Value_m": 0.1,
            "Target_Type": "target_speed",
            "Target_Value": 0.5,
            "Target_Unit": "m/s",
            "Known_Parameters_JSON": "{}",
        },
        {
            "Index": 2,
            "Prompt_ID": "synthetic_acceleration",
            "Video_File": "acceleration.mp4",
            "Tracking_JSON": "tracking_json/acceleration.json",
            "Module": "B1.2",
            "Metric": "acceleration",
            "Evaluator": "eval_acceleration",
            "Measurement_Method": "quadratic_fit",
            "Tracking_Objects": "object_center",
            "Num_Objects": 1,
            "View": "fixed_side",
            "Motion_Axis": "x",
            "Scale_Mode": "object_diameter",
            "Calibration_Object": "ball",
            "Calibration_Dimension": "diameter",
            "Calibration_Value_m": 0.1,
            "Target_Type": "target_acceleration",
            "Target_Value": 0.4,
            "Target_Unit": "m/s^2",
            "Known_Parameters_JSON": "{}",
        },
        {
            "Index": 3,
            "Prompt_ID": "synthetic_spring",
            "Video_File": "spring.mp4",
            "Tracking_JSON": "tracking_json/spring.json",
            "Module": "B3.5",
            "Metric": "spring_constant",
            "Evaluator": "eval_spring",
            "Measurement_Method": "period_detection",
            "Tracking_Objects": "object_center",
            "Num_Objects": 1,
            "View": "fixed_side",
            "Motion_Axis": "spring_axis",
            "Scale_Mode": "none",
            "Calibration_Object": "none",
            "Calibration_Dimension": "none",
            "Calibration_Value_m": None,
            "Target_Type": "spring_constant",
            "Target_Value": 4.0 * math.pi * math.pi * 0.2,
            "Target_Unit": "N/m",
            "Known_Parameters_JSON": json.dumps({"mass_kg": 0.2}),
        },
        {
            "Index": 4,
            "Prompt_ID": "synthetic_restitution",
            "Video_File": "restitution.mp4",
            "Tracking_JSON": "tracking_json/restitution.json",
            "Module": "B3.2",
            "Metric": "restitution_coefficient",
            "Evaluator": "eval_restitution",
            "Measurement_Method": "horizontal_velocity_ratio",
            "Tracking_Objects": "ball_center",
            "Num_Objects": 1,
            "View": "fixed_side",
            "Motion_Axis": "x",
            "Scale_Mode": "ratio_only",
            "Calibration_Object": "none",
            "Calibration_Dimension": "none",
            "Calibration_Value_m": None,
            "Target_Type": "restitution_coefficient",
            "Target_Value": 0.5,
            "Target_Unit": "none",
            "Known_Parameters_JSON": "{}",
        },
    ]

    meta = root / "synthetic_metadata.xlsx"
    pd.DataFrame(rows).to_excel(meta, index=False)
    return meta


def main() -> None:
    root = Path(__file__).resolve().parent / "synthetic_data"
    meta = build_fixture(root)
    output_dir = root / "eval_results"
    cmd = [
        sys.executable,
        "-m",
        "physics_eval.run_batch_eval",
        "--metadata",
        str(meta),
        "--json-root",
        str(root),
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[2])
    print((output_dir / "results.csv").read_text(encoding="utf-8-sig"))


if __name__ == "__main__":
    main()

