"""Public evaluation policy and dependency-free command help."""
from pathlib import Path
import subprocess
import sys

import pytest

from physics_eval.cli import parse_eval_args
from physics_eval.pipeline import resolve_tracking_initialization


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("mode", ["motion", "yolo", "yolo_then_motion"])
def test_formal_cli_has_no_detector_selector(mode):
    with pytest.raises(SystemExit) as exc:
        parse_eval_args(["--detector", mode])
    assert exc.value.code == 2


@pytest.mark.parametrize("mode", ["motion", "yolo"])
def test_row_overrides_cannot_bypass_fixed_initialization(mode):
    args = parse_eval_args([])
    row = {
        "Num_Objects": 1,
        "Known_Parameters_JSON": "{}",
        "Tracking_Overrides_JSON": '{"detector":"' + mode + '"}',
    }
    with pytest.raises(ValueError, match="fixes initialization"):
        resolve_tracking_initialization(row, args)


@pytest.mark.parametrize("command", ["eval", "score"])
def test_public_help_does_not_require_third_party_packages(command):
    completed = subprocess.run(
        [sys.executable, "-S", str(ROOT / "physeval.py"), command, "--help"],
        text=True, capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    if command == "eval":
        assert "--detector" not in completed.stdout
    else:
        assert "--scoring-protocol" not in completed.stdout


def test_public_defaults_keep_mask_evidence_and_explicit_run_options(tmp_path):
    args = parse_eval_args([
        "--workspace-root", str(tmp_path), "--vis", "--diagnostics",
        "--reuse-tracking", "--no-plots", "--area-action", "warn",
    ])
    assert args.detector == "yolo_then_motion"
    assert args.save_mask_png and args.area_stability_check and args.area_prefer_mask_area
    assert args.save_vis_video and args.save_diagnostics and args.save_auto_init_debug
    assert args.skip_tracking_if_exists
    assert not args.area_stability_save_plots
