"""Configuration and provenance for the single mask-quality scoring rule."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

MASK_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs/scoring/mask_quality.json"


def validate_mask_config(config: dict) -> dict:
    gamma = float(config["gamma"])
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and positive")
    if "flag_floor" in config:
        raise ValueError("flag_floor is no longer supported; mask component scores are not raised")
    if "flagged_missing_score" in config:
        raise ValueError("flagged_missing_score is no longer supported; usable mask diagnostics are required")
    start, end = config["window_fractions"]
    if not 0 <= start < end <= 1:
        raise ValueError("invalid mask diagnostic window")
    if not config["components"]:
        raise ValueError("mask scoring needs quality components")
    for spec in config["components"].values():
        good, bad = float(spec["good_value"]), float(spec["bad_value"])
        direction = spec["direction"]
        if not all(map(math.isfinite, (good, bad))) or direction not in {"high_bad", "low_bad"}:
            raise ValueError("invalid component thresholds")
        if not (good < bad if direction == "high_bad" else good > bad):
            raise ValueError("threshold ordering contradicts direction")
    return config


def load_mask_config() -> dict:
    return validate_mask_config(json.loads(MASK_CONFIG_PATH.read_text(encoding="utf-8")))


def mask_config_hash(config: dict) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()
