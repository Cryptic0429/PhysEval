"""Load and fingerprint declarative Qmask protocols."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs" / "scoring"


def validate_protocol(config: dict) -> dict:
    if not isinstance(config.get("protocol_id"), str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", config["protocol_id"]) or config["protocol_id"] in {".", ".."}:
        raise ValueError("protocol_id must be a safe filename component")
    gamma = float(config["gamma"])
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and positive")
    if config["aggregation"] not in {"min", "mean", "group_mean"}:
        raise ValueError("unknown aggregation")
    if config["object_aggregation"] not in {"min", "mean"}:
        raise ValueError("unknown object aggregation")
    if config["missing_policy"] not in {"legacy", "unavailable"}:
        raise ValueError("unknown missing policy")
    if not config["components"]:
        raise ValueError("protocol needs quality components")
    for spec in config["components"].values():
        good, bad = float(spec["good_value"]), float(spec["bad_value"])
        direction = spec["direction"]
        if not all(map(math.isfinite, (good, bad))) or direction not in {"high_bad", "low_bad"}:
            raise ValueError("invalid component thresholds")
        if not (good < bad if direction == "high_bad" else good > bad):
            raise ValueError("threshold ordering contradicts direction")
    if config["aggregation"] == "group_mean":
        members = []
        for group in config["groups"].values():
            if not group["components"] or not math.isfinite(float(group["weight"])) or group["weight"] <= 0:
                raise ValueError("groups need components and positive finite weights")
            members.extend(group["components"])
        if len(members) != len(set(members)) or set(members) != set(config["components"]):
            raise ValueError("groups must partition all components exactly once")
    return config


def load_protocol(name: str = "grouped_v2_candidate") -> dict:
    path = CONFIG_ROOT / f"{name}.json"
    if not path.is_file():
        path = Path(name).expanduser()
    return validate_protocol(json.loads(path.read_text(encoding="utf-8")))


def protocol_hash(config: dict) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()
