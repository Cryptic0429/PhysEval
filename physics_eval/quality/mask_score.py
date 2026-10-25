"""Mask reliability from minimum component and object scores."""
from __future__ import annotations

import math

from .config import load_mask_config, mask_config_hash, validate_mask_config


def finite(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def component_score(value, spec: dict, gamma: float):
    value = finite(value)
    if value is None:
        return None
    good, bad = spec["good_value"], spec["bad_value"]
    if spec["direction"] == "high_bad":
        if value <= good:
            return 1.0
        if value >= bad:
            return 0.0
        progress = (value - good) / max(bad - good, 1e-12)
    else:
        if value >= good:
            return 1.0
        if value <= bad:
            return 0.0
        progress = (good - value) / max(good - bad, 1e-12)
    return min(1.0, max(0.0, 1.0 - math.pow(progress, gamma)))


def compute_qmask(evidence: dict, config: dict | None = None) -> dict:
    """Score saved ``objects`` evidence without adding eligibility gates.

    Missing individual diagnostics are ignored, but object evidence and at least
    one computable component per object are required. Quality-control flags
    never raise component scores. Explicit diagnosed failures contribute zero.
    """
    config = load_mask_config() if config is None else validate_mask_config(config)
    objects = evidence.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError(
            "Mask diagnostics missing: mask_qc.objects must contain object diagnostics. "
            "Run eval with the corresponding masks before scoring."
        )
    reasons = []
    flagged = str(evidence.get("status") or "") == "flagged"
    scored, missing = [], []
    for obj in objects:
        if not isinstance(obj, dict):
            raise ValueError("Mask diagnostics invalid: each object record must be a dictionary")
        decision = str(obj.get("decision") or "")
        tokens = str(obj.get("reasons") or "")
        if decision and decision != "keep":
            flagged = True
        reasons.extend(part for part in tokens.split(";") if part)
        components = {}
        for name, spec in config["components"].items():
            score = component_score(obj.get(name), spec, config["gamma"])
            if score is None:
                missing.append({"object_id": obj.get("object_id"), "component": name})
            else:
                components[name] = score
        if "insufficient_valid_middle_frames" in tokens:
            components["valid_middle_frames"] = 0.0
        if "low_middle_coverage" in tokens and "middle_coverage" not in components:
            components["middle_coverage"] = 0.0
        if not components:
            raise ValueError(
                f"Mask diagnostics missing for object {obj.get('object_id')!r}: "
                "no computable quality components. Regenerate diagnostics from its masks."
            )
        score = min(components.values())
        scored.append({"object_id": obj.get("object_id"), "score": score,
                       "components": components, "decision": obj.get("decision")})
    score = min(item["score"] for item in scored)
    status = "fail" if score <= 0 else ("pass" if score >= 0.999 and not flagged else "weak")
    warnings = []
    if missing:
        warnings.append("missing_components_ignored")
    return {"score": score, "status": status, "objects": scored,
            "reasons": sorted(set(reasons)) if flagged else [], "config_hash": mask_config_hash(config),
            "missing_components": missing, "warnings": warnings,
            "window_fractions": config["window_fractions"],
            "evidence_source": evidence.get("evidence_source", "embedded_result")}
