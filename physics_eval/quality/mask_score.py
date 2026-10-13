"""Pure Qmask scoring; missing evidence is distinct from poor evidence."""
from __future__ import annotations

import math
from statistics import mean

from .protocol import protocol_hash, validate_protocol


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
    progress = max(0.0, min(1.0, (value - good) / (bad - good)))
    return max(0.0, min(1.0, 1.0 - progress ** gamma))


def compute_qmask(evidence: dict, protocol: dict, required_object_ids=None) -> dict:
    validate_protocol(protocol)
    legacy = protocol["missing_policy"] == "legacy"
    objects = evidence.get("evidence_objects", evidence.get("objects")) or []
    reasons = sorted({r for obj in objects for r in str(obj.get("reasons") or "").split(";") if r})
    qc_flagged = evidence.get("status") == "flagged" or any(
        obj.get("decision") and obj["decision"] != "keep" for obj in objects
    )
    actual = [str(obj.get("object_id")) for obj in objects]
    required = sorted(map(str, required_object_ids)) if required_object_ids is not None else sorted(actual)
    missing_objects = sorted(set(required) - set(actual))
    unexpected_objects = sorted(set(actual) - set(required))
    duplicate_objects = len(actual) != len(set(actual))
    scored = []
    missing = []
    warnings = []
    for obj in sorted(objects, key=lambda item: str(item.get("object_id"))):
        components = {}
        for name, spec in protocol["components"].items():
            score = component_score(obj.get(name), spec, protocol["gamma"])
            if score is None:
                missing.append({"object_id": obj.get("object_id"), "component": name})
            else:
                components[name] = score
        if legacy:
            tokens = str(obj.get("reasons") or "")
            if "insufficient_valid_middle_frames" in tokens:
                components["valid_middle_frames"] = 0.0
            if "low_middle_coverage" in tokens and "middle_coverage" not in components:
                components["middle_coverage"] = 0.0
        groups = {}
        if protocol["aggregation"] == "group_mean":
            for name, group in protocol["groups"].items():
                if all(key in components for key in group["components"]):
                    groups[name] = mean(components[key] for key in group["components"])
            score = (sum(groups[name] * group["weight"] for name, group in protocol["groups"].items())
                     / sum(group["weight"] for group in protocol["groups"].values())) if len(groups) == len(protocol["groups"]) else None
        elif components:
            score = min(components.values()) if protocol["aggregation"] == "min" else mean(components.values())
        else:
            score = (1.0 if obj.get("decision") in (None, "", "keep") else 0.0) if legacy else None
        scored.append({"object_id": obj.get("object_id"), "score": score,
                       "components": components, "groups": groups, "decision": obj.get("decision"),
                       "minimum_component": min(components.values()) if components else None})
    unavailable = not legacy and (not objects or missing or missing_objects or unexpected_objects or duplicate_objects)
    if legacy and (missing or not objects):
        warnings.append("legacy_missing_evidence_fallback")
    if unavailable:
        score, status = None, "score_unavailable"
    else:
        values = [obj["score"] for obj in scored]
        score = (min(values) if protocol["object_aggregation"] == "min" else mean(values)) if values else (0.0 if qc_flagged else 1.0)
        status = "fail" if score <= 0 else ("pass" if score >= 0.999 and not qc_flagged else "weak")
    return {"score": score, "status": status, "objects": scored, "reasons": reasons,
            "protocol_id": protocol["protocol_id"], "protocol_hash": protocol_hash(protocol),
            "window_policy": protocol["window_policy"], "required_object_ids": required,
            "missing_components": missing, "missing_objects": missing_objects,
            "unexpected_objects": unexpected_objects, "duplicate_objects": duplicate_objects,
            "warnings": warnings, "evidence_source": evidence.get("evidence_source", "embedded_result"),
            "diagnostic_version": evidence.get("diagnostic_version", "legacy_embedded_unversioned")}
