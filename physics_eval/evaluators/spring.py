from __future__ import annotations

import math
import json
import numpy as np

from physics_eval.evaluators.common import done, get_primary_track, parse_known_params, safe_eval
from physics_eval.utils.fitting import estimate_period, project_motion
from physics_eval.utils.quality import require_min_frames

CONFIDENCE_PROTOCOL = "period_confidence_v3_candidate"


def apply_period_confidence(legacy_result, tracking_df, metadata_row):
    """Keep legacy eligibility and measurements; audit quality only affects 1/.8 weight.

    Accepts a freshly computed or saved legacy result with Extra_JSON. Diagnostic
    errors are explicit weak evidence, never grounds to remove an eligible row.
    """
    result = dict(legacy_result)
    extra = json.loads(result.get("Extra_JSON") or "{}")
    previous_status = result["Status"]
    extra.update(period_protocol=CONFIDENCE_PROTOCOL,
                 eligibility_policy="legacy_membership_preserved",
                 legacy_status=previous_status, legacy_confidence=result.get("Fit_R2"))
    if previous_status not in {"valid", "weak_valid"}:
        extra.update(period_quality_status="not_evaluated", period_confidence=None,
                     period_quality_reasons=["legacy_ineligible_preserved"])
        result["Extra_JSON"] = json.dumps(extra, ensure_ascii=False, allow_nan=False)
        return result
    try:
        from physics_eval.utils.period_quality import estimate_period_quality
        track = get_primary_track(tracking_df)
        t = track["t"].to_numpy(float)
        frame_steps = np.diff(track["frame"].to_numpy(float))
        if len(t) < 2 or np.any(frame_steps <= 0):
            raise ValueError("insufficient_or_invalid_frame_indices")
        s, _ = project_motion(track, str(metadata_row.get("Motion_Axis") or "spring_axis"), scale=None)
        if extra.get("period_s") is None or not math.isfinite(extra["period_s"]) or extra["period_s"] <= 0:
            raise ValueError("missing_or_invalid_legacy_period")
        quality = estimate_period_quality(t, s, source=extra["period_method"],
            nominal_dt=float(np.median(np.diff(t) / frame_steps)), fixed_period_s=extra["period_s"])
        quality_status, confidence, diagnostics, reasons = quality.status, quality.confidence, quality.diagnostics, quality.reasons
    except Exception as exc:
        quality_status, confidence, diagnostics = "unavailable", 0.0, {}
        reasons = [f"quality_diagnostic_error:{type(exc).__name__}:{exc}"]
    # No new exclusion gate: poor/absent evidence gets the existing weak weight.
    result["Status"] = "valid" if quality_status == "valid" else "weak_valid"
    result["Fit_R2"] = diagnostics.get("fit_r2")
    result["Failure_Reason"] = "" if result["Status"] == "valid" else "period_quality_limited_eligibility_preserved"
    extra.update(period_confidence=confidence, period_quality_status=quality_status,
                 period_quality=diagnostics, period_quality_reasons=reasons,
                 observed_cycles=diagnostics.get("observed_cycles"),
                 spring_constant_N_per_m=result.get("Measured_Value"))
    result["Extra_JSON"] = json.dumps(extra, ensure_ascii=False, allow_nan=False)
    return result


def eval_spring(tracking_df, metadata_row, *, period_protocol="legacy_v1"):
    if period_protocol == CONFIDENCE_PROTOCOL:
        return apply_period_confidence(eval_spring(tracking_df, metadata_row), tracking_df, metadata_row)
    if period_protocol != "legacy_v1":
        raise ValueError(f"Unsupported spring period protocol: {period_protocol}")
    def _run():
        params = parse_known_params(metadata_row)
        mass = float(params.get("mass_kg"))
        if not math.isfinite(mass) or mass <= 0:
            raise ValueError("invalid_mass_kg")
        track = get_primary_track(tracking_df)
        require_min_frames(track, 12)
        t = track["t"].to_numpy(float)
        s, axis_used = project_motion(track, str(metadata_row.get("Motion_Axis") or "spring_axis"), scale=None)
        nominal_dt = None
        period, confidence, method = estimate_period(t, s)
        if period <= 0:
            raise ValueError("period_not_found")
        k = 4.0 * math.pi * math.pi * mass / (period * period)
        status = "valid" if confidence >= 0.2 else "weak_valid"
        return done(
            metadata_row,
            k,
            "N/m",
            confidence,
            None,
            status,
            "" if status == "valid" else "period_detection_weak",
            {"period_s": period, "period_method": method, "mass_kg": mass, "axis_used": axis_used},
        )

    result = safe_eval(metadata_row, _run)
    return result
