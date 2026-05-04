from __future__ import annotations

import math

from physics_eval.evaluators.common import done, get_primary_track, parse_known_params, safe_eval
from physics_eval.utils.fitting import estimate_period, project_motion
from physics_eval.utils.quality import require_min_frames


def eval_spring(tracking_df, metadata_row):
    def _run():
        params = parse_known_params(metadata_row)
        mass = float(params.get("mass_kg"))
        track = get_primary_track(tracking_df)
        require_min_frames(track, 12)
        t = track["t"].to_numpy(float)
        s, axis_used = project_motion(track, str(metadata_row.get("Motion_Axis") or "spring_axis"), scale=None)
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

    return safe_eval(metadata_row, _run)
