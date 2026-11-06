from __future__ import annotations

import numpy as np

from physics_eval.evaluators.common import done, g_from_params, get_primary_track, safe_eval
from physics_eval.calibration import require_scale
from physics_eval.math.fitting import central_velocity, project_motion, quadratic_fit
from physics_eval.quality.estimation_checks import one_dimensional_score, require_min_frames, status_from_r2


def eval_friction(tracking_df, metadata_row):
    def _run():
        track = get_primary_track(tracking_df)
        require_min_frames(track, 8)
        one_d = one_dimensional_score(track["cx_px"].to_numpy(float), track["cy_px"].to_numpy(float))
        if one_d < 0.75:
            raise ValueError("motion_not_1d")
        scale = require_scale(metadata_row, track)
        t = track["t"].to_numpy(float)
        q, axis_used = project_motion(track, str(metadata_row.get("Motion_Axis") or "x"))
        a2_px, b_px, c_px, r2, _ = quadratic_fit(t, q)
        v_px = central_velocity(t, q)
        speed_drop = abs(v_px[-1]) < abs(v_px[0]) if len(v_px) >= 2 else False
        status, reason = status_from_r2(r2)
        if not speed_drop and status == "valid":
            status, reason = "weak_valid", "velocity_not_decreasing"
        acc = 2.0 * a2_px * scale.scale_m_per_px
        mu = abs(acc) / max(abs(g_from_params(metadata_row)), 1e-12)
        return done(
            metadata_row,
            mu,
            "none",
            r2,
            scale.scale_m_per_px,
            status,
            reason,
            {"axis_used": axis_used, "acceleration_mps2": acc, "one_dimensional_score": one_d, "speed_drop": bool(speed_drop)},
        )

    return safe_eval(metadata_row, _run)
