from __future__ import annotations

import numpy as np

from physics_eval.evaluators.common import done, g_from_params, get_primary_track, safe_eval
from physics_eval.utils.calibration import require_scale
from physics_eval.utils.fitting import central_velocity, project_motion, quadratic_fit
from physics_eval.utils.quality import one_dimensional_score, require_min_frames, status_from_r2


def eval_friction(tracking_df, metadata_row):
    def _run():
        track = get_primary_track(tracking_df)
        require_min_frames(track, 8)
        one_d = one_dimensional_score(track["cx_px"].to_numpy(float), track["cy_px"].to_numpy(float))
        if one_d < 0.75:
            raise ValueError("motion_not_1d")
        scale = require_scale(metadata_row, track)
        t = track["t"].to_numpy(float)
        s, axis_used = project_motion(track, str(metadata_row.get("Motion_Axis") or "x"), scale.scale_m_per_px)
        a2, b, c, r2, _ = quadratic_fit(t, s)
        acc = 2.0 * a2
        mu = abs(acc) / max(abs(g_from_params(metadata_row)), 1e-12)
        v = central_velocity(t, s)
        speed_drop = abs(v[-1]) < abs(v[0]) if len(v) >= 2 else False
        status, reason = status_from_r2(r2)
        if not speed_drop and status == "valid":
            status, reason = "weak_valid", "velocity_not_decreasing"
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
