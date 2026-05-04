from __future__ import annotations

from physics_eval.evaluators.common import done, get_primary_track, numeric_target, safe_eval
from physics_eval.utils.calibration import require_scale
from physics_eval.utils.fitting import project_motion, quadratic_fit
from physics_eval.utils.quality import require_min_frames, status_from_r2


def eval_acceleration(tracking_df, metadata_row):
    def _run():
        track = get_primary_track(tracking_df)
        require_min_frames(track, 5)
        scale = require_scale(metadata_row, track)
        t = track["t"].to_numpy(float)
        s, axis_used = project_motion(track, str(metadata_row.get("Motion_Axis") or "auto"), scale.scale_m_per_px)
        a2, b, c, r2, _ = quadratic_fit(t, s)
        acc = 2.0 * a2
        target = numeric_target(metadata_row)
        if target is not None and target >= 0:
            acc = abs(acc)
        status, reason = status_from_r2(r2)
        return done(
            metadata_row,
            acc,
            "m/s^2",
            r2,
            scale.scale_m_per_px,
            status,
            reason,
            {"axis_used": axis_used, "quadratic_A": a2, "linear_B": b, "constant_C": c},
        )

    return safe_eval(metadata_row, _run)
