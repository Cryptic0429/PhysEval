from __future__ import annotations

from physics_eval.evaluators.common import done, get_primary_track, safe_eval
from physics_eval.calibration import require_scale
from physics_eval.math.fitting import linear_fit, project_motion
from physics_eval.quality.estimation_checks import require_min_frames, status_from_r2


def eval_velocity(tracking_df, metadata_row):
    def _run():
        track = get_primary_track(tracking_df)
        require_min_frames(track, 4)
        scale = require_scale(metadata_row, track)
        t = track["t"].to_numpy(float)
        q, axis_used = project_motion(track, str(metadata_row.get("Motion_Axis") or "auto"))
        v_px, intercept_px, r2, _ = linear_fit(t, q)
        status, reason = status_from_r2(r2)
        measured = abs(v_px) * scale.scale_m_per_px
        intercept = intercept_px * scale.scale_m_per_px
        return done(
            metadata_row,
            measured,
            "m/s",
            r2,
            scale.scale_m_per_px,
            status,
            reason,
            {"axis_used": axis_used, "intercept": intercept, "pixel_size_source": scale.source},
        )

    return safe_eval(metadata_row, _run)
