from __future__ import annotations

from physics_eval.evaluators.common import done, get_primary_track, safe_eval
from physics_eval.utils.calibration import require_scale
from physics_eval.utils.fitting import linear_fit, project_motion
from physics_eval.utils.quality import require_min_frames, status_from_r2


def eval_velocity(tracking_df, metadata_row):
    def _run():
        track = get_primary_track(tracking_df)
        require_min_frames(track, 4)
        scale = require_scale(metadata_row, track)
        t = track["t"].to_numpy(float)
        s, axis_used = project_motion(track, str(metadata_row.get("Motion_Axis") or "auto"), scale.scale_m_per_px)
        v, intercept, r2, _ = linear_fit(t, s)
        measured = abs(v)
        status, reason = status_from_r2(r2)
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
