from __future__ import annotations

from physics_eval.evaluators.common import done, get_primary_track, g_from_params, safe_eval
from physics_eval.utils.calibration import estimate_scale
from physics_eval.utils.fitting import project_motion, quadratic_fit
from physics_eval.utils.quality import require_min_frames, status_from_r2


def eval_gravity(tracking_df, metadata_row):
    def _run():
        track = get_primary_track(tracking_df)
        require_min_frames(track, 5)
        scale = estimate_scale(metadata_row, track)
        scale_value = scale.scale_m_per_px
        if scale_value is None and str(metadata_row.get("Scale_Mode") or "").lower() not in {"none", "ratio_only"}:
            raise ValueError("scale_unavailable")
        if scale_value is None:
            scale_value = 1.0
        t = track["t"].to_numpy(float)
        s, axis_used = project_motion(track, "y", scale_value, y_down_positive=True)
        a2, b, c, r2, _ = quadratic_fit(t, s)
        measured = abs(2.0 * a2)
        ref_g = g_from_params(metadata_row, default=9.8)
        status, reason = status_from_r2(r2)
        if ref_g == 0 and measured < 0.2:
            status, reason = "valid", ""
        return done(
            metadata_row,
            measured,
            "m/s^2" if scale.scale_m_per_px is not None else "px/s^2",
            r2,
            scale.scale_m_per_px,
            status,
            reason,
            {"axis_used": axis_used, "reference_g_mps2": ref_g, "quadratic_A": a2, "linear_B": b, "constant_C": c},
        )

    return safe_eval(metadata_row, _run)
