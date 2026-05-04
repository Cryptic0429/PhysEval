from __future__ import annotations

from physics_eval.evaluators.common import done, get_primary_track, parse_known_params, safe_eval
from physics_eval.utils.calibration import require_scale
from physics_eval.utils.fitting import quadratic_fit
from physics_eval.utils.quality import require_min_frames, status_from_r2


def eval_density_acceleration(tracking_df, metadata_row):
    def _run():
        params = parse_known_params(metadata_row)
        rho_f = float(params.get("liquid_density_kg_m3"))
        g = float(params.get("g", 9.8))
        track = get_primary_track(tracking_df)
        require_min_frames(track, 8)
        scale = require_scale(metadata_row, track)
        n = max(6, min(len(track), len(track) // 3))
        sub = track.iloc[:n]
        t = sub["t"].to_numpy(float)
        y_down = sub["cy_px"].to_numpy(float) * scale.scale_m_per_px
        a2, b, c, r2, _ = quadratic_fit(t, y_down)
        a_down = 2.0 * a2
        denom = 1.0 - a_down / max(g, 1e-12)
        if abs(denom) <= 1e-9:
            raise ValueError("fit_failed")
        rho_s = rho_f / denom
        status, reason = status_from_r2(r2)
        return done(
            metadata_row,
            rho_s,
            "kg/m^3",
            r2,
            scale.scale_m_per_px,
            status,
            reason,
            {"initial_window_frames": n, "a_down_mps2": a_down, "liquid_density_kg_m3": rho_f},
        )

    return safe_eval(metadata_row, _run)
