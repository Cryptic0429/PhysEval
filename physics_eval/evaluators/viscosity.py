from __future__ import annotations

import math

from physics_eval.evaluators.common import done, get_primary_track, parse_known_params, safe_eval
from physics_eval.utils.calibration import require_scale
from physics_eval.utils.fitting import terminal_velocity
from physics_eval.utils.quality import require_min_frames


def eval_viscosity(tracking_df, metadata_row):
    def _run():
        params = parse_known_params(metadata_row)
        rho_s = float(params.get("solid_density_kg_m3"))
        rho_f = float(params.get("liquid_density_kg_m3"))
        g = float(params.get("g", 9.8))
        diameter = float(metadata_row.get("Calibration_Value_m"))
        r = diameter / 2.0
        track = get_primary_track(tracking_df)
        require_min_frames(track, 12)
        scale = require_scale(metadata_row, track)
        t = track["t"].to_numpy(float)
        y_down = track["cy_px"].to_numpy(float) * scale.scale_m_per_px
        vt, confidence, window = terminal_velocity(t, y_down)
        if vt <= 1e-12:
            raise ValueError("terminal_velocity_not_found")
        eta = 2.0 * r * r * (rho_s - rho_f) * g / (9.0 * vt)
        if not math.isfinite(eta) or eta <= 0:
            raise ValueError("fit_failed")
        status = "valid" if confidence >= 0.75 else "weak_valid"
        return done(
            metadata_row,
            eta,
            "Pa*s",
            confidence,
            scale.scale_m_per_px,
            status,
            "" if status == "valid" else "terminal_velocity_weak",
            {"terminal_velocity_mps": vt, "terminal_window": list(window), "solid_density_kg_m3": rho_s, "liquid_density_kg_m3": rho_f},
        )

    return safe_eval(metadata_row, _run)
