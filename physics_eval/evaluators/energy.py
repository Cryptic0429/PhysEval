from __future__ import annotations

import numpy as np

from physics_eval.evaluators.common import done, g_from_params, get_primary_track, safe_eval
from physics_eval.utils.calibration import require_scale
from physics_eval.utils.fitting import central_velocity
from physics_eval.utils.quality import require_min_frames


def eval_energy(tracking_df, metadata_row):
    def _run():
        track = get_primary_track(tracking_df)
        require_min_frames(track, 10)
        scale = require_scale(metadata_row, track)
        t = track["t"].to_numpy(float)
        x = track["cx_px"].to_numpy(float) * scale.scale_m_per_px
        y_down = track["cy_px"].to_numpy(float) * scale.scale_m_per_px
        release_y = float(y_down[0])
        low_idx = int(np.argmax(y_down))
        if low_idx <= 1:
            raise ValueError("event_not_detected")
        delta_h = float(y_down[low_idx] - release_y)
        if delta_h <= 1e-5:
            raise ValueError("fit_failed")
        vx = central_velocity(t, x)
        vy = central_velocity(t, y_down)
        v_low = float(np.sqrt(vx[low_idx] ** 2 + vy[low_idx] ** 2))
        g = max(abs(g_from_params(metadata_row)), 1e-12)
        ratio = v_low * v_low / (2.0 * g * delta_h)
        return done(
            metadata_row,
            ratio,
            "none",
            None,
            scale.scale_m_per_px,
            "valid",
            "",
            {"delta_h_m": delta_h, "lowest_frame": int(track.iloc[low_idx]["frame"]), "v_lowest_mps": v_low, "g_mps2": g},
        )

    return safe_eval(metadata_row, _run)
