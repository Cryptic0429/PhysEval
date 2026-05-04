from __future__ import annotations

import numpy as np

from physics_eval.evaluators.common import done, get_primary_track, safe_eval
from physics_eval.utils.fitting import fit_slope_window, moving_average
from physics_eval.utils.quality import require_min_frames


def eval_restitution(tracking_df, metadata_row):
    def _run():
        track = get_primary_track(tracking_df)
        require_min_frames(track, 10)
        t = track["t"].to_numpy(float)
        x = track["cx_px"].to_numpy(float)
        vx = moving_average(np.gradient(x, t), 5)
        sign = np.sign(vx)
        flips = np.where((sign[:-1] * sign[1:] < 0) & (np.abs(vx[:-1]) > 1e-6) & (np.abs(vx[1:]) > 1e-6))[0]
        if len(flips) == 0:
            raise ValueError("event_not_detected")
        idx = int(flips[np.argmax(np.abs(vx[flips]) + np.abs(vx[flips + 1]))] + 1)
        w = max(4, min(10, len(t) // 6))
        v_before, r2_before = fit_slope_window(t, x, idx - w, idx - 1)
        v_after, r2_after = fit_slope_window(t, x, idx + 1, idx + w + 2)
        if abs(v_before) <= 1e-9:
            raise ValueError("event_not_detected")
        e = abs(v_after) / abs(v_before)
        r2 = min(r2_before, r2_after)
        status = "valid" if r2 >= 0.75 else "weak_valid"
        return done(
            metadata_row,
            e,
            "none",
            r2,
            None,
            status,
            "" if status == "valid" else "low_fit_r2",
            {"collision_frame": idx, "v_before_px_s": v_before, "v_after_px_s": v_after},
        )

    return safe_eval(metadata_row, _run)
