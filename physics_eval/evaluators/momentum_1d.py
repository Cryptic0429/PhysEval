from __future__ import annotations

import numpy as np

from physics_eval.evaluators.common import done, parse_known_params, safe_eval
from physics_eval.utils.fitting import fit_slope_window, pca_project
from physics_eval.utils.geometry import align_tracks_on_time, distance_xy
from physics_eval.utils.quality import require_min_frames
from physics_eval.utils.tracking import get_two_object_tracks


def eval_momentum_1d(tracking_df, metadata_row):
    def _run():
        params = parse_known_params(metadata_row)
        m1 = float(params.get("m1_kg"))
        m2 = float(params.get("m2_kg"))
        tr1, tr2 = get_two_object_tracks(tracking_df)
        require_min_frames(tr1, 8)
        require_min_frames(tr2, 8)
        merged = align_tracks_on_time(tr1, tr2)
        require_min_frames(merged, 8)
        t = merged["t"].to_numpy(float)
        d = distance_xy(
            merged["cx_px_1"].to_numpy(float),
            merged["cy_px_1"].to_numpy(float),
            merged["cx_px_2"].to_numpy(float),
            merged["cy_px_2"].to_numpy(float),
        )
        event_idx = int(np.argmin(d))
        w = max(4, min(10, len(t) // 5))
        if event_idx - w < 2 or event_idx + w + 2 > len(t):
            raise ValueError("event_not_detected")

        s1, axis = pca_project(merged["cx_px_1"].to_numpy(float), merged["cy_px_1"].to_numpy(float))
        axis_x, axis_y = axis
        pts2 = np.column_stack([merged["cx_px_2"].to_numpy(float), merged["cy_px_2"].to_numpy(float)])
        center1 = np.array([merged["cx_px_1"].mean(), merged["cy_px_1"].mean()])
        s2 = (pts2 - center1) @ axis

        v1_before, r2a = fit_slope_window(t, s1, event_idx - w, event_idx - 1)
        v2_before, r2b = fit_slope_window(t, s2, event_idx - w, event_idx - 1)
        v1_after, r2c = fit_slope_window(t, s1, event_idx + 1, event_idx + w + 2)
        v2_after, r2d = fit_slope_window(t, s2, event_idx + 1, event_idx + w + 2)
        p_before = m1 * v1_before + m2 * v2_before
        p_after = m1 * v1_after + m2 * v2_after
        err = abs(p_after - p_before) / max(abs(p_before), 1e-9)
        r2 = min(r2a, r2b, r2c, r2d)
        status = "valid" if r2 >= 0.7 else "weak_valid"
        return done(
            metadata_row,
            err,
            "relative_error",
            r2,
            None,
            status,
            "" if status == "valid" else "low_fit_r2",
            {
                "event_frame": int(merged.iloc[event_idx]["frame"]),
                "axis": [float(axis_x), float(axis_y)],
                "v1_before_px_s": v1_before,
                "v2_before_px_s": v2_before,
                "v1_after_px_s": v1_after,
                "v2_after_px_s": v2_after,
                "p_before_px_kg_s": p_before,
                "p_after_px_kg_s": p_after,
            },
        )

    return safe_eval(metadata_row, _run)

