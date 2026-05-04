from __future__ import annotations

import numpy as np


def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 1e-12:
        return 1.0 if ss_res <= 1e-12 else 0.0
    return 1.0 - ss_res / ss_tot


def polyfit_with_r2(t: np.ndarray, s: np.ndarray, degree: int) -> tuple[np.ndarray, float, np.ndarray]:
    t = np.asarray(t, dtype=float)
    s = np.asarray(s, dtype=float)
    good = np.isfinite(t) & np.isfinite(s)
    t = t[good]
    s = s[good]
    if len(t) < degree + 2:
        raise ValueError("insufficient points for polynomial fit")
    coeff = np.polyfit(t, s, degree)
    yhat = np.polyval(coeff, t)
    return coeff, r2_score(s, yhat), yhat


def linear_fit(t: np.ndarray, s: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    coeff, r2, yhat = polyfit_with_r2(t, s, 1)
    return float(coeff[0]), float(coeff[1]), r2, yhat


def quadratic_fit(t: np.ndarray, s: np.ndarray) -> tuple[float, float, float, float, np.ndarray]:
    coeff, r2, yhat = polyfit_with_r2(t, s, 2)
    return float(coeff[0]), float(coeff[1]), float(coeff[2]), r2, yhat


def pca_project(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.column_stack([x, y]).astype(float)
    center = np.nanmean(pts, axis=0)
    centered = pts - center
    _, _, vh = np.linalg.svd(centered[np.isfinite(centered).all(axis=1)], full_matrices=False)
    axis = vh[0]
    s = centered @ axis
    return s, axis


def project_motion(track, axis: str, scale: float | None = None, y_down_positive: bool = True) -> tuple[np.ndarray, str]:
    scale = 1.0 if scale is None else float(scale)
    axis_norm = (axis or "auto").lower()
    x = track["cx_px"].to_numpy(dtype=float) * scale
    y = track["cy_px"].to_numpy(dtype=float) * scale
    if not y_down_positive:
        y = -y
    if axis_norm == "x":
        return x, "x"
    if axis_norm in {"y", "vertical"}:
        return y, "y_down" if y_down_positive else "y_up"
    if axis_norm in {"2d", "auto", "spring_axis", "pca"}:
        s, pca_axis = pca_project(x, y)
        return s, f"pca[{pca_axis[0]:.4f},{pca_axis[1]:.4f}]"
    return x, "x"


def central_velocity(t: np.ndarray, s: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    s = np.asarray(s, dtype=float)
    if len(t) < 2:
        return np.full_like(s, np.nan)
    return np.gradient(s, t)


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if window <= 1 or len(x) < window:
        return x.copy()
    if window % 2 == 0:
        window -= 1
    pad = window // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(xp, kernel, mode="valid")[: len(x)]


def fit_slope_window(t: np.ndarray, s: np.ndarray, start: int, end: int) -> tuple[float, float]:
    start = max(0, int(start))
    end = min(len(t), int(end))
    if end - start < 3:
        raise ValueError("window too short for slope fit")
    slope, _, r2, _ = linear_fit(t[start:end], s[start:end])
    return slope, r2


def estimate_period(t: np.ndarray, s: np.ndarray) -> tuple[float, float, str]:
    t = np.asarray(t, dtype=float)
    s = np.asarray(s, dtype=float)
    if len(t) < 8:
        raise ValueError("insufficient frames for period detection")
    x = s - np.mean(s)
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        raise ValueError("non-positive time step")

    try:
        from scipy.signal import find_peaks

        min_distance = max(2, int(0.15 * len(x)))
        peaks, props = find_peaks(x, distance=min_distance, prominence=np.std(x) * 0.15)
        if len(peaks) >= 2:
            periods = np.diff(t[peaks])
            return float(np.median(periods)), float(len(peaks) / max(len(t), 1)), "peaks"
    except Exception:
        pass

    corr = np.correlate(x, x, mode="full")[len(x) - 1 :]
    corr[0] = 0
    min_lag = max(2, int(0.25 / dt))
    if len(corr) <= min_lag + 2:
        raise ValueError("period not found")
    local_peaks = [
        lag
        for lag in range(min_lag, len(corr) - 1)
        if corr[lag] > 0 and corr[lag] >= corr[lag - 1] and corr[lag] >= corr[lag + 1]
    ]
    if local_peaks:
        lag = int(local_peaks[0])
    else:
        lag = int(np.argmax(corr[min_lag:]) + min_lag)
    confidence = float(corr[lag] / max(np.max(np.abs(corr)), 1e-12))
    if confidence < 0.15:
        raise ValueError("period not found")
    return float(lag * dt), confidence, "autocorrelation"


def terminal_velocity(t: np.ndarray, s: np.ndarray) -> tuple[float, float, tuple[int, int]]:
    v = moving_average(central_velocity(t, s), 5)
    n = len(v)
    if n < 10:
        raise ValueError("insufficient frames for terminal velocity")
    window = max(5, n // 5)
    best = None
    for start in range(max(1, n - 2 * window), n - window + 1):
        seg = v[start : start + window]
        mean = float(np.nanmean(seg))
        std = float(np.nanstd(seg))
        if abs(mean) <= 1e-9:
            continue
        rel_std = std / abs(mean)
        score = rel_std + 0.02 * (n - start)
        if best is None or score < best[0]:
            best = (score, mean, rel_std, start, start + window)
    if best is None or best[2] > 0.35:
        raise ValueError("terminal velocity segment not found")
    return abs(float(best[1])), float(1.0 - min(best[2], 1.0)), (int(best[3]), int(best[4]))
