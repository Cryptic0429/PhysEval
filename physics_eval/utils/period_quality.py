"""Audit quality at a fixed legacy period; scores are heuristics, not probabilities."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import numpy as np
from scipy.optimize import minimize
from scipy.signal import find_peaks, savgol_filter


@dataclass(frozen=True)
class PeriodRules:
    min_samples: int = 12
    weak_cycles: float = 1.5
    valid_cycles: float = 3.0
    weak_supported_cycles: float = 1.0
    valid_supported_cycles: float = 2.5
    weak_fit_r2: float = 0.50
    valid_fit_r2: float = 0.85
    weak_interval_cv: float = 0.35
    valid_interval_cv: float = 0.15
    weak_coverage: float = 0.55
    valid_coverage: float = 0.85
    weak_max_gap_cycles: float = 0.60
    valid_max_gap_cycles: float = 0.25
    weak_samples_per_cycle: float = 8.0
    valid_samples_per_cycle: float = 12.0


DEFAULT_RULES = PeriodRules()
PROTOCOL = "period_confidence_v3_candidate"


@dataclass
class PeriodEstimate:
    period_s: float | None
    confidence: float
    method: str
    status: str
    diagnostics: dict
    reasons: list[str]

    def __iter__(self):
        # Existing three-value unpacking remains available.
        yield self.period_s
        yield self.confidence
        yield self.method


def classify_quality(d: dict, rules: PeriodRules = DEFAULT_RULES):
    """One source-independent rule set. Missing/NaN evidence cannot pass."""
    fields = ("observed_cycles", "supported_cycles", "fit_r2", "interval_cv",
              "coverage", "max_gap_cycles", "samples_per_cycle", "interval_count")
    if any(k not in d or d[k] is None or not np.isfinite(d[k]) for k in fields):
        return "invalid", 0.0, ["missing_or_nonfinite_quality_diagnostics"]
    if (any(d[k] < 0 for k in fields if k != "fit_r2") or not 0 <= d["coverage"] <= 1
            or d["fit_r2"] > 1.00000001 or d["supported_cycles"] > d["observed_cycles"] + 1e-8):
        return "invalid", 0.0, ["out_of_range_quality_diagnostics"]
    def failures(level):
        tests = {
            "observed_cycles": d["observed_cycles"] >= getattr(rules, level + "_cycles"),
            "supported_cycles": d["supported_cycles"] >= getattr(rules, level + "_supported_cycles"),
            "fit_r2": d["fit_r2"] >= getattr(rules, level + "_fit_r2"),
            "interval_cv": d["interval_cv"] <= getattr(rules, level + "_interval_cv"),
            "coverage": d["coverage"] >= getattr(rules, level + "_coverage"),
            "max_gap_cycles": d["max_gap_cycles"] <= getattr(rules, level + "_max_gap_cycles"),
            "samples_per_cycle": d["samples_per_cycle"] >= getattr(rules, level + "_samples_per_cycle"),
            "interval_count": d["interval_count"] >= (2 if level == "valid" else 1),
        }
        return [level + "_" + k for k, ok in tests.items() if not ok]
    weak_fail = failures("weak")
    valid_fail = failures("valid")
    score = float(np.clip(min(d["fit_r2"], d["coverage"],
                             np.exp(-3 * d["interval_cv"]),
                             d["supported_cycles"] / rules.valid_cycles), 0, 1))
    if weak_fail:
        return "invalid", score, weak_fail
    return ("weak_valid", score, valid_fail) if valid_fail else ("valid", score, [])


def estimate_period_quality(t, s, *, source, rules=DEFAULT_RULES, nominal_dt=None,
                            fixed_period_s):
    """Assess evidence for a supplied period; never estimate/replace that period.

    ``source`` records which legacy detector supplied T. Both sources run the
    same fixed-frequency damped fit and the same quality classification.
    """
    if source not in {"peaks", "autocorrelation"}:
        raise ValueError("unknown_legacy_period_source")
    config = asdict(rules)
    if any(not np.isfinite(v) or v <= 0 for v in config.values()):
        raise ValueError("invalid_period_rules")
    base = {"protocol": PROTOCOL, "period_fit_mode": "fixed_legacy_period",
            "rules": config,
            "rules_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
            "confidence_semantics": "heuristic_quality_score_not_probability"}
    def invalid(reason):
        return PeriodEstimate(float(fixed_period_s) if np.isfinite(fixed_period_s) and fixed_period_s > 0 else None,
                              0.0, source, "invalid", base.copy(), [reason])
    try:
        t, s = np.asarray(t, float), np.asarray(s, float)
    except (TypeError, ValueError):
        return invalid("non_numeric_input")
    if not np.isfinite(fixed_period_s) or fixed_period_s <= 0:
        return invalid("invalid_fixed_period")
    period = float(fixed_period_s)
    if t.ndim != 1 or s.ndim != 1 or len(t) != len(s) or len(t) < rules.min_samples:
        return invalid("insufficient_or_mismatched_samples")
    if not np.isfinite(t).all() or not np.isfinite(s).all():
        return invalid("nonfinite_input")
    if np.any(np.diff(t) <= 0):
        return invalid("non_increasing_time")
    t = t - t[0]
    dt, duration = float(np.median(np.diff(t))), float(t[-1])
    if duration <= 0:
        return invalid("insufficient_observation_duration")
    if nominal_dt is not None:
        if not np.isfinite(nominal_dt) or nominal_dt <= 0:
            return invalid("invalid_nominal_time_step")
        dt = float(nominal_dt)
    base["cadence_source"] = "provided" if nominal_dt is not None else "median_observed_time_step"
    trend = np.column_stack([np.ones(len(t)), t / duration])
    residual = s - trend @ np.linalg.lstsq(trend, s, rcond=None)[0]
    scale = float(np.std(residual))
    if scale <= max(1e-10, np.std(s) * 1e-6):
        return invalid("no_oscillatory_variation")
    y = (s - np.mean(s)) / scale
    n_grid = int(round(duration / dt)) + 1
    if n_grid > 200000:
        return invalid("time_grid_too_large")
    if n_grid < 5:
        return invalid("insufficient_cadence_samples")
    grid = np.linspace(0, duration, n_grid)
    grid_dt = float(grid[1] - grid[0])
    uniform = np.interp(grid, t, residual / scale)
    window = max(3, int(round(.10 * period / grid_dt)) | 1)
    smooth = savgol_filter(uniform, min(window, len(uniform) // 2 * 2 - 1), 2)
    peaks, _ = find_peaks(smooth, distance=max(2, int(.55 * period / grid_dt)),
                          prominence=.35 * np.std(smooth))
    intervals = np.diff(grid[peaks])
    cv = float(np.std(intervals) / np.mean(intervals)) if len(intervals) else None
    mismatch = float(abs(np.median(intervals) / period - 1)) if len(intervals) else None

    def fit(decay):
        envelope = np.exp(-float(decay) * t / duration)
        design = np.column_stack([trend, envelope * np.sin(2*np.pi*t/period),
                                   envelope * np.cos(2*np.pi*t/period)])
        coef = np.linalg.lstsq(design, y, rcond=None)[0]
        err = y - design @ coef
        return float(np.mean(err**2)), coef

    fits = [minimize(lambda z: fit(z[0])[0], [initial], method="L-BFGS-B", bounds=[(0, 8)])
             for initial in (0., 2.)]
    best = min(fits, key=lambda result: result.fun)
    if (not best.success and best.fun > 1e-12) or not np.isfinite(best.fun):
        return invalid("fixed_period_fit_failed")
    mse, coef = fit(best.x[0])
    amplitude = float(np.hypot(coef[-2], coef[-1]))
    noise = float(np.sqrt(mse))
    support_threshold = max(.05 * amplitude, 3 * noise)
    supported_duration = (0.0 if amplitude <= support_threshold else duration if best.x[0] < 1e-8
                          else duration * np.log(amplitude / support_threshold) / best.x[0])
    diagnostics = {**base, "observed_cycles": duration / period,
        "optimizer_success": bool(best.success), "optimizer_message": str(best.message),
        "supported_cycles": min(duration, supported_duration) / period,
        "fit_r2": float(1 - mse), "normalized_residual_rmse": noise,
        "interval_cv": max(cv, mismatch) if cv is not None else None,
        "raw_interval_cv": cv, "interval_period_mismatch": mismatch,
        "interval_count": len(intervals), "peak_count": len(peaks),
        "coverage": min(1., len(t) / n_grid),
        "max_gap_cycles": float(max(np.diff(t)) / period),
        "samples_per_cycle": period / dt, "decay_per_second": float(best.x[0] / duration),
        "observed_duration_s": duration, "sample_count": len(t),
        "candidate_period_s": period, "peak_min_distance_s": .55 * period}
    status, confidence, reasons = classify_quality(diagnostics, rules)
    return PeriodEstimate(period, confidence, source, status, diagnostics, reasons)
