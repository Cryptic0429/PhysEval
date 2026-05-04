from __future__ import annotations

import math
from typing import Callable

import pandas as pd

from physics_eval.utils.io import failure_result, finish_result, parse_known_params
from physics_eval.utils.tracking import get_object_track


def safe_eval(row: dict, func: Callable[[], dict]) -> dict:
    try:
        return func()
    except ValueError as exc:
        return failure_result(row, str(exc) or "failed")
    except Exception as exc:
        return failure_result(row, "failed", {"exception": f"{type(exc).__name__}: {exc}"})


def numeric_target(row: dict) -> float | None:
    val = row.get("Target_Value")
    if isinstance(val, (int, float)) and math.isfinite(float(val)):
        return float(val)
    try:
        parsed = float(str(val))
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def get_primary_track(tracking_df: pd.DataFrame) -> pd.DataFrame:
    return get_object_track(tracking_df)


def g_from_params(row: dict, default: float = 9.8) -> float:
    params = parse_known_params(row)
    if "g" in params:
        return float(params["g"])
    if "gravity" in params:
        return float(params["gravity"])
    if "reference_g_mps2" in params:
        return float(params["reference_g_mps2"])
    env = str(params.get("gravity_environment", "")).lower()
    if env == "moon":
        return 1.62
    if env == "mars":
        return 3.71
    if env in {"zero", "zero_gravity"}:
        return 0.0
    return default


def done(
    row: dict,
    measured: float,
    unit: str,
    r2: float | None,
    scale: float | None,
    status: str = "valid",
    reason: str = "",
    extra: dict | None = None,
) -> dict:
    return finish_result(row, measured, unit, r2, scale, status, reason, extra)

