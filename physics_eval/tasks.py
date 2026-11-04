"""Fixed execution defaults for the ten benchmark tasks.

These values used to be repeated in every metadata row. Explicit legacy fields
still take precedence, including empty values, to preserve existing behavior.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TaskDefinition:
    evaluator: str
    measurement_method: str


TASK_DEFINITIONS = {
    "velocity": TaskDefinition("eval_velocity", "linear_fit"),
    "acceleration": TaskDefinition("eval_acceleration", "quadratic_fit"),
    "gravity": TaskDefinition("eval_gravity", "vertical_quadratic_fit"),
    "friction_coefficient": TaskDefinition("eval_friction", "deceleration_fit"),
    "restitution_coefficient": TaskDefinition("eval_restitution", "horizontal_velocity_ratio"),
    "density_from_initial_acceleration": TaskDefinition("eval_density_acceleration", "initial_acceleration_fit"),
    "fluid_viscosity": TaskDefinition("eval_viscosity", "terminal_velocity"),
    "spring_constant": TaskDefinition("eval_spring", "period_detection"),
    "mechanical_energy_conservation": TaskDefinition("eval_energy", "height_speed_energy_ratio"),
    "momentum_conservation_1d": TaskDefinition("eval_momentum_1d", "pixel_velocity_momentum"),
}


def resolve_task_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with omitted execution fields restored for known tasks.

    Only absent keys receive defaults. Legacy overrides, blank values, unknown
    metrics and tracking-only inputs retain their original semantics.
    """
    resolved = dict(row)
    task = TASK_DEFINITIONS.get(str(row.get("Metric") or ""))
    if task is not None:
        resolved.setdefault("Evaluator", task.evaluator)
        resolved.setdefault("Measurement_Method", task.measurement_method)
    return resolved
