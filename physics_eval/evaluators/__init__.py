from physics_eval.evaluators.velocity import eval_velocity
from physics_eval.evaluators.acceleration import eval_acceleration
from physics_eval.evaluators.gravity import eval_gravity
from physics_eval.evaluators.friction import eval_friction
from physics_eval.evaluators.restitution import eval_restitution
from physics_eval.evaluators.density_acceleration import eval_density_acceleration
from physics_eval.evaluators.viscosity import eval_viscosity
from physics_eval.evaluators.spring import eval_spring
from physics_eval.evaluators.energy import eval_energy
from physics_eval.evaluators.momentum_1d import eval_momentum_1d


EVALUATOR_REGISTRY = {
    "eval_velocity": eval_velocity,
    "eval_acceleration": eval_acceleration,
    "eval_gravity": eval_gravity,
    "eval_friction": eval_friction,
    "eval_restitution": eval_restitution,
    "eval_density_acceleration": eval_density_acceleration,
    "eval_viscosity": eval_viscosity,
    "eval_spring": eval_spring,
    "eval_energy": eval_energy,
    "eval_momentum_1d": eval_momentum_1d,
}

