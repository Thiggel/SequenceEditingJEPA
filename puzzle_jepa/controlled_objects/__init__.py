"""Action-controlled rigid-object world for hierarchy experiments."""

from puzzle_jepa.controlled_objects.domain import (
    RigidAction,
    RigidObjectScene,
    RigidObjectTrajectory,
)
from puzzle_jepa.controlled_objects.generator import (
    ControlledObjectGenerator,
    ControlledObjectSpec,
)

__all__ = [
    "ControlledObjectGenerator",
    "ControlledObjectSpec",
    "RigidAction",
    "RigidObjectScene",
    "RigidObjectTrajectory",
]
