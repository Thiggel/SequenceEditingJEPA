"""Valid rigid-object world for hierarchical latent-model experiments."""

from puzzle_jepa.controlled_objects.domain import (
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
    "RigidObjectScene",
    "RigidObjectTrajectory",
]
