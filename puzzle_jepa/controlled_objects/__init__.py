"""Pixel-edit-controlled rigid-object world for hierarchy experiments."""

from puzzle_jepa.controlled_objects.domain import (
    PixelEdit,
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
    "PixelEdit",
    "RigidObjectScene",
    "RigidObjectTrajectory",
]
