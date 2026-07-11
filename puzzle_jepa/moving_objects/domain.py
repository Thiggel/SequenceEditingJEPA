from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class MovingObject:
    object_id: int
    shape_id: int
    color: int
    mask: np.ndarray
    row: int
    col: int
    velocity_row: int
    velocity_col: int
    angular_velocity: int


@dataclass(frozen=True, slots=True)
class MovingObjectTrajectory:
    states: np.ndarray
    object_maps: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    angular_velocities: np.ndarray
    shape_ids: np.ndarray
    colors: np.ndarray

    def __post_init__(self) -> None:
        if self.states.ndim != 3 or self.object_maps.shape != self.states.shape:
            raise ValueError("Moving-object states and ownership maps must be [T,H,W].")
        if self.positions.shape != self.velocities.shape or self.positions.ndim != 3:
            raise ValueError("Positions and velocities must be [T,N,2].")
        if self.positions.shape[:2] != (self.states.shape[0], self.shape_ids.shape[0]):
            raise ValueError("Trajectory metadata does not match frame/object counts.")
        if self.colors.shape != self.shape_ids.shape:
            raise ValueError("Each object must have one shape and color label.")
        if self.angular_velocities.shape != self.positions.shape[:2]:
            raise ValueError("Angular velocities must be [T,N].")

    @property
    def object_count(self) -> int:
        return int(self.shape_ids.shape[0])
