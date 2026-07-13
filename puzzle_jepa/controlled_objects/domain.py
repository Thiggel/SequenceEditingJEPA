from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PixelEdit:
    row: int
    col: int
    color: int

    def as_array(self) -> np.ndarray:
        return np.asarray((self.row, self.col, self.color), dtype=np.int64)


@dataclass(frozen=True, slots=True)
class RigidTransform:
    row: int
    col: int
    transform: int


@dataclass(frozen=True, slots=True)
class RigidObjectScene:
    grid: np.ndarray
    object_maps: np.ndarray
    shape_ids: np.ndarray
    colors: np.ndarray
    motion_ids: np.ndarray

    def __post_init__(self) -> None:
        if self.grid.ndim != 2 or self.object_maps.shape != self.grid.shape:
            raise ValueError("Scene grid and object map must be matching 2D arrays.")
        if (
            self.shape_ids.ndim != 1
            or self.colors.shape != self.shape_ids.shape
            or self.motion_ids.shape != self.shape_ids.shape
        ):
            raise ValueError("Each scene object must have one shape, color, and motion policy.")

    @property
    def object_count(self) -> int:
        return int(self.shape_ids.shape[0])


@dataclass(frozen=True, slots=True)
class RigidObjectTrajectory:
    states: np.ndarray
    actions: np.ndarray
    action_validity: np.ndarray
    scene: RigidObjectScene

    def __post_init__(self) -> None:
        if self.states.ndim != 3:
            raise ValueError("Trajectory states must be [T+1,H,W].")
        if self.actions.shape != (self.states.shape[0] - 1, 3):
            raise ValueError("Trajectory actions must be [T,3].")
        if self.action_validity.shape != self.actions.shape[:1]:
            raise ValueError("Trajectory validity must be [T].")

    @property
    def horizon(self) -> int:
        return int(self.actions.shape[0])
