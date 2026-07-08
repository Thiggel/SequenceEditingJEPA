from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class ActionOp(IntEnum):
    PAINT = 0
    ERASE = 1
    RECOLOR = 2


@dataclass(frozen=True, slots=True)
class LowLevelAction:
    op: ActionOp
    row: int
    col: int
    color: int

    def as_array(self) -> np.ndarray:
        return np.asarray([int(self.op), self.row, self.col, self.color], dtype=np.int64)


@dataclass(frozen=True, slots=True)
class ObjectSpec:
    object_id: int
    shape_type: str
    color: int
    mask: np.ndarray

    @property
    def area(self) -> int:
        return int(np.count_nonzero(self.mask))

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        rows, cols = np.where(self.mask)
        return int(rows.min()), int(cols.min()), int(rows.max()) + 1, int(cols.max()) + 1

    @property
    def centroid(self) -> tuple[float, float]:
        rows, cols = np.where(self.mask)
        return float(rows.mean()), float(cols.mean())


@dataclass(frozen=True, slots=True)
class SceneSpec:
    grid: np.ndarray
    objects: tuple[ObjectSpec, ...]

    @property
    def object_count(self) -> int:
        return len(self.objects)

    @property
    def object_map(self) -> np.ndarray:
        output = np.full(self.grid.shape, -1, dtype=np.int64)
        for obj in self.objects:
            output[obj.mask] = obj.object_id
        return output


@dataclass(frozen=True, slots=True)
class ObjectTrajectory:
    states: np.ndarray
    actions: tuple[LowLevelAction, ...]
    action_object_ids: tuple[int, ...]
    scene: SceneSpec
    kind: str
    semantic: bool

    def __post_init__(self) -> None:
        if self.states.ndim != 3:
            raise ValueError(f"states must be [T,H,W], got {self.states.shape}.")
        if len(self.actions) + 1 != self.states.shape[0]:
            raise ValueError("ObjectTrajectory requires one more state than action.")
        if len(self.action_object_ids) != len(self.actions):
            raise ValueError("action_object_ids must match actions.")


def apply_low_level_action(grid: np.ndarray, action: LowLevelAction) -> np.ndarray:
    output = np.asarray(grid, dtype=np.int64).copy()
    if not (0 <= action.row < output.shape[0] and 0 <= action.col < output.shape[1]):
        raise ValueError(f"Action cell {(action.row, action.col)} outside grid shape {output.shape}.")
    if action.op == ActionOp.PAINT:
        output[action.row, action.col] = action.color
    elif action.op == ActionOp.ERASE:
        output[action.row, action.col] = 0
    elif action.op == ActionOp.RECOLOR:
        output[action.row, action.col] = action.color
    else:
        raise ValueError(f"Unsupported action op {action.op}.")
    return output
