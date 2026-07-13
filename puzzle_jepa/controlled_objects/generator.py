from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from puzzle_jepa.controlled_objects.domain import (
    RigidAction,
    RigidObjectScene,
    RigidObjectTrajectory,
)


TRANSFORM_NAMES = ("noop", "up", "down", "left", "right", "cw", "ccw")
TRANSLATIONS = {
    1: (-1, 0),
    2: (1, 0),
    3: (0, -1),
    4: (0, 1),
}
SHAPE_MASKS = (
    np.asarray([[1, 1], [1, 1]], dtype=bool),
    np.asarray([[1, 1, 1]], dtype=bool),
    np.asarray([[1], [1], [1]], dtype=bool),
    np.asarray([[1, 0], [1, 1]], dtype=bool),
    np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.asarray([[1, 1, 1], [0, 1, 0]], dtype=bool),
)


@dataclass(frozen=True, slots=True)
class ControlledObjectSpec:
    grid_size: int = 16
    num_colors: int = 10
    object_count: int = 4
    trajectory_length: int = 64
    invalid_action_ratio: float = 0.1
    noop_ratio: float = 0.05
    require_state_change: bool = False
    max_scene_retries: int = 512


class ControlledObjectGenerator:
    def __init__(self, spec: ControlledObjectSpec):
        if spec.grid_size < 8:
            raise ValueError("Controlled-object grids require grid_size >= 8.")
        if not (1 <= spec.object_count < spec.num_colors):
            raise ValueError("Objects require distinct non-background colors.")
        if spec.trajectory_length < 1:
            raise ValueError("trajectory_length must be positive.")
        if not (0.0 <= spec.invalid_action_ratio < 1.0):
            raise ValueError("invalid_action_ratio must lie in [0,1).")
        if not (0.0 <= spec.noop_ratio < 1.0):
            raise ValueError("noop_ratio must lie in [0,1).")
        self.spec = spec

    def sample_scene(self, rng: np.random.Generator) -> RigidObjectScene:
        colors = rng.choice(
            np.arange(1, self.spec.num_colors),
            size=self.spec.object_count,
            replace=False,
        )
        for _ in range(self.spec.max_scene_retries):
            grid = np.zeros((self.spec.grid_size, self.spec.grid_size), dtype=np.int64)
            object_map = np.full_like(grid, -1)
            shape_ids = []
            placed = True
            for object_id, color in enumerate(colors):
                shape_id = int(rng.integers(0, len(SHAPE_MASKS)))
                mask = SHAPE_MASKS[shape_id]
                for _ in range(128):
                    row = int(rng.integers(0, self.spec.grid_size - mask.shape[0] + 1))
                    col = int(rng.integers(0, self.spec.grid_size - mask.shape[1] + 1))
                    region = grid[row : row + mask.shape[0], col : col + mask.shape[1]]
                    if np.any(region[mask] != 0):
                        continue
                    region[mask] = int(color)
                    map_region = object_map[
                        row : row + mask.shape[0], col : col + mask.shape[1]
                    ]
                    map_region[mask] = object_id
                    shape_ids.append(shape_id)
                    break
                else:
                    placed = False
                    break
            if placed:
                return RigidObjectScene(
                    grid=grid,
                    object_maps=object_map,
                    shape_ids=np.asarray(shape_ids, dtype=np.int64),
                    colors=np.asarray(colors, dtype=np.int64),
                )
        raise RuntimeError("Could not place a controlled rigid-object scene.")

    def sample_trajectory(
        self,
        rng: np.random.Generator,
        *,
        horizon: int | None = None,
    ) -> RigidObjectTrajectory:
        scene = self.sample_scene(rng)
        state = scene.grid.copy()
        states = [state]
        actions = []
        validity = []
        for _ in range(self.spec.trajectory_length if horizon is None else horizon):
            action = self.sample_action(state, rng)
            state, valid = self.apply_action(state, action)
            actions.append(action.as_array())
            validity.append(valid)
            states.append(state)
        return RigidObjectTrajectory(
            states=np.stack(states).astype(np.int64, copy=False),
            actions=np.stack(actions).astype(np.int64, copy=False),
            action_validity=np.asarray(validity, dtype=bool),
            scene=scene,
        )

    def sample_action(self, state: np.ndarray, rng: np.random.Generator) -> RigidAction:
        if not self.spec.require_state_change and rng.random() < self.spec.noop_ratio:
            return RigidAction(0, 0, 0)
        candidates = self.candidate_actions(state, include_invalid=True)
        legal = []
        effective = []
        invalid = []
        for action in candidates:
            next_state, valid = self.apply_action(state, action)
            if valid:
                legal.append(action)
                if np.any(next_state != state):
                    effective.append(action)
            else:
                invalid.append(action)
        if self.spec.require_state_change:
            if not effective:
                raise RuntimeError("Sampled scene has no state-changing rigid action.")
            return effective[int(rng.integers(0, len(effective)))]
        pool = invalid if invalid and rng.random() < self.spec.invalid_action_ratio else legal
        if not pool:
            return RigidAction(0, 0, 0)
        return pool[int(rng.integers(0, len(pool)))]

    def candidate_actions(
        self,
        state: np.ndarray,
        *,
        include_invalid: bool = False,
        state_changing_only: bool = False,
    ) -> tuple[RigidAction, ...]:
        actions = [] if state_changing_only else [RigidAction(0, 0, 0)]
        for color in np.unique(state[state != 0]):
            cells = np.argwhere(state == color)
            row, col = (int(value) for value in cells[np.lexsort((cells[:, 1], cells[:, 0]))][0])
            for transform in range(1, len(TRANSFORM_NAMES)):
                action = RigidAction(row, col, transform)
                next_state, valid = self.apply_action(state, action)
                changed = bool(np.any(next_state != state))
                if (include_invalid or valid) and (not state_changing_only or changed):
                    actions.append(action)
        return tuple(actions)

    def apply_action(self, state: np.ndarray, action: RigidAction) -> tuple[np.ndarray, bool]:
        if state.shape != (self.spec.grid_size, self.spec.grid_size):
            raise ValueError("State has the wrong controlled-object grid size.")
        if action.transform == 0:
            return state.copy(), True
        if not (0 <= action.transform < len(TRANSFORM_NAMES)):
            return state.copy(), False
        if not (0 <= action.row < self.spec.grid_size and 0 <= action.col < self.spec.grid_size):
            return state.copy(), False
        color = int(state[action.row, action.col])
        if color == 0:
            return state.copy(), False
        cells = np.argwhere(state == color)
        if not _is_connected(cells):
            return state.copy(), False
        top, left = cells.min(axis=0)
        bottom, right = cells.max(axis=0)
        mask = state[top : bottom + 1, left : right + 1] == color
        if action.transform in TRANSLATIONS:
            delta_row, delta_col = TRANSLATIONS[action.transform]
            target_top = int(top + delta_row)
            target_left = int(left + delta_col)
            target_mask = mask
        else:
            target_mask = np.rot90(mask, -1 if action.transform == 5 else 1)
            center_row2 = int(top + bottom)
            center_col2 = int(left + right)
            target_top = (center_row2 - (target_mask.shape[0] - 1)) // 2
            target_left = (center_col2 - (target_mask.shape[1] - 1)) // 2
        target_bottom = target_top + target_mask.shape[0]
        target_right = target_left + target_mask.shape[1]
        if (
            target_top < 0
            or target_left < 0
            or target_bottom > self.spec.grid_size
            or target_right > self.spec.grid_size
        ):
            return state.copy(), False
        cleared = state.copy()
        cleared[cleared == color] = 0
        region = cleared[target_top:target_bottom, target_left:target_right]
        if np.any(region[target_mask] != 0):
            return state.copy(), False
        region[target_mask] = color
        return cleared, True

    def replay(self, state: np.ndarray, actions: Iterable[RigidAction]) -> np.ndarray:
        output = state.copy()
        for action in actions:
            output, _ = self.apply_action(output, action)
        return output


def _is_connected(cells: np.ndarray) -> bool:
    if len(cells) == 0:
        return False
    remaining = {tuple(int(value) for value in cell) for cell in cells}
    stack = [remaining.pop()]
    visited = set(stack)
    while stack:
        row, col = stack.pop()
        for neighbor in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if neighbor in remaining:
                remaining.remove(neighbor)
                visited.add(neighbor)
                stack.append(neighbor)
    return len(visited) == len(cells)
