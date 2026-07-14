from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from puzzle_jepa.controlled_objects.domain import (
    RigidObjectScene,
    RigidObjectTrajectory,
    RigidTransform,
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
                motion_ids = rng.integers(
                    0, len(TRANSFORM_NAMES) - 1, size=self.spec.object_count
                )
                return RigidObjectScene(
                    grid=grid,
                    object_maps=object_map,
                    shape_ids=np.asarray(shape_ids, dtype=np.int64),
                    colors=np.asarray(colors, dtype=np.int64),
                    motion_ids=np.asarray(motion_ids, dtype=np.int64),
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
        requested_horizon = self.spec.trajectory_length if horizon is None else horizon
        macro_step = 0
        while len(actions) < requested_horizon:
            transform = self._deterministic_transform(state, scene, macro_step)
            target, valid = self._apply_rigid_transform(state, transform)
            if not valid or np.array_equal(target, state):
                continue
            state = target
            actions.append(transform.as_array())
            validity.append(True)
            states.append(state)
            macro_step += 1
        return RigidObjectTrajectory(
            states=np.stack(states).astype(np.int64, copy=False),
            actions=np.stack(actions).astype(np.int64, copy=False),
            action_validity=np.asarray(validity, dtype=bool),
            scene=scene,
        )

    def sample_action(self, state: np.ndarray, rng: np.random.Generator) -> RigidTransform:
        actions = self.candidate_actions(state, state_changing_only=True)
        return actions[int(rng.integers(0, len(actions)))]

    def candidate_actions(
        self,
        state: np.ndarray,
        *,
        include_invalid: bool = False,
        state_changing_only: bool = False,
    ) -> tuple[RigidTransform, ...]:
        del include_invalid, state_changing_only
        return self._candidate_rigid_transforms(state)

    def apply_action(
        self, state: np.ndarray, action: RigidTransform
    ) -> tuple[np.ndarray, bool]:
        if state.shape != (self.spec.grid_size, self.spec.grid_size):
            raise ValueError("State has the wrong controlled-object grid size.")
        if not (
            0 <= action.row < self.spec.grid_size
            and 0 <= action.col < self.spec.grid_size
        ):
            return state.copy(), False
        if not (1 <= action.transform < len(TRANSFORM_NAMES)):
            return state.copy(), False
        return self._apply_rigid_transform(state, action)

    def replay(self, state: np.ndarray, actions: Iterable[RigidTransform]) -> np.ndarray:
        output = state.copy()
        for action in actions:
            output, _ = self.apply_action(output, action)
        return output

    def _sample_transform(
        self, state: np.ndarray, rng: np.random.Generator
    ) -> RigidTransform:
        candidates = self._candidate_rigid_transforms(state)
        if not candidates:
            raise RuntimeError("Sampled scene has no state-changing rigid transform.")
        return candidates[int(rng.integers(0, len(candidates)))]

    def _deterministic_transform(
        self,
        state: np.ndarray,
        scene: RigidObjectScene,
        macro_step: int,
    ) -> RigidTransform:
        for object_offset in range(scene.object_count):
            object_id = (macro_step + object_offset) % scene.object_count
            color = int(scene.colors[object_id])
            cells = np.argwhere(state == color)
            if not len(cells):
                continue
            row, col = (int(value) for value in cells[np.lexsort((cells[:, 1], cells[:, 0]))][0])
            policy = int(scene.motion_ids[object_id])
            phase = macro_step // scene.object_count
            for transform_offset in range(6):
                transform = 1 + (policy + phase + transform_offset) % 6
                action = RigidTransform(row, col, transform)
                target, valid = self._apply_rigid_transform(state, action)
                if valid and not np.array_equal(target, state):
                    return action
        raise RuntimeError("Controlled scene has no valid deterministic rigid transform.")

    def _candidate_rigid_transforms(
        self, state: np.ndarray
    ) -> tuple[RigidTransform, ...]:
        actions = []
        seen_successors = set()
        for color in np.unique(state[state != 0]):
            cells = np.argwhere(state == color)
            row, col = (
                int(value)
                for value in cells[np.lexsort((cells[:, 1], cells[:, 0]))][0]
            )
            for transform in range(1, len(TRANSFORM_NAMES)):
                action = RigidTransform(row, col, transform)
                next_state, valid = self._apply_rigid_transform(state, action)
                if not valid or np.array_equal(next_state, state):
                    continue
                key = next_state.tobytes()
                if key in seen_successors:
                    continue
                seen_successors.add(key)
                actions.append(action)
        return tuple(actions)

    def _apply_rigid_transform(
        self, state: np.ndarray, action: RigidTransform
    ) -> tuple[np.ndarray, bool]:
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
