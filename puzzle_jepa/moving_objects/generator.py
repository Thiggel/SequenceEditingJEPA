from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from puzzle_jepa.moving_objects.domain import MovingObject, MovingObjectTrajectory


SHAPE_NAMES = ("square", "line_h", "line_v", "ell", "cross")
VELOCITIES = tuple(
    (row, col)
    for row in (-2, -1, 0, 1, 2)
    for col in (-2, -1, 0, 1, 2)
    if (row, col) != (0, 0) and max(abs(row), abs(col)) <= 2
)
ANGULAR_VELOCITIES = (-1, 0, 1)


@dataclass(frozen=True, slots=True)
class MovingObjectSpec:
    grid_size: int = 16
    num_colors: int = 10
    min_objects: int = 1
    max_objects: int = 4
    sequence_length: int = 12
    boundary_mode: str = "reflect"
    rotate_objects: bool = False
    max_scene_retries: int = 512


class MovingObjectGenerator:
    def __init__(self, spec: MovingObjectSpec):
        if spec.boundary_mode not in {"reflect", "wrap"}:
            raise ValueError("boundary_mode must be 'reflect' or 'wrap'.")
        if not (1 <= spec.min_objects <= spec.max_objects):
            raise ValueError("Object bounds must satisfy 1 <= min_objects <= max_objects.")
        if spec.grid_size < 8 or spec.sequence_length < 3:
            raise ValueError("Moving-object worlds require grid_size >= 8 and sequence_length >= 3.")
        if spec.num_colors < 3:
            raise ValueError("At least two foreground colors are required.")
        if spec.max_objects >= spec.num_colors:
            raise ValueError("The first motion gate requires one distinct foreground color per object.")
        self.spec = spec

    def sample_trajectory(self, rng: np.random.Generator) -> MovingObjectTrajectory:
        count = int(rng.integers(self.spec.min_objects, self.spec.max_objects + 1))
        for _ in range(self.spec.max_scene_retries):
            objects = self._sample_objects(rng, count)
            trajectory = self._rollout(objects)
            if trajectory is not None:
                return trajectory
        raise RuntimeError("Could not sample a collision-free moving-object trajectory.")

    def _sample_objects(self, rng: np.random.Generator, count: int) -> tuple[MovingObject, ...]:
        objects = []
        colors = rng.choice(np.arange(1, self.spec.num_colors), size=count, replace=False)
        for object_id in range(count):
            shape_id = int(rng.integers(0, len(SHAPE_NAMES)))
            mask = _shape_mask(shape_id)
            row = int(rng.integers(0, self.spec.grid_size - mask.shape[0] + 1))
            col = int(rng.integers(0, self.spec.grid_size - mask.shape[1] + 1))
            velocity_row, velocity_col = VELOCITIES[int(rng.integers(0, len(VELOCITIES)))]
            objects.append(
                MovingObject(
                    object_id=object_id,
                    shape_id=shape_id,
                    color=int(colors[object_id]),
                    mask=mask,
                    row=row,
                    col=col,
                    velocity_row=velocity_row,
                    velocity_col=velocity_col,
                    angular_velocity=(int(rng.choice((-1, 1))) if self.spec.rotate_objects else 0),
                )
            )
        return tuple(objects)

    def _rollout(self, initial: tuple[MovingObject, ...]) -> MovingObjectTrajectory | None:
        objects = initial
        frames = []
        maps = []
        positions = []
        velocities = []
        angular_velocities = []
        for _ in range(self.spec.sequence_length):
            rendered = _render(objects, self.spec.grid_size)
            if rendered is None:
                return None
            frame, object_map = rendered
            frames.append(frame)
            maps.append(object_map)
            positions.append([(obj.row, obj.col) for obj in objects])
            velocities.append([(obj.velocity_row, obj.velocity_col) for obj in objects])
            angular_velocities.append([obj.angular_velocity for obj in objects])
            objects = self._advance_collision_safe(objects)
        return MovingObjectTrajectory(
            states=np.stack(frames),
            object_maps=np.stack(maps),
            positions=np.asarray(positions, dtype=np.int64),
            velocities=np.asarray(velocities, dtype=np.int64),
            angular_velocities=np.asarray(angular_velocities, dtype=np.int64),
            shape_ids=np.asarray([obj.shape_id for obj in initial], dtype=np.int64),
            colors=np.asarray([obj.color for obj in initial], dtype=np.int64),
        )

    def _advance(self, obj: MovingObject) -> MovingObject:
        if self.spec.boundary_mode == "wrap":
            row = (obj.row + obj.velocity_row) % (self.spec.grid_size - obj.mask.shape[0] + 1)
            col = (obj.col + obj.velocity_col) % (self.spec.grid_size - obj.mask.shape[1] + 1)
            velocity_row, velocity_col = obj.velocity_row, obj.velocity_col
        else:
            row, velocity_row = _reflected_step(
                obj.row, obj.velocity_row, self.spec.grid_size - obj.mask.shape[0]
            )
            col, velocity_col = _reflected_step(
                obj.col, obj.velocity_col, self.spec.grid_size - obj.mask.shape[1]
            )
        advanced = MovingObject(
            object_id=obj.object_id,
            shape_id=obj.shape_id,
            color=obj.color,
            mask=obj.mask,
            row=row,
            col=col,
            velocity_row=velocity_row,
            velocity_col=velocity_col,
            angular_velocity=obj.angular_velocity,
        )
        return _rotate_advanced(advanced, self.spec.grid_size) if self.spec.rotate_objects else advanced

    def _advance_collision_safe(self, objects: tuple[MovingObject, ...]) -> tuple[MovingObject, ...]:
        proposed = tuple(self._advance(obj) for obj in objects)
        blocked = _overlap_ids(proposed)
        while blocked:
            resolved = tuple(
                _reverse_in_place(current) if current.object_id in blocked else candidate
                for current, candidate in zip(objects, proposed, strict=True)
            )
            extra = _overlap_ids(resolved) - blocked
            if not extra:
                return resolved
            blocked |= extra
        return proposed


def _shape_mask(shape_id: int) -> np.ndarray:
    masks = (
        np.ones((2, 2), dtype=bool),
        np.ones((1, 3), dtype=bool),
        np.ones((3, 1), dtype=bool),
        np.asarray([[1, 0], [1, 1]], dtype=bool),
        np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    )
    return masks[shape_id].copy()


def _reflected_step(position: int, velocity: int, maximum: int) -> tuple[int, int]:
    candidate = position + velocity
    if candidate < 0 or candidate > maximum:
        velocity = -velocity
        candidate = position + velocity
    return int(min(max(candidate, 0), maximum)), int(velocity)


def _render(objects: tuple[MovingObject, ...], grid_size: int) -> tuple[np.ndarray, np.ndarray] | None:
    frame = np.zeros((grid_size, grid_size), dtype=np.int64)
    object_map = np.full((grid_size, grid_size), -1, dtype=np.int64)
    for obj in objects:
        rows, cols = np.where(obj.mask)
        rows = rows + obj.row
        cols = cols + obj.col
        if np.any(object_map[rows, cols] >= 0):
            return None
        frame[rows, cols] = obj.color
        object_map[rows, cols] = obj.object_id
    return frame, object_map


def _overlap_ids(objects: tuple[MovingObject, ...]) -> set[int]:
    occupied: dict[tuple[int, int], int] = {}
    overlaps: set[int] = set()
    for obj in objects:
        for local_row, local_col in np.argwhere(obj.mask):
            cell = (int(local_row) + obj.row, int(local_col) + obj.col)
            if cell in occupied:
                overlaps.update((obj.object_id, occupied[cell]))
            else:
                occupied[cell] = obj.object_id
    return overlaps


def _reverse_in_place(obj: MovingObject) -> MovingObject:
    return MovingObject(
        object_id=obj.object_id,
        shape_id=obj.shape_id,
        color=obj.color,
        mask=obj.mask,
        row=obj.row,
        col=obj.col,
        velocity_row=-obj.velocity_row,
        velocity_col=-obj.velocity_col,
        angular_velocity=obj.angular_velocity,
    )


def _rotate_advanced(obj: MovingObject, grid_size: int) -> MovingObject:
    rotated = np.ascontiguousarray(np.rot90(obj.mask, k=obj.angular_velocity))
    center_row = obj.row + 0.5 * (obj.mask.shape[0] - 1)
    center_col = obj.col + 0.5 * (obj.mask.shape[1] - 1)
    row = int(round(center_row - 0.5 * (rotated.shape[0] - 1)))
    col = int(round(center_col - 0.5 * (rotated.shape[1] - 1)))
    row = min(max(row, 0), grid_size - rotated.shape[0])
    col = min(max(col, 0), grid_size - rotated.shape[1])
    return MovingObject(
        object_id=obj.object_id,
        shape_id=obj.shape_id,
        color=obj.color,
        mask=rotated,
        row=row,
        col=col,
        velocity_row=obj.velocity_row,
        velocity_col=obj.velocity_col,
        angular_velocity=obj.angular_velocity,
    )
