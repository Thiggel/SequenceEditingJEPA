from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import torch

from puzzle_jepa.moving_objects.generator import (
    ANGULAR_VELOCITIES,
    MovingObjectGenerator,
    SHAPE_NAMES,
    VELOCITIES,
)


@dataclass(slots=True)
class MovingObjectBatch:
    contexts: torch.Tensor
    future_contexts: torch.Tensor
    object_count: torch.Tensor
    visible_object_count: torch.Tensor
    future_visible_object_count: torch.Tensor
    shape_counts: torch.Tensor
    color_counts: torch.Tensor
    future_shape_counts: torch.Tensor
    future_color_counts: torch.Tensor
    velocity_counts: torch.Tensor
    angular_velocity_counts: torch.Tensor
    relations: torch.Tensor
    completion_features: torch.Tensor
    future_velocity_counts: torch.Tensor
    future_angular_velocity_counts: torch.Tensor
    future_relations: torch.Tensor
    future_completion_features: torch.Tensor
    current_grid: torch.Tensor

    def to(self, device: torch.device | str) -> "MovingObjectBatch":
        return MovingObjectBatch(
            **{field.name: getattr(self, field.name).to(device) for field in fields(self)}
        )


def sample_moving_object_batch(
    generator: MovingObjectGenerator,
    rng: np.random.Generator,
    *,
    batch_size: int,
    horizon: int,
    device: torch.device | str = "cpu",
) -> MovingObjectBatch:
    contexts = []
    futures = []
    counts = []
    visible_counts = []
    future_visible_counts = []
    shape_counts = []
    color_counts = []
    future_shape_counts = []
    future_color_counts = []
    velocity_counts = []
    angular_velocity_counts = []
    relations = []
    completion_features = []
    future_velocity_counts = []
    future_angular_velocity_counts = []
    future_relations = []
    future_completion_features = []
    current_grids = []
    required = horizon + 2
    for _ in range(batch_size):
        trajectory = generator.sample_trajectory(rng)
        if len(trajectory.states) < required:
            raise ValueError("sequence_length must be at least horizon + 2.")
        start = int(rng.integers(0, len(trajectory.states) - required + 1))
        contexts.append(trajectory.states[start : start + 2])
        futures.append(
            np.stack([trajectory.states[start + step : start + step + 2] for step in range(1, horizon + 1)])
        )
        state_index = start + 1
        counts.append(trajectory.object_count)
        visible_counts.append(_visible_object_count(trajectory.object_maps[state_index]))
        visible_ids = _visible_object_ids(trajectory.object_maps[state_index])
        future_visible_ids = _visible_object_ids(trajectory.object_maps[state_index + 1])
        future_visible_counts.append(len(future_visible_ids))
        shape_counts.append(
            np.bincount(trajectory.shape_ids[visible_ids], minlength=len(SHAPE_NAMES))
        )
        color_counts.append(
            np.bincount(trajectory.colors[visible_ids], minlength=generator.spec.num_colors)[1:]
        )
        future_shape_counts.append(
            np.bincount(trajectory.shape_ids[future_visible_ids], minlength=len(SHAPE_NAMES))
        )
        future_color_counts.append(
            np.bincount(
                trajectory.colors[future_visible_ids], minlength=generator.spec.num_colors
            )[1:]
        )
        if trajectory.kind == "motion":
            velocity_ids = [_velocity_id(tuple(values)) for values in trajectory.velocities[state_index]]
            velocity_counts.append(np.bincount(velocity_ids, minlength=len(VELOCITIES)))
            angular_velocity_counts.append(
                np.bincount(
                    [_angular_velocity_id(value) for value in trajectory.angular_velocities[state_index]],
                    minlength=len(ANGULAR_VELOCITIES),
                )
            )
        else:
            velocity_counts.append(np.zeros(len(VELOCITIES), dtype=np.int64))
            angular_velocity_counts.append(np.zeros(len(ANGULAR_VELOCITIES), dtype=np.int64))
        relations.append(
            _relation_features(
                trajectory.positions,
                state_index,
                generator.spec.grid_size,
                boundary_mode=generator.spec.boundary_mode,
                object_ids=visible_ids,
            )
        )
        completion_features.append(
            _completion_features(trajectory.completion[state_index, visible_ids])
        )
        if trajectory.kind == "motion":
            future_velocity_ids = [
                _velocity_id(tuple(values)) for values in trajectory.velocities[state_index + 1]
            ]
            future_velocity_counts.append(np.bincount(future_velocity_ids, minlength=len(VELOCITIES)))
            future_angular_velocity_counts.append(
                np.bincount(
                    [_angular_velocity_id(value) for value in trajectory.angular_velocities[state_index + 1]],
                    minlength=len(ANGULAR_VELOCITIES),
                )
            )
        else:
            future_velocity_counts.append(np.zeros(len(VELOCITIES), dtype=np.int64))
            future_angular_velocity_counts.append(np.zeros(len(ANGULAR_VELOCITIES), dtype=np.int64))
        future_relations.append(
            _relation_features(
                trajectory.positions,
                state_index + 1,
                generator.spec.grid_size,
                boundary_mode=generator.spec.boundary_mode,
                object_ids=future_visible_ids,
            )
        )
        future_completion_features.append(
            _completion_features(trajectory.completion[state_index + 1, future_visible_ids])
        )
        current_grids.append(trajectory.states[state_index])
    return MovingObjectBatch(
        contexts=torch.as_tensor(np.stack(contexts), dtype=torch.long, device=device),
        future_contexts=torch.as_tensor(np.stack(futures), dtype=torch.long, device=device),
        object_count=torch.as_tensor(counts, dtype=torch.long, device=device),
        visible_object_count=torch.as_tensor(visible_counts, dtype=torch.long, device=device),
        future_visible_object_count=torch.as_tensor(
            future_visible_counts, dtype=torch.long, device=device
        ),
        shape_counts=torch.as_tensor(np.stack(shape_counts), dtype=torch.float32, device=device),
        color_counts=torch.as_tensor(np.stack(color_counts), dtype=torch.float32, device=device),
        future_shape_counts=torch.as_tensor(
            np.stack(future_shape_counts), dtype=torch.float32, device=device
        ),
        future_color_counts=torch.as_tensor(
            np.stack(future_color_counts), dtype=torch.float32, device=device
        ),
        velocity_counts=torch.as_tensor(np.stack(velocity_counts), dtype=torch.float32, device=device),
        angular_velocity_counts=torch.as_tensor(
            np.stack(angular_velocity_counts), dtype=torch.float32, device=device
        ),
        relations=torch.as_tensor(np.stack(relations), dtype=torch.float32, device=device),
        completion_features=torch.as_tensor(
            np.stack(completion_features), dtype=torch.float32, device=device
        ),
        future_velocity_counts=torch.as_tensor(
            np.stack(future_velocity_counts), dtype=torch.float32, device=device
        ),
        future_angular_velocity_counts=torch.as_tensor(
            np.stack(future_angular_velocity_counts), dtype=torch.float32, device=device
        ),
        future_relations=torch.as_tensor(np.stack(future_relations), dtype=torch.float32, device=device),
        future_completion_features=torch.as_tensor(
            np.stack(future_completion_features), dtype=torch.float32, device=device
        ),
        current_grid=torch.as_tensor(np.stack(current_grids), dtype=torch.long, device=device),
    )


def _velocity_id(velocity: tuple[int, int]) -> int:
    return VELOCITIES.index(velocity)


def _angular_velocity_id(value: int) -> int:
    return ANGULAR_VELOCITIES.index(int(value))


def _visible_object_count(object_map: np.ndarray) -> int:
    return int(len(_visible_object_ids(object_map)))


def _visible_object_ids(object_map: np.ndarray) -> np.ndarray:
    return np.unique(object_map[object_map >= 0]).astype(np.int64)


def _completion_features(completion: np.ndarray) -> np.ndarray:
    values = np.asarray(completion, dtype=np.float32)
    return np.asarray(
        [values.mean(), values.min(), values.max(), values.std(), np.mean(values >= 1.0)],
        dtype=np.float32,
    )


def _relation_features(
    positions: np.ndarray,
    index: int,
    grid_size: int,
    *,
    boundary_mode: str = "reflect",
    object_ids: np.ndarray | None = None,
) -> np.ndarray:
    selected = slice(None) if object_ids is None else object_ids
    current = positions[index, selected].astype(np.float32)
    if len(current) < 2:
        return np.zeros(5, dtype=np.float32)
    previous = positions[max(0, index - 1), selected].astype(np.float32)
    pairs = [(left, right) for left in range(len(current)) for right in range(left + 1, len(current))]
    distances = np.asarray(
        [_pair_distance(current[a], current[b], grid_size, boundary_mode) for a, b in pairs],
        dtype=np.float32,
    )
    previous_distances = np.asarray(
        [_pair_distance(previous[a], previous[b], grid_size, boundary_mode) for a, b in pairs],
        dtype=np.float32,
    )
    same_row = np.asarray([abs(current[a, 0] - current[b, 0]) <= 1 for a, b in pairs], dtype=np.float32)
    same_col = np.asarray([abs(current[a, 1] - current[b, 1]) <= 1 for a, b in pairs], dtype=np.float32)
    return np.asarray(
        [
            distances.mean() / (2 * grid_size),
            distances.min() / (2 * grid_size),
            (distances < previous_distances).mean(),
            same_row.mean(),
            same_col.mean(),
        ],
        dtype=np.float32,
    )


def _pair_distance(left: np.ndarray, right: np.ndarray, grid_size: int, boundary_mode: str) -> float:
    delta = np.abs(left - right)
    if boundary_mode == "wrap":
        delta = np.minimum(delta, grid_size - delta)
    return float(delta.sum())
