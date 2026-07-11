from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import torch

from puzzle_jepa.moving_objects.generator import MovingObjectGenerator, SHAPE_NAMES, VELOCITIES


@dataclass(slots=True)
class MovingObjectBatch:
    contexts: torch.Tensor
    future_contexts: torch.Tensor
    object_count: torch.Tensor
    shape_counts: torch.Tensor
    color_counts: torch.Tensor
    velocity_counts: torch.Tensor
    relations: torch.Tensor
    future_velocity_counts: torch.Tensor
    future_relations: torch.Tensor
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
    shape_counts = []
    color_counts = []
    velocity_counts = []
    relations = []
    future_velocity_counts = []
    future_relations = []
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
        shape_counts.append(np.bincount(trajectory.shape_ids, minlength=len(SHAPE_NAMES)))
        color_counts.append(np.bincount(trajectory.colors, minlength=generator.spec.num_colors)[1:])
        velocity_ids = [_velocity_id(tuple(values)) for values in trajectory.velocities[state_index]]
        velocity_counts.append(np.bincount(velocity_ids, minlength=len(VELOCITIES)))
        relations.append(_relation_features(trajectory.positions, state_index, generator.spec.grid_size))
        future_velocity_ids = [
            _velocity_id(tuple(values)) for values in trajectory.velocities[state_index + 1]
        ]
        future_velocity_counts.append(np.bincount(future_velocity_ids, minlength=len(VELOCITIES)))
        future_relations.append(
            _relation_features(trajectory.positions, state_index + 1, generator.spec.grid_size)
        )
        current_grids.append(trajectory.states[state_index])
    return MovingObjectBatch(
        contexts=torch.as_tensor(np.stack(contexts), dtype=torch.long, device=device),
        future_contexts=torch.as_tensor(np.stack(futures), dtype=torch.long, device=device),
        object_count=torch.as_tensor(counts, dtype=torch.long, device=device),
        shape_counts=torch.as_tensor(np.stack(shape_counts), dtype=torch.float32, device=device),
        color_counts=torch.as_tensor(np.stack(color_counts), dtype=torch.float32, device=device),
        velocity_counts=torch.as_tensor(np.stack(velocity_counts), dtype=torch.float32, device=device),
        relations=torch.as_tensor(np.stack(relations), dtype=torch.float32, device=device),
        future_velocity_counts=torch.as_tensor(
            np.stack(future_velocity_counts), dtype=torch.float32, device=device
        ),
        future_relations=torch.as_tensor(np.stack(future_relations), dtype=torch.float32, device=device),
        current_grid=torch.as_tensor(np.stack(current_grids), dtype=torch.long, device=device),
    )


def _velocity_id(velocity: tuple[int, int]) -> int:
    return VELOCITIES.index(velocity)


def _relation_features(positions: np.ndarray, index: int, grid_size: int) -> np.ndarray:
    current = positions[index].astype(np.float32)
    if len(current) < 2:
        return np.zeros(5, dtype=np.float32)
    previous = positions[max(0, index - 1)].astype(np.float32)
    pairs = [(left, right) for left in range(len(current)) for right in range(left + 1, len(current))]
    distances = np.asarray([np.abs(current[a] - current[b]).sum() for a, b in pairs], dtype=np.float32)
    previous_distances = np.asarray([np.abs(previous[a] - previous[b]).sum() for a, b in pairs], dtype=np.float32)
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
