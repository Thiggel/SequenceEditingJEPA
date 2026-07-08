from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from puzzle_jepa.object_dynamics.domain import ObjectTrajectory
from puzzle_jepa.object_dynamics.generator import ObjectDynamicsGenerator


@dataclass(frozen=True, slots=True)
class ObjectDynamicsBatch:
    states: torch.Tensor
    actions: torch.Tensor
    futures: torch.Tensor
    object_count: torch.Tensor
    next_object_id: torch.Tensor
    valid_state: torch.Tensor
    completion: torch.Tensor


def sample_object_dynamics_batch(
    generator: ObjectDynamicsGenerator,
    rng: np.random.Generator,
    *,
    batch_size: int,
    horizon: int,
    device: str | torch.device = "cpu",
) -> ObjectDynamicsBatch:
    states = []
    actions = []
    futures = []
    object_count = []
    next_object_id = []
    valid_state = []
    completion = []
    max_objects = generator.spec.max_objects

    for _ in range(batch_size):
        trajectory = generator.sample_trajectory(rng, min_actions=horizon)
        start = int(rng.integers(0, len(trajectory.actions) - horizon + 1))
        states.append(trajectory.states[start])
        actions.append(np.stack([action.as_array() for action in trajectory.actions[start : start + horizon]]))
        futures.append(trajectory.states[start + 1 : start + horizon + 1])
        object_count.append(min(trajectory.scene.object_count, max_objects))
        next_id = trajectory.action_object_ids[start]
        next_object_id.append(max(0, min(max_objects, int(next_id) + 1)))
        valid_state.append(1.0 if trajectory.semantic else 0.0)
        completion.append(_completion_vector(trajectory, start, max_objects=max_objects))

    return ObjectDynamicsBatch(
        states=torch.as_tensor(np.stack(states), dtype=torch.long, device=device),
        actions=torch.as_tensor(np.stack(actions), dtype=torch.long, device=device),
        futures=torch.as_tensor(np.stack(futures), dtype=torch.long, device=device),
        object_count=torch.as_tensor(object_count, dtype=torch.long, device=device),
        next_object_id=torch.as_tensor(next_object_id, dtype=torch.long, device=device),
        valid_state=torch.as_tensor(valid_state, dtype=torch.float32, device=device),
        completion=torch.as_tensor(np.stack(completion), dtype=torch.float32, device=device),
    )


def _completion_vector(trajectory: ObjectTrajectory, state_index: int, *, max_objects: int) -> np.ndarray:
    state = trajectory.states[state_index]
    values = np.zeros((max_objects,), dtype=np.float32)
    for obj in trajectory.scene.objects[:max_objects]:
        target_cells = obj.mask
        correct = np.count_nonzero((state == obj.color) & target_cells)
        values[obj.object_id] = correct / max(1, obj.area)
    return values
