from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from puzzle_jepa.controlled_objects.domain import RigidObjectTrajectory
from puzzle_jepa.controlled_objects.generator import ControlledObjectGenerator


@dataclass(slots=True)
class ControlledObjectBatch:
    states: torch.Tensor
    actions: torch.Tensor
    action_validity: torch.Tensor

    def to(self, device: torch.device) -> ControlledObjectBatch:
        return ControlledObjectBatch(
            states=self.states.to(device),
            actions=self.actions.to(device),
            action_validity=self.action_validity.to(device),
        )


class ControlledTrajectoryDataset:
    def __init__(self, trajectories: tuple[RigidObjectTrajectory, ...]):
        if not trajectories:
            raise ValueError("A controlled trajectory dataset cannot be empty.")
        horizon = trajectories[0].horizon
        if any(trajectory.horizon != horizon for trajectory in trajectories):
            raise ValueError("All controlled trajectories must have the same horizon.")
        self.states = np.stack([trajectory.states for trajectory in trajectories])
        self.actions = np.stack([trajectory.actions for trajectory in trajectories])
        self.action_validity = np.stack(
            [trajectory.action_validity for trajectory in trajectories]
        )

    @property
    def trajectory_count(self) -> int:
        return int(self.states.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.actions.shape[1])

    def sample_batch(
        self,
        rng: np.random.Generator,
        *,
        batch_size: int,
        horizon: int,
        device: torch.device | None = None,
    ) -> ControlledObjectBatch:
        if not (1 <= horizon <= self.horizon):
            raise ValueError(f"Requested horizon {horizon} exceeds dataset horizon {self.horizon}.")
        trajectory_ids = rng.integers(0, self.trajectory_count, size=batch_size)
        max_start = self.horizon - horizon
        starts = rng.integers(0, max_start + 1, size=batch_size)
        state_windows = np.stack(
            [
                self.states[trajectory_id, start : start + horizon + 1]
                for trajectory_id, start in zip(trajectory_ids, starts, strict=True)
            ]
        )
        action_windows = np.stack(
            [
                self.actions[trajectory_id, start : start + horizon]
                for trajectory_id, start in zip(trajectory_ids, starts, strict=True)
            ]
        )
        validity_windows = np.stack(
            [
                self.action_validity[trajectory_id, start : start + horizon]
                for trajectory_id, start in zip(trajectory_ids, starts, strict=True)
            ]
        )
        return ControlledObjectBatch(
            states=torch.as_tensor(state_windows, dtype=torch.long, device=device),
            actions=torch.as_tensor(action_windows, dtype=torch.long, device=device),
            action_validity=torch.as_tensor(validity_windows, dtype=torch.bool, device=device),
        )


def build_controlled_dataset(
    generator: ControlledObjectGenerator,
    *,
    trajectory_count: int,
    seed: int,
) -> ControlledTrajectoryDataset:
    rng = np.random.default_rng(seed)
    return ControlledTrajectoryDataset(
        tuple(generator.sample_trajectory(rng) for _ in range(trajectory_count))
    )
