from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from puzzle_jepa.data.worlds import PuzzleExample, PuzzleWorld, SudokuWorld, WorldAction


@dataclass(frozen=True, slots=True)
class Transition:
    state: np.ndarray
    action: WorldAction
    next_state: np.ndarray
    goal: np.ndarray
    task_id: int
    clue_mask: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class TransitionBatch:
    states: torch.Tensor
    actions: torch.Tensor
    next_states: torch.Tensor
    goals: torch.Tensor
    clue_masks: torch.Tensor | None = None


def sample_oracle_transition(
    world: PuzzleWorld,
    example: PuzzleExample,
    rng: np.random.Generator | None = None,
) -> Transition:
    rng = rng or np.random.default_rng()
    state = world.validate_state(example.state)
    goal = world.validate_state(example.goal)
    improving = [
        action
        for action in world.legal_actions(state)
        if goal[action.row, action.col] == action.value and state[action.row, action.col] != goal[action.row, action.col]
    ]
    if not improving:
        raise ValueError("No oracle-improving action is available for this example.")
    action = improving[int(rng.integers(0, len(improving)))]
    return Transition(
        state=state.copy(),
        action=action,
        next_state=world.apply(state, action),
        goal=goal.copy(),
        task_id=world.task_id,
        clue_mask=world.clue_mask_from_puzzle(state) if isinstance(world, SudokuWorld) else None,
    )


def collate_transitions(transitions: list[Transition], device: str | torch.device = "cpu") -> TransitionBatch:
    if not transitions:
        raise ValueError("Cannot collate an empty transition list.")
    states = torch.as_tensor(np.stack([item.state for item in transitions]), dtype=torch.long, device=device)
    next_states = torch.as_tensor(np.stack([item.next_state for item in transitions]), dtype=torch.long, device=device)
    goals = torch.as_tensor(np.stack([item.goal for item in transitions]), dtype=torch.long, device=device)
    clue_masks = None
    if all(item.clue_mask is not None for item in transitions):
        clue_masks = torch.as_tensor(
            np.stack([item.clue_mask for item in transitions if item.clue_mask is not None]),
            dtype=torch.bool,
            device=device,
        )
    actions = torch.as_tensor(
        np.stack([item.action.as_array(item.task_id) for item in transitions]),
        dtype=torch.long,
        device=device,
    )
    return TransitionBatch(states=states, actions=actions, next_states=next_states, goals=goals, clue_masks=clue_masks)
