from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from puzzle_jepa.data.worlds import PuzzleWorld, WorldAction


class ActionScorer(Protocol):
    def score_actions_to_goal(
        self,
        state: torch.Tensor,
        actions: list[WorldAction],
        goal: torch.Tensor,
        task_id: int,
    ) -> torch.Tensor:
        ...


@dataclass(frozen=True, slots=True)
class PlanStep:
    state: np.ndarray
    action: WorldAction | None
    score: float


class SymbolicOracleScorer:
    """Small test/debug scorer that ranks actions by exact Hamming improvement to the oracle goal."""

    def __init__(self, world: PuzzleWorld):
        self.world = world

    def score_actions_to_goal(
        self,
        state: torch.Tensor,
        actions: list[WorldAction],
        goal: torch.Tensor,
        task_id: int,
    ) -> torch.Tensor:
        del task_id
        state_np = state.detach().cpu().numpy()
        goal_np = goal.detach().cpu().numpy()
        scores = []
        before = np.not_equal(state_np, goal_np).sum()
        for action in actions:
            try:
                after_state = self.world.apply(state_np, action)
                after = np.not_equal(after_state, goal_np).sum()
                scores.append(float(before - after))
            except ValueError:
                scores.append(float("-inf"))
        return torch.as_tensor(scores, dtype=torch.float32, device=state.device)


class LatentActionPlanner:
    def __init__(self, world: PuzzleWorld, scorer: ActionScorer, beam_size: int = 1, max_steps: int = 128):
        self.world = world
        self.scorer = scorer
        self.beam_size = int(beam_size)
        self.max_steps = int(max_steps)
        if self.beam_size <= 0:
            raise ValueError("beam_size must be positive.")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive.")

    def rank_actions(self, state: np.ndarray, goal: np.ndarray) -> list[tuple[WorldAction, float]]:
        state_arr = self.world.validate_state(state)
        goal_arr = self.world.validate_state(goal)
        actions = self.world.legal_actions(state_arr)
        if not actions:
            return []
        state_tensor = torch.as_tensor(state_arr, dtype=torch.long)
        goal_tensor = torch.as_tensor(goal_arr, dtype=torch.long)
        scores = self.scorer.score_actions_to_goal(state_tensor, actions, goal_tensor, self.world.task_id)
        order = scores.argsort(descending=True).tolist()
        return [(actions[idx], float(scores[idx].item())) for idx in order]

    def plan(self, state: np.ndarray, goal: np.ndarray) -> list[PlanStep]:
        start = self.world.validate_state(state)
        goal_arr = self.world.validate_state(goal)
        if self.world.is_goal(start, goal_arr):
            return [PlanStep(start.copy(), None, 0.0)]
        beam: list[tuple[np.ndarray, list[PlanStep], float]] = [(start.copy(), [PlanStep(start.copy(), None, 0.0)], 0.0)]
        for _ in range(self.max_steps):
            candidates: list[tuple[np.ndarray, list[PlanStep], float]] = []
            for current, trace, total_score in beam:
                for action, score in self.rank_actions(current, goal_arr)[: self.beam_size]:
                    try:
                        next_state = self.world.apply(current, action)
                    except ValueError:
                        continue
                    next_total = total_score + score
                    next_trace = [*trace, PlanStep(next_state.copy(), action, next_total)]
                    if self.world.is_goal(next_state, goal_arr):
                        return next_trace
                    candidates.append((next_state, next_trace, next_total))
            if not candidates:
                return beam[0][1]
            candidates.sort(key=lambda item: item[2], reverse=True)
            beam = candidates[: self.beam_size]
        return beam[0][1]
