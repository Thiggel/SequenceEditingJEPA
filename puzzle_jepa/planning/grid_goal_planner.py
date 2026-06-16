from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from puzzle_jepa.data.grid_goal_sudoku import apply_fill_action, legal_fill_actions
from puzzle_jepa.data.worlds import WorldAction
from puzzle_jepa.models.grid_goal_jepa import GridTokenGoalJEPA


ScoreMode = Literal["oracle_goal_distance", "predicted_goal_distance"]
TransitionMode = Literal["symbolic_reencode", "latent_rollout"]


@dataclass(frozen=True, slots=True)
class BeamMPCResult:
    solved: bool
    steps: int
    remaining_hamming: int
    actions: list[WorldAction]
    final_board: np.ndarray
    score_mode: str
    transition_mode: str
    beam_width: int
    beam_depth: int
    action_evals: int
    elapsed_seconds: float


def hamming_distance(board: np.ndarray, goal: np.ndarray) -> int:
    return int(np.not_equal(board, goal).sum())


@torch.no_grad()
def run_beam_mpc(
    model: GridTokenGoalJEPA,
    puzzle: np.ndarray,
    goal: np.ndarray,
    *,
    score_mode: ScoreMode,
    transition_mode: TransitionMode,
    beam_width: int,
    beam_depth: int,
    max_steps: int = 81,
    device: torch.device,
) -> BeamMPCResult:
    start = time.time()
    model.eval()
    current = np.asarray(puzzle, dtype=np.int64).copy()
    actions_taken: list[WorldAction] = []
    clue_mask = current != 0
    editable_mask = ~clue_mask
    active_mask = np.ones((9, 9), dtype=bool)
    context_latents, predicted_goal, oracle_goal = _prepare_goal_latents(
        model, current, goal, clue_mask, editable_mask, active_mask, device=device
    )
    action_evals = 0
    for _ in range(max_steps):
        if np.array_equal(current, goal) or not np.any(current == 0):
            break
        first, evals = beam_plan_once(
            model,
            current,
            goal,
            context_latents,
            predicted_goal,
            oracle_goal,
            clue_mask,
            editable_mask,
            active_mask,
            score_mode=score_mode,
            transition_mode=transition_mode,
            beam_width=beam_width,
            beam_depth=beam_depth,
            device=device,
        )
        action_evals += evals
        if first is None:
            break
        try:
            current = apply_fill_action(current, first, allow_conflicts=True)
        except ValueError:
            break
        actions_taken.append(first)
    return BeamMPCResult(
        solved=bool(np.array_equal(current, goal)),
        steps=len(actions_taken),
        remaining_hamming=hamming_distance(current, goal),
        actions=actions_taken,
        final_board=current,
        score_mode=score_mode,
        transition_mode=transition_mode,
        beam_width=beam_width,
        beam_depth=beam_depth,
        action_evals=action_evals,
        elapsed_seconds=time.time() - start,
    )


@torch.no_grad()
def beam_plan_once(
    model: GridTokenGoalJEPA,
    board: np.ndarray,
    goal: np.ndarray,
    context_latents: torch.Tensor,
    predicted_goal: torch.Tensor,
    oracle_goal: torch.Tensor,
    clue_mask: np.ndarray,
    editable_mask: np.ndarray,
    active_mask: np.ndarray,
    *,
    score_mode: ScoreMode,
    transition_mode: TransitionMode,
    beam_width: int,
    beam_depth: int,
    device: torch.device,
) -> tuple[WorldAction | None, int]:
    beam: list[tuple[float, np.ndarray, list[WorldAction], torch.Tensor | None]] = [(0.0, board.copy(), [], None)]
    best: tuple[float, list[WorldAction]] | None = None
    action_evals = 0
    for _ in range(max(1, int(beam_depth))):
        candidates: list[tuple[float, np.ndarray, list[WorldAction], torch.Tensor | None]] = []
        for _, node_board, seq, latent in beam:
            for action in legal_fill_actions(node_board, allow_conflicts=True):
                try:
                    leaf = apply_fill_action(node_board, action, allow_conflicts=True)
                except ValueError:
                    continue
                next_seq = [*seq, action]
                if transition_mode == "symbolic_reencode":
                    score, next_latent = score_board(
                        model,
                        leaf,
                        context_latents,
                        predicted_goal,
                        oracle_goal,
                        clue_mask,
                        editable_mask,
                        active_mask,
                        score_mode=score_mode,
                        device=device,
                    )
                else:
                    if latent is None:
                        _, latent = score_board(
                            model,
                            node_board,
                            context_latents,
                            predicted_goal,
                            oracle_goal,
                            clue_mask,
                            editable_mask,
                            active_mask,
                            score_mode=score_mode,
                            device=device,
                        )
                    action_t = torch.as_tensor([[action.row, action.col, action.value]], dtype=torch.long, device=device)
                    next_latent = model.predict_next(latent, action_t, context_latents)
                    goal_latent = oracle_goal if score_mode == "oracle_goal_distance" else predicted_goal
                    mask_t = torch.as_tensor(active_mask[None], dtype=torch.bool, device=device)
                    score = float(model.distance(next_latent, goal_latent, mask_t).item())
                action_evals += 1
                candidates.append((score, leaf, next_seq, next_latent))
                if best is None or score < best[0]:
                    best = (score, next_seq)
        if not candidates:
            break
        candidates.sort(key=lambda item: item[0])
        beam = candidates[: max(1, int(beam_width))]
        if beam[0][0] <= 1.0e-8:
            break
    if best is None or not best[1]:
        return None, action_evals
    return best[1][0], action_evals


@torch.no_grad()
def score_board(
    model: GridTokenGoalJEPA,
    board: np.ndarray,
    context_latents: torch.Tensor,
    predicted_goal: torch.Tensor,
    oracle_goal: torch.Tensor,
    clue_mask: np.ndarray,
    editable_mask: np.ndarray,
    active_mask: np.ndarray,
    *,
    score_mode: ScoreMode,
    device: torch.device,
) -> tuple[float, torch.Tensor]:
    board_t = torch.as_tensor(board[None], dtype=torch.long, device=device)
    clue_t = torch.as_tensor(clue_mask[None], dtype=torch.bool, device=device)
    edit_t = torch.as_tensor(editable_mask[None], dtype=torch.bool, device=device)
    active_t = torch.as_tensor(active_mask[None], dtype=torch.bool, device=device)
    latent = model.encode_state(board_t, context_latents, clue_t, edit_t, active_t)
    goal_latent = oracle_goal if score_mode == "oracle_goal_distance" else predicted_goal
    return float(model.distance(latent, goal_latent, active_t).item()), latent


@torch.no_grad()
def _prepare_goal_latents(
    model: GridTokenGoalJEPA,
    puzzle: np.ndarray,
    goal: np.ndarray,
    clue_mask: np.ndarray,
    editable_mask: np.ndarray,
    active_mask: np.ndarray,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    context_t = torch.as_tensor(puzzle[None], dtype=torch.long, device=device)
    clue_t = torch.as_tensor(clue_mask[None], dtype=torch.bool, device=device)
    edit_t = torch.as_tensor(editable_mask[None], dtype=torch.bool, device=device)
    active_t = torch.as_tensor(active_mask[None], dtype=torch.bool, device=device)
    goal_t = torch.as_tensor(goal[None], dtype=torch.long, device=device)
    context_latents = model.encode_context(context_t, clue_t, edit_t, active_t)
    predicted_goal = model.predict_goal(context_latents, active_t)
    oracle_goal = model.encode_state(goal_t, context_latents, clue_t, edit_t, active_t)
    return context_latents, predicted_goal, oracle_goal
