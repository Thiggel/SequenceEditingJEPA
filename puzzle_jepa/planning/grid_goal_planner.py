from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from puzzle_jepa.data.grid_goal_sudoku import apply_fill_action, legal_fill_actions
from puzzle_jepa.data.worlds import WorldAction
from puzzle_jepa.models.grid_goal_jepa import GridTokenGoalJEPA


ScoreMode = Literal[
    "oracle_goal_distance",
    "predicted_goal_distance",
    "oracle_goal_raw_euclidean_distance",
    "predicted_goal_raw_euclidean_distance",
    "oracle_goal_raw_squared_euclidean_distance",
    "predicted_goal_raw_squared_euclidean_distance",
    "oracle_goal_raw_cosine_distance",
    "predicted_goal_raw_cosine_distance",
    "oracle_goal_raw_hybrid_distance",
    "predicted_goal_raw_hybrid_distance",
    "oracle_goal_raw_euclidean_progress",
    "predicted_goal_raw_euclidean_progress",
    "oracle_goal_changed_cell_raw_euclidean_distance",
    "predicted_goal_changed_cell_raw_euclidean_distance",
    "oracle_goal_projected_euclidean_distance",
    "predicted_goal_projected_euclidean_distance",
]
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
    if transition_mode == "symbolic_reencode":
        return _beam_plan_once_symbolic(
            model,
            board,
            context_latents,
            predicted_goal,
            oracle_goal,
            clue_mask,
            editable_mask,
            active_mask,
            score_mode=score_mode,
            beam_width=beam_width,
            beam_depth=beam_depth,
            device=device,
        )
    return _beam_plan_once_latent(
        model,
        board,
        context_latents,
        predicted_goal,
        oracle_goal,
        active_mask,
        score_mode=score_mode,
        beam_width=beam_width,
        beam_depth=beam_depth,
        device=device,
    )


@torch.no_grad()
def _beam_plan_once_latent(
    model: GridTokenGoalJEPA,
    board: np.ndarray,
    context_latents: torch.Tensor,
    predicted_goal: torch.Tensor,
    oracle_goal: torch.Tensor,
    active_mask: np.ndarray,
    *,
    score_mode: ScoreMode,
    beam_width: int,
    beam_depth: int,
    device: torch.device,
    chunk_size: int = 2048,
) -> tuple[WorldAction | None, int]:
    clue_mask = board != 0
    editable_mask = ~clue_mask
    _, start_latent = score_board(
        model,
        board,
        context_latents,
        predicted_goal,
        oracle_goal,
        clue_mask,
        editable_mask,
        active_mask,
        score_mode=score_mode,
        device=device,
    )
    beam: list[tuple[float, np.ndarray, list[WorldAction], torch.Tensor | None]] = [(0.0, board.copy(), [], start_latent)]
    best: tuple[float, list[WorldAction]] | None = None
    action_evals = 0
    base_progress_mode = _progress_base_score_mode(score_mode)
    target_goal = _target_goal_latents(score_mode, predicted_goal, oracle_goal)
    for _ in range(max(1, int(beam_depth))):
        parent_latents: list[torch.Tensor] = []
        leaves: list[np.ndarray] = []
        seqs: list[list[WorldAction]] = []
        actions: list[WorldAction] = []
        for _, node_board, seq, latent in beam:
            if latent is None:
                raise RuntimeError("Latent rollout beam node is missing its latent state.")
            for action in legal_fill_actions(node_board, allow_conflicts=True):
                try:
                    leaf = apply_fill_action(node_board, action, allow_conflicts=True)
                except ValueError:
                    continue
                parent_latents.append(latent)
                leaves.append(leaf)
                seqs.append([*seq, action])
                actions.append(action)
        if not leaves:
            break
        candidates: list[tuple[float, np.ndarray, list[WorldAction], torch.Tensor | None]] = []
        mask_t = torch.as_tensor(active_mask[None], dtype=torch.bool, device=device)
        for start in range(0, len(leaves), chunk_size):
            end = min(len(leaves), start + chunk_size)
            parent_batch = torch.cat(parent_latents[start:end], dim=0)
            action_t = torch.as_tensor(
                [[action.row, action.col, action.value] for action in actions[start:end]],
                dtype=torch.long,
                device=device,
            )
            context_t = context_latents.expand(parent_batch.shape[0], -1, -1)
            next_latents = model.predict_next(parent_batch, action_t, context_t)
            chunk_mask = mask_t.expand(parent_batch.shape[0], -1, -1)
            if base_progress_mode is not None:
                next_score = raw_tokenwise_euclidean_distance(next_latents, _expand_tokens_like(target_goal, next_latents), chunk_mask)
                node_score = raw_tokenwise_euclidean_distance(parent_batch, _expand_tokens_like(target_goal, parent_batch), chunk_mask)
                score_values = next_score - node_score
            elif _is_changed_cell_score(score_mode):
                score_values = changed_cell_raw_euclidean_distances(next_latents, _expand_tokens_like(target_goal, next_latents), actions[start:end])
            else:
                score_values = latent_distance(model, next_latents, predicted_goal, oracle_goal, chunk_mask, score_mode)
            scores = [float(x) for x in score_values.detach().cpu().tolist()]
            for offset, score in enumerate(scores):
                idx = start + offset
                latent = next_latents[offset : offset + 1].detach()
                candidates.append((score, leaves[idx], seqs[idx], latent))
                if best is None or score < best[0]:
                    best = (score, seqs[idx])
            candidates.sort(key=lambda item: item[0])
            candidates = candidates[: max(1, int(beam_width))]
            action_evals += end - start
        beam = candidates
        if _can_early_stop(score_mode) and beam[0][0] <= 1.0e-8:
            break
    if best is None or not best[1]:
        return None, action_evals
    return best[1][0], action_evals


@torch.no_grad()
def _beam_plan_once_symbolic(
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
    beam_width: int,
    beam_depth: int,
    device: torch.device,
) -> tuple[WorldAction | None, int]:
    beam: list[tuple[float, np.ndarray, list[WorldAction], torch.Tensor | None]] = [(0.0, board.copy(), [], None)]
    best: tuple[float, list[WorldAction]] | None = None
    action_evals = 0
    base_progress_mode = _progress_base_score_mode(score_mode)
    target_goal = _target_goal_latents(score_mode, predicted_goal, oracle_goal)
    for _ in range(max(1, int(beam_depth))):
        leaves: list[np.ndarray] = []
        seqs: list[list[WorldAction]] = []
        actions: list[WorldAction] = []
        node_scores: list[float] = []
        for _, node_board, seq, _ in beam:
            node_score = 0.0
            if base_progress_mode is not None:
                node_score, _ = score_board(
                    model,
                    node_board,
                    context_latents,
                    predicted_goal,
                    oracle_goal,
                    clue_mask,
                    editable_mask,
                    active_mask,
                    score_mode=base_progress_mode,  # type: ignore[arg-type]
                    device=device,
                )
            for action in legal_fill_actions(node_board, allow_conflicts=True):
                try:
                    leaf = apply_fill_action(node_board, action, allow_conflicts=True)
                except ValueError:
                    continue
                leaves.append(leaf)
                seqs.append([*seq, action])
                actions.append(action)
                node_scores.append(node_score)
        if not leaves:
            break
        leaf_latents = encode_boards(
            model,
            leaves,
            context_latents,
            clue_mask,
            editable_mask,
            active_mask,
            device=device,
        )
        mask_t = torch.as_tensor(active_mask[None], dtype=torch.bool, device=device).expand(len(leaves), -1, -1)
        if base_progress_mode is not None:
            absolute = latent_distance(model, leaf_latents, predicted_goal, oracle_goal, mask_t, base_progress_mode)
            score_values = absolute - torch.as_tensor(node_scores, dtype=absolute.dtype, device=device)
        elif _is_changed_cell_score(score_mode):
            score_values = torch.stack(
                [changed_cell_raw_euclidean_distance(leaf_latents[i : i + 1], target_goal, action) for i, action in enumerate(actions)]
            ).reshape(-1)
        else:
            score_values = latent_distance(model, leaf_latents, predicted_goal, oracle_goal, mask_t, score_mode)
        scores = [float(x) for x in score_values.detach().cpu().tolist()]
        action_evals += len(leaves)
        candidates: list[tuple[float, np.ndarray, list[WorldAction], torch.Tensor | None]] = []
        for score, leaf, seq in zip(scores, leaves, seqs, strict=True):
            candidates.append((score, leaf, seq, None))
            if best is None or score < best[0]:
                best = (score, seq)
        candidates.sort(key=lambda item: item[0])
        beam = candidates[: max(1, int(beam_width))]
        if _can_early_stop(score_mode) and beam[0][0] <= 1.0e-8:
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
    return float(latent_distance(model, latent, predicted_goal, oracle_goal, active_t, score_mode).item()), latent


@torch.no_grad()
def encode_boards(
    model: GridTokenGoalJEPA,
    boards: list[np.ndarray],
    context_latents: torch.Tensor,
    clue_mask: np.ndarray,
    editable_mask: np.ndarray,
    active_mask: np.ndarray,
    *,
    device: torch.device,
    chunk_size: int = 512,
) -> torch.Tensor:
    board_t = torch.as_tensor(np.stack(boards), dtype=torch.long, device=device)
    count = board_t.shape[0]
    clue_t = torch.as_tensor(clue_mask[None], dtype=torch.bool, device=device).expand(count, -1, -1)
    edit_t = torch.as_tensor(editable_mask[None], dtype=torch.bool, device=device).expand(count, -1, -1)
    active_t = torch.as_tensor(active_mask[None], dtype=torch.bool, device=device).expand(count, -1, -1)
    context_t = context_latents.expand(count, -1, -1)
    latents = []
    for start in range(0, count, chunk_size):
        end = min(count, start + chunk_size)
        latents.append(model.encode_state(board_t[start:end], context_t[start:end], clue_t[start:end], edit_t[start:end], active_t[start:end]))
    return torch.cat(latents, dim=0)


def latent_distance(
    model: GridTokenGoalJEPA,
    latent: torch.Tensor,
    predicted_goal: torch.Tensor,
    oracle_goal: torch.Tensor,
    mask: torch.Tensor,
    score_mode: str,
) -> torch.Tensor:
    if mask.ndim == 3:
        mask = mask.reshape(mask.shape[0], -1)
    predicted_goal = _expand_tokens_like(predicted_goal, latent)
    oracle_goal = _expand_tokens_like(oracle_goal, latent)
    if score_mode == "predicted_goal_distance":
        return model.distance(latent, predicted_goal, mask)
    if score_mode == "oracle_goal_distance":
        return model.distance(latent, oracle_goal, mask)
    target_goal = _target_goal_latents(score_mode, predicted_goal, oracle_goal)
    metric = _score_metric_name(score_mode)
    if metric in {"raw_euclidean_distance", "raw_euclidean_progress"}:
        return raw_tokenwise_euclidean_distance(latent, target_goal, mask)
    if metric == "raw_squared_euclidean_distance":
        return raw_tokenwise_squared_euclidean_distance(latent, target_goal, mask)
    if metric == "raw_cosine_distance":
        return raw_tokenwise_cosine_distance(latent, target_goal, mask)
    if metric == "raw_hybrid_distance":
        return raw_tokenwise_euclidean_distance(latent, target_goal, mask) + raw_tokenwise_cosine_distance(latent, target_goal, mask)
    if metric == "projected_euclidean_distance":
        return projected_tokenwise_euclidean_distance(latent, target_goal, mask, model.distance_projector)
    if metric == "changed_cell_raw_euclidean_distance":
        return raw_tokenwise_euclidean_distance(latent, target_goal, mask)
    raise ValueError(f"Unknown score_mode: {score_mode}")


def _expand_tokens_like(tokens: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if tokens.shape == reference.shape:
        return tokens
    if tokens.shape[0] == 1 and tokens.shape[1:] == reference.shape[1:]:
        return tokens.expand(reference.shape[0], -1, -1)
    return tokens


def _target_goal_latents(score_mode: str, predicted_goal: torch.Tensor, oracle_goal: torch.Tensor) -> torch.Tensor:
    if score_mode.startswith("predicted_goal_"):
        return predicted_goal
    return oracle_goal


def _score_metric_name(score_mode: str) -> str:
    for prefix in ("oracle_goal_", "predicted_goal_"):
        if score_mode.startswith(prefix):
            return score_mode.removeprefix(prefix)
    return score_mode


def _is_progress_score(score_mode: str) -> bool:
    return _score_metric_name(score_mode) == "raw_euclidean_progress"


def _is_changed_cell_score(score_mode: str) -> bool:
    return _score_metric_name(score_mode) == "changed_cell_raw_euclidean_distance"


def _progress_base_score_mode(score_mode: str) -> str | None:
    if not _is_progress_score(score_mode):
        return None
    if score_mode.startswith("predicted_goal_"):
        return "predicted_goal_raw_euclidean_distance"
    return "oracle_goal_raw_euclidean_distance"


def _can_early_stop(score_mode: str) -> bool:
    return not _is_progress_score(score_mode)


def raw_tokenwise_euclidean_distance(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if a.shape != b.shape:
        raise ValueError(f"Raw distance inputs must have matching shapes, got {tuple(a.shape)} and {tuple(b.shape)}.")
    if mask.ndim == 3:
        mask = mask.reshape(mask.shape[0], -1)
    if mask.shape != a.shape[:-1]:
        raise ValueError(f"Raw distance mask must have shape {tuple(a.shape[:-1])}, got {tuple(mask.shape)}.")
    per_token = (a.float() - b.float()).square().sum(dim=-1).sqrt()
    weights = mask.float()
    return (per_token * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)


def raw_tokenwise_squared_euclidean_distance(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if a.shape != b.shape:
        raise ValueError(f"Raw squared distance inputs must have matching shapes, got {tuple(a.shape)} and {tuple(b.shape)}.")
    if mask.ndim == 3:
        mask = mask.reshape(mask.shape[0], -1)
    if mask.shape != a.shape[:-1]:
        raise ValueError(f"Raw squared distance mask must have shape {tuple(a.shape[:-1])}, got {tuple(mask.shape)}.")
    per_token = (a.float() - b.float()).square().sum(dim=-1)
    weights = mask.float()
    return (per_token * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)


def raw_tokenwise_cosine_distance(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if a.shape != b.shape:
        raise ValueError(f"Raw cosine distance inputs must have matching shapes, got {tuple(a.shape)} and {tuple(b.shape)}.")
    if mask.ndim == 3:
        mask = mask.reshape(mask.shape[0], -1)
    if mask.shape != a.shape[:-1]:
        raise ValueError(f"Raw cosine distance mask must have shape {tuple(a.shape[:-1])}, got {tuple(mask.shape)}.")
    per_token = 1.0 - torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1, eps=1.0e-6)
    weights = mask.float()
    return (per_token * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)


def projected_tokenwise_euclidean_distance(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor, projector: torch.nn.Module) -> torch.Tensor:
    if a.shape != b.shape:
        raise ValueError(f"Projected distance inputs must have matching shapes, got {tuple(a.shape)} and {tuple(b.shape)}.")
    if mask.ndim == 3:
        mask = mask.reshape(mask.shape[0], -1)
    if mask.shape != a.shape[:-1]:
        raise ValueError(f"Projected distance mask must have shape {tuple(a.shape[:-1])}, got {tuple(mask.shape)}.")
    per_token = (projector(a.float()) - projector(b.float())).square().sum(dim=-1).sqrt()
    weights = mask.float()
    return (per_token * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)


def changed_cell_raw_euclidean_distance(a: torch.Tensor, b: torch.Tensor, action: WorldAction) -> torch.Tensor:
    if a.shape != b.shape:
        raise ValueError(f"Changed-cell distance inputs must have matching shapes, got {tuple(a.shape)} and {tuple(b.shape)}.")
    idx = int(action.row) * 9 + int(action.col)
    if idx < 0 or idx >= a.shape[-2]:
        raise ValueError(f"Action cell index {idx} is outside token count {a.shape[-2]}.")
    return (a[..., idx, :].float() - b[..., idx, :].float()).square().sum(dim=-1).sqrt()


def changed_cell_raw_euclidean_distances(a: torch.Tensor, b: torch.Tensor, actions: list[WorldAction]) -> torch.Tensor:
    if a.shape != b.shape:
        raise ValueError(f"Changed-cell distance inputs must have matching shapes, got {tuple(a.shape)} and {tuple(b.shape)}.")
    if len(actions) != a.shape[0]:
        raise ValueError(f"Expected one action per batch item, got {len(actions)} actions for batch size {a.shape[0]}.")
    indices = torch.as_tensor([int(action.row) * 9 + int(action.col) for action in actions], dtype=torch.long, device=a.device)
    if bool(((indices < 0) | (indices >= a.shape[-2])).any().item()):
        raise ValueError(f"Action cell indices must be inside token count {a.shape[-2]}.")
    batch = torch.arange(a.shape[0], device=a.device)
    return (a[batch, indices, :].float() - b[batch, indices, :].float()).square().sum(dim=-1).sqrt()


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
