from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from puzzle_jepa.data.grid_goal_sudoku import apply_fill_action, corrupt_terminal, legal_fill_actions
from puzzle_jepa.data.worlds import PuzzleExample, WorldAction
from puzzle_jepa.models.grid_goal_jepa import GridTokenGoalJEPA
from puzzle_jepa.planning.grid_goal_planner import _prepare_goal_latents, score_board


@torch.no_grad()
def run_grid_goal_diagnostics(
    model: GridTokenGoalJEPA,
    examples: list[PuzzleExample],
    output_dir: Path,
    *,
    device: torch.device,
    seed: int = 0,
    max_examples: int = 32,
    panel_examples: int = 3,
    panel_steps: int = 5,
    panel_actions: int = 6,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    metrics: dict[str, Any] = {}
    metrics.update(_latent_geometry(model, examples[:max_examples], device=device))
    metrics.update(_trajectory_distance_metrics(model, examples[:max_examples], device=device))
    metrics.update(_action_rank_metrics(model, examples[:max_examples], device=device))
    metrics.update(_terminal_corruption_metrics(model, examples[:max_examples], rng, device=device))
    panels = _action_panels(
        model,
        examples[:panel_examples],
        rng,
        device=device,
        panel_steps=panel_steps,
        panel_actions=panel_actions,
    )
    (output_dir / "action_panels.json").write_text(json.dumps(panels, indent=2, sort_keys=True))
    (output_dir / "diagnostics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


@torch.no_grad()
def _latents_for_board(model: GridTokenGoalJEPA, example: PuzzleExample, board: np.ndarray, *, device: torch.device):
    clue_mask = example.state != 0
    editable_mask = ~clue_mask
    active_mask = np.ones((9, 9), dtype=bool)
    context, predicted_goal, oracle_goal = _prepare_goal_latents(
        model, example.state, example.goal, clue_mask, editable_mask, active_mask, device=device
    )
    _, board_latent = score_board(
        model,
        board,
        context,
        predicted_goal,
        oracle_goal,
        clue_mask,
        editable_mask,
        active_mask,
        score_mode="predicted_goal_distance",
        device=device,
    )
    return board_latent, predicted_goal, oracle_goal, context


def _latent_geometry(model: GridTokenGoalJEPA, examples: list[PuzzleExample], *, device: torch.device) -> dict[str, float]:
    tokens = []
    for example in examples:
        latent, _, _, _ = _latents_for_board(model, example, example.state, device=device)
        tokens.append(latent.squeeze(0).float().cpu())
    if not tokens:
        return {}
    x = torch.cat(tokens, dim=0)
    centered = x - x.mean(dim=0, keepdim=True)
    _, singular, _ = torch.linalg.svd(centered, full_matrices=False)
    effective_rank = float(torch.exp(-(singular / singular.sum().clamp_min(1e-12) * torch.log((singular / singular.sum().clamp_min(1e-12)).clamp_min(1e-12))).sum()).item())
    return {
        "latent_token_mean_abs": float(x.mean(dim=0).abs().mean().item()),
        "latent_token_std_mean": float(x.std(dim=0).mean().item()),
        "latent_token_std_min": float(x.std(dim=0).min().item()),
        "latent_effective_rank": effective_rank,
    }


def _trajectory_distance_metrics(model: GridTokenGoalJEPA, examples: list[PuzzleExample], *, device: torch.device) -> dict[str, float]:
    oracle_drops = []
    predicted_drops = []
    oracle_monotone = 0
    predicted_monotone = 0
    total = 0
    for example in examples:
        board = example.state.copy()
        clue_mask = example.state != 0
        editable_mask = ~clue_mask
        active_mask = np.ones((9, 9), dtype=bool)
        context, predicted_goal, oracle_goal = _prepare_goal_latents(
            model, example.state, example.goal, clue_mask, editable_mask, active_mask, device=device
        )
        prev_oracle, _ = score_board(model, board, context, predicted_goal, oracle_goal, clue_mask, editable_mask, active_mask, score_mode="oracle_goal_distance", device=device)
        prev_pred, _ = score_board(model, board, context, predicted_goal, oracle_goal, clue_mask, editable_mask, active_mask, score_mode="predicted_goal_distance", device=device)
        for row, col in np.argwhere(board == 0):
            board = apply_fill_action(board, WorldAction(int(row), int(col), int(example.goal[row, col])), allow_conflicts=True)
            cur_oracle, _ = score_board(model, board, context, predicted_goal, oracle_goal, clue_mask, editable_mask, active_mask, score_mode="oracle_goal_distance", device=device)
            cur_pred, _ = score_board(model, board, context, predicted_goal, oracle_goal, clue_mask, editable_mask, active_mask, score_mode="predicted_goal_distance", device=device)
            oracle_drops.append(prev_oracle - cur_oracle)
            predicted_drops.append(prev_pred - cur_pred)
            oracle_monotone += int(cur_oracle <= prev_oracle)
            predicted_monotone += int(cur_pred <= prev_pred)
            total += 1
            prev_oracle, prev_pred = cur_oracle, cur_pred
    return {
        "oracle_goal_distance_drop_mean": float(np.mean(oracle_drops)) if oracle_drops else 0.0,
        "predicted_goal_distance_drop_mean": float(np.mean(predicted_drops)) if predicted_drops else 0.0,
        "oracle_goal_distance_monotone_fraction": float(oracle_monotone / max(1, total)),
        "predicted_goal_distance_monotone_fraction": float(predicted_monotone / max(1, total)),
    }


def _action_rank_metrics(model: GridTokenGoalJEPA, examples: list[PuzzleExample], *, device: torch.device) -> dict[str, float]:
    oracle_top1 = []
    predicted_top1 = []
    oracle_pairwise = []
    predicted_pairwise = []
    for example in examples:
        board = example.state.copy()
        empty = np.argwhere(board == 0)
        for row, col in empty[: max(1, len(empty) // 2)]:
            board[int(row), int(col)] = int(example.goal[int(row), int(col)])
        actions = legal_fill_actions(board, allow_conflicts=True)
        if not actions:
            continue
        clue_mask = example.state != 0
        editable_mask = ~clue_mask
        active_mask = np.ones((9, 9), dtype=bool)
        context, predicted_goal, oracle_goal = _prepare_goal_latents(
            model, example.state, example.goal, clue_mask, editable_mask, active_mask, device=device
        )
        rows = []
        for action in actions:
            leaf = apply_fill_action(board, action, allow_conflicts=True)
            rows.append(
                (
                    action,
                    *[
                        score_board(model, leaf, context, predicted_goal, oracle_goal, clue_mask, editable_mask, active_mask, score_mode=mode, device=device)[0]
                        for mode in ("oracle_goal_distance", "predicted_goal_distance")
                    ],
                )
            )
        positives = [idx for idx, (action, _, _) in enumerate(rows) if action.value == int(example.goal[action.row, action.col])]
        if not positives:
            continue
        for offset, key in enumerate(("oracle", "predicted"), start=1):
            costs = np.asarray([row[offset] for row in rows])
            best = int(costs.argmin())
            pair = float(np.mean([costs[pos] < cost for pos in positives for cost in costs]))
            if key == "oracle":
                oracle_top1.append(float(best in positives))
                oracle_pairwise.append(pair)
            else:
                predicted_top1.append(float(best in positives))
                predicted_pairwise.append(pair)
    return {
        "oracle_goal_action_top1": float(np.mean(oracle_top1)) if oracle_top1 else 0.0,
        "predicted_goal_action_top1": float(np.mean(predicted_top1)) if predicted_top1 else 0.0,
        "oracle_goal_action_pairwise": float(np.mean(oracle_pairwise)) if oracle_pairwise else 0.0,
        "predicted_goal_action_pairwise": float(np.mean(predicted_pairwise)) if predicted_pairwise else 0.0,
    }


def _terminal_corruption_metrics(model: GridTokenGoalJEPA, examples: list[PuzzleExample], rng: np.random.Generator, *, device: torch.device) -> dict[str, float]:
    margins = []
    for example in examples:
        corrupt = corrupt_terminal(example.goal, rng)
        clue_mask = example.state != 0
        editable_mask = ~clue_mask
        active_mask = np.ones((9, 9), dtype=bool)
        context, predicted_goal, oracle_goal = _prepare_goal_latents(
            model, example.state, example.goal, clue_mask, editable_mask, active_mask, device=device
        )
        good, _ = score_board(model, example.goal, context, predicted_goal, oracle_goal, clue_mask, editable_mask, active_mask, score_mode="predicted_goal_distance", device=device)
        bad, _ = score_board(model, corrupt, context, predicted_goal, oracle_goal, clue_mask, editable_mask, active_mask, score_mode="predicted_goal_distance", device=device)
        margins.append(bad - good)
    return {
        "terminal_corruption_margin_mean": float(np.mean(margins)) if margins else 0.0,
        "terminal_corruption_margin_positive_fraction": float(np.mean([m > 0 for m in margins])) if margins else 0.0,
    }


def _action_panels(
    model: GridTokenGoalJEPA,
    examples: list[PuzzleExample],
    rng: np.random.Generator,
    *,
    device: torch.device,
    panel_steps: int,
    panel_actions: int,
) -> list[dict[str, Any]]:
    panels = []
    for ex_idx, example in enumerate(examples):
        board = example.state.copy()
        clue_mask = example.state != 0
        editable_mask = ~clue_mask
        active_mask = np.ones((9, 9), dtype=bool)
        context, predicted_goal, oracle_goal = _prepare_goal_latents(
            model, example.state, example.goal, clue_mask, editable_mask, active_mask, device=device
        )
        for step in range(panel_steps):
            actions = legal_fill_actions(board, allow_conflicts=True)
            if not actions:
                break
            positives = [a for a in actions if a.value == int(example.goal[a.row, a.col])]
            sampled = positives[:1]
            sampled.extend(actions[int(i)] for i in rng.choice(len(actions), size=min(panel_actions, len(actions)), replace=False))
            rows = []
            for action in sampled[:panel_actions]:
                leaf = apply_fill_action(board, action, allow_conflicts=True)
                rows.append(
                    {
                        "action": [action.row, action.col, action.value],
                        "is_target_digit": action.value == int(example.goal[action.row, action.col]),
                        "oracle_goal_distance": score_board(model, leaf, context, predicted_goal, oracle_goal, clue_mask, editable_mask, active_mask, score_mode="oracle_goal_distance", device=device)[0],
                        "predicted_goal_distance": score_board(model, leaf, context, predicted_goal, oracle_goal, clue_mask, editable_mask, active_mask, score_mode="predicted_goal_distance", device=device)[0],
                    }
                )
            panels.append({"example": ex_idx, "step": step, "actions": rows})
            if positives:
                board = apply_fill_action(board, positives[0], allow_conflicts=True)
    return panels
