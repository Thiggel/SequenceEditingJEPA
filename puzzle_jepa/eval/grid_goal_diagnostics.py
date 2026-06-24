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
    metrics.update(_latent_rollout_action_rank_metrics(model, examples[:max_examples], device=device))
    metrics.update(_rollout_drift_metrics(model, examples[:max_examples], device=device))
    metrics.update(_goal_alignment_metrics(model, examples[:max_examples], device=device))
    metrics.update(_distance_hamming_spearman_metrics(model, examples[:max_examples], device=device))
    metrics.update(_action_margin_by_depth_metrics(model, examples[:max_examples], device=device))
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


def _latent_rollout_action_rank_metrics(
    model: GridTokenGoalJEPA, examples: list[PuzzleExample], *, device: torch.device
) -> dict[str, float]:
    oracle_top1 = []
    predicted_top1 = []
    for example in examples:
        board = _half_filled_oracle_board(example)
        actions = legal_fill_actions(board, allow_conflicts=True)
        if not actions:
            continue
        clue_mask = example.state != 0
        editable_mask = ~clue_mask
        active_mask = np.ones((9, 9), dtype=bool)
        context, predicted_goal, oracle_goal = _prepare_goal_latents(
            model, example.state, example.goal, clue_mask, editable_mask, active_mask, device=device
        )
        _, current_latent = score_board(
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
        mask_t = torch.as_tensor(active_mask[None], dtype=torch.bool, device=device)
        rows = []
        for action in actions:
            action_t = torch.as_tensor([[action.row, action.col, action.value]], dtype=torch.long, device=device)
            next_latent = model.predict_next(current_latent, action_t, context)
            rows.append(
                (
                    action,
                    float(model.distance(next_latent, oracle_goal, mask_t).item()),
                    float(model.distance(next_latent, predicted_goal, mask_t).item()),
                )
            )
        positives = [idx for idx, (action, _, _) in enumerate(rows) if action.value == int(example.goal[action.row, action.col])]
        if not positives:
            continue
        oracle_top1.append(float(int(np.argmin([row[1] for row in rows])) in positives))
        predicted_top1.append(float(int(np.argmin([row[2] for row in rows])) in positives))
    predicted_top1_mean = float(np.mean(predicted_top1)) if predicted_top1 else 0.0
    return {
        "latent_rollout_oracle_goal_action_top1": float(np.mean(oracle_top1)) if oracle_top1 else 0.0,
        "latent_rollout_predicted_goal_action_top1": predicted_top1_mean,
        "latent_rollout_action_top1": predicted_top1_mean,
    }


def _rollout_drift_metrics(model: GridTokenGoalJEPA, examples: list[PuzzleExample], *, device: torch.device) -> dict[str, float]:
    configured = tuple(int(horizon) for horizon in getattr(model, "multi_step_horizons", ()) if int(horizon) > 0)
    horizons = tuple(sorted(set((1, 4, 8, 16) + configured)))
    values: dict[int, list[float]] = {horizon: [] for horizon in horizons}
    for example in examples:
        board = example.state.copy()
        actions = []
        boards = [board.copy()]
        for row, col in np.argwhere(board == 0)[: max(horizons)]:
            action = WorldAction(int(row), int(col), int(example.goal[row, col]))
            board = apply_fill_action(board, action, allow_conflicts=True)
            actions.append(action)
            boards.append(board.copy())
        if not actions:
            continue
        clue_mask = example.state != 0
        editable_mask = ~clue_mask
        active_mask = np.ones((9, 9), dtype=bool)
        context, predicted_goal, oracle_goal = _prepare_goal_latents(
            model, example.state, example.goal, clue_mask, editable_mask, active_mask, device=device
        )
        del predicted_goal, oracle_goal
        start_t = torch.as_tensor(boards[0][None], dtype=torch.long, device=device)
        clue_t = torch.as_tensor(clue_mask[None], dtype=torch.bool, device=device)
        edit_t = torch.as_tensor(editable_mask[None], dtype=torch.bool, device=device)
        active_t = torch.as_tensor(active_mask[None], dtype=torch.bool, device=device)
        rollout = model.encode_state(start_t, context, clue_t, edit_t, active_t)
        for step, action in enumerate(actions, start=1):
            action_t = torch.as_tensor([[action.row, action.col, action.value]], dtype=torch.long, device=device)
            rollout = model.predict_next(rollout, action_t, context)
            if step in values:
                target_t = torch.as_tensor(boards[step][None], dtype=torch.long, device=device)
                target = model.encode_state(target_t, context, clue_t, edit_t, active_t)
                values[step].append(float((rollout - target).square().mean().item()))
    metrics = {}
    for horizon, items in values.items():
        value = float(np.mean(items)) if items else 0.0
        metrics[f"latent_rollout_drift_mse_h{horizon}"] = value
        metrics[f"predictor_rollout_mse_h{horizon}"] = value
    return metrics


def _goal_alignment_metrics(model: GridTokenGoalJEPA, examples: list[PuzzleExample], *, device: torch.device) -> dict[str, float]:
    distances = []
    mses = []
    cosines = []
    for example in examples:
        clue_mask = example.state != 0
        editable_mask = ~clue_mask
        active_mask = np.ones((9, 9), dtype=bool)
        _, predicted_goal, oracle_goal = _prepare_goal_latents(
            model, example.state, example.goal, clue_mask, editable_mask, active_mask, device=device
        )
        mask_t = torch.as_tensor(active_mask[None], dtype=torch.bool, device=device)
        distance = float(model.distance(predicted_goal, oracle_goal, mask_t).item())
        distances.append(distance)
        mses.append(float((predicted_goal - oracle_goal).square().mean().item()))
        cos = torch.nn.functional.cosine_similarity(predicted_goal, oracle_goal, dim=-1)
        cosines.append(float(cos.mean().item()))
    distance_mean = float(np.mean(distances)) if distances else 0.0
    return {
        "predicted_oracle_goal_token_distance_mean": distance_mean,
        "predicted_vs_oracle_goal_distance": distance_mean,
        "goal_prediction_token_mse": float(np.mean(mses)) if mses else 0.0,
        "predicted_oracle_goal_token_cosine_mean": float(np.mean(cosines)) if cosines else 0.0,
    }


def _distance_hamming_spearman_metrics(
    model: GridTokenGoalJEPA, examples: list[PuzzleExample], *, device: torch.device
) -> dict[str, float]:
    oracle_pairs = []
    predicted_pairs = []
    for example in examples:
        board = example.state.copy()
        clue_mask = example.state != 0
        editable_mask = ~clue_mask
        active_mask = np.ones((9, 9), dtype=bool)
        context, predicted_goal, oracle_goal = _prepare_goal_latents(
            model, example.state, example.goal, clue_mask, editable_mask, active_mask, device=device
        )
        for row, col in np.argwhere(board == 0):
            hamming = float(np.not_equal(board, example.goal).sum())
            oracle_d, _ = score_board(model, board, context, predicted_goal, oracle_goal, clue_mask, editable_mask, active_mask, score_mode="oracle_goal_distance", device=device)
            pred_d, _ = score_board(model, board, context, predicted_goal, oracle_goal, clue_mask, editable_mask, active_mask, score_mode="predicted_goal_distance", device=device)
            oracle_pairs.append((hamming, oracle_d))
            predicted_pairs.append((hamming, pred_d))
            board = apply_fill_action(board, WorldAction(int(row), int(col), int(example.goal[row, col])), allow_conflicts=True)
    predicted = _spearman(predicted_pairs)
    return {
        "oracle_goal_distance_hamming_spearman": _spearman(oracle_pairs),
        "predicted_goal_distance_hamming_spearman": predicted,
        "distance_hamming_spearman": predicted,
    }


def _action_margin_by_depth_metrics(
    model: GridTokenGoalJEPA, examples: list[PuzzleExample], *, device: torch.device
) -> dict[str, float]:
    margins: dict[str, list[float]] = {"early": [], "middle": [], "late": []}
    for example in examples:
        empty = np.argwhere(example.state == 0)
        for label, frac in (("early", 0.25), ("middle", 0.5), ("late", 0.75)):
            board = example.state.copy()
            fill_count = min(len(empty), int(len(empty) * frac))
            for row, col in empty[:fill_count]:
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
            pos = []
            neg = []
            for action in actions:
                leaf = apply_fill_action(board, action, allow_conflicts=True)
                cost, _ = score_board(
                    model,
                    leaf,
                    context,
                    predicted_goal,
                    oracle_goal,
                    clue_mask,
                    editable_mask,
                    active_mask,
                    score_mode="predicted_goal_distance",
                    device=device,
                )
                if action.value == int(example.goal[action.row, action.col]):
                    pos.append(cost)
                else:
                    neg.append(cost)
            if pos and neg:
                margins[label].append(float(min(neg) - min(pos)))
    return {
        f"predicted_action_margin_{label}_mean": float(np.mean(items)) if items else 0.0
        for label, items in margins.items()
    }


def _terminal_corruption_metrics(model: GridTokenGoalJEPA, examples: list[PuzzleExample], rng: np.random.Generator, *, device: torch.device) -> dict[str, float]:
    margins = []
    margins_by_size: dict[int, list[float]] = {size: [] for size in range(1, 6)}
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
        for size in margins_by_size:
            sized_corrupt = corrupt_terminal(example.goal, rng, min_cells=size, max_cells=size)
            sized_bad, _ = score_board(model, sized_corrupt, context, predicted_goal, oracle_goal, clue_mask, editable_mask, active_mask, score_mode="predicted_goal_distance", device=device)
            margins_by_size[size].append(sized_bad - good)
    metrics = {
        "terminal_corruption_margin_mean": float(np.mean(margins)) if margins else 0.0,
        "terminal_corruption_margin_positive_fraction": float(np.mean([m > 0 for m in margins])) if margins else 0.0,
    }
    metrics.update(
        {
            f"terminal_corruption_margin_size_{size}_mean": float(np.mean(items)) if items else 0.0
            for size, items in margins_by_size.items()
        }
    )
    return metrics


def _half_filled_oracle_board(example: PuzzleExample) -> np.ndarray:
    board = example.state.copy()
    empty = np.argwhere(board == 0)
    for row, col in empty[: max(1, len(empty) // 2)]:
        board[int(row), int(col)] = int(example.goal[int(row), int(col)])
    return board


def _spearman(pairs: list[tuple[float, float]]) -> float:
    if len(pairs) < 2:
        return 0.0
    x = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    y = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt(np.square(rx).sum() * np.square(ry).sum()))
    return float((rx * ry).sum() / denom) if denom > 0.0 else 0.0


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = rank
        start = end
    return ranks


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
