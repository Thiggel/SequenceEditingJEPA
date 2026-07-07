from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from puzzle_jepa.data.grid_goal_sudoku import apply_fill_action, corrupt_terminal, legal_fill_actions
from puzzle_jepa.data.worlds import PuzzleExample, WorldAction
from puzzle_jepa.models.grid_goal_jepa import GridTokenGoalJEPA, _latent_active_mask
from puzzle_jepa.planning.grid_goal_planner import _predict_goal_for_board, _prepare_goal_latents, score_board


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
    metrics.update(_delta_action_probe_metrics(model, examples[:max_examples], device=device))
    metrics.update(_sd_progress_probe_metrics(model, examples[:max_examples], device=device))
    metrics.update(_rollout_drift_metrics(model, examples[:max_examples], device=device))
    metrics.update(_goal_alignment_metrics(model, examples[:max_examples], device=device))
    metrics.update(_goal_by_fill_depth_metrics(model, examples[:max_examples], device=device))
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
    context, predicted_goal, oracle_goal, initial_latents = _prepare_goal_latents(
        model, example.state, example.goal, clue_mask, editable_mask, active_mask, device=device
    )
    predicted_goal = _predict_goal_for_board(
        model,
        board,
        context,
        initial_latents,
        clue_mask,
        editable_mask,
        active_mask,
        device=device,
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
        context, predicted_goal, oracle_goal, initial_latents = _prepare_goal_latents(
            model, example.state, example.goal, clue_mask, editable_mask, active_mask, device=device
        )
        predicted_goal = _predict_goal_for_board(
            model, board, context, initial_latents, clue_mask, editable_mask, active_mask, device=device
        )
        prev_oracle, _ = score_board(model, board, context, predicted_goal, oracle_goal, clue_mask, editable_mask, active_mask, score_mode="oracle_goal_distance", device=device)
        prev_pred, _ = score_board(model, board, context, predicted_goal, oracle_goal, clue_mask, editable_mask, active_mask, score_mode="predicted_goal_distance", device=device)
        for row, col in np.argwhere(board == 0):
            board = apply_fill_action(board, WorldAction(int(row), int(col), int(example.goal[row, col])), allow_conflicts=True)
            predicted_goal = _predict_goal_for_board(
                model, board, context, initial_latents, clue_mask, editable_mask, active_mask, device=device
            )
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
        context, predicted_goal, oracle_goal, initial_latents = _prepare_goal_latents(
            model, example.state, example.goal, clue_mask, editable_mask, active_mask, device=device
        )
        predicted_goal = _predict_goal_for_board(
            model,
            board,
            context,
            initial_latents,
            clue_mask,
            editable_mask,
            active_mask,
            device=device,
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
        context, predicted_goal, oracle_goal, initial_latents = _prepare_goal_latents(
            model, example.state, example.goal, clue_mask, editable_mask, active_mask, device=device
        )
        predicted_goal = _predict_goal_for_board(
            model,
            board,
            context,
            initial_latents,
            clue_mask,
            editable_mask,
            active_mask,
            device=device,
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


def _delta_action_probe_metrics(model: GridTokenGoalJEPA, examples: list[PuzzleExample], *, device: torch.device) -> dict[str, float]:
    totals = {
        "ldad_target_delta_row_acc": [],
        "ldad_target_delta_col_acc": [],
        "ldad_target_delta_digit_acc": [],
        "ldad_target_delta_action_acc": [],
        "ldad_predicted_delta_row_acc": [],
        "ldad_predicted_delta_col_acc": [],
        "ldad_predicted_delta_digit_acc": [],
        "ldad_predicted_delta_action_acc": [],
        "target_delta_changed_cell_top1": [],
        "predicted_delta_changed_cell_top1": [],
        "target_delta_affected_f1": [],
        "predicted_delta_affected_f1": [],
    }
    for example in examples:
        boards, actions = _oracle_probe_sequence(example, max_steps=16)
        if len(actions) == 0:
            continue
        clue_mask = example.state != 0
        editable_mask = ~clue_mask
        active_mask = np.ones((9, 9), dtype=bool)
        context, _, oracle_goal, _ = _prepare_goal_latents(
            model, example.state, example.goal, clue_mask, editable_mask, active_mask, device=device
        )
        del oracle_goal
        boards_t = torch.as_tensor(np.stack(boards), dtype=torch.long, device=device)
        clue_t = torch.as_tensor(clue_mask[None], dtype=torch.bool, device=device).expand(len(boards), -1, -1)
        edit_t = torch.as_tensor(editable_mask[None], dtype=torch.bool, device=device).expand(len(boards), -1, -1)
        active_t = torch.as_tensor(active_mask[None], dtype=torch.bool, device=device).expand(len(boards), -1, -1)
        latents = model.encode_state(boards_t, context.expand(len(boards), -1, -1), clue_t, edit_t, active_t)
        actions_t = torch.as_tensor(np.asarray(actions), dtype=torch.long, device=device)[None]
        starts = latents[:-1][None]
        target_future = latents[1:][None]
        active_probe = torch.as_tensor(active_mask[None], dtype=torch.bool, device=device)
        predicted_future = model._delta_action_predicted_future(starts, actions_t, context, horizon=1)
        for prefix, future in (("ldad_target", target_future), ("ldad_predicted", predicted_future)):
            values = _decode_ldad_probe(model, future - starts, active_probe, actions_t)
            for key, value in values.items():
                totals[f"{prefix}_delta_{key}"].append(value)
        if latents.shape[-2] >= 81:
            for prefix, future in (("target", target_future), ("predicted", predicted_future)):
                locality = _delta_locality_probe(future - starts, actions_t)
                totals[f"{prefix}_delta_changed_cell_top1"].extend(locality["changed_cell_top1"])
                totals[f"{prefix}_delta_affected_f1"].extend(locality["affected_f1"])
    return {key: float(np.mean(values)) if values else 0.0 for key, values in totals.items()}


def _sd_progress_probe_metrics(model: GridTokenGoalJEPA, examples: list[PuzzleExample], *, device: torch.device) -> dict[str, float]:
    pairs = []
    monotone = []
    pairwise = []
    for example in examples:
        boards, _ = _oracle_probe_sequence(example, max_steps=16)
        if len(boards) < 2:
            continue
        clue_mask = example.state != 0
        editable_mask = ~clue_mask
        active_mask = np.ones((9, 9), dtype=bool)
        context, _, oracle_goal, _ = _prepare_goal_latents(
            model, example.state, example.goal, clue_mask, editable_mask, active_mask, device=device
        )
        boards_t = torch.as_tensor(np.stack(boards), dtype=torch.long, device=device)
        clue_t = torch.as_tensor(clue_mask[None], dtype=torch.bool, device=device).expand(len(boards), -1, -1)
        edit_t = torch.as_tensor(editable_mask[None], dtype=torch.bool, device=device).expand(len(boards), -1, -1)
        active_t = torch.as_tensor(active_mask[None], dtype=torch.bool, device=device).expand(len(boards), -1, -1)
        latents = model.encode_state(boards_t, context.expand(len(boards), -1, -1), clue_t, edit_t, active_t)
        goal_latents = oracle_goal.expand(len(boards), -1, -1)
        distances = model.sd_progress_distance(latents, goal_latents, active_t).detach().cpu().numpy()
        hammings = np.asarray([np.not_equal(board, example.goal).sum() for board in boards], dtype=np.float64)
        pairs.extend((float(hamming), float(distance)) for hamming, distance in zip(hammings, distances, strict=True))
        monotone.extend(float(cur <= prev) for prev, cur in zip(distances[:-1], distances[1:], strict=True))
        for before in range(len(distances)):
            for after in range(before + 1, len(distances)):
                if hammings[before] > hammings[after]:
                    pairwise.append(float(distances[before] > distances[after]))
    return {
        "sd_progress_distance_hamming_spearman": _spearman(pairs),
        "sd_progress_oracle_monotone_fraction": float(np.mean(monotone)) if monotone else 0.0,
        "sd_progress_pairwise_order_accuracy": float(np.mean(pairwise)) if pairwise else 0.0,
    }


def _decode_ldad_probe(
    model: GridTokenGoalJEPA,
    delta: torch.Tensor,
    active_mask: torch.Tensor,
    actions: torch.Tensor,
) -> dict[str, float]:
    steps = 1
    batch, starts, token_count = delta.shape[:3]
    flat_delta = delta.reshape(batch * starts, token_count, model.d_model)
    flat_mask = active_mask.expand(batch, *active_mask.shape[1:])
    flat_mask = flat_mask[:, None].expand(batch, starts, *flat_mask.shape[1:]).reshape(batch * starts, *flat_mask.shape[1:])
    action_sequence = actions[:, :starts].reshape(batch * starts, steps, 3)
    decoder_mask = _latent_active_mask(flat_mask, token_count=token_count)
    try:
        selected, selected_mask = model._select_delta_action_source(flat_delta, decoder_mask, action_sequence)
        row_logits, col_logits, digit_logits = model.delta_action_decoder(selected, selected_mask, steps)
    except (IndexError, ValueError):
        return {"row_acc": 0.0, "col_acc": 0.0, "digit_acc": 0.0, "action_acc": 0.0}
    row_pred = row_logits[:, 0].argmax(dim=-1)
    col_pred = col_logits[:, 0].argmax(dim=-1)
    digit_pred = digit_logits[:, 0].argmax(dim=-1)
    labels = action_sequence[:, 0]
    row_ok = row_pred == labels[:, 0].clamp(0, 8)
    col_ok = col_pred == labels[:, 1].clamp(0, 8)
    digit_ok = digit_pred == labels[:, 2].clamp(0, 9)
    return {
        "row_acc": float(row_ok.float().mean().item()),
        "col_acc": float(col_ok.float().mean().item()),
        "digit_acc": float(digit_ok.float().mean().item()),
        "action_acc": float((row_ok & col_ok & digit_ok).float().mean().item()),
    }


def _delta_locality_probe(delta: torch.Tensor, actions: torch.Tensor) -> dict[str, list[float]]:
    magnitudes = delta[..., :81, :].square().sum(dim=-1).squeeze(0)
    action_rows = actions[0, : magnitudes.shape[0], 0].detach().cpu().numpy()
    action_cols = actions[0, : magnitudes.shape[0], 1].detach().cpu().numpy()
    top1 = magnitudes.argmax(dim=-1).detach().cpu().numpy()
    topk = torch.topk(magnitudes, k=min(21, magnitudes.shape[-1]), dim=-1).indices.detach().cpu().numpy()
    changed = []
    affected_f1 = []
    for index, (row, col) in enumerate(zip(action_rows, action_cols, strict=True)):
        row = int(np.clip(row, 0, 8))
        col = int(np.clip(col, 0, 8))
        changed_position = row * 9 + col
        changed.append(float(int(top1[index]) == changed_position))
        affected = _affected_positions(row, col)
        predicted = set(int(item) for item in topk[index])
        overlap = len(affected & predicted)
        precision = overlap / max(1, len(predicted))
        recall = overlap / max(1, len(affected))
        affected_f1.append(0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall))
    return {"changed_cell_top1": changed, "affected_f1": affected_f1}


def _affected_positions(row: int, col: int) -> set[int]:
    positions = {row * 9 + item for item in range(9)}
    positions.update(item * 9 + col for item in range(9))
    block_row = (row // 3) * 3
    block_col = (col // 3) * 3
    positions.update((block_row + dr) * 9 + block_col + dc for dr in range(3) for dc in range(3))
    return positions


def _oracle_probe_sequence(example: PuzzleExample, *, max_steps: int) -> tuple[list[np.ndarray], list[list[int]]]:
    board = example.state.copy()
    boards = [board.copy()]
    actions = []
    for row, col in np.argwhere(board == 0)[:max_steps]:
        row = int(row)
        col = int(col)
        value = int(example.goal[row, col])
        actions.append([row, col, value])
        board = apply_fill_action(board, WorldAction(row, col, value), allow_conflicts=True)
        boards.append(board.copy())
    return boards, actions


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
        context, predicted_goal, oracle_goal, _ = _prepare_goal_latents(
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
        _, predicted_goal, oracle_goal, _ = _prepare_goal_latents(
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


def _goal_by_fill_depth_metrics(model: GridTokenGoalJEPA, examples: list[PuzzleExample], *, device: torch.device) -> dict[str, float]:
    fractions = (0.0, 0.25, 0.5, 0.75, 1.0)
    by_fraction: dict[float, dict[str, list[float]]] = {
        fraction: {"distance": [], "mse": [], "cosine": [], "q_step_rmse": [], "state_pred_distance": [], "state_oracle_distance": []}
        for fraction in fractions
    }
    q_to_refs = {"q_to_initial_mse": [], "q_to_half_mse": [], "q_to_goal_mse": []}
    for example in examples:
        clue_mask = example.state != 0
        editable_mask = ~clue_mask
        active_mask = np.ones((9, 9), dtype=bool)
        context, _, oracle_goal, initial_latents = _prepare_goal_latents(
            model, example.state, example.goal, clue_mask, editable_mask, active_mask, device=device
        )
        active_t = torch.as_tensor(active_mask[None], dtype=torch.bool, device=device)
        clue_t = torch.as_tensor(clue_mask[None], dtype=torch.bool, device=device)
        edit_t = torch.as_tensor(editable_mask[None], dtype=torch.bool, device=device)
        empty = np.argwhere(example.state == 0)
        previous_q: torch.Tensor | None = None
        encoded_refs: dict[float, torch.Tensor] = {}
        for fraction in fractions:
            board = example.state.copy()
            fill_count = min(len(empty), int(round(len(empty) * fraction)))
            for row, col in empty[:fill_count]:
                board[int(row), int(col)] = int(example.goal[int(row), int(col)])
            predicted_goal = _predict_goal_for_board(
                model,
                board,
                context,
                initial_latents,
                clue_mask,
                editable_mask,
                active_mask,
                device=device,
            )
            board_t = torch.as_tensor(board[None], dtype=torch.long, device=device)
            state_latent = model.encode_state(board_t, context, clue_t, edit_t, active_t)
            encoded_refs[fraction] = state_latent
            values = by_fraction[fraction]
            values["distance"].append(float(model.distance(predicted_goal, oracle_goal, active_t).item()))
            values["mse"].append(float((predicted_goal - oracle_goal).square().mean().item()))
            values["cosine"].append(float(torch.nn.functional.cosine_similarity(predicted_goal, oracle_goal, dim=-1).mean().item()))
            values["q_step_rmse"].append(
                0.0 if previous_q is None else float((predicted_goal - previous_q).square().mean().sqrt().item())
            )
            values["state_pred_distance"].append(float(model.distance(state_latent, predicted_goal, active_t).item()))
            values["state_oracle_distance"].append(float(model.distance(state_latent, oracle_goal, active_t).item()))
            previous_q = predicted_goal
        q0 = _predict_goal_for_board(
            model,
            example.state,
            context,
            initial_latents,
            clue_mask,
            editable_mask,
            active_mask,
            device=device,
        )
        q_to_refs["q_to_initial_mse"].append(float((q0 - encoded_refs[0.0]).square().mean().item()))
        q_to_refs["q_to_half_mse"].append(float((q0 - encoded_refs[0.5]).square().mean().item()))
        q_to_refs["q_to_goal_mse"].append(float((q0 - oracle_goal).square().mean().item()))
    metrics: dict[str, float] = {}
    for fraction, values in by_fraction.items():
        label = f"{int(round(100 * fraction)):03d}"
        metrics[f"goal_fill_{label}_pred_oracle_distance"] = float(np.mean(values["distance"])) if values["distance"] else 0.0
        metrics[f"goal_fill_{label}_pred_oracle_mse"] = float(np.mean(values["mse"])) if values["mse"] else 0.0
        metrics[f"goal_fill_{label}_pred_oracle_cosine"] = float(np.mean(values["cosine"])) if values["cosine"] else 0.0
        metrics[f"goal_fill_{label}_q_step_rmse"] = float(np.mean(values["q_step_rmse"])) if values["q_step_rmse"] else 0.0
        metrics[f"goal_fill_{label}_state_pred_distance"] = float(np.mean(values["state_pred_distance"])) if values["state_pred_distance"] else 0.0
        metrics[f"goal_fill_{label}_state_oracle_distance"] = float(np.mean(values["state_oracle_distance"])) if values["state_oracle_distance"] else 0.0
    metrics.update({key: float(np.mean(items)) if items else 0.0 for key, items in q_to_refs.items()})
    return metrics


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
        context, predicted_goal, oracle_goal, initial_latents = _prepare_goal_latents(
            model, example.state, example.goal, clue_mask, editable_mask, active_mask, device=device
        )
        for row, col in np.argwhere(board == 0):
            predicted_goal = _predict_goal_for_board(
                model,
                board,
                context,
                initial_latents,
                clue_mask,
                editable_mask,
                active_mask,
                device=device,
            )
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
            context, predicted_goal, oracle_goal, initial_latents = _prepare_goal_latents(
                model, example.state, example.goal, clue_mask, editable_mask, active_mask, device=device
            )
            predicted_goal = _predict_goal_for_board(
                model,
                board,
                context,
                initial_latents,
                clue_mask,
                editable_mask,
                active_mask,
                device=device,
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
        context, predicted_goal, oracle_goal, _ = _prepare_goal_latents(
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
        context, predicted_goal, oracle_goal, initial_latents = _prepare_goal_latents(
            model, example.state, example.goal, clue_mask, editable_mask, active_mask, device=device
        )
        for step in range(panel_steps):
            predicted_goal = _predict_goal_for_board(
                model,
                board,
                context,
                initial_latents,
                clue_mask,
                editable_mask,
                active_mask,
                device=device,
            )
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Grid-Token Goal-JEPA diagnostic probes.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from puzzle_jepa.eval.grid_goal_planner_matrix import load_checkpoint, load_eval_examples

    device = torch.device(args.device)
    model, config = load_checkpoint(args.checkpoint, device)
    examples = load_eval_examples(config, limit=int(args.examples))
    run_grid_goal_diagnostics(
        model,
        examples,
        args.output_dir,
        device=device,
        seed=int(args.seed),
        max_examples=int(args.examples),
    )


if __name__ == "__main__":
    main()
