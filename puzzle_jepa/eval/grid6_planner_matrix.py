from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from puzzle_jepa.data import SudokuWorld, WorldAction
from puzzle_jepa.eval.grid5_diagnostics import candidate_actions
from puzzle_jepa.models import CausalTrajectoryJEPA
from puzzle_jepa.train.grid0 import _build_world, _load_examples


@dataclass(slots=True)
class PlanResult:
    action: WorldAction | None
    score: float
    leaf_remaining_hamming: int
    leaf_terminal: bool


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_outputs(output_dir: Path, records: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    write_jsonl(output_dir / "planner_records.jsonl", records)
    (output_dir / "planner_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))


def load_model(run_root: Path, device: torch.device) -> tuple[CausalTrajectoryJEPA, dict[str, Any]]:
    checkpoint = torch.load(run_root / "checkpoint.pt", map_location=device)
    config = checkpoint["config"]
    world = _build_world(dict(config["task"]))
    model = CausalTrajectoryJEPA(vocab_size=world.vocab_size, **dict(config["model"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, config


def action_tensor(world: SudokuWorld, action: WorldAction, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(action.as_array(world.task_id), dtype=torch.long, device=device)


def action_array(world: SudokuWorld, actions: list[WorldAction], device: torch.device) -> torch.Tensor:
    if not actions:
        return torch.zeros(0, 4, dtype=torch.long, device=device)
    return torch.stack([action_tensor(world, action, device) for action in actions], dim=0)


def apply_action(world: SudokuWorld, board: np.ndarray, action: WorldAction, clue_mask: np.ndarray, action_mode: str) -> np.ndarray:
    return world.apply(
        board,
        action,
        clue_mask=clue_mask,
        allow_overwrite=action_mode == "mutable_overwrite",
        allow_conflicts=True,
    )


def action_pool(world: SudokuWorld, board: np.ndarray, clue_mask: np.ndarray, action_mode: str) -> list[WorldAction]:
    actions = candidate_actions(world, board, clue_mask)
    if action_mode == "mutable_overwrite":
        return actions
    if action_mode == "fill_empty":
        return [action for action in actions if int(board[action.row, action.col]) == 0]
    raise ValueError(f"unknown action_mode {action_mode!r}.")


def history_tensors(
    world: SudokuWorld,
    boards: list[np.ndarray],
    actions: list[WorldAction],
    initial_board: np.ndarray,
    clue_mask: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    states = torch.as_tensor(np.stack(boards)[None], dtype=torch.long, device=device)
    action_t = action_array(world, actions, device).unsqueeze(0)
    initial = torch.as_tensor(initial_board[None], dtype=torch.long, device=device)
    mask = torch.as_tensor(clue_mask[None], dtype=torch.bool, device=device)
    return states, action_t, initial, mask


@torch.no_grad()
def encode_history(
    model: CausalTrajectoryJEPA,
    world: SudokuWorld,
    boards: list[np.ndarray],
    actions: list[WorldAction],
    initial_board: np.ndarray,
    clue_mask: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    states, action_t, initial, mask = history_tensors(world, boards, actions, initial_board, clue_mask, device)
    return model.encode_context(states, action_t, initial_boards=initial, clue_masks=mask)[:, -1]


@torch.no_grad()
def encode_goal(
    model: CausalTrajectoryJEPA,
    world: SudokuWorld,
    goal: np.ndarray,
    initial_board: np.ndarray,
    clue_mask: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    states = torch.as_tensor(goal[None, None], dtype=torch.long, device=device)
    actions = torch.zeros(1, 0, 4, dtype=torch.long, device=device)
    initial = torch.as_tensor(initial_board[None], dtype=torch.long, device=device)
    mask = torch.as_tensor(clue_mask[None], dtype=torch.bool, device=device)
    return model.encode_context(states, actions, initial_boards=initial, clue_masks=mask)[:, 0]


def score_latent(model: CausalTrajectoryJEPA, latent: torch.Tensor, goal_latent: torch.Tensor, score_mode: str) -> float:
    if score_mode == "latent_goal":
        score = F.mse_loss(latent, goal_latent, reduction="none").mean(dim=-1)
    elif score_mode == "goal_energy":
        score = model.predict_goal_energy_from_latents(latent)
    else:
        raise ValueError(f"unknown score_mode {score_mode!r}.")
    return float(score.detach().cpu().item())


def rollout_boards(
    world: SudokuWorld,
    board: np.ndarray,
    actions: list[WorldAction],
    clue_mask: np.ndarray,
    action_mode: str,
) -> tuple[list[np.ndarray], list[WorldAction]]:
    boards: list[np.ndarray] = []
    applied: list[WorldAction] = []
    current = board.copy()
    for action in actions:
        try:
            current = apply_action(world, current, action, clue_mask, action_mode)
        except ValueError:
            break
        boards.append(current.copy())
        applied.append(action)
    return boards, applied


@torch.no_grad()
def predict_latent_rollout(
    model: CausalTrajectoryJEPA,
    current_latent: torch.Tensor,
    world: SudokuWorld,
    actions: list[WorldAction],
    device: torch.device,
) -> torch.Tensor:
    if not actions:
        return current_latent
    if len(actions) in model.horizons:
        chunk = action_array(world, actions, device).view(1, 1, len(actions), 4)
        return model.predict_horizon(current_latent[:, None], chunk, len(actions))[:, 0]
    latent = current_latent
    for action in actions:
        chunk = action_array(world, [action], device).view(1, 1, 1, 4)
        latent = model.predict_horizon(latent[:, None], chunk, 1)[:, 0]
    return latent


@torch.no_grad()
def score_sequence(
    model: CausalTrajectoryJEPA,
    world: SudokuWorld,
    history_boards: list[np.ndarray],
    history_actions: list[WorldAction],
    candidate: list[WorldAction],
    goal: np.ndarray,
    initial_board: np.ndarray,
    clue_mask: np.ndarray,
    *,
    transition_mode: str,
    score_mode: str,
    action_mode: str,
    aggregate: str,
    prefix_horizons: list[int],
    device: torch.device,
) -> tuple[float, np.ndarray, bool]:
    if not candidate:
        current_latent = encode_history(model, world, history_boards, history_actions, initial_board, clue_mask, device)
        goal_latent = encode_goal(model, world, goal, initial_board, clue_mask, device)
        return score_latent(model, current_latent, goal_latent, score_mode), history_boards[-1].copy(), False
    future_boards, applied = rollout_boards(world, history_boards[-1], candidate, clue_mask, action_mode)
    if not applied:
        return float("inf"), history_boards[-1].copy(), False
    goal_latent = encode_goal(model, world, goal, initial_board, clue_mask, device)
    if aggregate == "single":
        horizons = [len(applied)]
    elif aggregate == "mean":
        horizons = [h for h in prefix_horizons if h <= len(applied)]
        if not horizons:
            horizons = [len(applied)]
    else:
        raise ValueError(f"unknown aggregate mode {aggregate!r}.")
    scores = []
    for horizon in horizons:
        prefix_actions = applied[:horizon]
        if transition_mode == "symbolic_reencode":
            prefix_boards = future_boards[:horizon]
            latent = encode_history(
                model,
                world,
                [*history_boards, *prefix_boards],
                [*history_actions, *prefix_actions],
                initial_board,
                clue_mask,
                device,
            )
        elif transition_mode == "latent_rollout":
            current_latent = encode_history(model, world, history_boards, history_actions, initial_board, clue_mask, device)
            latent = predict_latent_rollout(model, current_latent, world, prefix_actions, device)
        else:
            raise ValueError(f"unknown transition_mode {transition_mode!r}.")
        scores.append(score_latent(model, latent, goal_latent, score_mode))
    terminal = bool(np.count_nonzero(future_boards[-1] == 0) == 0)
    return float(np.mean(scores)), future_boards[-1], terminal


def top_actions(
    model: CausalTrajectoryJEPA,
    world: SudokuWorld,
    history_boards: list[np.ndarray],
    history_actions: list[WorldAction],
    goal: np.ndarray,
    initial_board: np.ndarray,
    clue_mask: np.ndarray,
    *,
    transition_mode: str,
    score_mode: str,
    action_mode: str,
    branch_size: int,
    device: torch.device,
) -> list[WorldAction]:
    actions = action_pool(world, history_boards[-1], clue_mask, action_mode)
    if len(actions) <= max(1, branch_size):
        return actions
    scored = []
    for action in actions:
        score, _board, _terminal = score_sequence(
            model,
            world,
            history_boards,
            history_actions,
            [action],
            goal,
            initial_board,
            clue_mask,
            transition_mode=transition_mode,
            score_mode=score_mode,
            action_mode=action_mode,
            aggregate="single",
            prefix_horizons=[1],
            device=device,
        )
        scored.append((score, action))
    scored.sort(key=lambda item: item[0])
    return [action for _score, action in scored[: max(1, branch_size)]]


@torch.no_grad()
def beam_plan_once(
    model: CausalTrajectoryJEPA,
    world: SudokuWorld,
    history_boards: list[np.ndarray],
    history_actions: list[WorldAction],
    goal: np.ndarray,
    initial_board: np.ndarray,
    clue_mask: np.ndarray,
    *,
    horizon: int,
    transition_mode: str,
    score_mode: str,
    action_mode: str,
    aggregate: str,
    prefix_horizons: list[int],
    beam_width: int,
    branch_size: int,
    device: torch.device,
) -> PlanResult:
    frontier: list[tuple[list[WorldAction], float, np.ndarray, bool]] = [([], 0.0, history_boards[-1], False)]
    best = frontier[0]
    for _depth in range(max(1, int(horizon))):
        expanded = []
        for sequence, _score, _leaf, _terminal in frontier:
            boards, actions = rollout_boards(world, history_boards[-1], sequence, clue_mask, action_mode)
            node_history_boards = [*history_boards, *boards]
            node_history_actions = [*history_actions, *actions]
            actions_next = top_actions(
                model,
                world,
                node_history_boards,
                node_history_actions,
                goal,
                initial_board,
                clue_mask,
                transition_mode=transition_mode,
                score_mode=score_mode,
                action_mode=action_mode,
                branch_size=branch_size,
                device=device,
            )
            for action in actions_next:
                candidate = [*sequence, action]
                score, leaf, terminal = score_sequence(
                    model,
                    world,
                    history_boards,
                    history_actions,
                    candidate,
                    goal,
                    initial_board,
                    clue_mask,
                    transition_mode=transition_mode,
                    score_mode=score_mode,
                    action_mode=action_mode,
                    aggregate=aggregate,
                    prefix_horizons=prefix_horizons,
                    device=device,
                )
                expanded.append((candidate, score, leaf, terminal))
        if not expanded:
            break
        expanded.sort(key=lambda item: item[1])
        frontier = expanded[: max(1, beam_width)]
        best = frontier[0]
    sequence, score, leaf, terminal = best
    return PlanResult(
        action=sequence[0] if sequence else None,
        score=float(score),
        leaf_remaining_hamming=int(np.not_equal(leaf, goal).sum()),
        leaf_terminal=bool(terminal),
    )


@torch.no_grad()
def cem_plan_once(
    model: CausalTrajectoryJEPA,
    world: SudokuWorld,
    history_boards: list[np.ndarray],
    history_actions: list[WorldAction],
    goal: np.ndarray,
    initial_board: np.ndarray,
    clue_mask: np.ndarray,
    *,
    horizon: int,
    transition_mode: str,
    score_mode: str,
    action_mode: str,
    aggregate: str,
    prefix_horizons: list[int],
    candidates: int,
    elites: int,
    iterations: int,
    smoothing: float,
    rng: np.random.Generator,
    device: torch.device,
) -> PlanResult:
    mutable = np.argwhere(~clue_mask)
    if len(mutable) == 0:
        return PlanResult(None, float("inf"), int(np.not_equal(history_boards[-1], goal).sum()), False)
    horizon = max(1, int(horizon))
    pos_probs = np.full((horizon, len(mutable)), 1.0 / len(mutable), dtype=np.float64)
    val_probs = np.full((horizon, 9), 1.0 / 9.0, dtype=np.float64)
    best: tuple[list[WorldAction], float, np.ndarray, bool] | None = None
    for _ in range(max(1, int(iterations))):
        sampled: list[tuple[list[WorldAction], float, np.ndarray, bool, list[int], list[int]]] = []
        for _candidate in range(max(1, int(candidates))):
            sequence = []
            pos_indices = []
            value_indices = []
            for step in range(horizon):
                pos_idx = int(rng.choice(len(mutable), p=pos_probs[step]))
                value_idx = int(rng.choice(9, p=val_probs[step]))
                row, col = (int(x) for x in mutable[pos_idx])
                sequence.append(WorldAction(row, col, value_idx + 1))
                pos_indices.append(pos_idx)
                value_indices.append(value_idx)
            score, leaf, terminal = score_sequence(
                model,
                world,
                history_boards,
                history_actions,
                sequence,
                goal,
                initial_board,
                clue_mask,
                transition_mode=transition_mode,
                score_mode=score_mode,
                action_mode=action_mode,
                aggregate=aggregate,
                prefix_horizons=prefix_horizons,
                device=device,
            )
            sampled.append((sequence, score, leaf, terminal, pos_indices, value_indices))
        sampled.sort(key=lambda item: item[1])
        if best is None or sampled[0][1] < best[1]:
            best = (sampled[0][0], sampled[0][1], sampled[0][2], sampled[0][3])
        elite = sampled[: max(1, min(int(elites), len(sampled)))]
        new_pos = np.full_like(pos_probs, 1.0e-3)
        new_val = np.full_like(val_probs, 1.0e-3)
        for _sequence, _score, _leaf, _terminal, pos_indices, value_indices in elite:
            for step, pos_idx in enumerate(pos_indices):
                new_pos[step, pos_idx] += 1.0
            for step, value_idx in enumerate(value_indices):
                new_val[step, value_idx] += 1.0
        new_pos /= new_pos.sum(axis=1, keepdims=True)
        new_val /= new_val.sum(axis=1, keepdims=True)
        pos_probs = float(smoothing) * new_pos + (1.0 - float(smoothing)) * pos_probs
        val_probs = float(smoothing) * new_val + (1.0 - float(smoothing)) * val_probs
    if best is None:
        return PlanResult(None, float("inf"), int(np.not_equal(history_boards[-1], goal).sum()), False)
    sequence, score, leaf, terminal = best
    return PlanResult(sequence[0] if sequence else None, float(score), int(np.not_equal(leaf, goal).sum()), bool(terminal))


@torch.no_grad()
def mcts_plan_once(
    model: CausalTrajectoryJEPA,
    world: SudokuWorld,
    history_boards: list[np.ndarray],
    history_actions: list[WorldAction],
    goal: np.ndarray,
    initial_board: np.ndarray,
    clue_mask: np.ndarray,
    *,
    horizon: int,
    transition_mode: str,
    score_mode: str,
    action_mode: str,
    aggregate: str,
    prefix_horizons: list[int],
    simulations: int,
    branch_size: int,
    rng: np.random.Generator,
    device: torch.device,
) -> PlanResult:
    # Small diagnostic MCTS: sample branch-pruned rollouts and keep the best first action.
    best: tuple[list[WorldAction], float, np.ndarray, bool] | None = None
    for _ in range(max(1, int(simulations))):
        board = history_boards[-1].copy()
        sequence: list[WorldAction] = []
        for _depth in range(max(1, int(horizon))):
            boards, actions = rollout_boards(world, history_boards[-1], sequence, clue_mask, action_mode)
            node_history_boards = [*history_boards, *boards] if boards else history_boards
            node_history_actions = [*history_actions, *actions]
            choices = top_actions(
                model,
                world,
                node_history_boards,
                node_history_actions,
                goal,
                initial_board,
                clue_mask,
                transition_mode=transition_mode,
                score_mode=score_mode,
                action_mode=action_mode,
                branch_size=branch_size,
                device=device,
            )
            if not choices:
                break
            action = choices[int(rng.integers(0, len(choices)))]
            try:
                board = apply_action(world, board, action, clue_mask, action_mode)
            except ValueError:
                break
            sequence.append(action)
        score, leaf, terminal = score_sequence(
            model,
            world,
            history_boards,
            history_actions,
            sequence,
            goal,
            initial_board,
            clue_mask,
            transition_mode=transition_mode,
            score_mode=score_mode,
            action_mode=action_mode,
            aggregate=aggregate,
            prefix_horizons=prefix_horizons,
            device=device,
        )
        if best is None or score < best[1]:
            best = (sequence, score, leaf, terminal)
    if best is None:
        return PlanResult(None, float("inf"), int(np.not_equal(history_boards[-1], goal).sum()), False)
    sequence, score, leaf, terminal = best
    return PlanResult(sequence[0] if sequence else None, float(score), int(np.not_equal(leaf, goal).sum()), bool(terminal))


@torch.no_grad()
def run_closed_loop(
    model: CausalTrajectoryJEPA,
    world: SudokuWorld,
    example,
    *,
    planner: str,
    transition_mode: str,
    score_mode: str,
    action_mode: str,
    aggregate: str,
    horizon: int,
    prefix_horizons: list[int],
    max_steps: int,
    beam_width: int,
    branch_size: int,
    cem_candidates: int,
    cem_elites: int,
    cem_iterations: int,
    cem_smoothing: float,
    mcts_simulations: int,
    rng: np.random.Generator,
    device: torch.device,
) -> dict[str, Any]:
    clue_mask = world.clue_mask_from_puzzle(example.state)
    board = example.state.copy()
    history_boards = [board.copy()]
    history_actions: list[WorldAction] = []
    start_hamming = int(np.not_equal(board, example.goal).sum())
    first_root_goal = False
    first_leaf_remaining = None
    first_score = None
    steps = 0
    for step in range(max(1, int(max_steps))):
        if world.is_goal(board, example.goal):
            break
        plan_kwargs = dict(
            model=model,
            world=world,
            history_boards=history_boards,
            history_actions=history_actions,
            goal=example.goal,
            initial_board=example.state,
            clue_mask=clue_mask,
            horizon=min(max(1, int(horizon)), max(1, int(max_steps) - step)),
            transition_mode=transition_mode,
            score_mode=score_mode,
            action_mode=action_mode,
            aggregate=aggregate,
            prefix_horizons=prefix_horizons,
            device=device,
        )
        if planner == "beam":
            result = beam_plan_once(**plan_kwargs, beam_width=beam_width, branch_size=branch_size)
        elif planner == "cem":
            result = cem_plan_once(
                **plan_kwargs,
                candidates=cem_candidates,
                elites=cem_elites,
                iterations=cem_iterations,
                smoothing=cem_smoothing,
                rng=rng,
            )
        elif planner == "mcts":
            result = mcts_plan_once(**plan_kwargs, simulations=mcts_simulations, branch_size=branch_size, rng=rng)
        else:
            raise ValueError(f"unknown planner {planner!r}.")
        action = result.action
        if action is None:
            break
        if step == 0:
            first_root_goal = int(example.goal[action.row, action.col]) == int(action.value)
            first_leaf_remaining = int(result.leaf_remaining_hamming)
            first_score = float(result.score)
        try:
            board = apply_action(world, board, action, clue_mask, action_mode)
        except ValueError:
            break
        history_actions.append(action)
        history_boards.append(board.copy())
        steps += 1
    return {
        "start_hamming": start_hamming,
        "remaining_hamming": int(np.not_equal(board, example.goal).sum()),
        "steps": int(steps),
        "terminal": bool(np.count_nonzero(board == 0) == 0),
        "solved": bool(world.is_goal(board, example.goal)),
        "root_goal_value": bool(first_root_goal),
        "root_leaf_remaining_hamming": first_leaf_remaining,
        "root_score": first_score,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["mode"], []).append(record)
    summary = {}
    for mode, items in sorted(grouped.items()):
        summary[mode] = {
            "count": float(len(items)),
            "solve_rate": float(np.mean([item["solved"] for item in items])) if items else 0.0,
            "terminal_rate": float(np.mean([item["terminal"] for item in items])) if items else 0.0,
            "mean_remaining_hamming": float(np.mean([item["remaining_hamming"] for item in items])) if items else math.nan,
            "root_goal_value_rate": float(np.mean([item["root_goal_value"] for item in items])) if items else 0.0,
            "mean_root_leaf_remaining_hamming": float(
                np.mean([item["root_leaf_remaining_hamming"] for item in items if item["root_leaf_remaining_hamming"] is not None])
            )
            if any(item["root_leaf_remaining_hamming"] is not None for item in items)
            else math.nan,
        }
    return summary


def run_grid6_planner_matrix(
    *,
    run_root: Path,
    output_dir: Path,
    seed: int,
    examples: int,
    planners: list[str],
    transition_modes: list[str],
    score_modes: list[str],
    single_horizons: list[int],
    mean_horizons: list[int],
    action_mode: str,
    max_steps: int,
    beam_width: int,
    branch_size: int,
    cem_candidates: int,
    cem_elites: int,
    cem_iterations: int,
    cem_smoothing: float,
    mcts_simulations: int,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    model, config = load_model(run_root, device)
    world = _build_world(dict(config["task"]))
    if not isinstance(world, SudokuWorld):
        raise TypeError("Grid6 planner currently supports SudokuWorld.")
    eval_examples = _load_examples(dict(config["task"]), "eval")[: int(examples)]
    rng = np.random.default_rng(int(seed))
    records: list[dict[str, Any]] = []
    modes: list[tuple[str, int, list[int]]] = [(f"h{h}", int(h), [int(h)]) for h in single_horizons]
    if mean_horizons:
        modes.append(("mean_" + "_".join(str(h) for h in mean_horizons), max(mean_horizons), [int(h) for h in mean_horizons]))
    for planner in planners:
        for transition_mode in transition_modes:
            for score_mode in score_modes:
                for aggregate_name, horizon, prefixes in modes:
                    aggregate = "mean" if aggregate_name.startswith("mean_") else "single"
                    mode = f"{planner}_{transition_mode}_{score_mode}_{aggregate_name}"
                    for index, example in enumerate(eval_examples):
                        result = run_closed_loop(
                            model,
                            world,
                            example,
                            planner=planner,
                            transition_mode=transition_mode,
                            score_mode=score_mode,
                            action_mode=action_mode,
                            aggregate=aggregate,
                            horizon=horizon,
                            prefix_horizons=prefixes,
                            max_steps=max_steps,
                            beam_width=beam_width,
                            branch_size=branch_size,
                            cem_candidates=cem_candidates,
                            cem_elites=cem_elites,
                            cem_iterations=cem_iterations,
                            cem_smoothing=cem_smoothing,
                            mcts_simulations=mcts_simulations,
                            rng=rng,
                            device=device,
                        )
                        record = {
                            "example_index": index,
                            "mode": mode,
                            "planner": planner,
                            "transition_mode": transition_mode,
                            "score_mode": score_mode,
                            "aggregate": aggregate,
                            "horizon": horizon,
                            **result,
                        }
                        records.append(record)
                    write_outputs(output_dir, records, summarize(records))
    summary = {
        "run_root": str(run_root),
        "model": {
            "d_model": int(config["model"]["d_model"]),
            "horizons": list(config["model"]["horizons"]),
            "encoder_layers": int(config["model"]["encoder_layers"]),
            "predictor_layers": int(config["model"]["predictor_layers"]),
            "action_chunk_layers": int(config["model"]["action_chunk_layers"]),
            "action_dim": int(config["model"]["action_dim"]),
        },
        "planner": summarize(records),
    }
    write_outputs(output_dir, records, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=13001)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--planners", nargs="+", default=["beam", "cem", "mcts"])
    parser.add_argument("--transition-modes", nargs="+", default=["symbolic_reencode", "latent_rollout"])
    parser.add_argument("--score-modes", nargs="+", default=["latent_goal", "goal_energy"])
    parser.add_argument("--single-horizons", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--mean-horizons", nargs="*", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--action-mode", choices=["mutable_overwrite", "fill_empty"], default="mutable_overwrite")
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--branch-size", type=int, default=8)
    parser.add_argument("--cem-candidates", type=int, default=96)
    parser.add_argument("--cem-elites", type=int, default=12)
    parser.add_argument("--cem-iterations", type=int, default=4)
    parser.add_argument("--cem-smoothing", type=float, default=0.7)
    parser.add_argument("--mcts-simulations", type=int, default=64)
    args = parser.parse_args()
    summary = run_grid6_planner_matrix(
        run_root=args.run_root,
        output_dir=args.output_dir,
        seed=args.seed,
        examples=args.examples,
        planners=list(args.planners),
        transition_modes=list(args.transition_modes),
        score_modes=list(args.score_modes),
        single_horizons=list(args.single_horizons),
        mean_horizons=list(args.mean_horizons),
        action_mode=args.action_mode,
        max_steps=args.max_steps,
        beam_width=args.beam_width,
        branch_size=args.branch_size,
        cem_candidates=args.cem_candidates,
        cem_elites=args.cem_elites,
        cem_iterations=args.cem_iterations,
        cem_smoothing=args.cem_smoothing,
        mcts_simulations=args.mcts_simulations,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
