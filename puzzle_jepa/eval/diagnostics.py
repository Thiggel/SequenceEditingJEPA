from __future__ import annotations

import argparse
import json
import math
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from puzzle_jepa.data.worlds import MazeWorld, PuzzleExample, PuzzleWorld, SudokuWorld, WorldAction
from puzzle_jepa.models import ActionConditionedWorldModel


@dataclass(frozen=True, slots=True)
class RankRecord:
    depth_fraction: float
    prefix_steps: int
    trajectory_length: int
    rank: int
    legal_actions: int
    oracle_score: float
    best_score: float


@dataclass(frozen=True, slots=True)
class DriftRecord:
    example_index: int
    step: int
    trajectory_length: int
    terminal: bool
    latent_drift_mse: float
    predicted_goal_mse: float
    true_goal_mse: float
    energy_gap_mse: float


def run_diagnostics(
    run_root: Path,
    *,
    rank_examples: int,
    drift_examples: int,
    planning_examples: int,
    planning_beam_size: int,
    planning_branch_size: int,
    max_unroll_steps: int,
    horizons: list[int],
    depth_fractions: list[float],
    trace_examples: int,
    seed: int,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    checkpoint_path = run_root / "checkpoint.pt"
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    task_cfg = dict(config["task"])
    model_cfg = dict(config["model"])
    world = build_world(task_cfg)
    examples = load_examples(task_cfg, "eval")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActionConditionedWorldModel(vocab_size=world.vocab_size, **model_cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    rank_records = evaluate_action_ranks(
        model,
        world,
        examples,
        rng,
        num_examples=rank_examples,
        depth_fractions=depth_fractions,
    )
    drift_records, drift_summary = evaluate_latent_drift(
        model,
        world,
        examples,
        rng,
        num_examples=drift_examples,
        horizons=horizons,
        max_unroll_steps=max_unroll_steps,
    )
    planning = evaluate_latent_planning(
        model,
        world,
        examples,
        rng,
        num_examples=planning_examples,
        max_steps=max_unroll_steps,
        branch_size=planning_branch_size,
        beam_size=planning_beam_size,
    )
    traces = collect_planner_failure_traces(
        model,
        world,
        examples,
        rng,
        num_examples=trace_examples,
        max_steps=min(12, max_unroll_steps),
    )
    result = {
        "run_root": str(run_root),
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "task": world.name,
        "rank": summarize_rank_records(rank_records),
        "drift": drift_summary,
        "planning": planning,
        "planner_traces": traces,
    }
    destination = output_dir or (run_root / "diagnostics")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "diagnostics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (destination / "rank_records.jsonl").open("w") as handle:
        for record in rank_records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    with (destination / "drift_records.jsonl").open("w") as handle:
        for record in drift_records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    write_drift_plot(destination / "latent_energy_mse.png", drift_records)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


@torch.no_grad()
def evaluate_action_ranks(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    examples: list[PuzzleExample],
    rng: np.random.Generator,
    *,
    num_examples: int,
    depth_fractions: list[float],
) -> list[RankRecord]:
    records: list[RankRecord] = []
    if num_examples <= 0:
        return records
    for _ in range(num_examples):
        example = examples[int(rng.integers(0, len(examples)))]
        actions = oracle_action_sequence(world, example, rng)
        if not actions:
            continue
        for fraction in depth_fractions:
            prefix_steps = min(int(round(float(fraction) * (len(actions) - 1))), len(actions) - 1)
            clue_mask = clue_mask_for_planning(world, example.state)
            state = apply_prefix(world, example.state, actions, prefix_steps, clue_mask)
            oracle_action = actions[prefix_steps]
            rank = rank_action(model, world, state, example.goal, oracle_action, clue_mask)
            if rank is None:
                continue
            records.append(
                RankRecord(
                    depth_fraction=float(fraction),
                    prefix_steps=prefix_steps,
                    trajectory_length=len(actions),
                    **rank,
                )
            )
    return records


@torch.no_grad()
def evaluate_latent_drift(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    examples: list[PuzzleExample],
    rng: np.random.Generator,
    *,
    num_examples: int,
    horizons: list[int],
    max_unroll_steps: int,
) -> tuple[list[DriftRecord], dict[str, Any]]:
    records: list[DriftRecord] = []
    per_trace: list[dict[str, float]] = []
    if num_examples <= 0:
        return records, {}
    device = next(model.parameters()).device
    task_ids = torch.full((1,), world.task_id, dtype=torch.long, device=device)
    requested = {max(0, int(horizon)) for horizon in horizons}
    requested.discard(0)
    for example_index in range(num_examples):
        example = examples[int(rng.integers(0, len(examples)))]
        actions = oracle_action_sequence(world, example, rng)
        if not actions:
            continue
        clue_mask = clue_mask_for_planning(world, example.state)
        steps_to_run = min(len(actions), int(max_unroll_steps))
        record_steps = {step for step in requested if step <= steps_to_run}
        record_steps.add(steps_to_run)
        current_state = world.validate_state(example.state).copy()
        goal = world.validate_state(example.goal)
        state_tensor = torch.as_tensor(current_state[None, ...], dtype=torch.long, device=device)
        goal_tensor = torch.as_tensor(goal[None, ...], dtype=torch.long, device=device)
        latent = model.encoder(state_tensor, task_ids=task_ids)
        goal_latent = model.target_encoder(goal_tensor, task_ids=task_ids)
        predicted_energies = []
        true_energies = []
        for step, action in enumerate(actions[:steps_to_run], start=1):
            action_tensor = torch.as_tensor(
                [[world.task_id, action.row, action.col, action.value]],
                dtype=torch.long,
                device=device,
            )
            latent = model.predict_latent_from_latent(
                latent,
                action_tensor,
                height=world.height,
                width=world.width,
            )
            current_state = apply_planning_action(world, current_state, action, clue_mask)
            true_tensor = torch.as_tensor(current_state[None, ...], dtype=torch.long, device=device)
            true_latent = model.target_encoder(true_tensor, task_ids=task_ids)
            drift = float(F.mse_loss(latent, true_latent).detach().cpu().item())
            pred_energy = float(F.mse_loss(latent, goal_latent).detach().cpu().item())
            true_energy = float(F.mse_loss(true_latent, goal_latent).detach().cpu().item())
            predicted_energies.append(pred_energy)
            true_energies.append(true_energy)
            if step in record_steps:
                records.append(
                    DriftRecord(
                        example_index=example_index,
                        step=step,
                        trajectory_length=len(actions),
                        terminal=step == len(actions),
                        latent_drift_mse=drift,
                        predicted_goal_mse=pred_energy,
                        true_goal_mse=true_energy,
                        energy_gap_mse=pred_energy - true_energy,
                    )
                )
        per_trace.append(
            {
                "terminal_reached": float(steps_to_run == len(actions)),
                "predicted_energy_monotone_rate": monotone_nonincreasing_rate(predicted_energies),
                "true_energy_monotone_rate": monotone_nonincreasing_rate(true_energies),
                "predicted_energy_spearman_with_step": spearman_with_steps(predicted_energies),
                "true_energy_spearman_with_step": spearman_with_steps(true_energies),
            }
        )
    return records, summarize_drift_records(records, per_trace)


@torch.no_grad()
def evaluate_latent_planning(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    examples: list[PuzzleExample],
    rng: np.random.Generator,
    *,
    num_examples: int,
    max_steps: int,
    branch_size: int,
    beam_size: int,
) -> dict[str, Any]:
    if num_examples <= 0:
        return {}
    summaries = {
        "step_energy": [],
        "terminal_energy": [],
    }
    for _ in range(num_examples):
        example = examples[int(rng.integers(0, len(examples)))]
        for mode in summaries:
            summaries[mode].append(
                latent_beam_plan(
                    model,
                    world,
                    example,
                    max_steps=max_steps,
                    branch_size=branch_size,
                    beam_size=beam_size,
                    terminal_only_score=(mode == "terminal_energy"),
                )
            )
    return {
        mode: summarize_plan_summaries(items)
        for mode, items in summaries.items()
    }


@torch.no_grad()
def latent_beam_plan(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    example: PuzzleExample,
    *,
    max_steps: int,
    branch_size: int,
    beam_size: int,
    terminal_only_score: bool,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    start = world.validate_state(example.state).copy()
    goal = world.validate_state(example.goal)
    clue_mask = clue_mask_for_planning(world, start)
    task_ids = torch.full((1,), world.task_id, dtype=torch.long, device=device)
    goal_tensor = torch.as_tensor(goal[None, ...], dtype=torch.long, device=device)
    goal_latent = model.target_encoder(goal_tensor, task_ids=task_ids)
    start_tensor = torch.as_tensor(start[None, ...], dtype=torch.long, device=device)
    start_latent = model.encoder(start_tensor, task_ids=task_ids)
    active = [
        {
            "state": start,
            "latent": start_latent,
            "steps": 0,
            "energy": float(F.mse_loss(start_latent, goal_latent).detach().cpu().item()),
        }
    ]
    terminals = []
    limit = min(int(max_steps), terminal_step_limit(world, example, max_steps))
    for _ in range(limit):
        candidates = []
        for beam in active:
            state = beam["state"]
            if is_terminal_state(world, state, goal, clue_mask):
                terminals.append(beam)
                continue
            legal = legal_planning_actions(world, state, clue_mask)
            if not legal:
                continue
            scores = model.score_actions_to_goal(
                torch.as_tensor(state, dtype=torch.long),
                legal,
                torch.as_tensor(goal, dtype=torch.long),
                world.task_id,
            )
            order = scores.argsort(descending=True).detach().cpu().tolist()[: max(1, branch_size)]
            for index in order:
                action = legal[index]
                try:
                    next_state = apply_planning_action(world, state, action, clue_mask)
                except ValueError:
                    continue
                action_tensor = torch.as_tensor(
                    [[world.task_id, action.row, action.col, action.value]],
                    dtype=torch.long,
                    device=device,
                )
                next_latent = model.predict_latent_from_latent(
                    beam["latent"],
                    action_tensor,
                    height=world.height,
                    width=world.width,
                )
                energy = float(F.mse_loss(next_latent, goal_latent).detach().cpu().item())
                next_beam = {
                    "state": next_state,
                    "latent": next_latent,
                    "steps": int(beam["steps"]) + 1,
                    "energy": energy,
                }
                if is_terminal_state(world, next_state, goal, clue_mask):
                    terminals.append(next_beam)
                else:
                    candidates.append(next_beam)
        if not candidates:
            break
        candidates.sort(key=lambda item: float(item["energy"]))
        active = candidates[: max(1, beam_size)]
        if not terminal_only_score and any(world.is_goal(item["state"], goal) for item in active):
            break

    scored = terminals if terminal_only_score and terminals else [*terminals, *active]
    if not scored:
        scored = active
    best = min(scored, key=lambda item: float(item["energy"]))
    return {
        "solved": float(world.is_goal(best["state"], goal)),
        "terminal": float(is_terminal_state(world, best["state"], goal, clue_mask)),
        "steps": float(best["steps"]),
        "energy": float(best["energy"]),
        "remaining_hamming": float(np.not_equal(best["state"], goal).sum()),
    }


def summarize_plan_summaries(items: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "count": float(len(items)),
        "solve_rate": mean([item["solved"] for item in items]),
        "terminal_rate": mean([item["terminal"] for item in items]),
        "mean_steps": mean([item["steps"] for item in items]),
        "mean_energy": mean([item["energy"] for item in items]),
        "mean_remaining_hamming": mean([item["remaining_hamming"] for item in items]),
    }


def terminal_step_limit(world: PuzzleWorld, example: PuzzleExample, fallback: int) -> int:
    if isinstance(world, SudokuWorld):
        return int((example.state == 0).sum())
    if isinstance(world, MazeWorld):
        return max(1, int(((example.state == world.EMPTY) & (example.goal == world.PATH)).sum()))
    return int(fallback)


def is_terminal_state(
    world: PuzzleWorld,
    state: np.ndarray,
    goal: np.ndarray,
    clue_mask: np.ndarray | None,
) -> bool:
    if isinstance(world, SudokuWorld):
        if clue_mask is None:
            raise ValueError("Sudoku terminal check requires a clue mask.")
        return not np.any(world.validate_state(state)[~clue_mask] == 0)
    return world.is_goal(state, goal)


@torch.no_grad()
def collect_planner_failure_traces(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    examples: list[PuzzleExample],
    rng: np.random.Generator,
    *,
    num_examples: int,
    max_steps: int,
) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    for _ in range(num_examples):
        example = examples[int(rng.integers(0, len(examples)))]
        oracle_actions = oracle_action_sequence(world, example, rng)
        clue_mask = clue_mask_for_planning(world, example.state)
        current = world.validate_state(example.state).copy()
        events = []
        for step in range(min(max_steps, len(oracle_actions))):
            legal = legal_planning_actions(world, current, clue_mask)
            if not legal or world.is_goal(current, example.goal):
                break
            scores = model.score_actions_to_goal(
                torch.as_tensor(current, dtype=torch.long),
                legal,
                torch.as_tensor(example.goal, dtype=torch.long),
                world.task_id,
            )
            order = scores.argsort(descending=True).detach().cpu().tolist()
            chosen = legal[order[0]]
            oracle_action = oracle_actions[step]
            rank = None
            if oracle_action in legal:
                rank = order.index(legal.index(oracle_action)) + 1
            events.append(
                {
                    "step": step,
                    "chosen": asdict(chosen),
                    "oracle": asdict(oracle_action),
                    "oracle_rank": rank,
                    "legal_actions": len(legal),
                    "best_score": float(scores[order[0]].detach().cpu().item()),
                }
            )
            try:
                current = apply_planning_action(world, current, chosen, clue_mask)
            except ValueError:
                break
        traces.append(
            {
                "solved": world.is_goal(current, example.goal),
                "steps": len(events),
                "remaining_hamming": int(np.not_equal(current, example.goal).sum()),
                "events": events,
            }
        )
    return traces


def rank_action(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    state: np.ndarray,
    goal: np.ndarray,
    oracle_action: WorldAction,
    clue_mask: np.ndarray | None,
) -> dict[str, Any] | None:
    legal = legal_planning_actions(world, state, clue_mask)
    if oracle_action not in legal:
        return None
    scores = model.score_actions_to_goal(
        torch.as_tensor(state, dtype=torch.long),
        legal,
        torch.as_tensor(goal, dtype=torch.long),
        world.task_id,
    )
    order = scores.argsort(descending=True).detach().cpu().tolist()
    oracle_index = legal.index(oracle_action)
    rank = order.index(oracle_index) + 1
    return {
        "rank": int(rank),
        "legal_actions": len(legal),
        "oracle_score": float(scores[oracle_index].detach().cpu().item()),
        "best_score": float(scores[order[0]].detach().cpu().item()),
    }


def build_world(task_cfg: dict[str, Any]) -> PuzzleWorld:
    name = str(task_cfg["name"])
    if name == "sudoku":
        return SudokuWorld()
    if name == "maze":
        return MazeWorld(height=int(task_cfg.get("height", 30)), width=int(task_cfg.get("width", 30)))
    raise ValueError(f"Unknown task {name!r}.")


def load_examples(task_cfg: dict[str, Any], split_key: str) -> list[PuzzleExample]:
    from puzzle_jepa.data.hf_puzzles import HFPuzzleColumns, iter_hf_examples

    split = str(task_cfg[f"{split_key}_split"])
    limit = task_cfg.get(f"{split_key}_limit")
    return list(
        iter_hf_examples(
            repo_id=str(task_cfg["repo_id"]),
            split=split,
            world=build_world(task_cfg),
            columns=HFPuzzleColumns(
                puzzle=str(task_cfg.get("puzzle_column", "question")),
                solution=str(task_cfg.get("solution_column", "answer")),
            ),
            limit=None if limit is None else int(limit),
        )
    )


def oracle_action_sequence(
    world: PuzzleWorld,
    example: PuzzleExample,
    rng: np.random.Generator,
) -> list[WorldAction]:
    if isinstance(world, SudokuWorld):
        positions = np.argwhere(example.state == 0)
        if len(positions) == 0:
            return []
        order = rng.permutation(len(positions))
        return [
            WorldAction(int(positions[index, 0]), int(positions[index, 1]), int(example.goal[positions[index, 0], positions[index, 1]]))
            for index in order
        ]
    if isinstance(world, MazeWorld):
        return maze_path_actions(world, example)
    raise TypeError(f"Unsupported world type {type(world).__name__}.")


def maze_path_actions(world: MazeWorld, example: PuzzleExample) -> list[WorldAction]:
    puzzle = world.validate_state(example.state)
    goal = world.validate_state(example.goal)
    starts = np.argwhere(goal == world.START)
    goals = np.argwhere(goal == world.GOAL)
    if len(starts) != 1 or len(goals) != 1:
        return fallback_maze_path_actions(world, puzzle, goal)
    start = tuple(int(x) for x in starts[0])
    end = tuple(int(x) for x in goals[0])
    passable = {world.START, world.GOAL, world.PATH}
    queue: deque[tuple[int, int]] = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while queue:
        row, col = queue.popleft()
        if (row, col) == end:
            break
        for next_row, next_col in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            node = (next_row, next_col)
            if not (0 <= next_row < world.height and 0 <= next_col < world.width):
                continue
            if node in parent or int(goal[next_row, next_col]) not in passable:
                continue
            parent[node] = (row, col)
            queue.append(node)
    if end not in parent:
        return fallback_maze_path_actions(world, puzzle, goal)
    path = []
    node: tuple[int, int] | None = end
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return [
        WorldAction(row, col, world.PATH)
        for row, col in path
        if int(puzzle[row, col]) == world.EMPTY and int(goal[row, col]) == world.PATH
    ]


def fallback_maze_path_actions(world: MazeWorld, puzzle: np.ndarray, goal: np.ndarray) -> list[WorldAction]:
    positions = np.argwhere((puzzle == world.EMPTY) & (goal == world.PATH))
    return [WorldAction(int(row), int(col), world.PATH) for row, col in positions]


def apply_prefix(
    world: PuzzleWorld,
    state: np.ndarray,
    actions: list[WorldAction],
    prefix_steps: int,
    clue_mask: np.ndarray | None,
) -> np.ndarray:
    current = world.validate_state(state).copy()
    for action in actions[:prefix_steps]:
        current = apply_planning_action(world, current, action, clue_mask)
    return current


def clue_mask_for_planning(world: PuzzleWorld, initial_state: np.ndarray) -> np.ndarray | None:
    if isinstance(world, SudokuWorld):
        return world.clue_mask_from_puzzle(initial_state)
    return None


def legal_planning_actions(
    world: PuzzleWorld,
    state: np.ndarray,
    clue_mask: np.ndarray | None,
) -> list[WorldAction]:
    if isinstance(world, SudokuWorld):
        if clue_mask is None:
            raise ValueError("Sudoku planning requires a clue mask.")
        return world.legal_actions(
            state,
            clue_mask=clue_mask,
            allow_overwrite=True,
            allow_conflicts=True,
        )
    return world.legal_actions(state)


def apply_planning_action(
    world: PuzzleWorld,
    state: np.ndarray,
    action: WorldAction,
    clue_mask: np.ndarray | None,
) -> np.ndarray:
    if isinstance(world, SudokuWorld):
        if clue_mask is None:
            raise ValueError("Sudoku planning requires a clue mask.")
        return world.apply(
            state,
            action,
            clue_mask=clue_mask,
            allow_overwrite=True,
            allow_conflicts=True,
        )
    return world.apply(state, action)


def summarize_rank_records(records: list[RankRecord]) -> dict[str, Any]:
    by_depth: dict[str, dict[str, float]] = {}
    for depth in sorted({record.depth_fraction for record in records}):
        depth_records = [record for record in records if record.depth_fraction == depth]
        ranks = [record.rank for record in depth_records]
        by_depth[str(depth)] = {
            "count": len(ranks),
            "top1": mean([float(rank == 1) for rank in ranks]),
            "top5": mean([float(rank <= 5) for rank in ranks]),
            "mrr": mean([1.0 / rank for rank in ranks]),
            "mean_rank": mean(ranks),
            "median_rank": median(ranks),
            "mean_legal_actions": mean([record.legal_actions for record in depth_records]),
            "mean_score_margin": mean([record.best_score - record.oracle_score for record in depth_records]),
        }
    ranks = [record.rank for record in records]
    return {
        "count": len(records),
        "top1": mean([float(rank == 1) for rank in ranks]),
        "top5": mean([float(rank <= 5) for rank in ranks]),
        "mrr": mean([1.0 / rank for rank in ranks]),
        "mean_rank": mean(ranks),
        "median_rank": median(ranks),
        "by_depth_fraction": by_depth,
    }


def summarize_drift_records(records: list[DriftRecord], per_trace: list[dict[str, float]]) -> dict[str, Any]:
    by_step: dict[str, dict[str, float]] = {}
    for step in sorted({record.step for record in records}):
        step_records = [record for record in records if record.step == step]
        by_step[str(step)] = {
            "count": len(step_records),
            "latent_drift_mse": mean([record.latent_drift_mse for record in step_records]),
            "predicted_goal_mse": mean([record.predicted_goal_mse for record in step_records]),
            "true_goal_mse": mean([record.true_goal_mse for record in step_records]),
            "energy_gap_mse": mean([record.energy_gap_mse for record in step_records]),
            "terminal_rate": mean([float(record.terminal) for record in step_records]),
        }
    return {
        "count": len(records),
        "by_step": by_step,
        "terminal_reached_rate": mean([item["terminal_reached"] for item in per_trace]),
        "predicted_energy_monotone_rate": mean([item["predicted_energy_monotone_rate"] for item in per_trace]),
        "true_energy_monotone_rate": mean([item["true_energy_monotone_rate"] for item in per_trace]),
        "predicted_energy_spearman_with_step": mean([item["predicted_energy_spearman_with_step"] for item in per_trace]),
        "true_energy_spearman_with_step": mean([item["true_energy_spearman_with_step"] for item in per_trace]),
    }


def write_drift_plot(path: Path, records: list[DriftRecord]) -> None:
    if not records:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on cluster image.
        path.with_suffix(".plot_error.txt").write_text(f"matplotlib unavailable: {exc}\n")
        return
    steps = sorted({record.step for record in records})
    pred = [mean([record.predicted_goal_mse for record in records if record.step == step]) for step in steps]
    true = [mean([record.true_goal_mse for record in records if record.step == step]) for step in steps]
    drift = [mean([record.latent_drift_mse for record in records if record.step == step]) for step in steps]
    plt.figure(figsize=(7, 4))
    plt.plot(steps, pred, marker="o", label="predicted latent to goal")
    plt.plot(steps, true, marker="o", label="true encoded state to goal")
    plt.plot(steps, drift, marker="o", label="predicted latent drift")
    plt.xlabel("oracle unroll step")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def monotone_nonincreasing_rate(values: list[float]) -> float:
    if len(values) < 2:
        return math.nan
    return mean([float(values[index + 1] <= values[index]) for index in range(len(values) - 1)])


def spearman_with_steps(values: list[float]) -> float:
    if len(values) < 2:
        return math.nan
    steps = np.arange(1, len(values) + 1, dtype=np.float64)
    value_ranks = rankdata(np.asarray(values, dtype=np.float64))
    step_ranks = rankdata(steps)
    corr = np.corrcoef(step_ranks, value_ranks)[0, 1]
    return float(corr) if np.isfinite(corr) else math.nan


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    return ranks


def mean(values: list[float] | list[int]) -> float:
    clean = [float(value) for value in values if not math.isnan(float(value))]
    return float(np.mean(clean)) if clean else math.nan


def median(values: list[float] | list[int]) -> float:
    clean = [float(value) for value in values if not math.isnan(float(value))]
    return float(np.median(clean)) if clean else math.nan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Grid 1 JEPA planner/scorer diagnostics.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--rank-examples", type=int, default=64)
    parser.add_argument("--drift-examples", type=int, default=8)
    parser.add_argument("--planning-examples", type=int, default=4)
    parser.add_argument("--planning-beam-size", type=int, default=4)
    parser.add_argument("--planning-branch-size", type=int, default=8)
    parser.add_argument("--trace-examples", type=int, default=3)
    parser.add_argument("--max-unroll-steps", type=int, default=256)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 10, 20])
    parser.add_argument("--depth-fractions", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75])
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_diagnostics(
        args.run_root,
        rank_examples=args.rank_examples,
        drift_examples=args.drift_examples,
        planning_examples=args.planning_examples,
        planning_beam_size=args.planning_beam_size,
        planning_branch_size=args.planning_branch_size,
        max_unroll_steps=args.max_unroll_steps,
        horizons=args.horizons,
        depth_fractions=args.depth_fractions,
        trace_examples=args.trace_examples,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
