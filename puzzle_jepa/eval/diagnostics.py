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
class GoalRankRecord:
    depth_fraction: float
    prefix_steps: int
    trajectory_length: int
    best_goal_rank: int
    goal_actions: int
    legal_actions: int
    best_goal_score: float
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
    cem_examples: int,
    planning_beam_size: int,
    planning_branch_size: int,
    cem_population: int,
    cem_elite_frac: float,
    cem_iterations: int,
    cem_smoothing: float,
    cem_score: str,
    max_unroll_steps: int,
    horizons: list[int],
    depth_fractions: list[float],
    trace_examples: int,
    seed: int,
    reset_cadences: list[int] | None = None,
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
    goal_rank_records = evaluate_goal_action_ranks(
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
    planning, planning_records = evaluate_latent_planning(
        model,
        world,
        examples,
        rng,
        num_examples=planning_examples,
        max_steps=max_unroll_steps,
        branch_size=planning_branch_size,
        beam_size=planning_beam_size,
    )
    reencoded_planning, reencoded_planning_records = evaluate_reencoded_planning(
        model,
        world,
        examples,
        rng,
        num_examples=planning_examples,
        max_steps=max_unroll_steps,
        branch_size=planning_branch_size,
        beam_size=planning_beam_size,
    )
    paired_reset_planning: dict[str, Any] = {}
    paired_reset_planning_records: list[dict[str, Any]] = []
    if reset_cadences:
        paired_reset_planning, paired_reset_planning_records = evaluate_paired_reset_planning(
            model,
            world,
            examples,
            rng,
            num_examples=planning_examples,
            max_steps=max_unroll_steps,
            branch_size=planning_branch_size,
            beam_size=planning_beam_size,
            reset_cadences=reset_cadences,
        )
    cem_planning, cem_planning_records = evaluate_cem_planning(
        model,
        world,
        examples,
        rng,
        num_examples=cem_examples,
        max_steps=max_unroll_steps,
        population_size=cem_population,
        elite_frac=cem_elite_frac,
        iterations=cem_iterations,
        smoothing=cem_smoothing,
        score_mode=cem_score,
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
        "goal_rank": summarize_goal_rank_records(goal_rank_records),
        "drift": drift_summary,
        "planning": planning,
        "reencoded_planning": reencoded_planning,
        "paired_reset_planning": paired_reset_planning,
        "cem_planning": cem_planning,
        "planner_traces": traces,
    }
    destination = output_dir or (run_root / "diagnostics")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "diagnostics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (destination / "rank_records.jsonl").open("w") as handle:
        for record in rank_records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    with (destination / "goal_rank_records.jsonl").open("w") as handle:
        for record in goal_rank_records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    with (destination / "drift_records.jsonl").open("w") as handle:
        for record in drift_records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    with (destination / "latent_planning_records.jsonl").open("w") as handle:
        for record in planning_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    with (destination / "reencoded_planning_records.jsonl").open("w") as handle:
        for record in reencoded_planning_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    if paired_reset_planning_records:
        with (destination / "paired_reset_planning_records.jsonl").open("w") as handle:
            for record in paired_reset_planning_records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    if cem_planning_records:
        with (destination / "cem_planning_records.jsonl").open("w") as handle:
            for record in cem_planning_records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
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
def evaluate_goal_action_ranks(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    examples: list[PuzzleExample],
    rng: np.random.Generator,
    *,
    num_examples: int,
    depth_fractions: list[float],
) -> list[GoalRankRecord]:
    records: list[GoalRankRecord] = []
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
            rank = rank_goal_action_set(model, world, state, example.goal, clue_mask)
            if rank is None:
                continue
            records.append(
                GoalRankRecord(
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
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if num_examples <= 0:
        return {}, []
    summaries = {
        "step_energy": [],
        "terminal_energy": [],
    }
    for example_index in range(num_examples):
        example = examples[int(rng.integers(0, len(examples)))]
        for mode in summaries:
            plan = latent_beam_plan(
                model,
                world,
                example,
                max_steps=max_steps,
                branch_size=branch_size,
                beam_size=beam_size,
                terminal_only_score=(mode == "terminal_energy"),
            )
            plan["example_index"] = int(example_index)
            plan["mode"] = mode
            summaries[mode].append(plan)
    return (
        {
            mode: summarize_plan_summaries(items)
            for mode, items in summaries.items()
        },
        flatten_plan_records(summaries, planner="latent"),
    )


@torch.no_grad()
def evaluate_reencoded_planning(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    examples: list[PuzzleExample],
    rng: np.random.Generator,
    *,
    num_examples: int,
    max_steps: int,
    branch_size: int,
    beam_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if num_examples <= 0:
        return {}, []
    summaries = {
        "step_energy": [],
        "terminal_energy": [],
    }
    for example_index in range(num_examples):
        example = examples[int(rng.integers(0, len(examples)))]
        for mode in summaries:
            plan = reencoded_beam_plan(
                model,
                world,
                example,
                max_steps=max_steps,
                branch_size=branch_size,
                beam_size=beam_size,
                terminal_only_score=(mode == "terminal_energy"),
            )
            plan["example_index"] = int(example_index)
            plan["mode"] = mode
            summaries[mode].append(plan)
    return (
        {
            mode: summarize_plan_summaries(items)
            for mode, items in summaries.items()
        },
        flatten_plan_records(summaries, planner="reencoded"),
    )


@torch.no_grad()
def evaluate_paired_reset_planning(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    examples: list[PuzzleExample],
    rng: np.random.Generator,
    *,
    num_examples: int,
    max_steps: int,
    branch_size: int,
    beam_size: int,
    reset_cadences: list[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if num_examples <= 0:
        return {}, []
    cadences = sorted({int(cadence) for cadence in reset_cadences if int(cadence) > 0})
    if not cadences:
        return {}, []
    variants: list[tuple[str, str, int | None]] = [("latent_no_reset", "latent", None)]
    variants.extend((f"reset_every_{cadence}", "latent_reset", cadence) for cadence in cadences)
    variants.append(("reencoded", "reencoded", None))
    summaries: dict[str, dict[str, list[dict[str, Any]]]] = {
        name: {"step_energy": [], "terminal_energy": []}
        for name, _, _ in variants
    }
    sampled_examples = [examples[int(rng.integers(0, len(examples)))] for _ in range(num_examples)]
    for example_index, example in enumerate(sampled_examples):
        for mode in ("step_energy", "terminal_energy"):
            terminal_only_score = mode == "terminal_energy"
            for variant, planner, cadence in variants:
                if planner == "reencoded":
                    plan = reencoded_beam_plan(
                        model,
                        world,
                        example,
                        max_steps=max_steps,
                        branch_size=branch_size,
                        beam_size=beam_size,
                        terminal_only_score=terminal_only_score,
                    )
                else:
                    plan = latent_beam_plan(
                        model,
                        world,
                        example,
                        max_steps=max_steps,
                        branch_size=branch_size,
                        beam_size=beam_size,
                        terminal_only_score=terminal_only_score,
                        reset_cadence=cadence,
                    )
                plan["example_index"] = int(example_index)
                plan["mode"] = mode
                plan["variant"] = variant
                plan["planner"] = planner
                plan["reset_cadence"] = cadence
                summaries[variant][mode].append(plan)
    summary = {
        variant: {
            mode: summarize_plan_summaries(items)
            for mode, items in modes.items()
        }
        for variant, modes in summaries.items()
    }
    return summary, flatten_paired_plan_records(summaries)


@torch.no_grad()
def evaluate_cem_planning(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    examples: list[PuzzleExample],
    rng: np.random.Generator,
    *,
    num_examples: int,
    max_steps: int,
    population_size: int,
    elite_frac: float,
    iterations: int,
    smoothing: float,
    score_mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if num_examples <= 0:
        return {}, []
    summaries: dict[str, list[dict[str, Any]]] = {}
    for example_index in range(num_examples):
        example = examples[int(rng.integers(0, len(examples)))]
        plan = cem_plan(
            model,
            world,
            example,
            rng,
            max_steps=max_steps,
            population_size=population_size,
            elite_frac=elite_frac,
            iterations=iterations,
            smoothing=smoothing,
            score_mode=score_mode,
        )
        plan["example_index"] = int(example_index)
        summaries.setdefault(str(plan["score_mode"]), []).append(plan)
    return (
        {
            mode: summarize_plan_summaries(items)
            for mode, items in summaries.items()
        },
        flatten_cem_plan_records(summaries),
    )


@torch.no_grad()
def cem_plan(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    example: PuzzleExample,
    rng: np.random.Generator,
    *,
    max_steps: int,
    population_size: int,
    elite_frac: float,
    iterations: int,
    smoothing: float,
    score_mode: str,
) -> dict[str, Any]:
    start = world.validate_state(example.state).copy()
    goal = world.validate_state(example.goal)
    clue_mask = clue_mask_for_planning(world, start)
    limit = min(int(max_steps), terminal_step_limit(world, example, max_steps))
    score_mode = resolve_cem_score_mode(model, score_mode)
    if limit <= 0 or world.is_goal(start, goal):
        return {
            "solved": float(world.is_goal(start, goal)),
            "terminal": float(is_terminal_state(world, start, goal, clue_mask)),
            "steps": 0.0,
            "energy": 0.0,
            "remaining_hamming": float(np.not_equal(start, goal).sum()),
            "final_state": board_as_list(start),
            "goal_state": board_as_list(goal),
            "mismatches": mismatch_records(start, goal),
            "score_mode": score_mode,
            "population_size": float(population_size),
            "elite_frac": float(elite_frac),
            "iterations": float(iterations),
            "smoothing": float(smoothing),
        }

    action_space = cem_action_space(world, start, clue_mask)
    if not action_space["positions"] or not action_space["values"]:
        return {
            "solved": 0.0,
            "terminal": float(is_terminal_state(world, start, goal, clue_mask)),
            "steps": 0.0,
            "energy": math.inf,
            "remaining_hamming": float(np.not_equal(start, goal).sum()),
            "final_state": board_as_list(start),
            "goal_state": board_as_list(goal),
            "mismatches": mismatch_records(start, goal),
            "score_mode": score_mode,
            "population_size": float(population_size),
            "elite_frac": float(elite_frac),
            "iterations": float(iterations),
            "smoothing": float(smoothing),
        }

    horizon = max(1, limit)
    population_size = max(1, int(population_size))
    iterations = max(1, int(iterations))
    elite_count = max(1, min(population_size, int(math.ceil(population_size * float(elite_frac)))))
    smoothing = min(max(float(smoothing), 0.0), 1.0)
    cell_probs = np.full((horizon, len(action_space["positions"])), 1.0 / len(action_space["positions"]))
    value_probs = np.full((horizon, len(action_space["values"])), 1.0 / len(action_space["values"]))
    best_state = start.copy()
    best_actions: list[WorldAction] = []
    best_score = -math.inf

    for _ in range(iterations):
        candidates: list[dict[str, Any]] = []
        for _sample_index in range(population_size):
            candidate = sample_cem_rollout(
                world,
                start,
                goal,
                clue_mask,
                rng,
                cell_probs=cell_probs,
                value_probs=value_probs,
                action_space=action_space,
            )
            candidates.append(candidate)

        final_states = np.stack([item["state"] for item in candidates], axis=0)
        scores = cem_scores(
            model,
            world,
            final_states,
            goal,
            start,
            score_mode=score_mode,
        )
        for candidate, score in zip(candidates, scores, strict=True):
            candidate["score"] = float(score)
        candidates.sort(key=lambda item: float(item["score"]), reverse=True)
        if float(candidates[0]["score"]) > best_score:
            best_score = float(candidates[0]["score"])
            best_state = candidates[0]["state"]
            best_actions = list(candidates[0]["actions"])
        elites = candidates[:elite_count]
        elite_cell_probs, elite_value_probs = estimate_cem_distributions(elites, action_space, horizon)
        cell_probs = smoothing * cell_probs + (1.0 - smoothing) * elite_cell_probs
        value_probs = smoothing * value_probs + (1.0 - smoothing) * elite_value_probs
        cell_probs = normalize_categorical(cell_probs)
        value_probs = normalize_categorical(value_probs)

    return {
        "solved": float(world.is_goal(best_state, goal)),
        "terminal": float(is_terminal_state(world, best_state, goal, clue_mask)),
        "steps": float(len(best_actions)),
        "energy": float(-best_score),
        "remaining_hamming": float(np.not_equal(best_state, goal).sum()),
        "final_state": board_as_list(best_state),
        "goal_state": board_as_list(goal),
        "mismatches": mismatch_records(best_state, goal),
        "score_mode": score_mode,
        "population_size": float(population_size),
        "elite_frac": float(elite_frac),
        "iterations": float(iterations),
        "smoothing": float(smoothing),
    }


def resolve_cem_score_mode(model: ActionConditionedWorldModel, score_mode: str) -> str:
    mode = str(score_mode)
    if mode == "auto":
        return "goal_energy" if model.use_goal_energy_head else "latent_goal"
    if mode not in {"goal_energy", "latent_goal"}:
        raise ValueError("cem score mode must be 'auto', 'goal_energy', or 'latent_goal'.")
    if mode == "goal_energy" and not model.use_goal_energy_head:
        raise ValueError("CEM goal_energy scoring requires a model with use_goal_energy_head=True.")
    return mode


def cem_action_space(
    world: PuzzleWorld,
    initial_state: np.ndarray,
    clue_mask: np.ndarray | None,
) -> dict[str, Any]:
    if isinstance(world, SudokuWorld):
        if clue_mask is None:
            raise ValueError("Sudoku CEM planning requires a clue mask.")
        positions = [(int(row), int(col)) for row, col in np.argwhere(~clue_mask)]
        values = list(range(1, 10))
    else:
        legal = legal_planning_actions(world, initial_state, clue_mask)
        positions = sorted({(action.row, action.col) for action in legal})
        values = sorted({action.value for action in legal})
    position_to_index = {position: index for index, position in enumerate(positions)}
    value_to_index = {value: index for index, value in enumerate(values)}
    return {
        "positions": positions,
        "values": values,
        "position_to_index": position_to_index,
        "value_to_index": value_to_index,
    }


def sample_cem_rollout(
    world: PuzzleWorld,
    start: np.ndarray,
    goal: np.ndarray,
    clue_mask: np.ndarray | None,
    rng: np.random.Generator,
    *,
    cell_probs: np.ndarray,
    value_probs: np.ndarray,
    action_space: dict[str, Any],
) -> dict[str, Any]:
    current = world.validate_state(start).copy()
    actions: list[WorldAction] = []
    for step in range(cell_probs.shape[0]):
        if is_terminal_state(world, current, goal, clue_mask):
            break
        action = sample_cem_action(rng, cell_probs[step], value_probs[step], action_space)
        if action not in legal_planning_actions(world, current, clue_mask):
            replacement = sample_valid_cem_action(world, current, clue_mask, rng, cell_probs[step], value_probs[step], action_space)
            if replacement is None:
                break
            action = replacement
        current = apply_planning_action(world, current, action, clue_mask)
        actions.append(action)
    return {"state": current, "actions": actions}


def sample_cem_action(
    rng: np.random.Generator,
    cell_probs: np.ndarray,
    value_probs: np.ndarray,
    action_space: dict[str, Any],
) -> WorldAction:
    cell_index = int(rng.choice(len(action_space["positions"]), p=cell_probs))
    value_index = int(rng.choice(len(action_space["values"]), p=value_probs))
    row, col = action_space["positions"][cell_index]
    return WorldAction(row, col, int(action_space["values"][value_index]))


def sample_valid_cem_action(
    world: PuzzleWorld,
    state: np.ndarray,
    clue_mask: np.ndarray | None,
    rng: np.random.Generator,
    cell_probs: np.ndarray,
    value_probs: np.ndarray,
    action_space: dict[str, Any],
) -> WorldAction | None:
    legal = legal_planning_actions(world, state, clue_mask)
    if not legal:
        return None
    weights = np.asarray(
        [
            cell_probs[action_space["position_to_index"][(action.row, action.col)]]
            * value_probs[action_space["value_to_index"][action.value]]
            for action in legal
        ],
        dtype=np.float64,
    )
    total = float(weights.sum())
    if total <= 0.0 or not np.isfinite(total):
        return legal[int(rng.integers(0, len(legal)))]
    weights = weights / total
    return legal[int(rng.choice(len(legal), p=weights))]


@torch.no_grad()
def cem_scores(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    final_states: np.ndarray,
    goal: np.ndarray,
    initial_state: np.ndarray,
    *,
    score_mode: str,
) -> list[float]:
    goals = np.repeat(goal[None, ...], final_states.shape[0], axis=0)
    states_tensor = torch.as_tensor(final_states, dtype=torch.long)
    goals_tensor = torch.as_tensor(goals, dtype=torch.long)
    if score_mode == "goal_energy":
        initials = np.repeat(initial_state[None, ...], final_states.shape[0], axis=0)
        scores = model.score_states_to_goal(
            states_tensor,
            goals_tensor,
            world.task_id,
            initial_states=torch.as_tensor(initials, dtype=torch.long),
            use_goal_energy_head=True,
        )
    else:
        scores = model.score_states_to_goal(states_tensor, goals_tensor, world.task_id)
    return [float(item) for item in scores.detach().cpu().tolist()]


def estimate_cem_distributions(
    elites: list[dict[str, Any]],
    action_space: dict[str, Any],
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    min_prob = 1.0e-4
    cell_counts = np.full((horizon, len(action_space["positions"])), min_prob, dtype=np.float64)
    value_counts = np.full((horizon, len(action_space["values"])), min_prob, dtype=np.float64)
    for elite in elites:
        actions = elite["actions"]
        if not actions:
            continue
        for step in range(horizon):
            action = actions[min(step, len(actions) - 1)]
            cell_counts[step, action_space["position_to_index"][(action.row, action.col)]] += 1.0
            value_counts[step, action_space["value_to_index"][action.value]] += 1.0
    return normalize_categorical(cell_counts), normalize_categorical(value_counts)


def normalize_categorical(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    row_sums = values.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0.0] = 1.0
    return values / row_sums


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
    reset_cadence: int | None = None,
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
                next_steps = int(beam["steps"]) + 1
                if reset_cadence is not None and next_steps % int(reset_cadence) == 0:
                    next_tensor = torch.as_tensor(next_state[None, ...], dtype=torch.long, device=device)
                    next_latent = model.encoder(next_tensor, task_ids=task_ids)
                energy = float(F.mse_loss(next_latent, goal_latent).detach().cpu().item())
                next_beam = {
                    "state": next_state,
                    "latent": next_latent,
                    "steps": next_steps,
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
        "final_state": board_as_list(best["state"]),
        "goal_state": board_as_list(goal),
        "mismatches": mismatch_records(best["state"], goal),
    }


@torch.no_grad()
def reencoded_beam_plan(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    example: PuzzleExample,
    *,
    max_steps: int,
    branch_size: int,
    beam_size: int,
    terminal_only_score: bool,
) -> dict[str, Any]:
    start = world.validate_state(example.state).copy()
    goal = world.validate_state(example.goal)
    clue_mask = clue_mask_for_planning(world, start)
    active = [
        {
            "state": start,
            "steps": 0,
            "energy": encoded_state_energy(model, world, start, goal),
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
                next_beam = {
                    "state": next_state,
                    "steps": int(beam["steps"]) + 1,
                    "energy": encoded_state_energy(model, world, next_state, goal),
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
        "final_state": board_as_list(best["state"]),
        "goal_state": board_as_list(goal),
        "mismatches": mismatch_records(best["state"], goal),
    }


@torch.no_grad()
def encoded_state_energy(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    state: np.ndarray,
    goal: np.ndarray,
) -> float:
    device = next(model.parameters()).device
    task_ids = torch.full((1,), world.task_id, dtype=torch.long, device=device)
    state_tensor = torch.as_tensor(state[None, ...], dtype=torch.long, device=device)
    goal_tensor = torch.as_tensor(goal[None, ...], dtype=torch.long, device=device)
    latent = model.encoder(state_tensor, task_ids=task_ids)
    goal_latent = model.target_encoder(goal_tensor, task_ids=task_ids)
    return float(F.mse_loss(latent, goal_latent).detach().cpu().item())


def summarize_plan_summaries(items: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "count": float(len(items)),
        "solve_rate": mean([item["solved"] for item in items]),
        "terminal_rate": mean([item["terminal"] for item in items]),
        "mean_steps": mean([item["steps"] for item in items]),
        "mean_energy": mean([item["energy"] for item in items]),
        "mean_remaining_hamming": mean([item["remaining_hamming"] for item in items]),
    }


def flatten_plan_records(summaries: dict[str, list[dict[str, Any]]], *, planner: str) -> list[dict[str, Any]]:
    records = []
    for mode, items in summaries.items():
        for item in items:
            record = {
                "planner": planner,
                "mode": mode,
                "example_index": int(item["example_index"]),
                "solved": float(item["solved"]),
                "terminal": float(item["terminal"]),
                "steps": float(item["steps"]),
                "energy": float(item["energy"]),
                "remaining_hamming": float(item["remaining_hamming"]),
                "final_state": item["final_state"],
                "goal_state": item["goal_state"],
                "mismatches": item["mismatches"],
            }
            records.append(record)
    return records


def flatten_paired_plan_records(summaries: dict[str, dict[str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    records = []
    for _variant, modes in summaries.items():
        for mode, items in modes.items():
            for item in items:
                record = {
                    "planner": item["planner"],
                    "variant": item["variant"],
                    "mode": mode,
                    "reset_cadence": item["reset_cadence"],
                    "example_index": int(item["example_index"]),
                    "solved": float(item["solved"]),
                    "terminal": float(item["terminal"]),
                    "steps": float(item["steps"]),
                    "energy": float(item["energy"]),
                    "remaining_hamming": float(item["remaining_hamming"]),
                    "final_state": item["final_state"],
                    "goal_state": item["goal_state"],
                    "mismatches": item["mismatches"],
                }
                records.append(record)
    return records


def flatten_cem_plan_records(summaries: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    records = []
    for mode, items in summaries.items():
        for item in items:
            records.append(
                {
                    "planner": "cem",
                    "mode": mode,
                    "score_mode": item["score_mode"],
                    "example_index": int(item["example_index"]),
                    "solved": float(item["solved"]),
                    "terminal": float(item["terminal"]),
                    "steps": float(item["steps"]),
                    "energy": float(item["energy"]),
                    "remaining_hamming": float(item["remaining_hamming"]),
                    "population_size": float(item["population_size"]),
                    "elite_frac": float(item["elite_frac"]),
                    "iterations": float(item["iterations"]),
                    "smoothing": float(item["smoothing"]),
                    "final_state": item["final_state"],
                    "goal_state": item["goal_state"],
                    "mismatches": item["mismatches"],
                }
            )
    return records


def board_as_list(state: np.ndarray) -> list[list[int]]:
    return np.asarray(state, dtype=np.int64).tolist()


def mismatch_records(state: np.ndarray, goal: np.ndarray) -> list[dict[str, int]]:
    state_arr = np.asarray(state, dtype=np.int64)
    goal_arr = np.asarray(goal, dtype=np.int64)
    mismatches = []
    for row, col in np.argwhere(state_arr != goal_arr):
        mismatches.append(
            {
                "row": int(row),
                "col": int(col),
                "pred": int(state_arr[row, col]),
                "goal": int(goal_arr[row, col]),
            }
        )
    return mismatches


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


def rank_goal_action_set(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    state: np.ndarray,
    goal: np.ndarray,
    clue_mask: np.ndarray | None,
) -> dict[str, Any] | None:
    legal = legal_planning_actions(world, state, clue_mask)
    if not legal:
        return None
    goal_indices = goal_action_indices(world, state, goal, legal)
    if not goal_indices:
        return None
    scores = model.score_actions_to_goal(
        torch.as_tensor(state, dtype=torch.long),
        legal,
        torch.as_tensor(goal, dtype=torch.long),
        world.task_id,
    )
    order = scores.argsort(descending=True).detach().cpu().tolist()
    ranks = [order.index(index) + 1 for index in goal_indices]
    best_index = max(goal_indices, key=lambda index: float(scores[index].detach().cpu().item()))
    return {
        "best_goal_rank": int(min(ranks)),
        "goal_actions": len(goal_indices),
        "legal_actions": len(legal),
        "best_goal_score": float(scores[best_index].detach().cpu().item()),
        "best_score": float(scores[order[0]].detach().cpu().item()),
    }


def goal_action_indices(
    world: PuzzleWorld,
    state: np.ndarray,
    goal: np.ndarray,
    legal: list[WorldAction],
) -> list[int]:
    state_arr = world.validate_state(state)
    goal_arr = world.validate_state(goal)
    indices = []
    for index, action in enumerate(legal):
        if state_arr[action.row, action.col] == goal_arr[action.row, action.col]:
            continue
        if action.value == int(goal_arr[action.row, action.col]):
            indices.append(index)
    return indices


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
    if not records:
        return {"count": 0}
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


def summarize_goal_rank_records(records: list[GoalRankRecord]) -> dict[str, Any]:
    if not records:
        return {"count": 0}
    by_depth: dict[str, dict[str, float]] = {}
    for depth in sorted({record.depth_fraction for record in records}):
        depth_records = [record for record in records if record.depth_fraction == depth]
        ranks = [record.best_goal_rank for record in depth_records]
        by_depth[str(depth)] = {
            "count": len(ranks),
            "top1": mean([float(rank == 1) for rank in ranks]),
            "top5": mean([float(rank <= 5) for rank in ranks]),
            "mrr": mean([1.0 / rank for rank in ranks]),
            "mean_rank": mean(ranks),
            "median_rank": median(ranks),
            "mean_goal_actions": mean([record.goal_actions for record in depth_records]),
            "mean_legal_actions": mean([record.legal_actions for record in depth_records]),
            "mean_score_margin": mean([record.best_score - record.best_goal_score for record in depth_records]),
        }
    ranks = [record.best_goal_rank for record in records]
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
    parser.add_argument("--cem-examples", type=int, default=0)
    parser.add_argument("--planning-beam-size", type=int, default=4)
    parser.add_argument("--planning-branch-size", type=int, default=8)
    parser.add_argument("--cem-population", type=int, default=128)
    parser.add_argument("--cem-elite-frac", type=float, default=0.2)
    parser.add_argument("--cem-iterations", type=int, default=4)
    parser.add_argument("--cem-smoothing", type=float, default=0.7)
    parser.add_argument("--cem-score", choices=["auto", "goal_energy", "latent_goal"], default="auto")
    parser.add_argument("--trace-examples", type=int, default=3)
    parser.add_argument("--max-unroll-steps", type=int, default=256)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 10, 20])
    parser.add_argument("--depth-fractions", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75])
    parser.add_argument("--reset-cadences", type=int, nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_diagnostics(
        args.run_root,
        rank_examples=args.rank_examples,
        drift_examples=args.drift_examples,
        planning_examples=args.planning_examples,
        cem_examples=args.cem_examples,
        planning_beam_size=args.planning_beam_size,
        planning_branch_size=args.planning_branch_size,
        cem_population=args.cem_population,
        cem_elite_frac=args.cem_elite_frac,
        cem_iterations=args.cem_iterations,
        cem_smoothing=args.cem_smoothing,
        cem_score=args.cem_score,
        max_unroll_steps=args.max_unroll_steps,
        horizons=args.horizons,
        depth_fractions=args.depth_fractions,
        trace_examples=args.trace_examples,
        seed=args.seed,
        reset_cadences=args.reset_cadences,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
