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
    planning_score: str,
    planning_beam_size: int,
    planning_branch_size: int,
    cem_population: int,
    cem_elite_frac: float,
    cem_iterations: int,
    cem_smoothing: float,
    cem_score: str,
    cem_hierarchy_level: int,
    mcts_examples: int,
    mcts_simulations: int,
    mcts_depth: int,
    mcts_score: str,
    mcts_exploration: float,
    mcts_expansion_actions: int,
    mcts_debug_examples: int,
    mcts_debug_actions: int,
    subgoal_cem_examples: int,
    subgoal_hierarchy_level: int,
    subgoal_macro_horizon: int,
    subgoal_high_population: int,
    subgoal_low_population: int,
    subgoal_iterations: int,
    subgoal_elite_frac: float,
    subgoal_smoothing: float,
    subgoal_execute_steps: int,
    subgoal_prior_samples: int,
    subgoal_high_score: str,
    recursive_subgoal_examples: int,
    recursive_hierarchy_level: int,
    recursive_macro_horizon: int,
    recursive_high_population: int,
    recursive_low_population: int,
    recursive_iterations: int,
    recursive_elite_frac: float,
    recursive_smoothing: float,
    recursive_execute_steps: int,
    recursive_prior_samples: int,
    recursive_high_score: str,
    recursive_optimizer: str,
    recursive_gd_steps: int,
    recursive_gd_lr: float,
    recursive_reachability_weight: float,
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
    load_model_checkpoint_state(model, checkpoint["model"])
    model.eval()
    destination = output_dir or (run_root / "diagnostics")
    destination.mkdir(parents=True, exist_ok=True)

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
        planning_score=planning_score,
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
        planning_score=planning_score,
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
            planning_score=planning_score,
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
        hierarchy_level=cem_hierarchy_level,
    )
    mcts_planning, mcts_planning_records, mcts_debug_records = evaluate_mcts_planning(
        model,
        world,
        examples,
        rng,
        num_examples=mcts_examples,
        max_steps=max_unroll_steps,
        simulations=mcts_simulations,
        max_depth=mcts_depth,
        score_mode=mcts_score,
        exploration=mcts_exploration,
        expansion_actions=mcts_expansion_actions,
        debug_examples=mcts_debug_examples,
        debug_actions=mcts_debug_actions,
        stream_dir=destination,
    )
    subgoal_cem_planning, subgoal_cem_planning_records = evaluate_hierarchical_subgoal_cem_planning(
        model,
        world,
        examples,
        rng,
        num_examples=subgoal_cem_examples,
        max_steps=max_unroll_steps,
        hierarchy_level=subgoal_hierarchy_level,
        macro_horizon=subgoal_macro_horizon,
        high_population_size=subgoal_high_population,
        low_population_size=subgoal_low_population,
        elite_frac=subgoal_elite_frac,
        iterations=subgoal_iterations,
        smoothing=subgoal_smoothing,
        execute_steps=subgoal_execute_steps,
        prior_samples=subgoal_prior_samples,
        high_score_mode=subgoal_high_score,
    )
    recursive_subgoal_planning, recursive_subgoal_records = evaluate_recursive_hierarchical_subgoal_planning(
        model,
        world,
        examples,
        rng,
        num_examples=recursive_subgoal_examples,
        max_steps=max_unroll_steps,
        hierarchy_level=recursive_hierarchy_level,
        macro_horizon=recursive_macro_horizon,
        high_population_size=recursive_high_population,
        low_population_size=recursive_low_population,
        elite_frac=recursive_elite_frac,
        iterations=recursive_iterations,
        smoothing=recursive_smoothing,
        execute_steps=recursive_execute_steps,
        prior_samples=recursive_prior_samples,
        high_score_mode=recursive_high_score,
        optimizer=recursive_optimizer,
        gd_steps=recursive_gd_steps,
        gd_lr=recursive_gd_lr,
        reachability_weight=recursive_reachability_weight,
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
        "mcts_planning": mcts_planning,
        "hierarchical_subgoal_cem": subgoal_cem_planning,
        "recursive_hierarchical_subgoal": recursive_subgoal_planning,
        "planner_traces": traces,
    }
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
    if mcts_planning_records:
        with (destination / "mcts_planning_records.jsonl").open("w") as handle:
            for record in mcts_planning_records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    if mcts_debug_records:
        with (destination / "mcts_debug_records.jsonl").open("w") as handle:
            for record in mcts_debug_records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    if subgoal_cem_planning_records:
        with (destination / "hierarchical_subgoal_cem_records.jsonl").open("w") as handle:
            for record in subgoal_cem_planning_records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    if recursive_subgoal_records:
        with (destination / "recursive_hierarchical_subgoal_records.jsonl").open("w") as handle:
            for record in recursive_subgoal_records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    calibration_records = goal_energy_calibration_records(
        model,
        world,
        [*planning_records, *reencoded_planning_records, *paired_reset_planning_records],
    )
    if calibration_records:
        with (destination / "goal_energy_calibration_records.jsonl").open("w") as handle:
            for record in calibration_records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        write_goal_energy_calibration_plots(destination, calibration_records)
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
    planning_score: str = "latent_goal",
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
                planning_score=planning_score,
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
    planning_score: str = "latent_goal",
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
                planning_score=planning_score,
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
    planning_score: str = "latent_goal",
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
                        planning_score=planning_score,
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
                        planning_score=planning_score,
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
    hierarchy_level: int = 0,
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
            hierarchy_level=hierarchy_level,
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


@dataclass(slots=True)
class MCTSNode:
    state: np.ndarray
    parent: "MCTSNode | None"
    action: WorldAction | None
    depth: int
    untried_actions: list[WorldAction]
    children: dict[WorldAction, "MCTSNode"]
    visits: int = 0
    value_sum: float = 0.0
    leaf_score: float = math.nan
    oracle_leaf_energy: float = math.nan

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@torch.no_grad()
def evaluate_mcts_planning(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    examples: list[PuzzleExample],
    rng: np.random.Generator,
    *,
    num_examples: int,
    max_steps: int,
    simulations: int,
    max_depth: int,
    score_mode: str,
    exploration: float,
    expansion_actions: int,
    debug_examples: int,
    debug_actions: int,
    stream_dir: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if num_examples <= 0:
        return {}, [], []
    summaries = []
    debug_records = []
    resolved_score_mode = resolve_mcts_score_mode(model, score_mode)
    planning_handle = None
    debug_handle = None
    if stream_dir is not None:
        stream_dir.mkdir(parents=True, exist_ok=True)
        planning_handle = (stream_dir / "mcts_planning_records.jsonl").open("w")
        debug_handle = (stream_dir / "mcts_debug_records.jsonl").open("w")
    try:
        for example_index in range(num_examples):
            example = examples[int(rng.integers(0, len(examples)))]
            plan = mcts_plan(
                model,
                world,
                example,
                rng,
                max_steps=max_steps,
                simulations=simulations,
                max_depth=max_depth,
                score_mode=resolved_score_mode,
                exploration=exploration,
                expansion_actions=expansion_actions,
                collect_debug=example_index < max(0, int(debug_examples)),
                debug_actions=debug_actions,
            )
            plan_debug = plan.pop("debug_records", [])
            plan["example_index"] = int(example_index)
            summaries.append(plan)
            debug_records.extend(plan_debug)
            if planning_handle is not None:
                for record in flatten_mcts_plan_records([plan]):
                    planning_handle.write(json.dumps(record, sort_keys=True) + "\n")
                planning_handle.flush()
            if debug_handle is not None:
                for record in plan_debug:
                    debug_handle.write(json.dumps(record, sort_keys=True) + "\n")
                debug_handle.flush()
    finally:
        if planning_handle is not None:
            planning_handle.close()
        if debug_handle is not None:
            debug_handle.close()
    return (
        {resolved_score_mode: summarize_plan_summaries(summaries)},
        flatten_mcts_plan_records(summaries),
        debug_records,
    )


@torch.no_grad()
def mcts_plan(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    example: PuzzleExample,
    rng: np.random.Generator,
    *,
    max_steps: int,
    simulations: int,
    max_depth: int,
    score_mode: str,
    exploration: float,
    expansion_actions: int,
    collect_debug: bool = False,
    debug_actions: int = 8,
) -> dict[str, Any]:
    score_mode = resolve_mcts_score_mode(model, score_mode)
    current = world.validate_state(example.state).copy()
    goal = world.validate_state(example.goal)
    initial = current.copy()
    clue_mask = clue_mask_for_planning(world, initial)
    limit = min(int(max_steps), terminal_step_limit(world, example, max_steps))
    trajectory_states = [board_as_list(current)]
    debug_records: list[dict[str, Any]] = []
    score_cache: dict[tuple[str, bytes], float] = {}

    for step in range(max(0, limit)):
        if world.is_goal(current, goal) or is_terminal_state(world, current, goal, clue_mask):
            break
        root = build_mcts_tree(
            model,
            world,
            current,
            goal,
            initial,
            clue_mask,
            rng,
            simulations=simulations,
            max_depth=max_depth,
            score_mode=score_mode,
            exploration=exploration,
            expansion_actions=expansion_actions,
            score_cache=score_cache,
        )
        if collect_debug:
            debug_records.append(
                mcts_root_debug_record(
                    model,
                    world,
                    root,
                    goal,
                    initial,
                    step=step,
                    score_mode=score_mode,
                    debug_actions=debug_actions,
                    cache=score_cache,
                )
            )
        child = select_mcts_root_child(root)
        if child is None or child.action is None:
            break
        current = child.state.copy()
        trajectory_states.append(board_as_list(current))

    return {
        "solved": float(world.is_goal(current, goal)),
        "terminal": float(is_terminal_state(world, current, goal, clue_mask)),
        "steps": float(len(trajectory_states) - 1),
        "energy": float(-score_leaf_state(model, world, current, goal, initial, score_mode=score_mode, cache=score_cache)),
        "remaining_hamming": float(np.not_equal(current, goal).sum()),
        "final_state": board_as_list(current),
        "goal_state": board_as_list(goal),
        "trajectory_states": trajectory_states,
        "mismatches": mismatch_records(current, goal),
        "score_mode": score_mode,
        "simulations": float(simulations),
        "max_depth": float(max_depth),
        "exploration": float(exploration),
        "expansion_actions": float(expansion_actions),
        "debug_records": debug_records,
    }


@torch.no_grad()
def build_mcts_tree(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    root_state: np.ndarray,
    goal: np.ndarray,
    initial_state: np.ndarray,
    clue_mask: np.ndarray | None,
    rng: np.random.Generator,
    *,
    simulations: int,
    max_depth: int,
    score_mode: str,
    exploration: float,
    expansion_actions: int,
    score_cache: dict[tuple[str, bytes], float] | None = None,
) -> MCTSNode:
    expansion_actions = max(1, int(expansion_actions))
    root = MCTSNode(
        state=world.validate_state(root_state).copy(),
        parent=None,
        action=None,
        depth=0,
        untried_actions=shuffled_actions(
            legal_planning_actions(world, root_state, clue_mask),
            rng,
            limit=expansion_actions,
        ),
        children={},
    )
    for _ in range(max(1, int(simulations))):
        node = select_mcts_leaf(root, exploration=exploration)
        if node.depth < max(0, int(max_depth)) and node.untried_actions and not is_terminal_state(world, node.state, goal, clue_mask):
            node = expand_mcts_node(world, node, clue_mask, rng, expansion_actions=expansion_actions)
        value = score_leaf_state(
            model,
            world,
            node.state,
            goal,
            initial_state,
            score_mode=score_mode,
            cache=score_cache,
        )
        node.leaf_score = float(value)
        backup_mcts_value(node, float(value))
    return root


def select_mcts_leaf(root: MCTSNode, *, exploration: float) -> MCTSNode:
    node = root
    while not node.untried_actions and node.children:
        node = max(node.children.values(), key=lambda child: mcts_ucb_score(node, child, exploration=exploration))
    return node


def expand_mcts_node(
    world: PuzzleWorld,
    node: MCTSNode,
    clue_mask: np.ndarray | None,
    rng: np.random.Generator,
    *,
    expansion_actions: int,
) -> MCTSNode:
    action = node.untried_actions.pop()
    next_state = apply_planning_action(world, node.state, action, clue_mask)
    child = MCTSNode(
        state=next_state,
        parent=node,
        action=action,
        depth=node.depth + 1,
        untried_actions=shuffled_actions(
            legal_planning_actions(world, next_state, clue_mask),
            rng,
            limit=expansion_actions,
        ),
        children={},
    )
    node.children[action] = child
    return child


def mcts_ucb_score(parent: MCTSNode, child: MCTSNode, *, exploration: float) -> float:
    if child.visits <= 0:
        return math.inf
    bonus = float(exploration) * math.sqrt(math.log(max(parent.visits, 1) + 1.0) / child.visits)
    return child.mean_value + bonus


def backup_mcts_value(node: MCTSNode, value: float) -> None:
    current: MCTSNode | None = node
    while current is not None:
        current.visits += 1
        current.value_sum += float(value)
        current = current.parent


def select_mcts_root_child(root: MCTSNode) -> MCTSNode | None:
    if not root.children:
        return None
    return max(root.children.values(), key=lambda child: (child.visits, child.mean_value))


def shuffled_actions(
    actions: list[WorldAction],
    rng: np.random.Generator,
    *,
    limit: int | None = None,
) -> list[WorldAction]:
    if not actions:
        return []
    order = rng.permutation(len(actions))
    if limit is not None:
        order = order[: max(0, int(limit))]
    return [actions[int(index)] for index in order]


@torch.no_grad()
def score_leaf_state(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    state: np.ndarray,
    goal: np.ndarray,
    initial_state: np.ndarray,
    *,
    score_mode: str,
    cache: dict[tuple[str, bytes], float] | None = None,
) -> float:
    mode = resolve_mcts_score_mode(model, score_mode)
    cache_key = (mode, np.ascontiguousarray(state).tobytes())
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    if mode == "latent_goal":
        value = -encoded_state_energy(model, world, state, goal)
        if cache is not None:
            cache[cache_key] = float(value)
        return float(value)
    if mode in {"goal_energy", "goal_value"}:
        value = score_symbolic_states_to_goal(
            model,
            world,
            [state],
            goal,
            initial_state,
            planning_score=mode,
        )[0]
        if cache is not None:
            cache[cache_key] = float(value)
        return float(value)
    raise ValueError(f"Unsupported MCTS score mode: {score_mode}")


def resolve_mcts_score_mode(model: ActionConditionedWorldModel, score_mode: str) -> str:
    mode = str(score_mode)
    if mode == "auto":
        return "goal_energy" if model.use_goal_energy_head else "latent_goal"
    if mode not in {"goal_energy", "goal_value", "latent_goal"}:
        raise ValueError("MCTS score mode must be 'auto', 'goal_energy', 'goal_value', or 'latent_goal'.")
    if mode in {"goal_energy", "goal_value"} and not model.use_goal_energy_head:
        raise ValueError(f"{mode} MCTS scoring requires a checkpoint with use_goal_energy_head=True.")
    return mode


@torch.no_grad()
def mcts_root_debug_record(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    root: MCTSNode,
    goal: np.ndarray,
    initial_state: np.ndarray,
    *,
    step: int,
    score_mode: str,
    debug_actions: int,
    cache: dict[tuple[str, bytes], float] | None = None,
) -> dict[str, Any]:
    children = sorted(root.children.values(), key=lambda child: (child.visits, child.mean_value), reverse=True)
    root_actions = []
    for child in children[: max(0, int(debug_actions))]:
        action = child.action
        if action is None:
            continue
        leaf_score = score_leaf_state(model, world, child.state, goal, initial_state, score_mode=score_mode, cache=cache)
        oracle_energy = encoded_state_energy(model, world, child.state, goal)
        root_actions.append(
            {
                "action": asdict(action),
                "visits": int(child.visits),
                "mean_value": float(child.mean_value),
                "leaf_score": float(leaf_score),
                "oracle_leaf_energy": float(oracle_energy),
                "remaining_hamming": int(np.not_equal(child.state, goal).sum()),
                "writes_goal_value": bool(action.value == int(goal[action.row, action.col])),
            }
        )
    best = root_actions[0] if root_actions else None
    return {
        "step": int(step),
        "score_mode": score_mode,
        "root_visits": int(root.visits),
        "expanded_actions": int(len(root.children)),
        "best_action": None if best is None else best["action"],
        "best_writes_goal_value": None if best is None else bool(best["writes_goal_value"]),
        "actions": root_actions,
    }


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
    hierarchy_level: int = 0,
) -> dict[str, Any]:
    start = world.validate_state(example.state).copy()
    goal = world.validate_state(example.goal)
    clue_mask = clue_mask_for_planning(world, start)
    limit = min(int(max_steps), terminal_step_limit(world, example, max_steps))
    score_mode = resolve_cem_score_mode(model, score_mode, hierarchy_level=hierarchy_level)
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
            "hierarchy_level": float(hierarchy_level),
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
            "hierarchy_level": float(hierarchy_level),
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
            candidates=candidates,
            hierarchy_level=hierarchy_level,
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
        "hierarchy_level": float(hierarchy_level),
    }


def resolve_cem_score_mode(
    model: ActionConditionedWorldModel,
    score_mode: str,
    *,
    hierarchy_level: int = 0,
) -> str:
    mode = str(score_mode)
    if mode == "auto":
        return "goal_energy" if model.use_goal_energy_head else "latent_goal"
    if mode not in {"goal_energy", "latent_goal", "hierarchical_latent_goal"}:
        raise ValueError("cem score mode must be 'auto', 'goal_energy', 'latent_goal', or 'hierarchical_latent_goal'.")
    if mode == "goal_energy" and not model.use_goal_energy_head:
        raise ValueError("CEM goal_energy scoring requires a model with use_goal_energy_head=True.")
    if mode == "hierarchical_latent_goal" and int(hierarchy_level) <= 0:
        raise ValueError("hierarchical_latent_goal scoring requires --cem-hierarchy-level > 0.")
    if mode == "hierarchical_latent_goal" and int(hierarchy_level) >= model.hierarchy_levels:
        raise ValueError("cem hierarchy level must be lower than model.hierarchy_levels.")
    if mode == "hierarchical_latent_goal" and getattr(model, "checkpoint_missing_hierarchy_action_encoders", False):
        raise ValueError("hierarchical_latent_goal scoring requires a checkpoint with trained action encoders.")
    return mode


def resolve_planning_score_mode(model: ActionConditionedWorldModel, planning_score: str) -> str:
    mode = str(planning_score)
    if mode not in {"latent_goal", "goal_energy", "goal_value", "action_advantage"}:
        raise ValueError("planning score must be 'latent_goal', 'goal_energy', 'goal_value', or 'action_advantage'.")
    if mode in {"goal_energy", "goal_value"} and not model.use_goal_energy_head:
        raise ValueError(f"{mode} planning requires a model with use_goal_energy_head=True.")
    if mode == "action_advantage" and not model.use_action_value_head:
        raise ValueError("action_advantage planning requires a model with use_action_value_head=True.")
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
    candidates: list[dict[str, Any]] | None = None,
    hierarchy_level: int = 0,
) -> list[float]:
    if score_mode == "hierarchical_latent_goal":
        if candidates is None:
            raise ValueError("hierarchical_latent_goal scoring requires CEM candidates.")
        return hierarchical_cem_scores(
            model,
            world,
            initial_state,
            goal,
            candidates,
            hierarchy_level=hierarchy_level,
        )
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


@torch.no_grad()
def hierarchical_cem_scores(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    initial_state: np.ndarray,
    goal: np.ndarray,
    candidates: list[dict[str, Any]],
    *,
    hierarchy_level: int,
) -> list[float]:
    scores: list[float] = [math.nan for _ in candidates]
    state_tensor = torch.as_tensor(initial_state, dtype=torch.long)
    goal_tensor = torch.as_tensor(goal, dtype=torch.long)
    block = int(model.hierarchy_span) ** int(hierarchy_level)
    by_usable: dict[int, list[int]] = {}
    for index, candidate in enumerate(candidates):
        actions = candidate["actions"]
        usable = len(actions) - (len(actions) % block)
        by_usable.setdefault(usable, []).append(index)
    for usable, indices in by_usable.items():
        if usable <= 0:
            states = torch.as_tensor(np.stack([candidates[index]["state"] for index in indices]), dtype=torch.long)
            batch_scores = model.score_states_to_goal(states, goal_tensor, world.task_id)
        else:
            action_tensor = torch.as_tensor(
                [
                    [
                        [world.task_id, action.row, action.col, action.value]
                        for action in candidates[index]["actions"][:usable]
                    ]
                    for index in indices
                ],
                dtype=torch.long,
            )
            batch_scores = model.score_action_sequences_to_goal(
                state_tensor,
                action_tensor,
                goal_tensor,
                world.task_id,
                hierarchy_level=hierarchy_level,
            )
        for index, score in zip(indices, batch_scores.detach().cpu().tolist(), strict=True):
            scores[index] = float(score)
    return scores


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
def evaluate_hierarchical_subgoal_cem_planning(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    examples: list[PuzzleExample],
    rng: np.random.Generator,
    *,
    num_examples: int,
    max_steps: int,
    hierarchy_level: int,
    macro_horizon: int,
    high_population_size: int,
    low_population_size: int,
    elite_frac: float,
    iterations: int,
    smoothing: float,
    execute_steps: int,
    prior_samples: int,
    high_score_mode: str = "latent_goal",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if num_examples <= 0:
        return {}, []
    validate_hierarchical_subgoal_level(model, hierarchy_level)
    items: list[dict[str, Any]] = []
    for example_index in range(num_examples):
        example = examples[int(rng.integers(0, len(examples)))]
        plan = hierarchical_subgoal_cem_plan(
            model,
            world,
            example,
            rng,
            max_steps=max_steps,
            hierarchy_level=hierarchy_level,
            macro_horizon=macro_horizon,
            high_population_size=high_population_size,
            low_population_size=low_population_size,
            elite_frac=elite_frac,
            iterations=iterations,
            smoothing=smoothing,
            execute_steps=execute_steps,
            prior_samples=prior_samples,
            high_score_mode=high_score_mode,
        )
        plan["example_index"] = int(example_index)
        items.append(plan)
    return (
        {f"{high_score_mode}_subgoal": summarize_plan_summaries(items)},
        flatten_hierarchical_subgoal_cem_records(items),
    )


@torch.no_grad()
def hierarchical_subgoal_cem_plan(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    example: PuzzleExample,
    rng: np.random.Generator,
    *,
    max_steps: int,
    hierarchy_level: int,
    macro_horizon: int,
    high_population_size: int,
    low_population_size: int,
    elite_frac: float,
    iterations: int,
    smoothing: float,
    execute_steps: int,
    prior_samples: int,
    high_score_mode: str = "latent_goal",
) -> dict[str, Any]:
    validate_hierarchical_subgoal_level(model, hierarchy_level)
    high_score_mode = resolve_subgoal_high_score_mode(model, high_score_mode)
    start = world.validate_state(example.state).copy()
    goal = world.validate_state(example.goal)
    clue_mask = clue_mask_for_planning(world, start)
    limit = min(int(max_steps), terminal_step_limit(world, example, max_steps))
    block = int(model.hierarchy_span) ** int(hierarchy_level)
    high_population_size = max(1, int(high_population_size))
    low_population_size = max(1, int(low_population_size))
    iterations = max(1, int(iterations))
    execute_steps = max(1, int(execute_steps))
    macro_horizon = max(1, int(macro_horizon))
    smoothing = min(max(float(smoothing), 0.0), 1.0)

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
            "hierarchy_level": float(hierarchy_level),
            "macro_horizon": float(macro_horizon),
            "macro_block": float(block),
            "high_population_size": float(high_population_size),
            "low_population_size": float(low_population_size),
            "elite_frac": float(elite_frac),
            "iterations": float(iterations),
            "smoothing": float(smoothing),
            "execute_steps": float(execute_steps),
            "replans": 0.0,
            "mean_high_energy": 0.0,
            "mean_low_subgoal_energy": 0.0,
            "prior_samples": float(prior_samples),
            "high_score_mode": high_score_mode,
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
            "hierarchy_level": float(hierarchy_level),
            "macro_horizon": float(macro_horizon),
            "macro_block": float(block),
            "high_population_size": float(high_population_size),
            "low_population_size": float(low_population_size),
            "elite_frac": float(elite_frac),
            "iterations": float(iterations),
            "smoothing": float(smoothing),
            "execute_steps": float(execute_steps),
            "replans": 0.0,
            "mean_high_energy": math.inf,
            "mean_low_subgoal_energy": math.inf,
            "prior_samples": float(prior_samples),
            "high_score_mode": high_score_mode,
        }

    current = start.copy()
    steps = 0
    high_energies: list[float] = []
    low_energies: list[float] = []
    replans = 0
    while steps < limit and not is_terminal_state(world, current, goal, clue_mask):
        device = next(model.parameters()).device
        task_ids = torch.full((1,), world.task_id, dtype=torch.long, device=device)
        current_tensor = torch.as_tensor(current[None, ...], dtype=torch.long, device=device)
        initial_tensor = torch.as_tensor(start[None, ...], dtype=torch.long, device=device)
        goal_tensor = torch.as_tensor(goal[None, ...], dtype=torch.long, device=device)
        current_latent = model.encoder(current_tensor, task_ids=task_ids)
        initial_latent = model.encoder(initial_tensor, task_ids=task_ids)
        goal_latent = model.target_encoder(goal_tensor, task_ids=task_ids)
        prior = estimate_macro_action_prior(
            model,
            world,
            current,
            goal,
            clue_mask,
            rng,
            hierarchy_level=hierarchy_level,
            samples=prior_samples,
        )
        high_plan = high_level_subgoal_cem(
            model,
            current_latent,
            goal_latent,
            rng,
            hierarchy_level=hierarchy_level,
            macro_horizon=macro_horizon,
            population_size=high_population_size,
            elite_frac=elite_frac,
            iterations=iterations,
            smoothing=smoothing,
            prior=prior,
            initial_latent=initial_latent,
            score_mode=high_score_mode,
        )
        remaining = max(1, limit - steps)
        low_plan = low_level_subgoal_cem(
            model,
            world,
            current,
            goal,
            clue_mask,
            rng,
            subgoal_latent=high_plan["subgoal_latent"],
            horizon=min(block, remaining),
            population_size=low_population_size,
            elite_frac=elite_frac,
            iterations=iterations,
            smoothing=smoothing,
        )
        actions = list(low_plan["actions"])
        if not actions:
            break
        prefix = actions[: min(execute_steps, len(actions), limit - steps)]
        if not prefix:
            break
        for action in prefix:
            current = apply_planning_action(world, current, action, clue_mask)
        steps += len(prefix)
        replans += 1
        high_energies.append(float(high_plan["energy"]))
        low_energies.append(float(low_plan["energy"]))

    return {
        "solved": float(world.is_goal(current, goal)),
        "terminal": float(is_terminal_state(world, current, goal, clue_mask)),
        "steps": float(steps),
        "energy": float(encoded_state_energy(model, world, current, goal)),
        "remaining_hamming": float(np.not_equal(current, goal).sum()),
        "final_state": board_as_list(current),
        "goal_state": board_as_list(goal),
        "mismatches": mismatch_records(current, goal),
        "hierarchy_level": float(hierarchy_level),
        "macro_horizon": float(macro_horizon),
        "macro_block": float(block),
        "high_population_size": float(high_population_size),
        "low_population_size": float(low_population_size),
        "elite_frac": float(elite_frac),
        "iterations": float(iterations),
        "smoothing": float(smoothing),
        "execute_steps": float(execute_steps),
        "replans": float(replans),
        "mean_high_energy": mean(high_energies),
        "mean_low_subgoal_energy": mean(low_energies),
        "prior_samples": float(prior_samples),
        "high_score_mode": high_score_mode,
    }


@torch.no_grad()
def evaluate_recursive_hierarchical_subgoal_planning(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    examples: list[PuzzleExample],
    rng: np.random.Generator,
    *,
    num_examples: int,
    max_steps: int,
    hierarchy_level: int,
    macro_horizon: int,
    high_population_size: int,
    low_population_size: int,
    elite_frac: float,
    iterations: int,
    smoothing: float,
    execute_steps: int,
    prior_samples: int,
    high_score_mode: str = "latent_goal",
    optimizer: str = "cem",
    gd_steps: int = 32,
    gd_lr: float = 0.05,
    reachability_weight: float = 0.01,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if num_examples <= 0:
        return {}, []
    validate_hierarchical_subgoal_level(model, hierarchy_level)
    optimizer = resolve_subgoal_high_optimizer(optimizer)
    high_score_mode = resolve_subgoal_high_score_mode(model, high_score_mode)
    items: list[dict[str, Any]] = []
    for example_index in range(num_examples):
        example = examples[int(rng.integers(0, len(examples)))]
        plan = recursive_hierarchical_subgoal_plan(
            model,
            world,
            example,
            rng,
            max_steps=max_steps,
            hierarchy_level=hierarchy_level,
            macro_horizon=macro_horizon,
            high_population_size=high_population_size,
            low_population_size=low_population_size,
            elite_frac=elite_frac,
            iterations=iterations,
            smoothing=smoothing,
            execute_steps=execute_steps,
            prior_samples=prior_samples,
            high_score_mode=high_score_mode,
            optimizer=optimizer,
            gd_steps=gd_steps,
            gd_lr=gd_lr,
            reachability_weight=reachability_weight,
        )
        plan["example_index"] = int(example_index)
        items.append(plan)
    mode = f"{high_score_mode}_{optimizer}_recursive_subgoal"
    return (
        {mode: summarize_plan_summaries(items)},
        flatten_recursive_hierarchical_subgoal_records(items),
    )


@torch.no_grad()
def recursive_hierarchical_subgoal_plan(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    example: PuzzleExample,
    rng: np.random.Generator,
    *,
    max_steps: int,
    hierarchy_level: int,
    macro_horizon: int,
    high_population_size: int,
    low_population_size: int,
    elite_frac: float,
    iterations: int,
    smoothing: float,
    execute_steps: int,
    prior_samples: int,
    high_score_mode: str = "latent_goal",
    optimizer: str = "cem",
    gd_steps: int = 32,
    gd_lr: float = 0.05,
    reachability_weight: float = 0.01,
) -> dict[str, Any]:
    validate_hierarchical_subgoal_level(model, hierarchy_level)
    high_score_mode = resolve_subgoal_high_score_mode(model, high_score_mode)
    optimizer = resolve_subgoal_high_optimizer(optimizer)
    start = world.validate_state(example.state).copy()
    goal = world.validate_state(example.goal)
    clue_mask = clue_mask_for_planning(world, start)
    limit = min(int(max_steps), terminal_step_limit(world, example, max_steps))
    block = int(model.hierarchy_span) ** int(hierarchy_level)
    high_population_size = max(1, int(high_population_size))
    low_population_size = max(1, int(low_population_size))
    iterations = max(1, int(iterations))
    execute_steps = max(1, int(execute_steps))
    macro_horizon = max(1, int(macro_horizon))
    gd_steps = max(1, int(gd_steps))
    gd_lr = max(float(gd_lr), 1.0e-8)
    smoothing = min(max(float(smoothing), 0.0), 1.0)

    metadata = {
        "hierarchy_level": float(hierarchy_level),
        "macro_horizon": float(macro_horizon),
        "macro_block": float(block),
        "high_population_size": float(high_population_size),
        "low_population_size": float(low_population_size),
        "elite_frac": float(elite_frac),
        "iterations": float(iterations),
        "smoothing": float(smoothing),
        "execute_steps": float(execute_steps),
        "prior_samples": float(prior_samples),
        "high_score_mode": high_score_mode,
        "optimizer": optimizer,
        "gd_steps": float(gd_steps),
        "gd_lr": float(gd_lr),
        "reachability_weight": float(reachability_weight),
    }
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
            "replans": 0.0,
            "mean_top_energy": 0.0,
            "mean_recursive_energy": 0.0,
            "mean_leaf_subgoal_energy": 0.0,
            **metadata,
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
            "replans": 0.0,
            "mean_top_energy": math.inf,
            "mean_recursive_energy": math.inf,
            "mean_leaf_subgoal_energy": math.inf,
            **metadata,
        }

    current = start.copy()
    steps = 0
    replans = 0
    top_energies: list[float] = []
    recursive_energies: list[float] = []
    leaf_energies: list[float] = []
    while steps < limit and not is_terminal_state(world, current, goal, clue_mask):
        device = next(model.parameters()).device
        task_ids = torch.full((1,), world.task_id, dtype=torch.long, device=device)
        initial_tensor = torch.as_tensor(start[None, ...], dtype=torch.long, device=device)
        goal_tensor = torch.as_tensor(goal[None, ...], dtype=torch.long, device=device)
        initial_latent = model.encoder(initial_tensor, task_ids=task_ids)
        goal_latent = model.target_encoder(goal_tensor, task_ids=task_ids)
        action_plan = recursive_subgoal_action_plan(
            model,
            world,
            current,
            goal,
            start,
            clue_mask,
            rng,
            target_latent=goal_latent,
            initial_latent=initial_latent,
            level=hierarchy_level,
            top_level=hierarchy_level,
            top_score_mode=high_score_mode,
            optimizer=optimizer,
            macro_horizon=macro_horizon,
            high_population_size=high_population_size,
            low_population_size=low_population_size,
            elite_frac=elite_frac,
            iterations=iterations,
            smoothing=smoothing,
            prior_samples=prior_samples,
            gd_steps=gd_steps,
            gd_lr=gd_lr,
            reachability_weight=reachability_weight,
            steps_remaining=limit - steps,
        )
        actions = list(action_plan["actions"])
        if not actions:
            break
        prefix = actions[: min(execute_steps, len(actions), limit - steps)]
        if not prefix:
            break
        for action in prefix:
            current = apply_planning_action(world, current, action, clue_mask)
        steps += len(prefix)
        replans += 1
        level_records = action_plan.get("level_records", [])
        if level_records:
            top_energies.append(float(level_records[0]["energy"]))
            recursive_energies.extend(float(record["energy"]) for record in level_records)
            leaf_energies.append(float(level_records[-1]["energy"]))

    return {
        "solved": float(world.is_goal(current, goal)),
        "terminal": float(is_terminal_state(world, current, goal, clue_mask)),
        "steps": float(steps),
        "energy": float(encoded_state_energy(model, world, current, goal)),
        "remaining_hamming": float(np.not_equal(current, goal).sum()),
        "final_state": board_as_list(current),
        "goal_state": board_as_list(goal),
        "mismatches": mismatch_records(current, goal),
        "replans": float(replans),
        "mean_top_energy": mean(top_energies),
        "mean_recursive_energy": mean(recursive_energies),
        "mean_leaf_subgoal_energy": mean(leaf_energies),
        **metadata,
    }


@torch.no_grad()
def recursive_subgoal_action_plan(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    current: np.ndarray,
    goal: np.ndarray,
    initial_state: np.ndarray,
    clue_mask: np.ndarray | None,
    rng: np.random.Generator,
    *,
    target_latent: torch.Tensor,
    initial_latent: torch.Tensor,
    level: int,
    top_level: int,
    top_score_mode: str,
    optimizer: str,
    macro_horizon: int,
    high_population_size: int,
    low_population_size: int,
    elite_frac: float,
    iterations: int,
    smoothing: float,
    prior_samples: int,
    gd_steps: int,
    gd_lr: float,
    reachability_weight: float,
    steps_remaining: int,
) -> dict[str, Any]:
    if level <= 0:
        low_plan = low_level_subgoal_cem(
            model,
            world,
            current,
            goal,
            clue_mask,
            rng,
            subgoal_latent=target_latent,
            horizon=min(max(1, int(model.hierarchy_span)), max(1, int(steps_remaining))),
            population_size=low_population_size,
            elite_frac=elite_frac,
            iterations=iterations,
            smoothing=smoothing,
        )
        return {
            "actions": list(low_plan["actions"]),
            "level_records": [
                {
                    "level": 0,
                    "optimizer": "cem_discrete",
                    "score_mode": "latent_goal",
                    "macro_horizon": float(min(max(1, int(model.hierarchy_span)), max(1, int(steps_remaining)))),
                    "energy": float(low_plan["energy"]),
                }
            ],
        }

    device = next(model.parameters()).device
    task_ids = torch.full((1,), world.task_id, dtype=torch.long, device=device)
    current_tensor = torch.as_tensor(current[None, ...], dtype=torch.long, device=device)
    current_latent = model.encoder(current_tensor, task_ids=task_ids)
    level_score_mode = top_score_mode if level == top_level else "latent_goal"
    level_horizon = max(1, int(macro_horizon) if level == top_level else int(model.hierarchy_span))
    prior = estimate_macro_action_prior(
        model,
        world,
        current,
        goal,
        clue_mask,
        rng,
        hierarchy_level=level,
        samples=prior_samples,
    )
    high_plan = optimize_high_level_subgoal(
        model,
        current_latent,
        target_latent,
        rng,
        hierarchy_level=level,
        macro_horizon=level_horizon,
        population_size=high_population_size,
        elite_frac=elite_frac,
        iterations=iterations,
        smoothing=smoothing,
        prior=prior,
        initial_latent=initial_latent,
        score_mode=level_score_mode,
        optimizer=optimizer,
        gd_steps=gd_steps,
        gd_lr=gd_lr,
        reachability_weight=reachability_weight,
    )
    lower = recursive_subgoal_action_plan(
        model,
        world,
        current,
        goal,
        initial_state,
        clue_mask,
        rng,
        target_latent=high_plan["subgoal_latent"],
        initial_latent=initial_latent,
        level=level - 1,
        top_level=top_level,
        top_score_mode=top_score_mode,
        optimizer=optimizer,
        macro_horizon=macro_horizon,
        high_population_size=high_population_size,
        low_population_size=low_population_size,
        elite_frac=elite_frac,
        iterations=iterations,
        smoothing=smoothing,
        prior_samples=prior_samples,
        gd_steps=gd_steps,
        gd_lr=gd_lr,
        reachability_weight=reachability_weight,
        steps_remaining=steps_remaining,
    )
    return {
        "actions": list(lower["actions"]),
        "level_records": [
            {
                "level": int(level),
                "optimizer": optimizer,
                "score_mode": level_score_mode,
                "macro_horizon": float(level_horizon),
                "energy": float(high_plan["energy"]),
            },
            *lower.get("level_records", []),
        ],
    }


def validate_hierarchical_subgoal_level(model: ActionConditionedWorldModel, hierarchy_level: int) -> None:
    if int(hierarchy_level) <= 0:
        raise ValueError("hierarchical subgoal CEM requires --subgoal-hierarchy-level > 0.")
    if int(hierarchy_level) >= model.hierarchy_levels:
        raise ValueError("subgoal hierarchy level must be lower than model.hierarchy_levels.")
    if getattr(model, "checkpoint_missing_hierarchy_action_encoders", False):
        raise ValueError("hierarchical subgoal CEM requires a checkpoint with trained action encoders.")


def resolve_subgoal_high_score_mode(model: ActionConditionedWorldModel, high_score_mode: str) -> str:
    mode = str(high_score_mode)
    if mode not in {"latent_goal", "goal_energy", "goal_value", "macro_action_advantage"}:
        raise ValueError(
            "subgoal high score must be 'latent_goal', 'goal_energy', 'goal_value', or 'macro_action_advantage'."
        )
    if mode in {"goal_energy", "goal_value"} and not model.use_goal_energy_head:
        raise ValueError(f"{mode} subgoal scoring requires a model with use_goal_energy_head=True.")
    if mode == "macro_action_advantage" and not model.use_macro_action_value_head:
        raise ValueError("macro_action_advantage subgoal scoring requires model.use_macro_action_value_head=True.")
    return mode


def resolve_subgoal_high_optimizer(optimizer: str) -> str:
    mode = str(optimizer)
    if mode not in {"cem", "gd", "gd_reachability"}:
        raise ValueError("recursive subgoal optimizer must be 'cem', 'gd', or 'gd_reachability'.")
    return mode


def optimize_high_level_subgoal(
    model: ActionConditionedWorldModel,
    current_latent: torch.Tensor,
    goal_latent: torch.Tensor,
    rng: np.random.Generator,
    *,
    hierarchy_level: int,
    macro_horizon: int,
    population_size: int,
    elite_frac: float,
    iterations: int,
    smoothing: float,
    prior: torch.Tensor | None,
    initial_latent: torch.Tensor | None,
    score_mode: str,
    optimizer: str,
    gd_steps: int,
    gd_lr: float,
    reachability_weight: float,
) -> dict[str, Any]:
    optimizer = resolve_subgoal_high_optimizer(optimizer)
    if optimizer == "cem":
        return high_level_subgoal_cem(
            model,
            current_latent,
            goal_latent,
            rng,
            hierarchy_level=hierarchy_level,
            macro_horizon=macro_horizon,
            population_size=population_size,
            elite_frac=elite_frac,
            iterations=iterations,
            smoothing=smoothing,
            prior=prior,
            initial_latent=initial_latent,
            score_mode=score_mode,
        )
    return high_level_subgoal_gradient(
        model,
        current_latent,
        goal_latent,
        hierarchy_level=hierarchy_level,
        macro_horizon=macro_horizon,
        prior=prior,
        initial_latent=initial_latent,
        score_mode=score_mode,
        steps=gd_steps,
        lr=gd_lr,
        reachability_weight=reachability_weight if optimizer == "gd_reachability" else 0.0,
    )


@torch.no_grad()
def high_level_subgoal_cem(
    model: ActionConditionedWorldModel,
    current_latent: torch.Tensor,
    goal_latent: torch.Tensor,
    rng: np.random.Generator,
    *,
    hierarchy_level: int,
    macro_horizon: int,
    population_size: int,
    elite_frac: float,
    iterations: int,
    smoothing: float,
    prior: torch.Tensor | None,
    initial_latent: torch.Tensor | None = None,
    score_mode: str = "latent_goal",
) -> dict[str, Any]:
    del rng
    device = current_latent.device
    hidden_size = int(current_latent.shape[-1])
    population_size = max(1, int(population_size))
    macro_horizon = max(1, int(macro_horizon))
    iterations = max(1, int(iterations))
    elite_count = max(1, min(population_size, int(math.ceil(population_size * float(elite_frac)))))
    smoothing = min(max(float(smoothing), 0.0), 1.0)
    if prior is not None and prior.numel() > 0:
        prior = prior.to(device=device, dtype=current_latent.dtype)
        base_mu = prior.mean(dim=0)
        if prior.shape[0] > 1:
            base_std = prior.std(dim=0, unbiased=False).clamp_min(1.0e-3)
        else:
            base_std = torch.ones(hidden_size, dtype=current_latent.dtype, device=device)
    else:
        base_mu = torch.zeros(hidden_size, dtype=current_latent.dtype, device=device)
        base_std = torch.ones(hidden_size, dtype=current_latent.dtype, device=device)
    mu = base_mu.unsqueeze(0).expand(macro_horizon, -1).clone()
    std = base_std.unsqueeze(0).expand(macro_horizon, -1).clone()
    best_energy = math.inf
    best_subgoal = current_latent.detach().clone()
    best_sequence = torch.zeros(macro_horizon, hidden_size, dtype=current_latent.dtype, device=device)
    goal_batch = goal_latent.expand(population_size, -1, -1)
    current_batch = current_latent.expand(population_size, -1, -1)
    if score_mode in {"goal_energy", "goal_value", "macro_action_advantage"}:
        if initial_latent is None:
            raise ValueError("initial_latent is required for learned high-level subgoal scoring.")
        initial_batch = initial_latent.expand(population_size, -1, -1)
    else:
        initial_batch = None

    for _ in range(iterations):
        samples = mu.unsqueeze(0) + std.unsqueeze(0) * torch.randn(
            population_size,
            macro_horizon,
            hidden_size,
            dtype=current_latent.dtype,
            device=device,
        )
        latents = current_batch.clone()
        first_subgoals = None
        macro_scores = []
        for step in range(macro_horizon):
            if score_mode == "macro_action_advantage":
                if initial_batch is None:
                    raise RuntimeError("initial_batch was not prepared for macro_action_advantage scoring.")
                macro_scores.append(
                    model.predict_macro_action_value_from_latents(latents, initial_batch, samples[:, step])
                )
            latents = model.predict_latent_from_abstract_action(
                latents,
                samples[:, step],
                level=hierarchy_level,
            )
            if step == 0:
                first_subgoals = latents
        if score_mode == "latent_goal":
            energies = F.mse_loss(latents, goal_batch, reduction="none").mean(dim=(1, 2))
        elif score_mode == "goal_energy":
            if initial_batch is None:
                raise RuntimeError("initial_batch was not prepared for goal_energy scoring.")
            energies = model.predict_goal_energy_from_latents(latents, initial_batch)
        elif score_mode == "goal_value":
            if initial_batch is None:
                raise RuntimeError("initial_batch was not prepared for goal_value scoring.")
            energies = -model.predict_goal_energy_from_latents(latents, initial_batch)
        elif score_mode == "macro_action_advantage":
            if not macro_scores:
                raise RuntimeError("macro_action_advantage scoring did not produce any step scores.")
            energies = -torch.stack(macro_scores, dim=0).sum(dim=0)
        else:
            raise ValueError(f"Unsupported high-level score mode: {score_mode}")
        order = torch.argsort(energies)
        best_index = int(order[0].detach().cpu().item())
        energy = float(energies[best_index].detach().cpu().item())
        if energy < best_energy:
            best_energy = energy
            if first_subgoals is None:
                raise RuntimeError("high-level CEM did not produce a subgoal.")
            best_subgoal = first_subgoals[best_index : best_index + 1].detach().clone()
            best_sequence = samples[best_index].detach().clone()
        elites = samples[order[:elite_count]]
        elite_mu = elites.mean(dim=0)
        elite_std = elites.std(dim=0, unbiased=False).clamp_min(1.0e-3)
        mu = smoothing * mu + (1.0 - smoothing) * elite_mu
        std = smoothing * std + (1.0 - smoothing) * elite_std

    return {
        "subgoal_latent": best_subgoal,
        "energy": float(best_energy),
        "latent_action_sequence": best_sequence,
        "high_score_mode": score_mode,
    }


def high_level_subgoal_gradient(
    model: ActionConditionedWorldModel,
    current_latent: torch.Tensor,
    goal_latent: torch.Tensor,
    *,
    hierarchy_level: int,
    macro_horizon: int,
    prior: torch.Tensor | None,
    initial_latent: torch.Tensor | None = None,
    score_mode: str = "latent_goal",
    steps: int = 32,
    lr: float = 0.05,
    reachability_weight: float = 0.0,
) -> dict[str, Any]:
    device = current_latent.device
    dtype = current_latent.dtype
    hidden_size = int(current_latent.shape[-1])
    macro_horizon = max(1, int(macro_horizon))
    steps = max(1, int(steps))
    lr = max(float(lr), 1.0e-8)
    reachability_weight = max(float(reachability_weight), 0.0)
    if score_mode in {"goal_energy", "goal_value", "macro_action_advantage"} and initial_latent is None:
        raise ValueError("initial_latent is required for learned high-level subgoal scoring.")
    if prior is not None and prior.numel() > 0:
        prior = prior.to(device=device, dtype=dtype)
        base_mu = prior.mean(dim=0)
        if prior.shape[0] > 1:
            base_std = prior.std(dim=0, unbiased=False).clamp_min(1.0e-3)
        else:
            base_std = torch.ones(hidden_size, dtype=dtype, device=device)
    else:
        base_mu = torch.zeros(hidden_size, dtype=dtype, device=device)
        base_std = torch.ones(hidden_size, dtype=dtype, device=device)
    actions = base_mu.unsqueeze(0).expand(macro_horizon, -1).clone().detach().requires_grad_(True)
    first_moment = torch.zeros_like(actions)
    second_moment = torch.zeros_like(actions)
    best_energy = math.inf
    best_subgoal = current_latent.detach().clone()
    best_sequence = actions.detach().clone()

    with torch.enable_grad():
        for step_index in range(steps):
            energy, first_subgoal = high_level_subgoal_energy(
                model,
                current_latent,
                goal_latent,
                actions,
                hierarchy_level=hierarchy_level,
                initial_latent=initial_latent,
                score_mode=score_mode,
            )
            if reachability_weight > 0.0:
                if prior is not None and prior.numel() > 0:
                    penalty = ((actions - base_mu.unsqueeze(0)) / base_std.unsqueeze(0)).pow(2).mean()
                else:
                    penalty = actions.pow(2).mean()
                energy = energy + reachability_weight * penalty
            energy_value = float(energy.detach().cpu().item())
            if energy_value < best_energy:
                best_energy = energy_value
                best_subgoal = first_subgoal.detach().clone()
                best_sequence = actions.detach().clone()
            (gradient,) = torch.autograd.grad(energy, actions, allow_unused=False)
            with torch.no_grad():
                beta1 = 0.9
                beta2 = 0.999
                first_moment.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                second_moment.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
                corrected_first = first_moment / (1.0 - beta1 ** (step_index + 1))
                corrected_second = second_moment / (1.0 - beta2 ** (step_index + 1))
                actions.sub_(lr * corrected_first / (corrected_second.sqrt() + 1.0e-8))

    return {
        "subgoal_latent": best_subgoal,
        "energy": float(best_energy),
        "latent_action_sequence": best_sequence,
        "high_score_mode": score_mode,
    }


def high_level_subgoal_energy(
    model: ActionConditionedWorldModel,
    current_latent: torch.Tensor,
    goal_latent: torch.Tensor,
    actions: torch.Tensor,
    *,
    hierarchy_level: int,
    initial_latent: torch.Tensor | None,
    score_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if actions.ndim != 2:
        raise ValueError("gradient high-level actions must have shape [macro_horizon, hidden].")
    latents = current_latent
    first_subgoal = current_latent
    macro_scores = []
    for step in range(actions.shape[0]):
        action = actions[step : step + 1]
        if score_mode == "macro_action_advantage":
            if initial_latent is None:
                raise RuntimeError("initial_latent is required for macro_action_advantage scoring.")
            macro_scores.append(model.predict_macro_action_value_from_latents(latents, initial_latent, action))
        latents = model.predict_latent_from_abstract_action(latents, action, level=hierarchy_level)
        if step == 0:
            first_subgoal = latents
    if score_mode == "latent_goal":
        energy = F.mse_loss(latents, goal_latent, reduction="mean")
    elif score_mode == "goal_energy":
        if initial_latent is None:
            raise RuntimeError("initial_latent is required for goal_energy scoring.")
        energy = model.predict_goal_energy_from_latents(latents, initial_latent).mean()
    elif score_mode == "goal_value":
        if initial_latent is None:
            raise RuntimeError("initial_latent is required for goal_value scoring.")
        energy = -model.predict_goal_energy_from_latents(latents, initial_latent).mean()
    elif score_mode == "macro_action_advantage":
        if not macro_scores:
            raise RuntimeError("macro_action_advantage scoring did not produce any step scores.")
        energy = -torch.stack(macro_scores, dim=0).sum()
    else:
        raise ValueError(f"Unsupported high-level score mode: {score_mode}")
    return energy, first_subgoal


@torch.no_grad()
def estimate_macro_action_prior(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    start: np.ndarray,
    goal: np.ndarray,
    clue_mask: np.ndarray | None,
    rng: np.random.Generator,
    *,
    hierarchy_level: int,
    samples: int,
) -> torch.Tensor | None:
    samples = int(samples)
    if samples <= 0:
        return None
    action_space = cem_action_space(world, start, clue_mask)
    if not action_space["positions"] or not action_space["values"]:
        return None
    block = int(model.hierarchy_span) ** int(hierarchy_level)
    cell_probs = np.full((block, len(action_space["positions"])), 1.0 / len(action_space["positions"]))
    value_probs = np.full((block, len(action_space["values"])), 1.0 / len(action_space["values"]))
    encoded_sequences: list[list[list[int]]] = []
    max_attempts = max(samples * 4, samples)
    for _ in range(max_attempts):
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
        actions = candidate["actions"]
        if len(actions) != block:
            continue
        encoded_sequences.append(
            [
                [world.task_id, action.row, action.col, action.value]
                for action in actions
            ]
        )
        if len(encoded_sequences) >= samples:
            break
    if not encoded_sequences:
        return None
    device = next(model.parameters()).device
    action_tensor = torch.as_tensor(encoded_sequences, dtype=torch.long, device=device)
    return model.encode_hierarchy_action(action_tensor, level=hierarchy_level).detach()


@torch.no_grad()
def low_level_subgoal_cem(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    start: np.ndarray,
    goal: np.ndarray,
    clue_mask: np.ndarray | None,
    rng: np.random.Generator,
    *,
    subgoal_latent: torch.Tensor,
    horizon: int,
    population_size: int,
    elite_frac: float,
    iterations: int,
    smoothing: float,
) -> dict[str, Any]:
    action_space = cem_action_space(world, start, clue_mask)
    if not action_space["positions"] or not action_space["values"]:
        return {"state": start.copy(), "actions": [], "energy": math.inf}
    horizon = max(1, int(horizon))
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
        candidates = [
            sample_cem_rollout(
                world,
                start,
                goal,
                clue_mask,
                rng,
                cell_probs=cell_probs,
                value_probs=value_probs,
                action_space=action_space,
            )
            for _sample_index in range(population_size)
        ]
        scores = primitive_subgoal_scores(
            model,
            world,
            start,
            candidates,
            subgoal_latent=subgoal_latent,
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

    return {"state": best_state, "actions": best_actions, "energy": float(-best_score)}


@torch.no_grad()
def primitive_subgoal_scores(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    initial_state: np.ndarray,
    candidates: list[dict[str, Any]],
    *,
    subgoal_latent: torch.Tensor,
) -> list[float]:
    device = next(model.parameters()).device
    scores: list[float] = [math.nan for _ in candidates]
    state_tensor = torch.as_tensor(initial_state[None, ...], dtype=torch.long, device=device)
    task_ids = torch.full((1,), world.task_id, dtype=torch.long, device=device)
    start_latent = model.encoder(state_tensor, task_ids=task_ids)
    height, width = int(initial_state.shape[-2]), int(initial_state.shape[-1])
    by_length: dict[int, list[int]] = {}
    for index, candidate in enumerate(candidates):
        by_length.setdefault(len(candidate["actions"]), []).append(index)
    for length, indices in by_length.items():
        latents = start_latent.expand(len(indices), -1, -1).clone()
        if length > 0:
            action_tensor = torch.as_tensor(
                [
                    [
                        [world.task_id, action.row, action.col, action.value]
                        for action in candidates[index]["actions"]
                    ]
                    for index in indices
                ],
                dtype=torch.long,
                device=device,
            )
            for step in range(length):
                latents = model.predict_latent_from_latent(
                    latents,
                    action_tensor[:, step],
                    height=height,
                    width=width,
                )
        target = subgoal_latent.expand(len(indices), -1, -1)
        batch_scores = -F.mse_loss(latents, target, reduction="none").mean(dim=(1, 2))
        for index, score in zip(indices, batch_scores.detach().cpu().tolist(), strict=True):
            scores[index] = float(score)
    return scores


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
    planning_score: str = "latent_goal",
) -> dict[str, Any]:
    planning_score = resolve_planning_score_mode(model, planning_score)
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
            "trajectory_states": [board_as_list(start)],
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
            next_items: list[tuple[WorldAction, np.ndarray, float]] = []
            if planning_score in {"goal_energy", "goal_value"}:
                next_states: list[np.ndarray] = []
                next_actions: list[WorldAction] = []
                for action in legal:
                    try:
                        next_states.append(apply_planning_action(world, state, action, clue_mask))
                        next_actions.append(action)
                    except ValueError:
                        continue
                scores = score_symbolic_states_to_goal(
                    model,
                    world,
                    next_states,
                    goal,
                    start,
                    planning_score=planning_score,
                )
                order = np.argsort(np.asarray(scores, dtype=np.float64))[::-1].tolist()[: max(1, branch_size)]
                next_items = [
                    (next_actions[index], next_states[index], float(scores[index]))
                    for index in order
                ]
            elif planning_score == "action_advantage":
                scores = model.score_actions_with_value_head(
                    torch.as_tensor(state, dtype=torch.long),
                    torch.as_tensor(start, dtype=torch.long),
                    legal,
                    world.task_id,
                )
                order = scores.argsort(descending=True).detach().cpu().tolist()[: max(1, branch_size)]
                for index in order:
                    action = legal[index]
                    try:
                        next_state = apply_planning_action(world, state, action, clue_mask)
                    except ValueError:
                        continue
                    next_items.append((action, next_state, float(scores[index].detach().cpu().item())))
            else:
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
                    next_items.append((action, next_state, float(scores[index].detach().cpu().item())))
            for action, next_state, score in next_items:
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
                if planning_score == "goal_energy":
                    energy = float(-score)
                elif planning_score == "goal_value":
                    energy = float(-score)
                elif planning_score == "action_advantage":
                    energy = float(-score)
                else:
                    energy = float(F.mse_loss(next_latent, goal_latent).detach().cpu().item())
                next_beam = {
                    "state": next_state,
                    "latent": next_latent,
                    "steps": next_steps,
                    "energy": energy,
                    "trajectory_states": [*beam["trajectory_states"], board_as_list(next_state)],
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
        "trajectory_states": best["trajectory_states"],
        "mismatches": mismatch_records(best["state"], goal),
        "planning_score": planning_score,
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
    planning_score: str = "latent_goal",
) -> dict[str, Any]:
    planning_score = resolve_planning_score_mode(model, planning_score)
    start = world.validate_state(example.state).copy()
    goal = world.validate_state(example.goal)
    clue_mask = clue_mask_for_planning(world, start)
    active = [
        {
            "state": start,
            "steps": 0,
            "energy": encoded_state_energy(model, world, start, goal),
            "trajectory_states": [board_as_list(start)],
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
            next_states: list[np.ndarray] = []
            next_actions: list[WorldAction] = []
            for action in legal:
                try:
                    next_states.append(apply_planning_action(world, state, action, clue_mask))
                    next_actions.append(action)
                except ValueError:
                    continue
            if planning_score == "action_advantage":
                scores_tensor = model.score_actions_with_value_head(
                    torch.as_tensor(state, dtype=torch.long),
                    torch.as_tensor(start, dtype=torch.long),
                    next_actions,
                    world.task_id,
                )
                scores = [float(item) for item in scores_tensor.detach().cpu().tolist()]
            else:
                scores = score_symbolic_states_to_goal(
                    model,
                    world,
                    next_states,
                    goal,
                    start,
                    planning_score=planning_score,
                )
            order = np.argsort(np.asarray(scores, dtype=np.float64))[::-1].tolist()[: max(1, branch_size)]
            for index in order:
                next_state = next_states[index]
                next_beam = {
                    "state": next_state,
                    "steps": int(beam["steps"]) + 1,
                    "energy": float(-scores[index]),
                    "trajectory_states": [*beam["trajectory_states"], board_as_list(next_state)],
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
        "trajectory_states": best["trajectory_states"],
        "mismatches": mismatch_records(best["state"], goal),
        "planning_score": planning_score,
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


@torch.no_grad()
def score_symbolic_states_to_goal(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    states: list[np.ndarray],
    goal: np.ndarray,
    initial_state: np.ndarray,
    *,
    planning_score: str,
) -> list[float]:
    if not states:
        return []
    mode = resolve_planning_score_mode(model, planning_score)
    state_array = np.stack(states, axis=0)
    goal_array = np.repeat(goal[None, ...], len(states), axis=0)
    if mode == "action_advantage":
        raise ValueError("action_advantage scores actions, not symbolic states.")
    if mode in {"goal_energy", "goal_value"}:
        initial_array = np.repeat(initial_state[None, ...], len(states), axis=0)
        values = model.predict_goal_energy(
            torch.as_tensor(state_array, dtype=torch.long),
            torch.as_tensor(initial_array, dtype=torch.long),
            world.task_id,
        )
        scores = -values if mode == "goal_energy" else values
    else:
        scores = model.score_states_to_goal(
            torch.as_tensor(state_array, dtype=torch.long),
            torch.as_tensor(goal_array, dtype=torch.long),
            world.task_id,
        )
    return [float(item) for item in scores.detach().cpu().tolist()]


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
                "planning_score": item.get("planning_score", "latent_goal"),
                "final_state": item["final_state"],
                "goal_state": item["goal_state"],
                "trajectory_states": item.get("trajectory_states", []),
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
                    "planning_score": item.get("planning_score", "latent_goal"),
                    "final_state": item["final_state"],
                    "goal_state": item["goal_state"],
                    "trajectory_states": item.get("trajectory_states", []),
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
                    "hierarchy_level": float(item.get("hierarchy_level", 0.0)),
                    "final_state": item["final_state"],
                    "goal_state": item["goal_state"],
                    "mismatches": item["mismatches"],
                }
            )
    return records


def flatten_mcts_plan_records(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for item in items:
        records.append(
            {
                "planner": "mcts",
                "score_mode": item["score_mode"],
                "example_index": int(item["example_index"]),
                "solved": float(item["solved"]),
                "terminal": float(item["terminal"]),
                "steps": float(item["steps"]),
                "energy": float(item["energy"]),
                "remaining_hamming": float(item["remaining_hamming"]),
                "simulations": float(item["simulations"]),
                "max_depth": float(item["max_depth"]),
                "exploration": float(item["exploration"]),
                "expansion_actions": float(item["expansion_actions"]),
                "final_state": item["final_state"],
                "goal_state": item["goal_state"],
                "trajectory_states": item.get("trajectory_states", []),
                "mismatches": item["mismatches"],
            }
        )
    return records


def flatten_hierarchical_subgoal_cem_records(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for item in items:
        records.append(
            {
                "planner": "hierarchical_subgoal_cem",
                "mode": f"{item.get('high_score_mode', 'latent_goal')}_subgoal",
                "high_score_mode": item.get("high_score_mode", "latent_goal"),
                "example_index": int(item["example_index"]),
                "solved": float(item["solved"]),
                "terminal": float(item["terminal"]),
                "steps": float(item["steps"]),
                "energy": float(item["energy"]),
                "remaining_hamming": float(item["remaining_hamming"]),
                "hierarchy_level": float(item["hierarchy_level"]),
                "macro_horizon": float(item["macro_horizon"]),
                "macro_block": float(item["macro_block"]),
                "high_population_size": float(item["high_population_size"]),
                "low_population_size": float(item["low_population_size"]),
                "elite_frac": float(item["elite_frac"]),
                "iterations": float(item["iterations"]),
                "smoothing": float(item["smoothing"]),
                "execute_steps": float(item["execute_steps"]),
                "replans": float(item["replans"]),
                "mean_high_energy": float(item["mean_high_energy"]),
                "mean_low_subgoal_energy": float(item["mean_low_subgoal_energy"]),
                "prior_samples": float(item["prior_samples"]),
                "final_state": item["final_state"],
                "goal_state": item["goal_state"],
                "mismatches": item["mismatches"],
            }
        )
    return records


def flatten_recursive_hierarchical_subgoal_records(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for item in items:
        records.append(
            {
                "planner": "recursive_hierarchical_subgoal",
                "mode": f"{item.get('high_score_mode', 'latent_goal')}_{item.get('optimizer', 'cem')}_recursive_subgoal",
                "high_score_mode": item.get("high_score_mode", "latent_goal"),
                "optimizer": item.get("optimizer", "cem"),
                "example_index": int(item["example_index"]),
                "solved": float(item["solved"]),
                "terminal": float(item["terminal"]),
                "steps": float(item["steps"]),
                "energy": float(item["energy"]),
                "remaining_hamming": float(item["remaining_hamming"]),
                "hierarchy_level": float(item["hierarchy_level"]),
                "macro_horizon": float(item["macro_horizon"]),
                "macro_block": float(item["macro_block"]),
                "high_population_size": float(item["high_population_size"]),
                "low_population_size": float(item["low_population_size"]),
                "elite_frac": float(item["elite_frac"]),
                "iterations": float(item["iterations"]),
                "smoothing": float(item["smoothing"]),
                "execute_steps": float(item["execute_steps"]),
                "replans": float(item["replans"]),
                "mean_top_energy": float(item["mean_top_energy"]),
                "mean_recursive_energy": float(item["mean_recursive_energy"]),
                "mean_leaf_subgoal_energy": float(item["mean_leaf_subgoal_energy"]),
                "prior_samples": float(item["prior_samples"]),
                "gd_steps": float(item["gd_steps"]),
                "gd_lr": float(item["gd_lr"]),
                "reachability_weight": float(item["reachability_weight"]),
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


def load_model_checkpoint_state(model: ActionConditionedWorldModel, state: dict[str, torch.Tensor]) -> None:
    load_result = model.load_state_dict(state, strict=False)
    missing = list(load_result.missing_keys)
    unexpected = list(load_result.unexpected_keys)
    allowed_missing = [key for key in missing if key.startswith("higher_action_encoders.")]
    if unexpected or len(allowed_missing) != len(missing):
        raise RuntimeError(
            "Checkpoint/model mismatch. "
            f"missing={missing}, unexpected={unexpected}"
        )
    setattr(model, "checkpoint_missing_hierarchy_action_encoders", bool(allowed_missing))


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


@torch.no_grad()
def goal_energy_calibration_records(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    plan_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not model.use_goal_energy_head:
        return []
    records: list[dict[str, Any]] = []
    for plan in plan_records:
        trajectory = plan.get("trajectory_states") or []
        if not trajectory:
            continue
        states = np.asarray(trajectory, dtype=np.int64)
        goal = np.asarray(plan["goal_state"], dtype=np.int64)
        initial = states[0]
        predicted = predict_goal_energy_values(model, world, states, initial)
        true = encoded_state_energies(model, world, states, goal)
        remaining = np.not_equal(states, goal[None, ...]).sum(axis=(1, 2))
        for step, (pred, target, hamming) in enumerate(zip(predicted, true, remaining, strict=True)):
            records.append(
                {
                    "planner": plan["planner"],
                    "mode": plan["mode"],
                    "variant": plan.get("variant", ""),
                    "reset_cadence": plan.get("reset_cadence"),
                    "example_index": int(plan["example_index"]),
                    "step": int(step),
                    "predicted_goal_energy": float(pred),
                    "true_goal_mse": float(target),
                    "signed_error": float(pred - target),
                    "absolute_error": float(abs(pred - target)),
                    "remaining_hamming": float(hamming),
                    "solved": float(plan["solved"]),
                    "terminal": float(plan["terminal"]),
                    "planning_score": plan.get("planning_score", "latent_goal"),
                }
            )
    return records


@torch.no_grad()
def predict_goal_energy_values(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    states: np.ndarray,
    initial_state: np.ndarray,
) -> list[float]:
    device = next(model.parameters()).device
    state_tensor = torch.as_tensor(states, dtype=torch.long, device=device)
    initial_tensor = torch.as_tensor(
        np.repeat(initial_state[None, ...], len(states), axis=0),
        dtype=torch.long,
        device=device,
    )
    task_ids = torch.full((len(states),), world.task_id, dtype=torch.long, device=device)
    values = model.predict_goal_energy(state_tensor, initial_tensor, task_ids)
    return [float(item) for item in values.detach().cpu().tolist()]


@torch.no_grad()
def encoded_state_energies(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    states: np.ndarray,
    goal: np.ndarray,
) -> list[float]:
    device = next(model.parameters()).device
    task_ids = torch.full((len(states),), world.task_id, dtype=torch.long, device=device)
    state_tensor = torch.as_tensor(states, dtype=torch.long, device=device)
    goal_tensor = torch.as_tensor(
        np.repeat(goal[None, ...], len(states), axis=0),
        dtype=torch.long,
        device=device,
    )
    latents = model.encoder(state_tensor, task_ids=task_ids)
    goal_latents = model.target_encoder(goal_tensor, task_ids=task_ids)
    values = F.mse_loss(latents, goal_latents, reduction="none").mean(dim=(1, 2))
    return [float(item) for item in values.detach().cpu().tolist()]


def write_goal_energy_calibration_plots(destination: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on cluster image.
        (destination / "goal_energy_calibration.plot_error.txt").write_text(
            f"matplotlib unavailable: {exc}\n"
        )
        return

    keys = sorted({(item["planner"], item["mode"], item.get("variant", "")) for item in records})
    plt.figure(figsize=(8, 5))
    for planner, mode, variant in keys:
        subset = [
            item for item in records
            if item["planner"] == planner and item["mode"] == mode and item.get("variant", "") == variant
        ]
        steps = sorted({int(item["step"]) for item in subset})
        errors = [
            mean([item["absolute_error"] for item in subset if int(item["step"]) == step])
            for step in steps
        ]
        label = f"{planner}/{variant or mode}"
        plt.plot(steps, errors, label=label)
    plt.xlabel("planning step")
    plt.ylabel("mean absolute energy error")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(destination / "goal_energy_abs_error_by_step.png")
    plt.close()

    example_keys = []
    for item in records:
        key = (item["planner"], item["mode"], item.get("variant", ""), int(item["example_index"]))
        if key not in example_keys:
            example_keys.append(key)
        if len(example_keys) >= 6:
            break
    for index, (planner, mode, variant, example_index) in enumerate(example_keys):
        subset = [
            item for item in records
            if item["planner"] == planner
            and item["mode"] == mode
            and item.get("variant", "") == variant
            and int(item["example_index"]) == example_index
        ]
        subset.sort(key=lambda item: int(item["step"]))
        steps = [int(item["step"]) for item in subset]
        predicted = [float(item["predicted_goal_energy"]) for item in subset]
        true = [float(item["true_goal_mse"]) for item in subset]
        hamming = [float(item["remaining_hamming"]) for item in subset]
        plt.figure(figsize=(8, 5))
        plt.plot(steps, predicted, marker="o", label="predicted goal energy")
        plt.plot(steps, true, marker="o", label="true goal latent MSE")
        plt.plot(steps, hamming, marker=".", label="remaining Hamming")
        plt.xlabel("planning step")
        plt.ylabel("value")
        plt.title(f"{planner}/{variant or mode} example {example_index}")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(destination / f"goal_energy_example_{index}.png")
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
    parser.add_argument(
        "--planning-score",
        choices=["latent_goal", "goal_energy", "goal_value", "action_advantage"],
        default="latent_goal",
    )
    parser.add_argument("--planning-beam-size", type=int, default=4)
    parser.add_argument("--planning-branch-size", type=int, default=8)
    parser.add_argument("--cem-population", type=int, default=128)
    parser.add_argument("--cem-elite-frac", type=float, default=0.2)
    parser.add_argument("--cem-iterations", type=int, default=4)
    parser.add_argument("--cem-smoothing", type=float, default=0.7)
    parser.add_argument(
        "--cem-score",
        choices=["auto", "goal_energy", "latent_goal", "hierarchical_latent_goal"],
        default="auto",
    )
    parser.add_argument("--cem-hierarchy-level", type=int, default=0)
    parser.add_argument("--mcts-examples", type=int, default=0)
    parser.add_argument("--mcts-simulations", type=int, default=512)
    parser.add_argument("--mcts-depth", type=int, default=8)
    parser.add_argument(
        "--mcts-score",
        choices=["auto", "goal_energy", "goal_value", "latent_goal"],
        default="auto",
    )
    parser.add_argument("--mcts-exploration", type=float, default=1.0)
    parser.add_argument("--mcts-expansion-actions", type=int, default=64)
    parser.add_argument("--mcts-debug-examples", type=int, default=3)
    parser.add_argument("--mcts-debug-actions", type=int, default=12)
    parser.add_argument("--subgoal-cem-examples", type=int, default=0)
    parser.add_argument("--subgoal-hierarchy-level", type=int, default=1)
    parser.add_argument("--subgoal-macro-horizon", type=int, default=3)
    parser.add_argument("--subgoal-high-population", type=int, default=128)
    parser.add_argument("--subgoal-low-population", type=int, default=128)
    parser.add_argument("--subgoal-iterations", type=int, default=4)
    parser.add_argument("--subgoal-elite-frac", type=float, default=0.2)
    parser.add_argument("--subgoal-smoothing", type=float, default=0.7)
    parser.add_argument("--subgoal-execute-steps", type=int, default=1)
    parser.add_argument("--subgoal-prior-samples", type=int, default=64)
    parser.add_argument(
        "--subgoal-high-score",
        choices=["latent_goal", "goal_energy", "goal_value", "macro_action_advantage"],
        default="latent_goal",
    )
    parser.add_argument("--recursive-subgoal-examples", type=int, default=0)
    parser.add_argument("--recursive-hierarchy-level", type=int, default=2)
    parser.add_argument("--recursive-macro-horizon", type=int, default=5)
    parser.add_argument("--recursive-high-population", type=int, default=128)
    parser.add_argument("--recursive-low-population", type=int, default=128)
    parser.add_argument("--recursive-iterations", type=int, default=4)
    parser.add_argument("--recursive-elite-frac", type=float, default=0.2)
    parser.add_argument("--recursive-smoothing", type=float, default=0.7)
    parser.add_argument("--recursive-execute-steps", type=int, default=1)
    parser.add_argument("--recursive-prior-samples", type=int, default=64)
    parser.add_argument(
        "--recursive-high-score",
        choices=["latent_goal", "goal_energy", "goal_value", "macro_action_advantage"],
        default="latent_goal",
    )
    parser.add_argument(
        "--recursive-optimizer",
        choices=["cem", "gd", "gd_reachability"],
        default="cem",
    )
    parser.add_argument("--recursive-gd-steps", type=int, default=32)
    parser.add_argument("--recursive-gd-lr", type=float, default=0.05)
    parser.add_argument("--recursive-reachability-weight", type=float, default=0.01)
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
        planning_score=args.planning_score,
        planning_beam_size=args.planning_beam_size,
        planning_branch_size=args.planning_branch_size,
        cem_population=args.cem_population,
        cem_elite_frac=args.cem_elite_frac,
        cem_iterations=args.cem_iterations,
        cem_smoothing=args.cem_smoothing,
        cem_score=args.cem_score,
        cem_hierarchy_level=args.cem_hierarchy_level,
        mcts_examples=args.mcts_examples,
        mcts_simulations=args.mcts_simulations,
        mcts_depth=args.mcts_depth,
        mcts_score=args.mcts_score,
        mcts_exploration=args.mcts_exploration,
        mcts_expansion_actions=args.mcts_expansion_actions,
        mcts_debug_examples=args.mcts_debug_examples,
        mcts_debug_actions=args.mcts_debug_actions,
        subgoal_cem_examples=args.subgoal_cem_examples,
        subgoal_hierarchy_level=args.subgoal_hierarchy_level,
        subgoal_macro_horizon=args.subgoal_macro_horizon,
        subgoal_high_population=args.subgoal_high_population,
        subgoal_low_population=args.subgoal_low_population,
        subgoal_iterations=args.subgoal_iterations,
        subgoal_elite_frac=args.subgoal_elite_frac,
        subgoal_smoothing=args.subgoal_smoothing,
        subgoal_execute_steps=args.subgoal_execute_steps,
        subgoal_prior_samples=args.subgoal_prior_samples,
        subgoal_high_score=args.subgoal_high_score,
        recursive_subgoal_examples=args.recursive_subgoal_examples,
        recursive_hierarchy_level=args.recursive_hierarchy_level,
        recursive_macro_horizon=args.recursive_macro_horizon,
        recursive_high_population=args.recursive_high_population,
        recursive_low_population=args.recursive_low_population,
        recursive_iterations=args.recursive_iterations,
        recursive_elite_frac=args.recursive_elite_frac,
        recursive_smoothing=args.recursive_smoothing,
        recursive_execute_steps=args.recursive_execute_steps,
        recursive_prior_samples=args.recursive_prior_samples,
        recursive_high_score=args.recursive_high_score,
        recursive_optimizer=args.recursive_optimizer,
        recursive_gd_steps=args.recursive_gd_steps,
        recursive_gd_lr=args.recursive_gd_lr,
        recursive_reachability_weight=args.recursive_reachability_weight,
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
