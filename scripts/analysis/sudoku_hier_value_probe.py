from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from puzzle_jepa.data.worlds import PuzzleExample, PuzzleWorld, SudokuWorld, WorldAction
from puzzle_jepa.eval.diagnostics import (
    apply_planning_action,
    build_mcts_tree,
    build_world,
    clue_mask_for_planning,
    encoded_state_energy,
    estimate_macro_action_prior,
    high_level_subgoal_cem,
    high_level_subgoal_gradient,
    is_terminal_state,
    legal_planning_actions,
    load_examples,
    load_model_checkpoint_state,
    low_level_subgoal_cem,
    mcts_plan,
    mcts_root_debug_record,
    optimize_high_level_subgoal,
    oracle_action_sequence,
    recursive_hierarchical_subgoal_plan,
    score_leaf_state,
    score_symbolic_states_to_goal,
    terminal_step_limit,
)
from puzzle_jepa.models import ActionConditionedWorldModel


DEFAULT_RUN_ROOT = Path("/home/vault/c107fa/c107fa12/sequence-editing/runs")


def load_run(run_root: Path, *, device: torch.device) -> tuple[ActionConditionedWorldModel, PuzzleWorld, list[PuzzleExample], dict[str, Any]]:
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
    model = ActionConditionedWorldModel(vocab_size=world.vocab_size, **model_cfg).to(device)
    load_model_checkpoint_state(model, checkpoint["model"])
    model.eval()
    return model, world, examples, config


def score_states_raw(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    states: list[np.ndarray],
    initial: np.ndarray,
) -> list[float]:
    device = next(model.parameters()).device
    state_tensor = torch.as_tensor(np.stack(states, axis=0), dtype=torch.long, device=device)
    initial_tensor = torch.as_tensor(
        np.repeat(initial[None, ...], len(states), axis=0),
        dtype=torch.long,
        device=device,
    )
    task_ids = torch.full((len(states),), world.task_id, dtype=torch.long, device=device)
    with torch.no_grad():
        values = model.predict_goal_energy(state_tensor, initial_tensor, task_ids)
    return [float(item) for item in values.detach().cpu().tolist()]


def terminal_candidates(
    example: PuzzleExample,
    rng: np.random.Generator,
    *,
    variants_per_level: int,
) -> list[dict[str, Any]]:
    start = np.asarray(example.state, dtype=np.int64)
    goal = np.asarray(example.goal, dtype=np.int64)
    mutable = np.argwhere(start == 0)
    candidates: list[dict[str, Any]] = [
        {
            "kind": "true_solution",
            "state": goal.copy(),
            "wrong_cells": 0,
            "changed_cells": [],
        }
    ]
    for wrong_count in (1, 4, 16):
        if len(mutable) < wrong_count:
            continue
        for variant in range(variants_per_level):
            state = goal.copy()
            choice = rng.choice(len(mutable), size=wrong_count, replace=False)
            changed = []
            for idx in choice:
                row, col = (int(mutable[idx, 0]), int(mutable[idx, 1]))
                allowed = [value for value in range(1, 10) if value != int(goal[row, col])]
                value = int(rng.choice(allowed))
                state[row, col] = value
                changed.append({"row": row, "col": col, "goal": int(goal[row, col]), "value": value})
            candidates.append(
                {
                    "kind": f"corrupt_{wrong_count}",
                    "state": state,
                    "wrong_cells": int(np.not_equal(state, goal).sum()),
                    "changed_cells": changed,
                    "variant": variant,
                }
            )
    for variant in range(variants_per_level):
        state = start.copy()
        for row, col in mutable:
            state[int(row), int(col)] = int(rng.integers(1, 10))
        if np.array_equal(state, goal) and len(mutable) > 0:
            row, col = (int(mutable[0, 0]), int(mutable[0, 1]))
            state[row, col] = (int(goal[row, col]) % 9) + 1
        candidates.append(
            {
                "kind": "random_completion",
                "state": state,
                "wrong_cells": int(np.not_equal(state, goal).sum()),
                "changed_cells": [],
                "variant": variant,
            }
        )
    return candidates


def terminal_discrimination_probe(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    examples: list[PuzzleExample],
    rng: np.random.Generator,
    *,
    score_mode: str,
    num_examples: int,
    variants_per_level: int,
) -> dict[str, Any]:
    rows = []
    pair_margins_by_kind: dict[str, list[float]] = {}
    true_better_by_kind: dict[str, list[float]] = {}
    for example_index, example in enumerate(examples[:num_examples]):
        candidates = terminal_candidates(example, rng, variants_per_level=variants_per_level)
        raw_scores = score_states_raw(model, world, [item["state"] for item in candidates], example.state)
        oracle_energies = [encoded_state_energy(model, world, item["state"], example.goal) for item in candidates]
        true_score = raw_scores[0]
        true_oracle = oracle_energies[0]
        for candidate, raw_score, oracle_energy in zip(candidates, raw_scores, oracle_energies, strict=True):
            if score_mode == "goal_value":
                oriented_margin = true_score - raw_score
                true_better = true_score > raw_score
            elif score_mode == "goal_energy":
                oriented_margin = raw_score - true_score
                true_better = true_score < raw_score
            else:
                raise ValueError(f"Unsupported terminal score mode: {score_mode}")
            row = {
                "example_index": int(example_index),
                "kind": candidate["kind"],
                "wrong_cells": int(candidate["wrong_cells"]),
                "raw_score": float(raw_score),
                "sigmoid": float(1.0 / (1.0 + math.exp(-max(min(raw_score, 60.0), -60.0)))),
                "oracle_latent_energy": float(oracle_energy),
                "true_raw_score": float(true_score),
                "true_oracle_latent_energy": float(true_oracle),
                "true_better": None if candidate["kind"] == "true_solution" else bool(true_better),
                "oriented_margin_vs_true": 0.0 if candidate["kind"] == "true_solution" else float(oriented_margin),
                "changed_cells": candidate.get("changed_cells", [])[:4],
            }
            rows.append(row)
            if candidate["kind"] != "true_solution":
                pair_margins_by_kind.setdefault(candidate["kind"], []).append(float(oriented_margin))
                true_better_by_kind.setdefault(candidate["kind"], []).append(float(true_better))
    margins = [margin for values in pair_margins_by_kind.values() for margin in values]
    wins = [value for values in true_better_by_kind.values() for value in values]
    return {
        "score_mode": score_mode,
        "summary": {
            "examples": int(num_examples),
            "candidate_rows": int(len(rows)),
            "true_better_rate": mean(wins),
            "mean_oriented_margin_vs_wrong": mean(margins),
            "median_oriented_margin_vs_wrong": median(margins),
            "by_kind": {
                kind: {
                    "count": len(values),
                    "true_better_rate": mean(true_better_by_kind[kind]),
                    "mean_oriented_margin_vs_true": mean(values),
                    "median_oriented_margin_vs_true": median(values),
                }
                for kind, values in sorted(pair_margins_by_kind.items())
            },
        },
        "examples": rows[: min(len(rows), 40)],
    }


def local_action_probe(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    examples: list[PuzzleExample],
    rng: np.random.Generator,
    *,
    score_mode: str,
    num_examples: int,
    depths: list[float],
    debug_actions: int,
) -> dict[str, Any]:
    records = []
    best_goal_flags = []
    best_goal_ranks = []
    for example_index, example in enumerate(examples[:num_examples]):
        oracle_actions = oracle_action_sequence(world, example, rng)
        clue_mask = clue_mask_for_planning(world, example.state)
        for depth in depths:
            prefix_steps = int(round(depth * len(oracle_actions)))
            state = np.asarray(example.state, dtype=np.int64).copy()
            for action in oracle_actions[:prefix_steps]:
                state = apply_planning_action(world, state, action, clue_mask)
            legal = legal_planning_actions(world, state, clue_mask)
            if not legal:
                continue
            next_states = [apply_planning_action(world, state, action, clue_mask) for action in legal]
            scores = score_symbolic_states_to_goal(model, world, next_states, example.goal, example.state, planning_score=score_mode)
            goal_indices = [
                index
                for index, action in enumerate(legal)
                if int(state[action.row, action.col]) != int(example.goal[action.row, action.col])
                and action.value == int(example.goal[action.row, action.col])
            ]
            order = np.argsort(np.asarray(scores, dtype=np.float64))[::-1].tolist()
            best_goal_rank = None
            best_goal_score = None
            if goal_indices:
                ranks = [order.index(index) + 1 for index in goal_indices]
                best_goal_rank = int(min(ranks))
                best_goal_index = max(goal_indices, key=lambda index: scores[index])
                best_goal_score = float(scores[best_goal_index])
                best_goal_flags.append(float(best_goal_rank == 1))
                best_goal_ranks.append(float(best_goal_rank))
            top_actions = []
            for rank, index in enumerate(order[:debug_actions], start=1):
                action = legal[index]
                top_actions.append(
                    {
                        "rank": int(rank),
                        "action": asdict(action),
                        "score": float(scores[index]),
                        "raw_value_or_neg_energy": float(scores[index]),
                        "writes_goal_value": bool(action.value == int(example.goal[action.row, action.col])),
                        "cell_current": int(state[action.row, action.col]),
                        "cell_goal": int(example.goal[action.row, action.col]),
                        "next_remaining_hamming": int(np.not_equal(next_states[index], example.goal).sum()),
                        "next_oracle_latent_energy": float(encoded_state_energy(model, world, next_states[index], example.goal)),
                    }
                )
            records.append(
                {
                    "example_index": int(example_index),
                    "depth_fraction": float(depth),
                    "prefix_steps": int(prefix_steps),
                    "legal_actions": int(len(legal)),
                    "goal_actions": int(len(goal_indices)),
                    "best_goal_rank": best_goal_rank,
                    "best_goal_score": best_goal_score,
                    "best_score": float(scores[order[0]]),
                    "best_writes_goal_value": bool(top_actions[0]["writes_goal_value"]),
                    "top_actions": top_actions,
                }
            )
    return {
        "score_mode": score_mode,
        "summary": {
            "records": len(records),
            "best_goal_top1_rate": mean(best_goal_flags),
            "mean_best_goal_rank": mean(best_goal_ranks),
            "median_best_goal_rank": median(best_goal_ranks),
        },
        "examples": records[: min(len(records), 20)],
    }


def rollout_abstract_sequence(
    model: ActionConditionedWorldModel,
    current_latent: torch.Tensor,
    sequence: torch.Tensor,
    *,
    level: int,
) -> torch.Tensor:
    latents = current_latent
    for step in range(sequence.shape[0]):
        latents = model.predict_latent_from_abstract_action(latents, sequence[step : step + 1], level=level)
    return latents


def count_action_quality(
    world: PuzzleWorld,
    start: np.ndarray,
    goal: np.ndarray,
    clue_mask: np.ndarray | None,
    actions: list[WorldAction],
) -> dict[str, Any]:
    current = np.asarray(start, dtype=np.int64).copy()
    correct = 0
    wrong = 0
    overwrite_correct = 0
    trace = []
    for action in actions:
        was_goal = int(current[action.row, action.col]) == int(goal[action.row, action.col])
        writes_goal = action.value == int(goal[action.row, action.col])
        if writes_goal:
            correct += 1
        else:
            wrong += 1
        if was_goal and not writes_goal:
            overwrite_correct += 1
        current = apply_planning_action(world, current, action, clue_mask)
        trace.append(
            {
                "action": asdict(action),
                "writes_goal_value": bool(writes_goal),
                "overwrites_correct_cell": bool(was_goal and not writes_goal),
                "remaining_hamming": int(np.not_equal(current, goal).sum()),
            }
        )
    return {
        "correct_writes": int(correct),
        "wrong_writes": int(wrong),
        "overwrite_correct_cells": int(overwrite_correct),
        "after_remaining_hamming": int(np.not_equal(current, goal).sum()),
        "action_trace": trace[:8],
    }


def top_level_probe(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    examples: list[PuzzleExample],
    rng: np.random.Generator,
    *,
    score_modes: list[str],
    optimizers: list[str],
    num_examples: int,
    hierarchy_level: int,
    macro_horizon: int,
    population: int,
    iterations: int,
    low_population: int,
    gd_steps: int,
    gd_lr: float,
    reachability_weight: float,
    prior_samples: int,
) -> dict[str, Any]:
    records = []
    device = next(model.parameters()).device
    for example_index, example in enumerate(examples[:num_examples]):
        clue_mask = clue_mask_for_planning(world, example.state)
        task_ids = torch.full((1,), world.task_id, dtype=torch.long, device=device)
        start_tensor = torch.as_tensor(example.state[None, ...], dtype=torch.long, device=device)
        goal_tensor = torch.as_tensor(example.goal[None, ...], dtype=torch.long, device=device)
        current_latent = model.encoder(start_tensor, task_ids=task_ids)
        initial_latent = model.encoder(start_tensor, task_ids=task_ids)
        goal_latent = model.target_encoder(goal_tensor, task_ids=task_ids)
        initial_goal_mse = float(F.mse_loss(current_latent, goal_latent).detach().cpu().item())
        prior = estimate_macro_action_prior(
            model,
            world,
            example.state,
            example.goal,
            clue_mask,
            rng,
            hierarchy_level=hierarchy_level,
            samples=prior_samples,
        )
        for score_mode in score_modes:
            if score_mode in {"goal_energy", "goal_value"} and not model.use_goal_energy_head:
                continue
            if score_mode == "macro_action_advantage" and not model.use_macro_action_value_head:
                continue
            for optimizer in optimizers:
                if optimizer == "cem":
                    high = high_level_subgoal_cem(
                        model,
                        current_latent,
                        goal_latent,
                        rng,
                        hierarchy_level=hierarchy_level,
                        macro_horizon=macro_horizon,
                        population_size=population,
                        elite_frac=0.2,
                        iterations=iterations,
                        smoothing=0.5,
                        prior=prior,
                        initial_latent=initial_latent,
                        score_mode=score_mode,
                    )
                else:
                    high = high_level_subgoal_gradient(
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
                final_latent = rollout_abstract_sequence(
                    model,
                    current_latent,
                    high["latent_action_sequence"],
                    level=hierarchy_level,
                )
                final_goal_mse = float(F.mse_loss(final_latent, goal_latent).detach().cpu().item())
                first_subgoal_goal_mse = float(F.mse_loss(high["subgoal_latent"], goal_latent).detach().cpu().item())
                raw_goal_head = None
                if model.use_goal_energy_head:
                    with torch.no_grad():
                        raw_goal_head = float(model.predict_goal_energy_from_latents(final_latent, initial_latent).detach().cpu().item())
                low = low_level_subgoal_cem(
                    model,
                    world,
                    example.state,
                    example.goal,
                    clue_mask,
                    rng,
                    subgoal_latent=high["subgoal_latent"],
                    horizon=min(int(model.hierarchy_span) ** hierarchy_level, terminal_step_limit(world, example, 81)),
                    population_size=low_population,
                    elite_frac=0.2,
                    iterations=iterations,
                    smoothing=0.5,
                )
                quality = count_action_quality(world, example.state, example.goal, clue_mask, list(low["actions"]))
                records.append(
                    {
                        "example_index": int(example_index),
                        "score_mode": score_mode,
                        "optimizer": optimizer,
                        "high_energy": float(high["energy"]),
                        "initial_latent_goal_mse": initial_goal_mse,
                        "predicted_final_latent_goal_mse": final_goal_mse,
                        "first_subgoal_goal_mse": first_subgoal_goal_mse,
                        "raw_goal_head_at_predicted_final": raw_goal_head,
                        "low_subgoal_energy": float(low["energy"]),
                        "low_action_count": int(len(low["actions"])),
                        **quality,
                    }
                )
    return {
        "summary": {
            "records": len(records),
            "mean_predicted_final_latent_goal_mse": mean([r["predicted_final_latent_goal_mse"] for r in records]),
            "mean_first_subgoal_goal_mse": mean([r["first_subgoal_goal_mse"] for r in records]),
            "mean_low_correct_writes": mean([r["correct_writes"] for r in records]),
            "mean_low_wrong_writes": mean([r["wrong_writes"] for r in records]),
        },
        "examples": records,
    }


def recursive_planning_probe(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    examples: list[PuzzleExample],
    rng: np.random.Generator,
    *,
    score_mode: str,
    optimizers: list[str],
    num_examples: int,
    max_steps: int,
    hierarchy_level: int,
    macro_horizon: int,
    population: int,
    low_population: int,
    iterations: int,
    gd_steps: int,
    gd_lr: float,
    reachability_weight: float,
    prior_samples: int,
) -> dict[str, Any]:
    records = []
    for example_index, example in enumerate(examples[:num_examples]):
        for optimizer in optimizers:
            record = recursive_hierarchical_subgoal_plan(
                model,
                world,
                example,
                rng,
                max_steps=max_steps,
                hierarchy_level=hierarchy_level,
                macro_horizon=macro_horizon,
                high_population_size=population,
                low_population_size=low_population,
                elite_frac=0.2,
                iterations=iterations,
                smoothing=0.5,
                execute_steps=1,
                prior_samples=prior_samples,
                high_score_mode=score_mode,
                optimizer=optimizer,
                gd_steps=gd_steps,
                gd_lr=gd_lr,
                reachability_weight=reachability_weight,
            )
            record["example_index"] = int(example_index)
            records.append(record)
    return {
        "summary": {
            "records": len(records),
            "solve_rate": mean([r["solved"] for r in records]),
            "terminal_rate": mean([r["terminal"] for r in records]),
            "mean_steps": mean([r["steps"] for r in records]),
            "mean_remaining_hamming": mean([r["remaining_hamming"] for r in records]),
        },
        "examples": records,
    }


def tree_stats(root: Any, world: PuzzleWorld, goal: np.ndarray, clue_mask: np.ndarray | None) -> dict[str, Any]:
    stack = [root]
    nodes = 0
    max_depth = 0
    terminal_nodes = 0
    solved_nodes = 0
    scored_nodes = 0
    while stack:
        node = stack.pop()
        nodes += 1
        max_depth = max(max_depth, int(node.depth))
        terminal = is_terminal_state(world, node.state, goal, clue_mask)
        terminal_nodes += int(terminal)
        solved_nodes += int(world.is_goal(node.state, goal))
        scored_nodes += int(not math.isnan(float(node.leaf_score)))
        stack.extend(node.children.values())
    return {
        "nodes": int(nodes),
        "max_depth_reached": int(max_depth),
        "terminal_nodes": int(terminal_nodes),
        "solved_nodes": int(solved_nodes),
        "scored_nodes": int(scored_nodes),
    }


def mcts_probe(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    examples: list[PuzzleExample],
    rng: np.random.Generator,
    *,
    score_modes: list[str],
    num_examples: int,
    simulations: list[int],
    expansion_actions: list[int],
    debug_actions: int,
    run_full_plan: bool,
) -> dict[str, Any]:
    root_records = []
    plan_records = []
    for example_index, example in enumerate(examples[:num_examples]):
        goal = np.asarray(example.goal, dtype=np.int64)
        initial = np.asarray(example.state, dtype=np.int64)
        clue_mask = clue_mask_for_planning(world, initial)
        depth = terminal_step_limit(world, example, 81)
        for score_mode in score_modes:
            if score_mode in {"goal_energy", "goal_value"} and not model.use_goal_energy_head:
                continue
            for sims in simulations:
                for expansion in expansion_actions:
                    cache: dict[tuple[str, bytes], float] = {}
                    started = time.time()
                    root = build_mcts_tree(
                        model,
                        world,
                        initial,
                        goal,
                        initial,
                        clue_mask,
                        rng,
                        simulations=sims,
                        max_depth=depth,
                        score_mode=score_mode,
                        exploration=1.4,
                        expansion_actions=expansion,
                        score_cache=cache,
                    )
                    elapsed = time.time() - started
                    debug = mcts_root_debug_record(
                        model,
                        world,
                        root,
                        goal,
                        initial,
                        step=0,
                        score_mode=score_mode,
                        debug_actions=debug_actions,
                        cache=cache,
                    )
                    root_records.append(
                        {
                            "example_index": int(example_index),
                            "score_mode": score_mode,
                            "simulations": int(sims),
                            "expansion_actions": int(expansion),
                            "terminal_depth": int(depth),
                            "elapsed_sec": float(elapsed),
                            **tree_stats(root, world, goal, clue_mask),
                            "root_debug": debug,
                        }
                    )
                    if run_full_plan:
                        started = time.time()
                        plan = mcts_plan(
                            model,
                            world,
                            example,
                            rng,
                            max_steps=depth,
                            simulations=sims,
                            max_depth=depth,
                            score_mode=score_mode,
                            exploration=1.4,
                            expansion_actions=expansion,
                            collect_debug=True,
                            debug_actions=debug_actions,
                        )
                        plan["example_index"] = int(example_index)
                        plan["elapsed_sec"] = float(time.time() - started)
                        plan_records.append(plan)
    return {
        "root_summary": {
            "records": len(root_records),
            "mean_max_depth_reached": mean([r["max_depth_reached"] for r in root_records]),
            "mean_terminal_nodes": mean([r["terminal_nodes"] for r in root_records]),
            "mean_solved_nodes": mean([r["solved_nodes"] for r in root_records]),
        },
        "root_examples": root_records,
        "plan_summary": {
            "records": len(plan_records),
            "solve_rate": mean([r["solved"] for r in plan_records]),
            "terminal_rate": mean([r["terminal"] for r in plan_records]),
            "mean_remaining_hamming": mean([r["remaining_hamming"] for r in plan_records]),
            "mean_elapsed_sec": mean([r["elapsed_sec"] for r in plan_records]),
        },
        "plan_examples": plan_records,
    }


def mean(values: list[float]) -> float | None:
    return None if not values else float(np.mean(np.asarray(values, dtype=np.float64)))


def median(values: list[float]) -> float | None:
    return None if not values else float(np.median(np.asarray(values, dtype=np.float64)))


def parse_int_list(text: str) -> list[int]:
    return [int(item) for item in text.split(",") if item.strip()]


def parse_str_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--state-value-run", default="sudoku_jepa_5m_hier_value_l3_span4_state_value")
    parser.add_argument("--terminal-energy-run", default="sudoku_jepa_5m_hier_value_l3_span4_terminal_energy")
    parser.add_argument("--macro-run", default="sudoku_jepa_5m_hier_value_l3_span4_macro_action_advantage")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--terminal-examples", type=int, default=8)
    parser.add_argument("--terminal-variants", type=int, default=4)
    parser.add_argument("--local-action-examples", type=int, default=4)
    parser.add_argument("--top-examples", type=int, default=1)
    parser.add_argument("--recursive-examples", type=int, default=1)
    parser.add_argument("--run-recursive", action="store_true")
    parser.add_argument("--mcts-examples", type=int, default=1)
    parser.add_argument("--hierarchy-level", type=int, default=2)
    parser.add_argument("--macro-horizon", type=int, default=5)
    parser.add_argument("--population", type=int, default=512)
    parser.add_argument("--low-population", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--gd-steps", type=int, default=64)
    parser.add_argument("--gd-lr", type=float, default=0.05)
    parser.add_argument("--reachability-weight", type=float, default=0.01)
    parser.add_argument("--prior-samples", type=int, default=64)
    parser.add_argument("--mcts-simulations", default="256,1024")
    parser.add_argument("--mcts-expansion-actions", default="16,64")
    parser.add_argument("--mcts-full-plan", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output: dict[str, Any] = {
        "metadata": {
            "seed": int(args.seed),
            "device": str(device),
            "run_root": str(args.run_root),
        }
    }

    state_model, state_world, state_examples, state_config = load_run(args.run_root / args.state_value_run, device=device)
    output["metadata"]["state_value_run"] = args.state_value_run
    output["metadata"]["state_value_model"] = {
        key: state_config["model"].get(key)
        for key in ("use_goal_energy_head", "use_macro_action_value_head", "hierarchy_levels", "hierarchy_span")
    }
    output["state_value_terminal_discrimination"] = terminal_discrimination_probe(
        state_model,
        state_world,
        state_examples,
        rng,
        score_mode="goal_value",
        num_examples=args.terminal_examples,
        variants_per_level=args.terminal_variants,
    )
    output["state_value_local_action_ranking"] = local_action_probe(
        state_model,
        state_world,
        state_examples,
        rng,
        score_mode="goal_value",
        num_examples=args.local_action_examples,
        depths=[0.0, 0.5, 0.9],
        debug_actions=8,
    )
    output["state_value_top_level"] = top_level_probe(
        state_model,
        state_world,
        state_examples,
        rng,
        score_modes=["latent_goal", "goal_value"],
        optimizers=["cem", "gd", "gd_reachability"],
        num_examples=args.top_examples,
        hierarchy_level=args.hierarchy_level,
        macro_horizon=args.macro_horizon,
        population=args.population,
        iterations=args.iterations,
        low_population=args.low_population,
        gd_steps=args.gd_steps,
        gd_lr=args.gd_lr,
        reachability_weight=args.reachability_weight,
        prior_samples=args.prior_samples,
    )
    if args.run_recursive:
        output["state_value_recursive"] = recursive_planning_probe(
            state_model,
            state_world,
            state_examples,
            rng,
            score_mode="goal_value",
            optimizers=["cem", "gd", "gd_reachability"],
            num_examples=args.recursive_examples,
            max_steps=81,
            hierarchy_level=args.hierarchy_level,
            macro_horizon=args.macro_horizon,
            population=args.population,
            low_population=args.low_population,
            iterations=args.iterations,
            gd_steps=args.gd_steps,
            gd_lr=args.gd_lr,
            reachability_weight=args.reachability_weight,
            prior_samples=args.prior_samples,
        )
    output["state_value_mcts"] = mcts_probe(
        state_model,
        state_world,
        state_examples,
        rng,
        score_modes=["goal_value", "latent_goal"],
        num_examples=args.mcts_examples,
        simulations=parse_int_list(args.mcts_simulations),
        expansion_actions=parse_int_list(args.mcts_expansion_actions),
        debug_actions=8,
        run_full_plan=bool(args.mcts_full_plan),
    )
    del state_model
    torch.cuda.empty_cache()

    terminal_model, terminal_world, terminal_examples, terminal_config = load_run(
        args.run_root / args.terminal_energy_run,
        device=device,
    )
    output["metadata"]["terminal_energy_run"] = args.terminal_energy_run
    output["metadata"]["terminal_energy_model"] = {
        key: terminal_config["model"].get(key)
        for key in ("use_goal_energy_head", "use_macro_action_value_head", "hierarchy_levels", "hierarchy_span")
    }
    output["terminal_energy_terminal_discrimination"] = terminal_discrimination_probe(
        terminal_model,
        terminal_world,
        terminal_examples,
        rng,
        score_mode="goal_energy",
        num_examples=args.terminal_examples,
        variants_per_level=args.terminal_variants,
    )
    del terminal_model
    torch.cuda.empty_cache()

    macro_model, macro_world, macro_examples, macro_config = load_run(args.run_root / args.macro_run, device=device)
    output["metadata"]["macro_run"] = args.macro_run
    output["metadata"]["macro_model"] = {
        key: macro_config["model"].get(key)
        for key in ("use_goal_energy_head", "use_macro_action_value_head", "hierarchy_levels", "hierarchy_span")
    }
    output["macro_top_level"] = top_level_probe(
        macro_model,
        macro_world,
        macro_examples,
        rng,
        score_modes=["latent_goal", "macro_action_advantage"],
        optimizers=["cem", "gd", "gd_reachability"],
        num_examples=args.top_examples,
        hierarchy_level=args.hierarchy_level,
        macro_horizon=args.macro_horizon,
        population=args.population,
        iterations=args.iterations,
        low_population=args.low_population,
        gd_steps=args.gd_steps,
        gd_lr=args.gd_lr,
        reachability_weight=args.reachability_weight,
        prior_samples=args.prior_samples,
    )
    if args.run_recursive:
        output["macro_recursive"] = recursive_planning_probe(
            macro_model,
            macro_world,
            macro_examples,
            rng,
            score_mode="macro_action_advantage",
            optimizers=["cem", "gd", "gd_reachability"],
            num_examples=args.recursive_examples,
            max_steps=81,
            hierarchy_level=args.hierarchy_level,
            macro_horizon=args.macro_horizon,
            population=args.population,
            low_population=args.low_population,
            iterations=args.iterations,
            gd_steps=args.gd_steps,
            gd_lr=args.gd_lr,
            reachability_weight=args.reachability_weight,
            prior_samples=args.prior_samples,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
