from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from puzzle_jepa.data.worlds import PuzzleExample, PuzzleWorld, WorldAction
from puzzle_jepa.eval.diagnostics import (
    apply_planning_action,
    build_world,
    cem_action_space,
    clue_mask_for_planning,
    encoded_state_energy,
    high_level_subgoal_cem,
    high_level_subgoal_gradient,
    is_terminal_state,
    legal_planning_actions,
    load_examples,
    load_model_checkpoint_state,
    low_level_subgoal_cem,
    oracle_action_sequence,
    sample_cem_rollout,
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


def oracle_prefix_state(
    world: PuzzleWorld,
    example: PuzzleExample,
    rng: np.random.Generator,
    *,
    depth_fraction: float,
) -> tuple[np.ndarray, list[WorldAction], np.ndarray | None]:
    actions = oracle_action_sequence(world, example, rng)
    prefix_steps = int(round(float(depth_fraction) * len(actions)))
    clue_mask = clue_mask_for_planning(world, example.state)
    current = np.asarray(example.state, dtype=np.int64).copy()
    for action in actions[:prefix_steps]:
        current = apply_planning_action(world, current, action, clue_mask)
    return current, actions[:prefix_steps], clue_mask


def oracle_complete_after_action(
    world: PuzzleWorld,
    current: np.ndarray,
    initial: np.ndarray,
    goal: np.ndarray,
    clue_mask: np.ndarray | None,
    action: WorldAction,
) -> np.ndarray:
    next_state = apply_planning_action(world, current, action, clue_mask)
    terminal = np.asarray(goal, dtype=np.int64).copy()
    mutable = np.asarray(initial, dtype=np.int64) == 0
    filled_mutable = mutable & (next_state != 0)
    terminal[filled_mutable] = next_state[filled_mutable]
    return terminal


def terminal_projected_action_ranking(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    examples: list[PuzzleExample],
    rng: np.random.Generator,
    *,
    score_mode: str,
    num_examples: int,
    depth_fractions: list[float],
    debug_actions: int,
) -> dict[str, Any]:
    records = []
    action_top1 = []
    action_ranks = []
    cell_top1 = []
    cell_ranks = []
    for example_index, example in enumerate(examples[:num_examples]):
        for depth in depth_fractions:
            current, _prefix, clue_mask = oracle_prefix_state(world, example, rng, depth_fraction=depth)
            legal = legal_planning_actions(world, current, clue_mask)
            terminals = [
                oracle_complete_after_action(world, current, example.state, example.goal, clue_mask, action)
                for action in legal
            ]
            scores = score_symbolic_states_to_goal(
                model,
                world,
                terminals,
                example.goal,
                example.state,
                planning_score=score_mode,
            )
            order = np.argsort(np.asarray(scores, dtype=np.float64))[::-1].tolist()
            solution_indices = [index for index, terminal in enumerate(terminals) if np.array_equal(terminal, example.goal)]
            best_solution_rank = None
            if solution_indices:
                ranks = [order.index(index) + 1 for index in solution_indices]
                best_solution_rank = int(min(ranks))
                action_top1.append(float(best_solution_rank == 1))
                action_ranks.append(float(best_solution_rank))

            by_cell: dict[tuple[int, int], list[int]] = {}
            for index, action in enumerate(legal):
                by_cell.setdefault((action.row, action.col), []).append(index)
            cell_records = []
            for (row, col), indices in by_cell.items():
                true_indices = [index for index in indices if np.array_equal(terminals[index], example.goal)]
                if not true_indices:
                    continue
                cell_order = sorted(indices, key=lambda index: float(scores[index]), reverse=True)
                true_rank = cell_order.index(true_indices[0]) + 1
                cell_top1.append(float(true_rank == 1))
                cell_ranks.append(float(true_rank))
                if len(cell_records) < 6:
                    cell_records.append(
                        {
                            "row": int(row),
                            "col": int(col),
                            "goal": int(example.goal[row, col]),
                            "current": int(current[row, col]),
                            "true_value_rank_within_cell": int(true_rank),
                            "values": [
                                {
                                    "value": int(legal[index].value),
                                    "score": float(scores[index]),
                                    "terminal_wrong_cells": int(np.not_equal(terminals[index], example.goal).sum()),
                                    "is_solution": bool(np.array_equal(terminals[index], example.goal)),
                                }
                                for index in cell_order[:9]
                            ],
                        }
                    )

            top_actions = []
            for rank, index in enumerate(order[:debug_actions], start=1):
                action = legal[index]
                terminal = terminals[index]
                top_actions.append(
                    {
                        "rank": int(rank),
                        "action": asdict(action),
                        "score": float(scores[index]),
                        "is_solution_terminal": bool(np.array_equal(terminal, example.goal)),
                        "terminal_wrong_cells": int(np.not_equal(terminal, example.goal).sum()),
                        "oracle_terminal_energy": float(encoded_state_energy(model, world, terminal, example.goal)),
                        "cell_current": int(current[action.row, action.col]),
                        "cell_goal": int(example.goal[action.row, action.col]),
                    }
                )

            records.append(
                {
                    "example_index": int(example_index),
                    "depth_fraction": float(depth),
                    "legal_actions": int(len(legal)),
                    "solution_terminal_actions": int(len(solution_indices)),
                    "best_solution_rank": best_solution_rank,
                    "best_score": float(scores[order[0]]) if order else math.nan,
                    "best_is_solution_terminal": bool(order and np.array_equal(terminals[order[0]], example.goal)),
                    "top_actions": top_actions,
                    "sample_cell_value_rankings": cell_records,
                }
            )
    return {
        "score_mode": score_mode,
        "summary": {
            "records": int(len(records)),
            "solution_action_top1_rate": mean(action_top1),
            "mean_best_solution_action_rank": mean(action_ranks),
            "median_best_solution_action_rank": median(action_ranks),
            "cell_true_value_top1_rate": mean(cell_top1),
            "mean_cell_true_value_rank": mean(cell_ranks),
            "median_cell_true_value_rank": median(cell_ranks),
        },
        "examples": records[: min(20, len(records))],
    }


def terminal_candidates_for_nearest(example: PuzzleExample, *, max_cells: int | None = None) -> list[dict[str, Any]]:
    initial = np.asarray(example.state, dtype=np.int64)
    goal = np.asarray(example.goal, dtype=np.int64)
    mutable = np.argwhere(initial == 0)
    if max_cells is not None:
        mutable = mutable[: max(0, int(max_cells))]
    candidates = [{"kind": "true_solution", "state": goal.copy(), "wrong_cells": 0, "changed": None}]
    for row, col in mutable:
        row_i, col_i = int(row), int(col)
        for value in range(1, 10):
            if value == int(goal[row_i, col_i]):
                continue
            state = goal.copy()
            state[row_i, col_i] = value
            candidates.append(
                {
                    "kind": "one_cell_corrupt",
                    "state": state,
                    "wrong_cells": 1,
                    "changed": {"row": row_i, "col": col_i, "goal": int(goal[row_i, col_i]), "value": int(value)},
                }
            )
    return candidates


@torch.no_grad()
def encode_terminal_candidates(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    candidates: list[dict[str, Any]],
) -> torch.Tensor:
    device = next(model.parameters()).device
    states = torch.as_tensor(np.stack([item["state"] for item in candidates], axis=0), dtype=torch.long, device=device)
    task_ids = torch.full((len(candidates),), world.task_id, dtype=torch.long, device=device)
    return model.target_encoder(states, task_ids=task_ids)


def high_level_terminal_nearest_probe(
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
    gd_steps: int,
    gd_lr: float,
) -> dict[str, Any]:
    records = []
    device = next(model.parameters()).device
    for example_index, example in enumerate(examples[:num_examples]):
        task_ids = torch.full((1,), world.task_id, dtype=torch.long, device=device)
        start_tensor = torch.as_tensor(example.state[None, ...], dtype=torch.long, device=device)
        goal_tensor = torch.as_tensor(example.goal[None, ...], dtype=torch.long, device=device)
        current_latent = model.encoder(start_tensor, task_ids=task_ids)
        initial_latent = model.encoder(start_tensor, task_ids=task_ids)
        goal_latent = model.target_encoder(goal_tensor, task_ids=task_ids)
        terminal_candidates = terminal_candidates_for_nearest(example)
        terminal_latents = encode_terminal_candidates(model, world, terminal_candidates)
        for score_mode in score_modes:
            if score_mode in {"goal_energy", "goal_value"} and not model.use_goal_energy_head:
                continue
            if score_mode == "macro_action_advantage" and not model.use_macro_action_value_head:
                continue
            for optimizer in optimizers:
                if optimizer == "cem":
                    plan = high_level_subgoal_cem(
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
                        prior=None,
                        initial_latent=initial_latent,
                        score_mode=score_mode,
                    )
                else:
                    plan = high_level_subgoal_gradient(
                        model,
                        current_latent,
                        goal_latent,
                        hierarchy_level=hierarchy_level,
                        macro_horizon=macro_horizon,
                        prior=None,
                        initial_latent=initial_latent,
                        score_mode=score_mode,
                        steps=gd_steps,
                        lr=gd_lr,
                        reachability_weight=0.01 if optimizer == "gd_reachability" else 0.0,
                    )
                final_latent = current_latent
                for step in range(plan["latent_action_sequence"].shape[0]):
                    final_latent = model.predict_latent_from_abstract_action(
                        final_latent,
                        plan["latent_action_sequence"][step : step + 1],
                        level=hierarchy_level,
                    )
                distances = F.mse_loss(
                    final_latent.expand(terminal_latents.shape[0], -1, -1),
                    terminal_latents,
                    reduction="none",
                ).mean(dim=(1, 2))
                order = torch.argsort(distances).detach().cpu().tolist()
                true_rank = int(order.index(0) + 1)
                nearest = terminal_candidates[order[0]]
                records.append(
                    {
                        "example_index": int(example_index),
                        "score_mode": score_mode,
                        "optimizer": optimizer,
                        "top_energy": float(plan["energy"]),
                        "true_terminal_rank_among_one_cell_corruptions": true_rank,
                        "true_terminal_distance": float(distances[0].detach().cpu().item()),
                        "nearest_terminal_kind": nearest["kind"],
                        "nearest_terminal_wrong_cells": int(nearest["wrong_cells"]),
                        "nearest_terminal_changed": nearest["changed"],
                        "nearest_terminal_distance": float(distances[order[0]].detach().cpu().item()),
                        "predicted_final_goal_mse": float(F.mse_loss(final_latent, goal_latent).detach().cpu().item()),
                    }
                )
    return {
        "summary": {
            "records": int(len(records)),
            "true_terminal_top1_rate": mean([float(r["true_terminal_rank_among_one_cell_corruptions"] == 1) for r in records]),
            "mean_true_terminal_rank": mean([float(r["true_terminal_rank_among_one_cell_corruptions"]) for r in records]),
            "median_true_terminal_rank": median([float(r["true_terminal_rank_among_one_cell_corruptions"]) for r in records]),
        },
        "examples": records,
    }


def encode_action_sequences(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    sequences: list[list[WorldAction]],
    *,
    level: int,
) -> torch.Tensor:
    device = next(model.parameters()).device
    action_tensor = torch.as_tensor(
        [
            [[world.task_id, action.row, action.col, action.value] for action in sequence]
            for sequence in sequences
        ],
        dtype=torch.long,
        device=device,
    )
    return model.encode_hierarchy_action(action_tensor, level=level)


def sample_macro_codebook(
    world: PuzzleWorld,
    example: PuzzleExample,
    rng: np.random.Generator,
    *,
    block: int,
    samples: int,
) -> list[dict[str, Any]]:
    clue_mask = clue_mask_for_planning(world, example.state)
    action_space = cem_action_space(world, example.state, clue_mask)
    cell_probs = np.full((block, len(action_space["positions"])), 1.0 / len(action_space["positions"]))
    value_probs = np.full((block, len(action_space["values"])), 1.0 / len(action_space["values"]))
    items: list[dict[str, Any]] = []
    oracle = oracle_action_sequence(world, example, rng)
    if len(oracle) >= block:
        items.append({"kind": "oracle_chunk", "actions": oracle[:block]})
    max_attempts = max(samples * 4, samples)
    for _ in range(max_attempts):
        rollout = sample_cem_rollout(
            world,
            example.state,
            example.goal,
            clue_mask,
            rng,
            cell_probs=cell_probs,
            value_probs=value_probs,
            action_space=action_space,
        )
        actions = list(rollout["actions"])
        if len(actions) != block:
            continue
        items.append({"kind": "random_chunk", "actions": actions})
        if len(items) >= samples:
            break
    return items


@torch.no_grad()
def codebook_subgoal_probe(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    examples: list[PuzzleExample],
    rng: np.random.Generator,
    *,
    score_mode: str,
    num_examples: int,
    hierarchy_level: int,
    samples: int,
    low_population: int,
    iterations: int,
) -> dict[str, Any]:
    records = []
    device = next(model.parameters()).device
    block = int(model.hierarchy_span) ** int(hierarchy_level)
    for example_index, example in enumerate(examples[:num_examples]):
        items = sample_macro_codebook(world, example, rng, block=block, samples=samples)
        if not items:
            continue
        action_latents = encode_action_sequences(model, world, [item["actions"] for item in items], level=hierarchy_level)
        task_ids = torch.full((1,), world.task_id, dtype=torch.long, device=device)
        start_tensor = torch.as_tensor(example.state[None, ...], dtype=torch.long, device=device)
        goal_tensor = torch.as_tensor(example.goal[None, ...], dtype=torch.long, device=device)
        current_latent = model.encoder(start_tensor, task_ids=task_ids)
        initial_latent = model.encoder(start_tensor, task_ids=task_ids)
        goal_latent = model.target_encoder(goal_tensor, task_ids=task_ids)
        next_latents = model.predict_latent_from_abstract_action(
            current_latent.expand(action_latents.shape[0], -1, -1),
            action_latents,
            level=hierarchy_level,
        )
        if score_mode == "latent_goal":
            energies = F.mse_loss(next_latents, goal_latent.expand_as(next_latents), reduction="none").mean(dim=(1, 2))
        elif score_mode == "goal_value":
            energies = -model.predict_goal_energy_from_latents(next_latents, initial_latent)
        elif score_mode == "macro_action_advantage":
            energies = -model.predict_macro_action_value_from_latents(
                current_latent.expand(action_latents.shape[0], -1, -1),
                initial_latent.expand(action_latents.shape[0], -1, -1),
                action_latents,
                level=hierarchy_level,
            )
        else:
            raise ValueError(score_mode)
        best_index = int(torch.argmin(energies).detach().cpu().item())
        best = items[best_index]
        clue_mask = clue_mask_for_planning(world, example.state)
        low = low_level_subgoal_cem(
            model,
            world,
            example.state,
            example.goal,
            clue_mask,
            rng,
            subgoal_latent=next_latents[best_index : best_index + 1],
            horizon=block,
            population_size=low_population,
            elite_frac=0.2,
            iterations=iterations,
            smoothing=0.5,
        )
        executed = np.asarray(example.state, dtype=np.int64).copy()
        correct = 0
        wrong = 0
        for action in best["actions"]:
            correct += int(action.value == int(example.goal[action.row, action.col]))
            wrong += int(action.value != int(example.goal[action.row, action.col]))
            executed = apply_planning_action(world, executed, action, clue_mask)
        low_correct = sum(int(a.value == int(example.goal[a.row, a.col])) for a in low["actions"])
        low_wrong = len(low["actions"]) - low_correct
        oracle_indices = [index for index, item in enumerate(items) if item["kind"] == "oracle_chunk"]
        oracle_rank = None
        if oracle_indices:
            order = torch.argsort(energies).detach().cpu().tolist()
            oracle_rank = int(order.index(oracle_indices[0]) + 1)
        records.append(
            {
                "example_index": int(example_index),
                "score_mode": score_mode,
                "codebook_size": int(len(items)),
                "oracle_chunk_rank": oracle_rank,
                "selected_kind": best["kind"],
                "selected_energy": float(energies[best_index].detach().cpu().item()),
                "selected_correct_writes": int(correct),
                "selected_wrong_writes": int(wrong),
                "selected_remaining_hamming_after_execute": int(np.not_equal(executed, example.goal).sum()),
                "low_cem_correct_writes": int(low_correct),
                "low_cem_wrong_writes": int(low_wrong),
                "low_cem_remaining_hamming": int(np.not_equal(low["state"], example.goal).sum()),
                "selected_first_actions": [asdict(action) for action in best["actions"][:5]],
                "low_cem_first_actions": [asdict(action) for action in low["actions"][:5]],
            }
        )
    return {
        "summary": {
            "records": int(len(records)),
            "oracle_chunk_top1_rate": mean([
                float(record["oracle_chunk_rank"] == 1)
                for record in records
                if record["oracle_chunk_rank"] is not None
            ]),
            "mean_oracle_chunk_rank": mean([
                float(record["oracle_chunk_rank"])
                for record in records
                if record["oracle_chunk_rank"] is not None
            ]),
            "selected_oracle_rate": mean([float(record["selected_kind"] == "oracle_chunk") for record in records]),
            "mean_selected_correct_writes": mean([float(record["selected_correct_writes"]) for record in records]),
            "mean_low_cem_correct_writes": mean([float(record["low_cem_correct_writes"]) for record in records]),
        },
        "examples": records,
    }


def mean(values: list[float]) -> float | None:
    return None if not values else float(np.mean(np.asarray(values, dtype=np.float64)))


def median(values: list[float]) -> float | None:
    return None if not values else float(np.median(np.asarray(values, dtype=np.float64)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--state-value-run", default="sudoku_jepa_5m_hier_value_l3_span4_state_value")
    parser.add_argument("--terminal-energy-run", default="sudoku_jepa_5m_hier_value_l3_span4_terminal_energy")
    parser.add_argument("--macro-run", default="sudoku_jepa_5m_hier_value_l3_span4_macro_action_advantage")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--examples", type=int, default=16)
    parser.add_argument("--top-examples", type=int, default=2)
    parser.add_argument("--hierarchy-level", type=int, default=2)
    parser.add_argument("--macro-horizon", type=int, default=5)
    parser.add_argument("--population", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--gd-steps", type=int, default=48)
    parser.add_argument("--gd-lr", type=float, default=0.05)
    parser.add_argument("--codebook-samples", type=int, default=128)
    parser.add_argument("--low-population", type=int, default=256)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output: dict[str, Any] = {"metadata": {"seed": int(args.seed), "device": str(device)}}

    state_model, state_world, state_examples, _state_config = load_run(args.run_root / args.state_value_run, device=device)
    output["state_value_terminal_projected_actions"] = terminal_projected_action_ranking(
        state_model,
        state_world,
        state_examples,
        rng,
        score_mode="goal_value",
        num_examples=args.examples,
        depth_fractions=[0.0, 0.5, 0.9],
        debug_actions=8,
    )
    output["state_value_high_level_terminal_nearest"] = high_level_terminal_nearest_probe(
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
        gd_steps=args.gd_steps,
        gd_lr=args.gd_lr,
    )
    output["state_value_codebook_subgoal"] = codebook_subgoal_probe(
        state_model,
        state_world,
        state_examples,
        rng,
        score_mode="goal_value",
        num_examples=args.top_examples,
        hierarchy_level=args.hierarchy_level,
        samples=args.codebook_samples,
        low_population=args.low_population,
        iterations=args.iterations,
    )
    del state_model
    torch.cuda.empty_cache()

    terminal_model, terminal_world, terminal_examples, _terminal_config = load_run(
        args.run_root / args.terminal_energy_run,
        device=device,
    )
    output["terminal_energy_terminal_projected_actions"] = terminal_projected_action_ranking(
        terminal_model,
        terminal_world,
        terminal_examples,
        rng,
        score_mode="goal_energy",
        num_examples=args.examples,
        depth_fractions=[0.0, 0.5, 0.9],
        debug_actions=8,
    )
    del terminal_model
    torch.cuda.empty_cache()

    macro_model, macro_world, macro_examples, _macro_config = load_run(args.run_root / args.macro_run, device=device)
    output["macro_high_level_terminal_nearest"] = high_level_terminal_nearest_probe(
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
        gd_steps=args.gd_steps,
        gd_lr=args.gd_lr,
    )
    output["macro_codebook_subgoal"] = codebook_subgoal_probe(
        macro_model,
        macro_world,
        macro_examples,
        rng,
        score_mode="macro_action_advantage",
        num_examples=args.top_examples,
        hierarchy_level=args.hierarchy_level,
        samples=args.codebook_samples,
        low_population=args.low_population,
        iterations=args.iterations,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
