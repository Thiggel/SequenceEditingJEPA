from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW

from puzzle_jepa.data import (
    HFPuzzleColumns,
    MazeWorld,
    PuzzleExample,
    SudokuWorld,
    collate_rollouts,
    collate_transitions,
    iter_hf_examples,
    sample_curriculum_rollout_transition,
    sample_curriculum_transition,
    sample_random_mutable_transition,
    sample_oracle_partial_transition,
)
from puzzle_jepa.data.trajectories import Transition
from puzzle_jepa.data.worlds import PuzzleWorld, WorldAction
from puzzle_jepa.models import ActionConditionedWorldModel


def run_grid0(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", 0))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_cfg = dict(config["task"])
    data_cfg = dict(config.get("data", {}))
    train_cfg = dict(config["training"])
    eval_cfg = dict(config["eval"])
    output_dir = Path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    world = _build_world(task_cfg)
    train_examples = _load_examples(task_cfg, "train")
    eval_examples = _load_examples(task_cfg, "eval")
    model = ActionConditionedWorldModel(vocab_size=world.vocab_size, **dict(config["model"])).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.05)),
    )
    use_amp = bool(train_cfg.get("bf16", True)) and device.type == "cuda"
    metrics_path = output_dir / "metrics.jsonl"
    max_steps = int(train_cfg["max_steps"])
    batch_size = int(train_cfg["batch_size"])
    train_oracle_probability = float(train_cfg.get("oracle_probability", data_cfg.get("oracle_probability", 1.0)))
    eval_oracle_probability = float(eval_cfg.get("oracle_probability", 1.0))
    eval_every = int(train_cfg.get("eval_every_steps", max_steps))
    save_every = int(train_cfg.get("save_every_steps", eval_every))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    rollout_steps = int(train_cfg.get("rollout_steps", 1))
    rollout_weight = float(train_cfg.get("rollout_weight", 1.0 if rollout_steps > 1 else 0.0))
    rollout_batch_size = int(train_cfg.get("rollout_batch_size", batch_size))
    rollout_oracle_probability = float(train_cfg.get("rollout_oracle_probability", 1.0))
    goal_energy_weight = float(train_cfg.get("goal_energy_weight", 0.0))
    goal_energy_contrastive_weight = float(train_cfg.get("goal_energy_contrastive_weight", 0.0))
    goal_energy_monotonicity_weight = float(train_cfg.get("goal_energy_monotonicity_weight", 0.0))
    goal_terminal_weight = float(train_cfg.get("goal_terminal_correctness_weight", 0.0))
    goal_ranking_weight = float(train_cfg.get("goal_ranking_weight", 0.0))
    goal_local_regression_weight = float(train_cfg.get("goal_local_regression_weight", 0.0))
    goal_local_margin_weight = float(train_cfg.get("goal_local_margin_weight", 0.0))
    action_advantage_weight = float(train_cfg.get("action_advantage_weight", 0.0))
    macro_action_advantage_weight = float(train_cfg.get("macro_action_advantage_weight", 0.0))
    latent_progress_weight = float(train_cfg.get("latent_progress_weight", 0.0))
    action_policy_weight = float(train_cfg.get("action_policy_weight", 0.0))
    goal_energy_aux_batch_size = int(train_cfg.get("goal_energy_aux_batch_size", batch_size))
    goal_energy_positive_count = int(train_cfg.get("goal_energy_positive_count", 1))
    goal_energy_negative_count = int(train_cfg.get("goal_energy_negative_count", 8))
    hierarchy_weight = float(train_cfg.get("hierarchy_weight", 0.0))
    hierarchy_batch_size = int(train_cfg.get("hierarchy_batch_size", train_cfg.get("rollout_batch_size", batch_size)))
    hierarchy_rollout_steps = int(
        train_cfg.get("hierarchy_rollout_steps", _max_hierarchy_horizon(model))
    )
    hierarchy_oracle_probability = float(train_cfg.get("hierarchy_oracle_probability", 1.0))
    latest_metrics: dict[str, Any] = {}

    with (output_dir / "config.json").open("w") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)

    for step in range(1, max_steps + 1):
        model.train()
        batch = _sample_batch(world, train_examples, rng, batch_size, device, train_oracle_probability)
        loss_weights = _build_transition_loss_weights(world, batch.actions, batch.states.shape[-2:], train_cfg)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            output = model(
                batch.states,
                batch.actions,
                batch.next_states,
                loss_weights=loss_weights,
                goals=batch.goals,
                initial_states=_initial_states_from_batch(world, batch),
                goal_energy_weight=goal_energy_weight,
                goal_energy_target_scale=float(train_cfg.get("goal_energy_target_scale", 1.0)),
            )
            loss = output.loss
            rollout_loss = None
            hierarchy_loss = None
            energy_aux_loss = None
            ranking_loss = None
            if rollout_steps > 1 and rollout_weight > 0.0:
                rollout_batch = _sample_rollout_batch(
                    world,
                    train_examples,
                    rng,
                    rollout_batch_size,
                    rollout_steps,
                    device,
                    oracle_probability=rollout_oracle_probability,
                )
                rollout_output = model.rollout_loss(
                    rollout_batch.states,
                    rollout_batch.actions,
                    rollout_batch.target_states,
                )
                rollout_loss = rollout_output.loss
                loss = loss + rollout_weight * rollout_loss
            if hierarchy_weight > 0.0:
                hierarchy_steps = max(1, hierarchy_rollout_steps)
                hierarchy_batch = _sample_rollout_batch(
                    world,
                    train_examples,
                    rng,
                    hierarchy_batch_size,
                    hierarchy_steps,
                    device,
                    oracle_probability=hierarchy_oracle_probability,
                )
                hierarchy_output = model.hierarchy_loss(
                    hierarchy_batch.states,
                    hierarchy_batch.actions,
                    hierarchy_batch.target_states,
                )
                hierarchy_loss = hierarchy_output.loss
                loss = loss + hierarchy_weight * hierarchy_loss
            if goal_energy_contrastive_weight > 0.0 or goal_energy_monotonicity_weight > 0.0:
                energy_output = _goal_energy_auxiliary_loss(
                    model,
                    world,
                    train_examples,
                    rng,
                    train_cfg,
                    device,
                    goal_energy_aux_batch_size,
                    goal_energy_positive_count,
                    goal_energy_negative_count,
                    goal_energy_contrastive_weight,
                    goal_energy_monotonicity_weight,
                )
                energy_aux_loss = energy_output.loss
                loss = loss + energy_aux_loss
            if goal_terminal_weight > 0.0:
                terminal_output = _goal_terminal_correctness_loss(
                    model,
                    world,
                    train_examples,
                    rng,
                    train_cfg,
                    device,
                )
                energy_aux_loss = terminal_output.loss if energy_aux_loss is None else energy_aux_loss + terminal_output.loss
                loss = loss + terminal_output.loss
            if goal_ranking_weight > 0.0:
                ranking_loss = _goal_energy_ranking_loss(
                    model,
                    world,
                    train_examples,
                    rng,
                    train_cfg,
                    device,
                )
                scaled_ranking_loss = goal_ranking_weight * ranking_loss
                energy_aux_loss = scaled_ranking_loss if energy_aux_loss is None else energy_aux_loss + scaled_ranking_loss
                loss = loss + scaled_ranking_loss
            if goal_local_regression_weight > 0.0:
                local_loss = _goal_local_score_regression_loss(
                    model,
                    world,
                    train_examples,
                    rng,
                    train_cfg,
                    device,
                )
                scaled_local = goal_local_regression_weight * local_loss
                energy_aux_loss = scaled_local if energy_aux_loss is None else energy_aux_loss + scaled_local
                loss = loss + scaled_local
            if goal_local_margin_weight > 0.0:
                margin_loss = _goal_local_margin_loss(
                    model,
                    world,
                    train_examples,
                    rng,
                    train_cfg,
                    device,
                )
                scaled_margin = goal_local_margin_weight * margin_loss
                energy_aux_loss = scaled_margin if energy_aux_loss is None else energy_aux_loss + scaled_margin
                loss = loss + scaled_margin
            if action_advantage_weight > 0.0:
                advantage_output = _action_advantage_loss(
                    model,
                    world,
                    train_examples,
                    rng,
                    train_cfg,
                    device,
                )
                scaled_advantage = action_advantage_weight * advantage_output.loss
                energy_aux_loss = scaled_advantage if energy_aux_loss is None else energy_aux_loss + scaled_advantage
                loss = loss + scaled_advantage
            if macro_action_advantage_weight > 0.0:
                macro_output = _macro_action_advantage_loss(
                    model,
                    world,
                    train_examples,
                    rng,
                    train_cfg,
                    device,
                )
                scaled_macro = macro_action_advantage_weight * macro_output.loss
                energy_aux_loss = scaled_macro if energy_aux_loss is None else energy_aux_loss + scaled_macro
                loss = loss + scaled_macro
            if latent_progress_weight > 0.0:
                progress_loss = _latent_progress_loss(
                    model,
                    world,
                    train_examples,
                    rng,
                    train_cfg,
                    device,
                )
                scaled_progress = latent_progress_weight * progress_loss
                energy_aux_loss = scaled_progress if energy_aux_loss is None else energy_aux_loss + scaled_progress
                loss = loss + scaled_progress
            if action_policy_weight > 0.0:
                if not model.use_action_policy_head:
                    raise ValueError("training.action_policy_weight requires model.use_action_policy_head=true.")
                energy_transitions = _sample_transitions(
                    world,
                    train_examples,
                    rng,
                    goal_energy_aux_batch_size,
                    oracle_probability=1.0,
                )
                energy_batch = collate_transitions(energy_transitions, device=device)
                policy_output = model.action_policy_loss(energy_batch.states, energy_batch.actions)
                loss = loss + action_policy_weight * policy_output.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        model.sync_target()

        if step == 1 or step % eval_every == 0 or step == max_steps:
            latest_metrics = _evaluate(
                model,
                world,
                eval_examples,
                eval_cfg,
                seed + step,
                device,
                eval_oracle_probability,
            )
            latest_metrics.update(
                {
                    "step": step,
                    "train_loss": float(loss.detach().cpu().item()),
                    "train_one_step_loss": float(output.loss.detach().cpu().item()),
                    "train_rollout_loss": (
                        0.0 if rollout_loss is None else float(rollout_loss.detach().cpu().item())
                    ),
                    "train_hierarchy_loss": (
                        0.0 if hierarchy_loss is None else float(hierarchy_loss.detach().cpu().item())
                    ),
                    "train_goal_energy_aux_loss": (
                        0.0 if energy_aux_loss is None else float(energy_aux_loss.detach().cpu().item())
                    ),
                    "train_goal_ranking_loss": (
                        0.0 if ranking_loss is None else float(ranking_loss.detach().cpu().item())
                    ),
                    "rollout_steps": rollout_steps,
                    "rollout_weight": rollout_weight,
                    "rollout_batch_size": rollout_batch_size if rollout_steps > 1 else 0,
                    "rollout_oracle_probability": rollout_oracle_probability if rollout_steps > 1 else 0.0,
                    "goal_energy_weight": goal_energy_weight,
                    "goal_energy_contrastive_loss": str(train_cfg.get("goal_energy_contrastive_loss", "none")),
                    "goal_energy_contrastive_weight": goal_energy_contrastive_weight,
                    "goal_energy_monotonicity_weight": goal_energy_monotonicity_weight,
                    "goal_terminal_correctness_weight": goal_terminal_weight,
                    "goal_ranking_weight": goal_ranking_weight,
                    "goal_ranking_label": str(train_cfg.get("goal_ranking_label", "none")),
                    "goal_local_regression_weight": goal_local_regression_weight,
                    "goal_local_margin_weight": goal_local_margin_weight,
                    "action_advantage_weight": action_advantage_weight,
                    "macro_action_advantage_weight": macro_action_advantage_weight,
                    "latent_progress_weight": latent_progress_weight,
                    "goal_energy_positive_count": goal_energy_positive_count,
                    "goal_energy_negative_count": goal_energy_negative_count,
                    "action_policy_weight": action_policy_weight,
                    "hierarchy_weight": hierarchy_weight,
                    "hierarchy_levels": int(model.hierarchy_levels),
                    "hierarchy_span": int(model.hierarchy_span),
                    "hierarchy_rollout_steps": hierarchy_rollout_steps if hierarchy_weight > 0.0 else 0,
                    "hierarchy_batch_size": hierarchy_batch_size if hierarchy_weight > 0.0 else 0,
                    "hierarchy_oracle_probability": hierarchy_oracle_probability if hierarchy_weight > 0.0 else 0.0,
                    "loss_weighting": str(train_cfg.get("loss_weighting", "uniform")),
                    "param_count": _param_count(model),
                    "trainable_param_count": _param_count(model, trainable_only=True),
                    "train_oracle_probability": train_oracle_probability,
                    "eval_oracle_probability": eval_oracle_probability,
                }
            )
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(latest_metrics, sort_keys=True) + "\n")
            print(json.dumps(latest_metrics, sort_keys=True), flush=True)

        if step % save_every == 0 or step == max_steps:
            _save_checkpoint(output_dir / f"checkpoint-{step}.pt", model, optimizer, step, latest_metrics, config)
            _save_checkpoint(output_dir / "checkpoint.pt", model, optimizer, step, latest_metrics, config)

    final_path = output_dir / "metrics.json"
    with final_path.open("w") as handle:
        json.dump(latest_metrics, handle, indent=2, sort_keys=True)
    return latest_metrics


def _build_world(task_cfg: dict[str, Any]) -> PuzzleWorld:
    name = str(task_cfg["name"])
    if name == "sudoku":
        return SudokuWorld()
    if name == "maze":
        return MazeWorld(height=int(task_cfg.get("height", 30)), width=int(task_cfg.get("width", 30)))
    raise ValueError(f"Unknown task {name!r}.")


def _load_examples(task_cfg: dict[str, Any], split_key: str) -> list[PuzzleExample]:
    name = str(task_cfg["name"])
    world: PuzzleWorld
    if name == "sudoku":
        world = SudokuWorld()
    elif name == "maze":
        world = MazeWorld(height=int(task_cfg.get("height", 30)), width=int(task_cfg.get("width", 30)))
    else:
        raise ValueError(f"Unknown task {name!r}.")
    split = str(task_cfg[f"{split_key}_split"])
    limit = task_cfg.get(f"{split_key}_limit")
    return list(
        iter_hf_examples(
            repo_id=str(task_cfg["repo_id"]),
            split=split,
            world=world,
            columns=HFPuzzleColumns(
                puzzle=str(task_cfg.get("puzzle_column", "question")),
                solution=str(task_cfg.get("solution_column", "answer")),
            ),
            limit=None if limit is None else int(limit),
        )
    )


def _sample_batch(
    world: PuzzleWorld,
    examples: list[PuzzleExample],
    rng: np.random.Generator,
    batch_size: int,
    device: torch.device,
    oracle_probability: float = 1.0,
):
    transitions = _sample_transitions(world, examples, rng, batch_size, oracle_probability)
    return collate_transitions(transitions, device=device)


def _sample_transitions(
    world: PuzzleWorld,
    examples: list[PuzzleExample],
    rng: np.random.Generator,
    batch_size: int,
    oracle_probability: float = 1.0,
) -> list[Transition]:
    return [
        sample_curriculum_transition(
            world,
            examples[int(rng.integers(0, len(examples)))],
            rng,
            oracle_probability=oracle_probability,
        )
        for _ in range(batch_size)
    ]


def _sample_rollout_batch(
    world: PuzzleWorld,
    examples: list[PuzzleExample],
    rng: np.random.Generator,
    batch_size: int,
    steps: int,
    device: torch.device,
    oracle_probability: float = 1.0,
):
    rollouts = [
        sample_curriculum_rollout_transition(
            world,
            examples[int(rng.integers(0, len(examples)))],
            rng,
            steps=steps,
            oracle_probability=oracle_probability,
        )
        for _ in range(batch_size)
    ]
    return collate_rollouts(rollouts, device=device)


def _build_transition_loss_weights(
    world: PuzzleWorld,
    actions: torch.Tensor,
    shape: torch.Size | tuple[int, int],
    train_cfg: dict[str, Any],
) -> torch.Tensor | None:
    mode = str(train_cfg.get("loss_weighting", "uniform"))
    if mode == "uniform":
        return None
    height, width = int(shape[0]), int(shape[1])
    base = float(train_cfg.get("base_cell_weight", 1.0))
    changed = float(train_cfg.get("changed_cell_weight", 8.0))
    context = float(train_cfg.get("context_cell_weight", 2.0))
    rows = actions[:, 1].clamp(0, height - 1)
    cols = actions[:, 2].clamp(0, width - 1)
    batch = actions.shape[0]
    weights = torch.full((batch, height, width), base, dtype=torch.float32, device=actions.device)
    batch_indices = torch.arange(batch, device=actions.device)
    if mode == "changed_only":
        weights.zero_()
        weights[batch_indices, rows, cols] = 1.0
        return weights
    if mode != "local_context":
        raise ValueError(f"Unknown loss_weighting mode {mode!r}.")
    if isinstance(world, SudokuWorld):
        for index in range(batch):
            row = int(rows[index].item())
            col = int(cols[index].item())
            block_row = 3 * (row // 3)
            block_col = 3 * (col // 3)
            weights[index, row, :] = context
            weights[index, :, col] = context
            weights[index, block_row : block_row + 3, block_col : block_col + 3] = context
    else:
        radius = int(train_cfg.get("context_radius", 1))
        for index in range(batch):
            row = int(rows[index].item())
            col = int(cols[index].item())
            row_start = max(0, row - radius)
            row_end = min(height, row + radius + 1)
            col_start = max(0, col - radius)
            col_end = min(width, col + radius + 1)
            weights[index, row_start:row_end, col_start:col_end] = context
    weights[batch_indices, rows, cols] = changed
    return weights


def _goal_energy_auxiliary_loss(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    train_examples: list[PuzzleExample],
    rng: np.random.Generator,
    train_cfg: dict[str, Any],
    device: torch.device,
    batch_size: int,
    positive_count: int,
    negative_count: int,
    contrastive_weight: float,
    monotonicity_weight: float,
):
    energy_transitions = _sample_goal_energy_aux_transitions(world, train_examples, rng, train_cfg, batch_size)
    energy_batch = collate_transitions(energy_transitions, device=device)
    positive_mode = str(train_cfg.get("goal_energy_positive_mode", "single_oracle"))
    if positive_mode == "all_goal_correct":
        positive_states = _sample_local_positive_states(
            world,
            energy_transitions,
            rng,
            max(1, positive_count),
            device,
        )
    elif positive_mode == "single_oracle":
        positive_states = energy_batch.next_states
    else:
        raise ValueError("goal_energy_positive_mode must be 'single_oracle' or 'all_goal_correct'.")
    negative_states = _sample_local_negative_states(
        world,
        energy_transitions,
        rng,
        negative_count,
        device,
        exclude_goal_correct=positive_mode == "all_goal_correct",
    )
    return model.goal_energy_loss(
        energy_batch.states,
        energy_batch.goals,
        task_ids=energy_batch.actions[:, 0],
        initial_states=_initial_states_from_batch(world, energy_batch),
        positive_states=positive_states,
        negative_states=negative_states,
        contrastive_loss=str(train_cfg.get("goal_energy_contrastive_loss", "margin")),
        contrastive_temperature=float(train_cfg.get("goal_energy_contrastive_temperature", 0.1)),
        contrastive_margin=float(train_cfg.get("goal_energy_contrastive_margin", 0.05)),
        contrastive_weight=contrastive_weight,
        monotonicity_weight=monotonicity_weight,
        monotonicity_margin=float(train_cfg.get("goal_energy_monotonicity_margin", 0.0)),
        regression_weight=float(train_cfg.get("goal_energy_aux_regression_weight", 0.0)),
        target_scale=float(train_cfg.get("goal_energy_target_scale", 1.0)),
    )


def _goal_terminal_correctness_loss(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    train_examples: list[PuzzleExample],
    rng: np.random.Generator,
    train_cfg: dict[str, Any],
    device: torch.device,
):
    if not model.use_goal_energy_head:
        raise ValueError("terminal correctness training requires model.use_goal_energy_head=true.")
    batch_size = int(train_cfg.get("goal_terminal_batch_size", train_cfg.get("goal_energy_aux_batch_size", 64)))
    positive_fraction = float(train_cfg.get("goal_terminal_positive_fraction", 0.5))
    positive_count = max(1, min(batch_size - 1, int(round(batch_size * positive_fraction))))
    negative_count = batch_size - positive_count
    examples = [train_examples[int(rng.integers(0, len(train_examples)))] for _ in range(batch_size)]
    positive_states = [world.validate_state(example.goal).copy() for example in examples[:positive_count]]
    negative_states = [
        sample_random_mutable_transition(world, example, rng).next_state
        for example in examples[positive_count:]
    ]
    states = torch.as_tensor(np.stack(positive_states + negative_states), dtype=torch.long, device=device)
    goals = torch.as_tensor(np.stack([world.validate_state(example.goal) for example in examples]), dtype=torch.long, device=device)
    if isinstance(world, SudokuWorld):
        initial = torch.as_tensor(
            np.stack([world.validate_state(example.state) for example in examples]),
            dtype=torch.long,
            device=device,
        )
    else:
        initial = states
    task_ids = torch.full((batch_size,), world.task_id, dtype=torch.long, device=device)
    return model.goal_energy_loss(
        states,
        goals,
        task_ids=task_ids,
        initial_states=initial,
        terminal_correctness_weight=float(train_cfg.get("goal_terminal_correctness_weight", 1.0)),
        terminal_target_mode=str(train_cfg.get("goal_terminal_target_mode", "binary")),
        terminal_discount=float(train_cfg.get("goal_terminal_discount", 0.99)),
        regression_weight=float(train_cfg.get("goal_terminal_regression_weight", 0.0)),
    )


def _goal_energy_ranking_loss(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    train_examples: list[PuzzleExample],
    rng: np.random.Generator,
    train_cfg: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    if not model.use_goal_energy_head:
        raise ValueError("ranking training requires model.use_goal_energy_head=true.")
    batch_size = int(train_cfg.get("goal_ranking_batch_size", train_cfg.get("goal_energy_aux_batch_size", 16)))
    candidate_count = int(train_cfg.get("goal_ranking_candidates", 128))
    if batch_size <= 0 or candidate_count <= 1:
        raise ValueError("goal ranking needs a positive batch size and at least two candidates.")
    transitions = _sample_goal_energy_aux_transitions(world, train_examples, rng, train_cfg, batch_size)
    candidate_rows: list[np.ndarray] = []
    goal_rows: list[np.ndarray] = []
    initial_rows: list[np.ndarray] = []
    remaining_rows: list[np.ndarray] = []
    for transition in transitions:
        candidates = _successors_for_ranking_query(world, transition)
        if not candidates:
            candidates = [transition.next_state.copy()]
        indices = rng.choice(len(candidates), size=candidate_count, replace=len(candidates) < candidate_count)
        sampled = np.stack([candidates[int(index)] for index in indices])
        candidate_rows.append(sampled)
        goal = world.validate_state(transition.goal)
        goal_rows.append(np.broadcast_to(goal, sampled.shape).copy())
        if isinstance(world, SudokuWorld):
            if transition.clue_mask is None:
                raise ValueError("Sudoku ranking queries require a clue mask.")
            initial = world.validate_state(transition.state) * transition.clue_mask
        else:
            initial = world.validate_state(transition.state)
        initial_rows.append(np.broadcast_to(initial, sampled.shape).copy())
        remaining_rows.append((sampled != goal).reshape(sampled.shape[0], -1).sum(axis=1).astype(np.float32))
    states = torch.as_tensor(np.stack(candidate_rows), dtype=torch.long, device=device)
    goals = torch.as_tensor(np.stack(goal_rows), dtype=torch.long, device=device)
    initial_states = torch.as_tensor(np.stack(initial_rows), dtype=torch.long, device=device)
    remaining = torch.as_tensor(np.stack(remaining_rows), dtype=torch.float32, device=device)
    query_count, candidates_per_query = states.shape[:2]
    flat_states = states.reshape(query_count * candidates_per_query, *states.shape[2:])
    flat_goals = goals.reshape(query_count * candidates_per_query, *goals.shape[2:])
    flat_initials = initial_states.reshape(query_count * candidates_per_query, *initial_states.shape[2:])
    task_ids = torch.full((query_count * candidates_per_query,), world.task_id, dtype=torch.long, device=device)
    pred_energy = model.predict_goal_energy(flat_states, flat_initials, task_ids).reshape(query_count, candidates_per_query)
    pred_scores = -pred_energy
    label_mode = str(train_cfg.get("goal_ranking_label", "remaining_discounted"))
    if label_mode == "remaining_discounted":
        gamma = min(max(float(train_cfg.get("goal_ranking_discount", 0.99)), 0.0), 1.0)
        relevance = torch.pow(torch.full_like(remaining, gamma), remaining)
    elif label_mode == "remaining_wrong":
        relevance = -remaining
    elif label_mode == "latent_goal_distance":
        with torch.no_grad():
            flat_task_ids = task_ids
            state_latents = model.target_encoder(flat_states, task_ids=flat_task_ids)
            goal_latents = model.target_encoder(flat_goals, task_ids=flat_task_ids)
            latent_distance = torch.nn.functional.mse_loss(
                state_latents,
                goal_latents,
                reduction="none",
            ).mean(dim=(1, 2)).reshape(query_count, candidates_per_query)
        relevance = -latent_distance
    else:
        raise ValueError(
            "goal_ranking_label must be 'remaining_discounted', 'remaining_wrong', or 'latent_goal_distance'."
        )
    label_temperature = max(float(train_cfg.get("goal_ranking_label_temperature", 1.0)), 1e-6)
    score_temperature = max(float(train_cfg.get("goal_ranking_score_temperature", 0.05)), 1e-6)
    target_logits = _standardize_query_values(relevance) / label_temperature
    target_probs = torch.softmax(target_logits, dim=1)
    log_pred_probs = torch.log_softmax(pred_scores / score_temperature, dim=1)
    return -(target_probs * log_pred_probs).sum(dim=1).mean()


def _ranking_query_tensors(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    train_examples: list[PuzzleExample],
    rng: np.random.Generator,
    train_cfg: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = int(train_cfg.get("goal_ranking_batch_size", train_cfg.get("goal_energy_aux_batch_size", 16)))
    candidate_count = int(train_cfg.get("goal_ranking_candidates", 128))
    transitions = _sample_goal_energy_aux_transitions(world, train_examples, rng, train_cfg, batch_size)
    candidate_rows: list[np.ndarray] = []
    goal_rows: list[np.ndarray] = []
    initial_rows: list[np.ndarray] = []
    current_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    for transition in transitions:
        state = world.validate_state(transition.state)
        actions = _successor_actions_for_ranking_query(world, transition)
        candidates = []
        kept_actions = []
        for action in actions:
            try:
                if isinstance(world, SudokuWorld):
                    if transition.clue_mask is None:
                        raise ValueError("Sudoku ranking queries require a clue mask.")
                    candidates.append(
                        world.apply(state, action, clue_mask=transition.clue_mask, allow_overwrite=True, allow_conflicts=True)
                    )
                else:
                    candidates.append(world.apply(state, action))
                kept_actions.append(action)
            except ValueError:
                continue
        if not candidates:
            candidates = [transition.next_state.copy()]
            kept_actions = [transition.action]
        indices = rng.choice(len(candidates), size=candidate_count, replace=len(candidates) < candidate_count)
        sampled = np.stack([candidates[int(index)] for index in indices])
        sampled_actions = np.asarray(
            [[world.task_id, kept_actions[int(index)].row, kept_actions[int(index)].col, kept_actions[int(index)].value] for index in indices],
            dtype=np.int64,
        )
        candidate_rows.append(sampled)
        current_rows.append(np.broadcast_to(state, sampled.shape).copy())
        action_rows.append(sampled_actions)
        goal = world.validate_state(transition.goal)
        goal_rows.append(np.broadcast_to(goal, sampled.shape).copy())
        if isinstance(world, SudokuWorld):
            if transition.clue_mask is None:
                raise ValueError("Sudoku ranking queries require a clue mask.")
            initial = world.validate_state(transition.state) * transition.clue_mask
        else:
            initial = state
        initial_rows.append(np.broadcast_to(initial, sampled.shape).copy())
    states = torch.as_tensor(np.stack(candidate_rows), dtype=torch.long, device=device)
    currents = torch.as_tensor(np.stack(current_rows), dtype=torch.long, device=device)
    actions = torch.as_tensor(np.stack(action_rows), dtype=torch.long, device=device)
    goals = torch.as_tensor(np.stack(goal_rows), dtype=torch.long, device=device)
    initials = torch.as_tensor(np.stack(initial_rows), dtype=torch.long, device=device)
    task_ids = torch.full((states.shape[0] * states.shape[1],), world.task_id, dtype=torch.long, device=device)
    return states, currents, actions, goals, initials, task_ids


def _goal_local_score_regression_loss(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    train_examples: list[PuzzleExample],
    rng: np.random.Generator,
    train_cfg: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    states, _currents, _actions, goals, initials, task_ids = _ranking_query_tensors(
        model, world, train_examples, rng, train_cfg, device
    )
    query_count, candidates = states.shape[:2]
    flat_states = states.reshape(query_count * candidates, *states.shape[2:])
    flat_goals = goals.reshape(query_count * candidates, *goals.shape[2:])
    flat_initials = initials.reshape(query_count * candidates, *initials.shape[2:])
    pred_energy = model.predict_goal_energy(flat_states, flat_initials, task_ids).reshape(query_count, candidates)
    pred_score = -pred_energy
    label_mode = str(train_cfg.get("goal_local_regression_label", "latent_goal_distance"))
    if label_mode == "remaining_wrong":
        remaining = (states != goals).flatten(start_dim=2).sum(dim=2).float()
        target_score = -remaining
    else:
        with torch.no_grad():
            state_latents = model.target_encoder(flat_states, task_ids=task_ids)
            goal_latents = model.target_encoder(flat_goals, task_ids=task_ids)
            distance = torch.nn.functional.mse_loss(state_latents, goal_latents, reduction="none").mean(dim=(1, 2))
            target_score = -distance.reshape(query_count, candidates)
    pred_norm = _standardize_query_values(pred_score)
    target_norm = _standardize_query_values(target_score)
    return torch.nn.functional.mse_loss(pred_norm, target_norm)


def _goal_local_margin_loss(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    train_examples: list[PuzzleExample],
    rng: np.random.Generator,
    train_cfg: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    states, _currents, _actions, goals, initials, task_ids = _ranking_query_tensors(
        model, world, train_examples, rng, train_cfg, device
    )
    query_count, candidates = states.shape[:2]
    flat_states = states.reshape(query_count * candidates, *states.shape[2:])
    flat_initials = initials.reshape(query_count * candidates, *initials.shape[2:])
    pred_energy = model.predict_goal_energy(flat_states, flat_initials, task_ids).reshape(query_count, candidates)
    remaining = (states != goals).flatten(start_dim=2).sum(dim=2)
    best_remaining = remaining.min(dim=1, keepdim=True).values
    positive = remaining == best_remaining
    negative = remaining > best_remaining
    losses = []
    margin = float(train_cfg.get("goal_local_margin", 0.1))
    for row in range(query_count):
        pos = pred_energy[row][positive[row]]
        neg = pred_energy[row][negative[row]]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        losses.append(torch.nn.functional.relu(margin + pos[:, None] - neg[None, :]).mean())
    if not losses:
        return pred_energy.sum() * 0.0
    return torch.stack(losses).mean()


def _action_advantage_loss(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    train_examples: list[PuzzleExample],
    rng: np.random.Generator,
    train_cfg: dict[str, Any],
    device: torch.device,
) -> Any:
    if not model.use_action_value_head:
        raise ValueError("training.action_advantage_weight requires model.use_action_value_head=true.")
    states, currents, actions, goals, initials, task_ids = _ranking_query_tensors(
        model, world, train_examples, rng, train_cfg, device
    )
    query_count, candidates = states.shape[:2]
    flat_states = states.reshape(query_count * candidates, *states.shape[2:])
    flat_currents = currents.reshape(query_count * candidates, *currents.shape[2:])
    flat_goals = goals.reshape(query_count * candidates, *goals.shape[2:])
    flat_initials = initials.reshape(query_count * candidates, *initials.shape[2:])
    flat_actions = actions.reshape(query_count * candidates, actions.shape[-1])
    with torch.no_grad():
        current_latents = model.target_encoder(flat_currents, task_ids=task_ids)
        next_latents = model.target_encoder(flat_states, task_ids=task_ids)
        goal_latents = model.target_encoder(flat_goals, task_ids=task_ids)
        current_energy = torch.nn.functional.mse_loss(current_latents, goal_latents, reduction="none").mean(dim=(1, 2))
        next_energy = torch.nn.functional.mse_loss(next_latents, goal_latents, reduction="none").mean(dim=(1, 2))
        advantage = (current_energy - next_energy).reshape(query_count, candidates)
        if bool(train_cfg.get("action_advantage_standardize", True)):
            advantage = _standardize_query_values(advantage)
        targets = advantage.reshape(query_count * candidates)
    return model.action_value_loss(flat_currents, flat_initials, flat_actions, targets)


def _macro_action_advantage_loss(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    train_examples: list[PuzzleExample],
    rng: np.random.Generator,
    train_cfg: dict[str, Any],
    device: torch.device,
) -> Any:
    if not model.use_macro_action_value_head:
        raise ValueError("training.macro_action_advantage_weight requires model.use_macro_action_value_head=true.")
    level = int(train_cfg.get("macro_action_advantage_level", model.hierarchy_levels - 1))
    if level <= 0 or level >= model.hierarchy_levels:
        raise ValueError("macro_action_advantage_level must be in [1, hierarchy_levels).")
    steps = int(model.hierarchy_span) ** level
    batch_size = int(train_cfg.get("macro_action_advantage_batch_size", train_cfg.get("hierarchy_batch_size", 64)))
    rollout_batch = _sample_rollout_batch(world, train_examples, rng, batch_size, steps, device)
    return model.macro_action_value_loss(
        rollout_batch.states,
        _initial_states_from_rollout_batch(world, rollout_batch),
        rollout_batch.actions,
        rollout_batch.goals,
        level=level,
        standardize=bool(train_cfg.get("macro_action_advantage_standardize", True)),
    )


def _latent_progress_loss(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    train_examples: list[PuzzleExample],
    rng: np.random.Generator,
    train_cfg: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    batch_size = int(train_cfg.get("latent_progress_batch_size", train_cfg.get("goal_energy_aux_batch_size", 64)))
    transitions = _sample_goal_energy_aux_transitions(world, train_examples, rng, train_cfg, batch_size)
    batch = collate_transitions(transitions, device=device)
    task_ids = batch.actions[:, 0]
    state_latents = model.encoder(batch.states, task_ids=task_ids)
    goal_latents = model.encoder(batch.goals, task_ids=task_ids)
    distance = torch.nn.functional.mse_loss(state_latents, goal_latents, reduction="none").mean(dim=(1, 2))
    remaining = (batch.states != batch.goals).flatten(start_dim=1).float().mean(dim=1)
    scale = float(train_cfg.get("latent_progress_target_scale", 1.0))
    return torch.nn.functional.mse_loss(distance, remaining * scale)


def _successors_for_ranking_query(world: PuzzleWorld, transition: Transition) -> list[np.ndarray]:
    state = world.validate_state(transition.state)
    actions = _successor_actions_for_ranking_query(world, transition)
    candidates: list[np.ndarray] = []
    for action in actions:
        try:
            if isinstance(world, SudokuWorld):
                candidates.append(
                    world.apply(
                        state,
                        action,
                        clue_mask=transition.clue_mask,
                        allow_overwrite=True,
                        allow_conflicts=True,
                    )
                )
            else:
                candidates.append(world.apply(state, action))
        except ValueError:
            continue
    return candidates


def _successor_actions_for_ranking_query(world: PuzzleWorld, transition: Transition) -> list[WorldAction]:
    state = world.validate_state(transition.state)
    if isinstance(world, SudokuWorld):
        if transition.clue_mask is None:
            raise ValueError("Sudoku ranking queries require a clue mask.")
        return world.legal_actions(
            state,
            clue_mask=transition.clue_mask,
            allow_overwrite=True,
            allow_conflicts=True,
        )
    return world.legal_actions(state)


def _standardize_query_values(values: torch.Tensor) -> torch.Tensor:
    centered = values - values.mean(dim=1, keepdim=True)
    scale = centered.std(dim=1, keepdim=True).clamp_min(1e-6)
    return centered / scale


def _sample_goal_energy_aux_transitions(
    world: PuzzleWorld,
    examples: list[PuzzleExample],
    rng: np.random.Generator,
    train_cfg: dict[str, Any],
    batch_size: int,
) -> list[Transition]:
    mode = str(train_cfg.get("goal_energy_aux_sampling", "random"))
    if mode == "random":
        return _sample_transitions(world, examples, rng, batch_size, oracle_probability=1.0)
    if mode != "stratified_puzzle":
        raise ValueError("goal_energy_aux_sampling must be 'random' or 'stratified_puzzle'.")
    puzzles = int(train_cfg.get("goal_energy_aux_puzzles", 16))
    states_per_puzzle = int(train_cfg.get("goal_energy_aux_states_per_puzzle", 4))
    if puzzles <= 0 or states_per_puzzle <= 0:
        raise ValueError("stratified goal-energy auxiliary sampling needs positive puzzle/state counts.")
    selected = rng.choice(len(examples), size=puzzles, replace=len(examples) < puzzles)
    transitions: list[Transition] = []
    for example_index in selected:
        example = examples[int(example_index)]
        for _ in range(states_per_puzzle):
            transitions.append(sample_oracle_partial_transition(world, example, rng))
    if len(transitions) == batch_size:
        return transitions
    if len(transitions) > batch_size:
        chosen = rng.choice(len(transitions), size=batch_size, replace=False)
        return [transitions[int(index)] for index in chosen]
    while len(transitions) < batch_size:
        example = examples[int(selected[int(rng.integers(0, len(selected)))])]
        transitions.append(sample_oracle_partial_transition(world, example, rng))
    return transitions


def _initial_states_from_batch(world: PuzzleWorld, batch) -> torch.Tensor:
    if isinstance(world, SudokuWorld) and batch.clue_masks is not None:
        return batch.states * batch.clue_masks.to(dtype=batch.states.dtype)
    return batch.states


def _initial_states_from_rollout_batch(world: PuzzleWorld, batch) -> torch.Tensor:
    if isinstance(world, SudokuWorld) and batch.clue_masks is not None:
        return batch.states * batch.clue_masks.to(dtype=batch.states.dtype)
    return batch.states


def _sample_local_negative_states(
    world: PuzzleWorld,
    transitions: list[Transition],
    rng: np.random.Generator,
    negative_count: int,
    device: torch.device,
    *,
    exclude_goal_correct: bool = False,
) -> torch.Tensor:
    if negative_count <= 0:
        raise ValueError("goal_energy_negative_count must be positive when contrastive energy loss is enabled.")
    rows: list[np.ndarray] = []
    for transition in transitions:
        candidates = _negative_successors_for_transition(world, transition, exclude_goal_correct=exclude_goal_correct)
        if not candidates:
            candidates = [transition.state.copy()]
        indices = rng.choice(len(candidates), size=negative_count, replace=len(candidates) < negative_count)
        rows.append(np.stack([candidates[int(index)] for index in indices]))
    return torch.as_tensor(np.stack(rows), dtype=torch.long, device=device)


def _sample_local_positive_states(
    world: PuzzleWorld,
    transitions: list[Transition],
    rng: np.random.Generator,
    positive_count: int,
    device: torch.device,
) -> torch.Tensor:
    if positive_count <= 0:
        raise ValueError("goal_energy_positive_count must be positive when all_goal_correct positives are enabled.")
    rows: list[np.ndarray] = []
    for transition in transitions:
        candidates = _positive_successors_for_transition(world, transition)
        if not candidates:
            candidates = [transition.next_state.copy()]
        indices = rng.choice(len(candidates), size=positive_count, replace=len(candidates) < positive_count)
        rows.append(np.stack([candidates[int(index)] for index in indices]))
    return torch.as_tensor(np.stack(rows), dtype=torch.long, device=device)


def _positive_successors_for_transition(world: PuzzleWorld, transition: Transition) -> list[np.ndarray]:
    state = world.validate_state(transition.state)
    goal = world.validate_state(transition.goal)
    if isinstance(world, SudokuWorld):
        if transition.clue_mask is None:
            raise ValueError("Sudoku positives require a clue mask.")
        actions = world.legal_actions(
            state,
            clue_mask=transition.clue_mask,
            allow_overwrite=True,
            allow_conflicts=True,
        )
    else:
        actions = world.legal_actions(state)
    candidates: list[np.ndarray] = []
    for action in actions:
        if state[action.row, action.col] == goal[action.row, action.col]:
            continue
        if action.value != int(goal[action.row, action.col]):
            continue
        try:
            if isinstance(world, SudokuWorld):
                next_state = world.apply(
                    state,
                    action,
                    clue_mask=transition.clue_mask,
                    allow_overwrite=True,
                    allow_conflicts=True,
                )
            else:
                next_state = world.apply(state, action)
        except ValueError:
            continue
        candidates.append(next_state)
    return candidates


def _negative_successors_for_transition(
    world: PuzzleWorld,
    transition: Transition,
    *,
    exclude_goal_correct: bool = False,
) -> list[np.ndarray]:
    state = world.validate_state(transition.state)
    actions: list[WorldAction]
    if isinstance(world, SudokuWorld):
        if transition.clue_mask is None:
            raise ValueError("Sudoku contrastive negatives require a clue mask.")
        actions = world.legal_actions(
            state,
            clue_mask=transition.clue_mask,
            allow_overwrite=True,
            allow_conflicts=True,
        )
    else:
        actions = world.legal_actions(state)
    candidates: list[np.ndarray] = []
    for action in actions:
        if action == transition.action:
            continue
        if exclude_goal_correct and action.value == int(transition.goal[action.row, action.col]):
            continue
        try:
            if isinstance(world, SudokuWorld):
                next_state = world.apply(
                    state,
                    action,
                    clue_mask=transition.clue_mask,
                    allow_overwrite=True,
                    allow_conflicts=True,
                )
            else:
                next_state = world.apply(state, action)
        except ValueError:
            continue
        if not np.array_equal(next_state, transition.next_state):
            candidates.append(next_state)
    return candidates


def _max_hierarchy_horizon(model: ActionConditionedWorldModel) -> int:
    return int(model.hierarchy_span) ** max(0, int(model.hierarchy_levels) - 1)


@torch.no_grad()
def _evaluate(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    examples: list[PuzzleExample],
    eval_cfg: dict[str, Any],
    seed: int,
    device: torch.device,
    oracle_probability: float = 1.0,
) -> dict[str, Any]:
    model.eval()
    rng = np.random.default_rng(seed)
    num_batches = int(eval_cfg.get("num_batches", 2))
    batch_size = int(eval_cfg.get("batch_size", 32))
    losses = []
    top1 = []
    ranks = []
    for _ in range(num_batches):
        batch = _sample_batch(world, examples, rng, batch_size, device, oracle_probability)
        output = model(batch.states, batch.actions, batch.next_states)
        losses.append(float(output.loss.detach().cpu().item()))
        rank_metrics = _oracle_action_ranks(model, world, batch, int(eval_cfg.get("rank_examples", 8)))
        top1.extend(rank_metrics["top1"])
        ranks.extend(rank_metrics["ranks"])
    plan = _planning_eval(model, world, examples, rng, eval_cfg, device)
    distance = _latent_distance_eval(model, world, examples, rng, eval_cfg, device)
    goal_energy = _goal_energy_eval(model, world, examples, rng, eval_cfg, device)
    return {
        "eval_loss": float(np.mean(losses)),
        "oracle_action_top1": float(np.mean(top1)) if top1 else 0.0,
        "oracle_action_mean_rank": float(np.mean(ranks)) if ranks else math.inf,
        **plan,
        **distance,
        **goal_energy,
    }


@torch.no_grad()
def _oracle_action_ranks(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    batch,
    max_examples: int,
) -> dict[str, list[float]]:
    ranks: list[float] = []
    top1: list[float] = []
    states = batch.states.detach().cpu().numpy()
    actions_tensor = batch.actions.detach().cpu().numpy()
    clue_masks = None if batch.clue_masks is None else batch.clue_masks.detach().cpu().numpy()
    for index in range(min(max_examples, len(states))):
        state = states[index]
        clue_mask = None if clue_masks is None else clue_masks[index]
        oracle = WorldAction(
            int(actions_tensor[index, 1]),
            int(actions_tensor[index, 2]),
            int(actions_tensor[index, 3]),
        )
        actions = _legal_planning_actions(world, state, clue_mask)
        if not actions:
            continue
        scores = model.score_actions_to_goal(
            batch.states[index],
            actions,
            batch.goals[index],
            world.task_id,
        )
        order = scores.argsort(descending=True).detach().cpu().tolist()
        action_order = [actions[i] for i in order]
        try:
            rank = action_order.index(oracle) + 1
        except ValueError:
            continue
        ranks.append(float(rank))
        top1.append(float(rank == 1))
    return {"ranks": ranks, "top1": top1}


@torch.no_grad()
def _planning_eval(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    examples: list[PuzzleExample],
    rng: np.random.Generator,
    eval_cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    del device
    max_examples = int(eval_cfg.get("planning_examples", 4))
    max_steps = int(eval_cfg.get("max_plan_steps", 32))
    horizons = eval_cfg.get("planning_horizons", [1])
    if isinstance(horizons, int):
        horizons = [horizons]
    branch_size = int(eval_cfg.get("planning_branching", 1))
    metrics: dict[str, Any] = {}
    for horizon in [int(item) for item in horizons]:
        solved = []
        lengths = []
        for _ in range(max_examples):
            example = examples[int(rng.integers(0, len(examples)))]
            final_state, steps = _plan_fixed_horizon(
                model,
                world,
                example.state,
                example.goal,
                clue_mask=_clue_mask_for_planning(world, example.state),
                horizon=max(1, horizon),
                branch_size=branch_size,
                max_steps=max_steps,
            )
            solved.append(float(world.is_goal(final_state, example.goal)))
            lengths.append(float(steps))
        metrics[f"planning_h{horizon}_solve_rate"] = float(np.mean(solved)) if solved else 0.0
        metrics[f"planning_h{horizon}_mean_steps"] = float(np.mean(lengths)) if lengths else 0.0
    return metrics


@torch.no_grad()
def _plan_fixed_horizon(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    state: np.ndarray,
    goal: np.ndarray,
    clue_mask: np.ndarray | None,
    horizon: int,
    branch_size: int,
    max_steps: int,
) -> tuple[np.ndarray, int]:
    current = world.validate_state(state).copy()
    goal_arr = world.validate_state(goal)
    for step in range(max_steps):
        if world.is_goal(current, goal_arr):
            return current, step
        ranked = _rank_actions_fixed_horizon(model, world, current, goal_arr, clue_mask, horizon, branch_size)
        if not ranked:
            return current, step
        try:
            current = _apply_planning_action(world, current, ranked[0][0], clue_mask)
        except ValueError:
            return current, step
    return current, max_steps


@torch.no_grad()
def _rank_actions_fixed_horizon(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    state: np.ndarray,
    goal: np.ndarray,
    clue_mask: np.ndarray | None,
    horizon: int,
    branch_size: int,
) -> list[tuple[WorldAction, float]]:
    if horizon <= 1:
        return _rank_model_actions(model, world, state, goal, clue_mask, limit=None)

    first_actions = _rank_model_actions(model, world, state, goal, clue_mask, limit=max(1, branch_size))
    frontier: list[tuple[np.ndarray, WorldAction]] = []
    for action, _score in first_actions:
        try:
            frontier.append((_apply_planning_action(world, state, action, clue_mask), action))
        except ValueError:
            continue
    for _depth in range(1, horizon):
        next_frontier: list[tuple[np.ndarray, WorldAction]] = []
        for current, first_action in frontier:
            if world.is_goal(current, goal):
                next_frontier.append((current, first_action))
                continue
            for action, _score in _rank_model_actions(model, world, current, goal, clue_mask, limit=max(1, branch_size)):
                try:
                    next_frontier.append((_apply_planning_action(world, current, action, clue_mask), first_action))
                except ValueError:
                    continue
        if not next_frontier:
            break
        frontier = next_frontier

    if not frontier:
        return []
    terminal_states = torch.as_tensor(np.stack([item[0] for item in frontier]), dtype=torch.long)
    goals = torch.as_tensor(np.stack([goal for _state, _action in frontier]), dtype=torch.long)
    scores = model.score_states_to_goal(terminal_states, goals, world.task_id).detach().cpu().tolist()
    best_by_first: dict[WorldAction, float] = {}
    for (_terminal, first_action), score in zip(frontier, scores, strict=True):
        score = float(score)
        if world.is_goal(_terminal, goal):
            score = float("inf")
        best_by_first[first_action] = max(best_by_first.get(first_action, float("-inf")), score)
    return sorted(best_by_first.items(), key=lambda item: item[1], reverse=True)


@torch.no_grad()
def _rank_model_actions(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    state: np.ndarray,
    goal: np.ndarray,
    clue_mask: np.ndarray | None,
    limit: int | None,
) -> list[tuple[WorldAction, float]]:
    actions = _legal_planning_actions(world, state, clue_mask)
    if not actions:
        return []
    scores = model.score_actions_to_goal(
        torch.as_tensor(state, dtype=torch.long),
        actions,
        torch.as_tensor(goal, dtype=torch.long),
        world.task_id,
    )
    order = scores.argsort(descending=True).detach().cpu().tolist()
    if limit is not None:
        order = order[:limit]
    return [(actions[index], float(scores[index].item())) for index in order]


def _clue_mask_for_planning(world: PuzzleWorld, initial_state: np.ndarray) -> np.ndarray | None:
    if isinstance(world, SudokuWorld):
        return world.clue_mask_from_puzzle(initial_state)
    return None


def _legal_planning_actions(
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


def _apply_planning_action(
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


@torch.no_grad()
def _latent_distance_eval(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    examples: list[PuzzleExample],
    rng: np.random.Generator,
    eval_cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    del device
    num_examples = int(eval_cfg.get("distance_examples", 0))
    if num_examples <= 0:
        return {}

    metrics: dict[str, Any] = {}
    samplers = {
        "oracle": sample_oracle_partial_transition,
        "random": sample_random_mutable_transition,
    }
    for name, sampler in samplers.items():
        deltas = []
        improves = []
        for _ in range(num_examples):
            transition = sampler(world, examples[int(rng.integers(0, len(examples)))], rng)
            states = torch.as_tensor(np.stack([transition.state, transition.next_state]), dtype=torch.long)
            goals = torch.as_tensor(np.stack([transition.goal, transition.goal]), dtype=torch.long)
            before, after = model.score_states_to_goal(states, goals, transition.task_id).detach().cpu().tolist()
            delta = float(after - before)
            deltas.append(delta)
            improves.append(float(delta > 0.0))
        metrics[f"{name}_latent_delta"] = float(np.mean(deltas)) if deltas else 0.0
        metrics[f"{name}_latent_improve_rate"] = float(np.mean(improves)) if improves else 0.0
    return metrics


@torch.no_grad()
def _goal_energy_eval(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    examples: list[PuzzleExample],
    rng: np.random.Generator,
    eval_cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    if not model.use_goal_energy_head:
        return {}
    num_examples = int(eval_cfg.get("goal_energy_examples", eval_cfg.get("distance_examples", 0)))
    if num_examples <= 0:
        return {}
    transitions = [
        sample_curriculum_transition(
            world,
            examples[int(rng.integers(0, len(examples)))],
            rng,
            oracle_probability=float(eval_cfg.get("oracle_probability", 1.0)),
        )
        for _ in range(num_examples)
    ]
    batch = collate_transitions(transitions, device=device)
    task_ids = batch.actions[:, 0]
    initial_states = _initial_states_from_batch(world, batch)
    pred = model.predict_goal_energy(batch.states, initial_states, task_ids)
    state_latents = model.target_encoder(batch.states, task_ids=task_ids)
    goal_latents = model.target_encoder(batch.goals, task_ids=task_ids)
    target = torch.nn.functional.mse_loss(state_latents, goal_latents, reduction="none").mean(dim=(1, 2))
    mse = torch.nn.functional.mse_loss(pred, target)
    return {
        "goal_energy_eval_mse": float(mse.detach().cpu().item()),
        "goal_energy_eval_pred_mean": float(pred.detach().mean().cpu().item()),
        "goal_energy_eval_target_mean": float(target.detach().mean().cpu().item()),
    }


def _param_count(model: torch.nn.Module, trainable_only: bool = False) -> int:
    params = model.parameters()
    if trainable_only:
        return sum(param.numel() for param in params if param.requires_grad)
    return sum(param.numel() for param in params)


def _save_checkpoint(
    path: Path,
    model: ActionConditionedWorldModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> None:
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
        },
        path,
    )


@hydra.main(version_base=None, config_path="../../configs/puzzle", config_name="grid0_sudoku_jepa_5m_oracle_smoke")
def main(cfg: DictConfig) -> None:
    config = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(config, dict):
        raise TypeError("Hydra config must resolve to a mapping.")
    print(json.dumps(run_grid0(config), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
