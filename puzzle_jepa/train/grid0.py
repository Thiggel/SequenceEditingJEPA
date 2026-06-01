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
    sample_curriculum_transition,
    sample_random_mutable_transition,
    sample_oracle_rollout_transition,
    sample_oracle_partial_transition,
)
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
    goal_energy_weight = float(train_cfg.get("goal_energy_weight", 0.0))
    hierarchy_weight = float(train_cfg.get("hierarchy_weight", 0.0))
    hierarchy_batch_size = int(train_cfg.get("hierarchy_batch_size", train_cfg.get("rollout_batch_size", batch_size)))
    hierarchy_rollout_steps = int(
        train_cfg.get("hierarchy_rollout_steps", _max_hierarchy_horizon(model))
    )
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
            )
            loss = output.loss
            rollout_loss = None
            hierarchy_loss = None
            if rollout_steps > 1 and rollout_weight > 0.0:
                rollout_batch = _sample_rollout_batch(world, train_examples, rng, rollout_batch_size, rollout_steps, device)
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
                )
                hierarchy_output = model.hierarchy_loss(
                    hierarchy_batch.states,
                    hierarchy_batch.actions,
                    hierarchy_batch.target_states,
                )
                hierarchy_loss = hierarchy_output.loss
                loss = loss + hierarchy_weight * hierarchy_loss
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
                    "rollout_steps": rollout_steps,
                    "rollout_weight": rollout_weight,
                    "rollout_batch_size": rollout_batch_size if rollout_steps > 1 else 0,
                    "goal_energy_weight": goal_energy_weight,
                    "hierarchy_weight": hierarchy_weight,
                    "hierarchy_levels": int(model.hierarchy_levels),
                    "hierarchy_span": int(model.hierarchy_span),
                    "hierarchy_rollout_steps": hierarchy_rollout_steps if hierarchy_weight > 0.0 else 0,
                    "hierarchy_batch_size": hierarchy_batch_size if hierarchy_weight > 0.0 else 0,
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
    transitions = [
        sample_curriculum_transition(
            world,
            examples[int(rng.integers(0, len(examples)))],
            rng,
            oracle_probability=oracle_probability,
        )
        for _ in range(batch_size)
    ]
    return collate_transitions(transitions, device=device)


def _sample_rollout_batch(
    world: PuzzleWorld,
    examples: list[PuzzleExample],
    rng: np.random.Generator,
    batch_size: int,
    steps: int,
    device: torch.device,
):
    rollouts = [
        sample_oracle_rollout_transition(
            world,
            examples[int(rng.integers(0, len(examples)))],
            rng,
            steps=steps,
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


def _initial_states_from_batch(world: PuzzleWorld, batch) -> torch.Tensor:
    if isinstance(world, SudokuWorld) and batch.clue_masks is not None:
        return batch.states * batch.clue_masks.to(dtype=batch.states.dtype)
    return batch.states


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
