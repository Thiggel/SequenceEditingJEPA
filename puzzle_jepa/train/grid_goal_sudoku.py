from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW

from puzzle_jepa.data.grid_goal_sudoku import (
    collate_grid_goal_sudoku_trajectories,
    corrupt_terminal,
    sample_random_grid_goal_sudoku_trajectory,
    sample_grid_goal_sudoku_trajectory,
)
from puzzle_jepa.data.hf_puzzles import HFPuzzleColumns, iter_hf_examples
from puzzle_jepa.data.worlds import PuzzleExample, SudokuWorld
from puzzle_jepa.eval.grid_goal_diagnostics import run_grid_goal_diagnostics
from puzzle_jepa.models.grid_goal_jepa import GridTokenGoalJEPA


ABLATIONS: dict[str, dict[str, Any]] = {
    "M0_full": {},
    "R1_no_context_masks": {"use_context_masks": False},
    "R2_mean_pooled_distance": {"distance_mode": "mean_pooled"},
    "R3_k1_only": {"multi_step_horizons": [1]},
    "R3_k4": {"multi_step_horizons": [1, 4]},
    "R3_k8": {"multi_step_horizons": [1, 4, 8]},
    "R3_k16": {"multi_step_horizons": [1, 4, 8, 16]},
    "R4_no_goal_nce": {"goal_nce_weight": 0.0},
    "R5_no_progress_rank": {"progress_rank_weight": 0.0},
    "R6_no_action_rank": {"action_rank_weight": 0.0},
    "R7_no_terminal_corrupt": {"terminal_corrupt_weight": 0.0},
    "R8_no_sigreg": {"sigreg_weight": 0.0},
    "R9_no_temporal_straightening": {"temporal_straightening_weight": 0.0},
}


def run_grid_goal_sudoku(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", 0))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    ablation = str(config.get("ablation", "M0_full"))
    config = _apply_ablation(config, ablation)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))

    train_examples = _load_examples(dict(config["task"]), split_key="train_split")
    eval_examples = _load_examples(dict(config["task"]), split_key="eval_split")
    model = GridTokenGoalJEPA(**dict(config["model"])).to(device)
    peak_lr = float(config["training"]["learning_rate"])
    optimizer = AdamW(
        model.parameters(),
        lr=peak_lr,
        weight_decay=float(config["training"].get("weight_decay", 1.0e-3)),
    )
    max_steps = int(config["training"]["max_steps"])
    batch_size = int(config["training"].get("batch_size", 64))
    gradient_accumulation_steps = int(config["training"].get("gradient_accumulation_steps", 1))
    if gradient_accumulation_steps <= 0:
        raise ValueError("training.gradient_accumulation_steps must be positive.")
    warmup_steps = int(config["training"].get("warmup_steps", 0))
    min_lr_ratio = float(config["training"].get("min_lr_ratio", 0.1))
    eval_every = int(config["training"].get("eval_every_steps", 1000))
    save_every = int(config["training"].get("save_every_steps", 5000))
    oracle_probability = float(config["training"].get("oracle_probability", 0.5))
    grad_clip = float(config["training"].get("grad_clip", 1.0))
    use_amp = bool(config["training"].get("bf16", True)) and device.type == "cuda"
    metrics_path = output_dir / "metrics.jsonl"
    latest: dict[str, Any] = {}

    for step in range(1, max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = []
        for _micro_step in range(gradient_accumulation_steps):
            batch = _sample_batch(
                train_examples,
                rng,
                batch_size=batch_size,
                oracle_probability=oracle_probability,
                allow_overwrite=bool(config["training"].get("allow_overwrite", False)),
                editable_noise_probability=float(config["training"].get("editable_noise_probability", 0.0)),
                random_max_steps=config["training"].get("random_max_steps"),
                counterfactual_branches=int(config["training"].get("counterfactual_branches", 0)),
                counterfactual_depth=int(config["training"].get("counterfactual_depth", 1)),
                counterfactual_max_pairs=int(config["training"].get("counterfactual_max_pairs", 0)),
                device=device,
            )
            try:
                rank_sample = _sample_rank_actions(batch.boards, batch.goals, rng, masks=batch.masks, device=device)
            except TypeError as exc:
                if "unexpected keyword argument 'masks'" not in str(exc):
                    raise
                rank_sample = _sample_rank_actions(batch.boards, batch.goals, rng, device=device)
            if len(rank_sample) == 2:
                action_rank_states = batch.boards[:, 0]
                positive_actions, negative_actions = rank_sample
            else:
                action_rank_states, positive_actions, negative_actions = rank_sample
            corrupt_goals = torch.as_tensor(
                np.stack([corrupt_terminal(goal.cpu().numpy(), rng) for goal in batch.goals]),
                dtype=torch.long,
                device=device,
            )
            if not bool(config.get("use_context_masks", True)):
                batch = _zero_context_masks(batch)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                output = model(
                    batch.boards,
                    batch.actions,
                    batch.context,
                    batch.clue_mask,
                    batch.editable_mask,
                    batch.active_mask,
                    batch.goals,
                    masks=batch.masks,
                    oracle_mask=batch.oracle_mask,
                    action_rank_states=action_rank_states,
                    positive_actions=positive_actions,
                    negative_actions=negative_actions,
                    corrupt_goals=corrupt_goals,
                    counterfactual_states=batch.counterfactual_states,
                    counterfactual_actions=batch.counterfactual_actions,
                    counterfactual_next_boards=batch.counterfactual_next_boards,
                    counterfactual_mask=batch.counterfactual_mask,
                    counterfactual_action_sequences=batch.counterfactual_action_sequences,
                    counterfactual_future_boards=batch.counterfactual_future_boards,
                    counterfactual_step_mask=batch.counterfactual_step_mask,
                )
            if not torch.isfinite(output.loss.detach()):
                raise FloatingPointError(f"Non-finite loss at step {step}: {float(output.loss.detach().cpu())}")
            (output.loss / gradient_accumulation_steps).backward()
            outputs.append(output)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        grad_norm_value = float(grad_norm.detach().cpu())
        if not np.isfinite(grad_norm_value):
            raise FloatingPointError(f"Non-finite gradient norm at step {step}: {grad_norm_value}")
        lr = _scheduled_lr(
            step=step,
            max_steps=max_steps,
            peak_lr=peak_lr,
            warmup_steps=warmup_steps,
            min_lr_ratio=min_lr_ratio,
        )
        _set_optimizer_lr(optimizer, lr)
        optimizer.step()
        if hasattr(model, "update_ema_target_encoder"):
            model.update_ema_target_encoder()
        output = outputs[-1]
        if step == 1 or step % eval_every == 0 or step == max_steps:
            latest = {
                "step": step,
                "ablation": ablation,
                "train_loss": float(output.loss.detach().cpu()),
                "train_dynamics_loss": _output_scalar(output, "dynamics_loss"),
                "train_dense_future_loss": _output_scalar(output, "dense_future_loss"),
                "train_counterfactual_dynamics_loss": _output_scalar(output, "counterfactual_dynamics_loss"),
                "train_hierarchy_loss": _output_scalar(output, "hierarchy_loss"),
                "train_sigreg_loss": _output_scalar(output, "sigreg_loss"),
                "train_goal_mse_loss": _output_scalar(output, "goal_mse_loss"),
                "train_goal_nce_loss": _output_scalar(output, "goal_nce_loss"),
                "train_goal_distance_field_loss": _output_scalar(output, "goal_distance_field_loss"),
                "train_waypoint_loss": _output_scalar(output, "waypoint_loss"),
                "train_waypoint_final_loss": _output_scalar(output, "waypoint_final_loss"),
                "train_progress_rank_loss": _output_scalar(output, "progress_rank_loss"),
                "train_action_rank_loss": _output_scalar(output, "action_rank_loss"),
                "train_policy_prior_loss": _output_scalar(output, "policy_prior_loss"),
                "train_delta_action_loss": _output_scalar(output, "delta_action_loss"),
                "train_metric_geometry_loss": _output_scalar(output, "metric_geometry_loss"),
                "train_metric_goal_mse_loss": _output_scalar(output, "metric_goal_mse_loss"),
                "train_bad_state_loss": _output_scalar(output, "bad_state_loss"),
                "train_bad_margin_loss": _output_scalar(output, "bad_margin_loss"),
                "train_temporal_straightening_loss": _output_scalar(output, "temporal_straightening_loss"),
                "train_terminal_corrupt_loss": _output_scalar(output, "terminal_corrupt_loss"),
                "learning_rate": lr,
                "peak_learning_rate": peak_lr,
                "warmup_steps": warmup_steps,
                "min_lr_ratio": min_lr_ratio,
                "grad_clip": grad_clip,
                "grad_norm_pre_clip": grad_norm_value,
                "micro_batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "effective_batch_size": batch_size * gradient_accumulation_steps,
                "param_count": _param_count(model),
            }
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(latest, sort_keys=True) + "\n")
            print(json.dumps(latest, sort_keys=True), flush=True)
        if step % save_every == 0 or step == max_steps:
            _save_checkpoint(output_dir / f"checkpoint-{step}.pt", model, optimizer, step, latest, config)
            _save_checkpoint(output_dir / "checkpoint.pt", model, optimizer, step, latest, config)

    diagnostics = run_grid_goal_diagnostics(
        model,
        eval_examples[: int(config["eval"].get("diagnostic_examples", 32))],
        output_dir,
        device=device,
        seed=seed + 500,
        panel_examples=int(config["eval"].get("panel_examples", 3)),
        panel_steps=int(config["eval"].get("panel_steps", 5)),
        panel_actions=int(config["eval"].get("panel_actions", 6)),
    )
    latest.update(diagnostics)
    (output_dir / "metrics.json").write_text(json.dumps(latest, indent=2, sort_keys=True))
    return latest


def _apply_ablation(config: dict[str, Any], ablation: str) -> dict[str, Any]:
    if ablation not in ABLATIONS:
        raise ValueError(f"Unknown ablation {ablation!r}; choices are {sorted(ABLATIONS)}.")
    config = json.loads(json.dumps(config))
    config["ablation"] = ablation
    for key, value in ABLATIONS[ablation].items():
        if key in {"use_context_masks", "distance_mode"}:
            if key == "distance_mode":
                config["model"][key] = value
            else:
                config[key] = value
        else:
            config["model"][key] = value
    return config


def _load_examples(task_cfg: dict[str, Any], *, split_key: str) -> list[PuzzleExample]:
    world = SudokuWorld()
    columns = HFPuzzleColumns(
        puzzle=str(task_cfg.get("puzzle_column", "question")),
        solution=str(task_cfg.get("solution_column", "answer")),
    )
    limit_key = "train_limit" if split_key == "train_split" else "eval_limit"
    limit = task_cfg.get(limit_key)
    return list(iter_hf_examples(str(task_cfg["repo_id"]), str(task_cfg[split_key]), world, columns, None if limit is None else int(limit)))


def _sample_batch(
    examples: list[PuzzleExample],
    rng: np.random.Generator,
    *,
    batch_size: int,
    oracle_probability: float,
    allow_overwrite: bool,
    editable_noise_probability: float,
    random_max_steps: int | None,
    counterfactual_branches: int,
    counterfactual_depth: int,
    counterfactual_max_pairs: int,
    device: torch.device,
):
    trajectories = []
    for _ in range(batch_size):
        example = examples[int(rng.integers(0, len(examples)))]
        if rng.random() < oracle_probability:
            trajectories.append(
                sample_grid_goal_sudoku_trajectory(
                    example,
                    rng,
                    allow_conflicts=True,
                    allow_overwrite=allow_overwrite,
                    editable_noise_probability=editable_noise_probability,
                    counterfactual_branches=counterfactual_branches,
                    counterfactual_depth=counterfactual_depth,
                    counterfactual_max_pairs=counterfactual_max_pairs,
                )
            )
        else:
            trajectories.append(
                sample_random_grid_goal_sudoku_trajectory(
                    example,
                    rng,
                    allow_conflicts=True,
                    allow_overwrite=allow_overwrite,
                    max_steps=random_max_steps,
                    counterfactual_branches=counterfactual_branches,
                    counterfactual_depth=counterfactual_depth,
                    counterfactual_max_pairs=counterfactual_max_pairs,
                )
            )
    return collate_grid_goal_sudoku_trajectories(trajectories, device=device)


def _sample_rank_actions(
    boards: torch.Tensor,
    goals: torch.Tensor,
    rng: np.random.Generator,
    *,
    masks: torch.Tensor | None = None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if boards.ndim == 3:
        rank_boards = boards
    elif boards.ndim == 4:
        selected = []
        masks_cpu = None if masks is None else masks.cpu().numpy()
        for batch_index, sequence in enumerate(boards.cpu().numpy()):
            valid_steps = np.arange(sequence.shape[0])
            if masks_cpu is not None:
                valid_steps = valid_steps[masks_cpu[batch_index]]
            candidates = [int(step) for step in valid_steps if np.any(sequence[int(step)] == 0)]
            if not candidates:
                candidates = [int(valid_steps[0])] if len(valid_steps) else [0]
            selected.append(sequence[candidates[int(rng.integers(0, len(candidates)))]])
        rank_boards = torch.as_tensor(np.stack(selected), dtype=boards.dtype, device=device)
    else:
        raise ValueError(f"Expected boards [batch, rows, cols] or [batch, frames, rows, cols], got {tuple(boards.shape)}.")
    positives = []
    negatives = []
    for board, goal in zip(rank_boards.cpu().numpy(), goals.cpu().numpy(), strict=True):
        empty = np.argwhere(board == 0)
        if len(empty) == 0:
            positives.append([0, 0, 0])
            negatives.append([0, 0, 0])
            continue
        row, col = (int(x) for x in empty[int(rng.integers(0, len(empty)))])
        correct = int(goal[row, col])
        wrong = int(rng.integers(1, 10))
        if wrong == correct:
            wrong = 1 + (wrong % 9)
        positives.append([row, col, correct])
        negatives.append([row, col, wrong])
    return (
        rank_boards,
        torch.as_tensor(positives, dtype=torch.long, device=device),
        torch.as_tensor(negatives, dtype=torch.long, device=device),
    )


def _zero_context_masks(batch):
    return type(batch)(
        boards=batch.boards,
        actions=batch.actions,
        context=torch.zeros_like(batch.context),
        clue_mask=torch.zeros_like(batch.clue_mask),
        editable_mask=torch.ones_like(batch.editable_mask),
        active_mask=batch.active_mask,
        goals=batch.goals,
        masks=batch.masks,
        oracle_mask=batch.oracle_mask,
        counterfactual_states=batch.counterfactual_states,
        counterfactual_actions=batch.counterfactual_actions,
        counterfactual_next_boards=batch.counterfactual_next_boards,
        counterfactual_mask=batch.counterfactual_mask,
        counterfactual_action_sequences=batch.counterfactual_action_sequences,
        counterfactual_future_boards=batch.counterfactual_future_boards,
        counterfactual_step_mask=batch.counterfactual_step_mask,
    )


def _save_checkpoint(path: Path, model: GridTokenGoalJEPA, optimizer: torch.optim.Optimizer, step: int, metrics: dict[str, Any], config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "metrics": metrics, "config": config}, path)


def _param_count(model: torch.nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def _scheduled_lr(
    *,
    step: int,
    max_steps: int,
    peak_lr: float,
    warmup_steps: int,
    min_lr_ratio: float,
) -> float:
    if warmup_steps > 0 and step <= warmup_steps:
        return peak_lr * step / warmup_steps
    decay_steps = max(1, max_steps - max(0, warmup_steps))
    decay_step = min(decay_steps, max(0, step - max(0, warmup_steps)))
    cosine = 0.5 * (1.0 + np.cos(np.pi * decay_step / decay_steps))
    return peak_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def _output_scalar(output: Any, name: str) -> float:
    value = getattr(output, name, None)
    if value is None:
        return 0.0
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


@hydra.main(version_base=None, config_path="../../configs/puzzle", config_name="grid_goal_sudoku")
def main(cfg: DictConfig) -> None:
    print(json.dumps(run_grid_goal_sudoku(OmegaConf.to_container(cfg, resolve=True)), sort_keys=True))


if __name__ == "__main__":
    main()
