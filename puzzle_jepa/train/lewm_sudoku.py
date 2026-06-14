from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW

from puzzle_jepa.data.hf_puzzles import HFPuzzleColumns, iter_hf_examples
from puzzle_jepa.data.lewm_sudoku import collate_sudoku_trajectories, sample_sudoku_trajectory
from puzzle_jepa.data.worlds import PuzzleExample, SudokuWorld
from puzzle_jepa.eval.lewm_diagnostics import DEFAULT_PROJECTION_HORIZONS, run_lewm_diagnostic_bundle
from puzzle_jepa.eval.lewm_planner_matrix import run_planner_matrix
from puzzle_jepa.models.lewm import LeWMSudokuModel


def run_lewm_sudoku(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", 0))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))

    task_cfg = dict(config["task"])
    train_cfg = dict(config["training"])
    eval_cfg = dict(config.get("eval", {}))
    train_examples = _load_examples(task_cfg, split_key="train_split")
    eval_examples = _load_examples(task_cfg, split_key="eval_split")
    max_frames = max(_max_trajectory_frames(train_examples), _max_trajectory_frames(eval_examples))
    model_max_history = int(config["model"].get("max_history", 82))

    model = LeWMSudokuModel(**dict(config["model"])).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 5.0e-5)),
        weight_decay=float(train_cfg.get("weight_decay", 1.0e-3)),
    )
    max_steps = int(train_cfg["max_steps"])
    batch_size = int(train_cfg.get("batch_size", 128))
    num_frames_raw = train_cfg.get("num_frames")
    num_frames = None if num_frames_raw is None else int(num_frames_raw)
    required_history = max_frames if num_frames is None else num_frames
    if required_history > model_max_history:
        raise ValueError(
            f"Training needs {required_history} frames, but model.max_history={model_max_history}."
        )
    oracle_probability = float(train_cfg.get("oracle_probability", 0.5))
    eval_every = int(train_cfg.get("eval_every_steps", max_steps))
    save_every = int(train_cfg.get("save_every_steps", eval_every))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    use_amp = bool(train_cfg.get("bf16", True)) and device.type == "cuda"
    metrics_path = output_dir / "metrics.jsonl"
    latest: dict[str, Any] = {}

    for step in range(1, max_steps + 1):
        model.train()
        batch = _sample_batch(
            train_examples,
            rng,
            batch_size=batch_size,
            num_frames=num_frames,
            oracle_probability=oracle_probability,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            output = model(batch.boards, batch.actions, batch.goals, masks=batch.masks)
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if step == 1 or step % eval_every == 0 or step == max_steps:
            latest = _eval_losses(
                model,
                eval_examples,
                rng,
                batch_size=int(eval_cfg.get("batch_size", min(128, batch_size))),
                num_frames=num_frames,
                oracle_probability=float(eval_cfg.get("oracle_probability", 1.0)),
                device=device,
                use_amp=use_amp,
            )
            latest.update(
                {
                    "step": step,
                    "train_loss": float(output.loss.detach().cpu()),
                    "train_prediction_loss": float(output.prediction_loss.cpu()),
                    "train_sigreg_loss": float(output.sigreg_loss.cpu()),
                    "train_value_loss": float(output.value_loss.cpu()),
                    "train_weighted_prediction_loss": float(output.prediction_loss.cpu()),
                    "train_weighted_sigreg_loss": float(
                        output.sigreg_loss.cpu() * float(config["model"].get("sigreg_weight", 0.1))
                    ),
                    "train_weighted_value_loss": float(
                        output.value_loss.cpu() * float(config["model"].get("value_weight", 1.0))
                    ),
                    "learning_rate": float(train_cfg.get("learning_rate", 5.0e-5)),
                    "batch_size": batch_size,
                    "num_frames": num_frames,
                    "max_loaded_trajectory_frames": max_frames,
                    "oracle_probability": oracle_probability,
                    "param_count": _param_count(model),
                    "trainable_param_count": _param_count(model, trainable_only=True),
                    "sigreg_weight": float(config["model"].get("sigreg_weight", 0.1)),
                    "sigreg_projections": int(config["model"].get("sigreg_projections", 1024)),
                    "stop_gradient_target": bool(config["model"].get("stop_gradient_target", False)),
                }
            )
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(latest, sort_keys=True) + "\n")
            print(json.dumps(latest, sort_keys=True), flush=True)

        if step % save_every == 0 or step == max_steps:
            _save_checkpoint(output_dir / f"checkpoint-{step}.pt", model, optimizer, step, latest, config)
            _save_checkpoint(output_dir / "checkpoint.pt", model, optimizer, step, latest, config)

    diagnostics = _run_diagnostics(model, eval_examples, eval_cfg, output_dir, device=device, seed=seed)
    latest.update(diagnostics)
    (output_dir / "metrics.json").write_text(json.dumps(latest, indent=2, sort_keys=True))
    return latest


def _load_examples(task_cfg: dict[str, Any], *, split_key: str) -> list[PuzzleExample]:
    world = SudokuWorld()
    columns = HFPuzzleColumns(
        puzzle=str(task_cfg.get("puzzle_column", "question")),
        solution=str(task_cfg.get("solution_column", "answer")),
    )
    limit_key = "train_limit" if split_key == "train_split" else "eval_limit"
    limit = task_cfg.get(limit_key)
    return list(
        iter_hf_examples(
            str(task_cfg["repo_id"]),
            str(task_cfg[split_key]),
            world,
            columns,
            limit=None if limit is None else int(limit),
        )
    )


def _max_trajectory_frames(examples: list[PuzzleExample]) -> int:
    if not examples:
        raise ValueError("At least one Sudoku example is required.")
    return max(int(np.count_nonzero(example.state == 0)) + 1 for example in examples)


def _sample_batch(
    examples: list[PuzzleExample],
    rng: np.random.Generator,
    *,
    batch_size: int,
    num_frames: int | None,
    oracle_probability: float,
    device: torch.device,
):
    trajectories = [
        sample_sudoku_trajectory(
            examples[int(rng.integers(0, len(examples)))],
            rng,
            num_frames=num_frames,
            oracle_probability=oracle_probability,
            allow_conflicts=True,
        )
        for _ in range(batch_size)
    ]
    return collate_sudoku_trajectories(trajectories, device=device)


@torch.no_grad()
def _eval_losses(
    model: LeWMSudokuModel,
    examples: list[PuzzleExample],
    rng: np.random.Generator,
    *,
    batch_size: int,
    num_frames: int | None,
    oracle_probability: float,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float]:
    model.eval()
    batch = _sample_batch(
        examples,
        rng,
        batch_size=batch_size,
        num_frames=num_frames,
        oracle_probability=oracle_probability,
        device=device,
    )
    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
        output = model(batch.boards, batch.actions, batch.goals, masks=batch.masks)
    metrics = {
        "eval_loss": float(output.loss.detach().cpu()),
        "eval_prediction_loss": float(output.prediction_loss.cpu()),
        "eval_sigreg_loss": float(output.sigreg_loss.cpu()),
        "eval_value_loss": float(output.value_loss.cpu()),
    }
    metrics.update(_batch_diagnostic_metrics(output, batch.masks, batch.oracle_mask))
    return metrics


@torch.no_grad()
def _run_diagnostics(
    model: LeWMSudokuModel,
    examples: list[PuzzleExample],
    eval_cfg: dict[str, Any],
    output_dir: Path,
    *,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    model.eval()
    diagnostics = run_lewm_diagnostic_bundle(
        model,
        examples,
        output_dir,
        device=device,
        seed=seed + 500,
        latent_examples=int(eval_cfg.get("latent_examples", 128)),
        trajectory_examples=int(eval_cfg.get("trajectory_examples", 32)),
        rank_examples=int(eval_cfg.get("rank_examples", 16)),
        panel_examples=int(eval_cfg.get("panel_examples", 3)),
        panel_steps=int(eval_cfg.get("panel_steps", 5)),
        panel_actions=int(eval_cfg.get("panel_actions", 6)),
        projection_horizons=_parse_int_tuple(eval_cfg.get("projection_horizons", DEFAULT_PROJECTION_HORIZONS)),
        write_plots=bool(eval_cfg.get("write_diagnostic_plots", True)),
    )
    diagnostics_path = output_dir / "diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True))
    if bool(eval_cfg.get("run_planner_matrix", True)):
        run_planner_matrix(
            model,
            examples,
            output_path=output_dir / "planner_matrix.jsonl",
            device=device,
            seed=seed + 1000,
            max_examples=int(eval_cfg.get("planner_examples", 8)),
            max_steps=int(eval_cfg.get("max_plan_steps", 81)),
            fast=bool(eval_cfg.get("fast_planner_matrix", False)),
        )
    return diagnostics


def _batch_diagnostic_metrics(output, masks: torch.Tensor, oracle_mask: torch.Tensor) -> dict[str, float]:
    transition_mask = masks[:, :-1] & masks[:, 1:]
    transition_error = (output.predicted_embeddings[:, :-1] - output.embeddings[:, 1:]).square().mean(dim=-1)
    value_error = (output.predicted_goal_distances.detach() - output.goal_distances).square()
    value_abs_error = (output.predicted_goal_distances.detach() - output.goal_distances).abs()
    metrics = {
        "eval_transition_count": float(transition_mask.sum().cpu()),
        "eval_frame_count": float(masks.sum().cpu()),
        "eval_oracle_fraction": float(oracle_mask.float().mean().cpu()),
        "eval_prediction_mse_early": _masked_window_mean(transition_error, transition_mask, 0.0, 1.0 / 3.0),
        "eval_prediction_mse_middle": _masked_window_mean(transition_error, transition_mask, 1.0 / 3.0, 2.0 / 3.0),
        "eval_prediction_mse_late": _masked_window_mean(transition_error, transition_mask, 2.0 / 3.0, 1.0),
        "eval_value_mae": _masked_mean(value_abs_error, masks),
        "eval_value_rmse": float(np.sqrt(_masked_mean(value_error, masks))),
        "eval_goal_distance_mean": _masked_mean(output.goal_distances, masks),
        "eval_goal_distance_std": _masked_std(output.goal_distances, masks),
        "eval_predicted_goal_distance_mean": _masked_mean(output.predicted_goal_distances.detach(), masks),
        "eval_predicted_goal_distance_std": _masked_std(output.predicted_goal_distances.detach(), masks),
        "eval_value_corr": _masked_corr(output.goal_distances, output.predicted_goal_distances.detach(), masks),
    }
    for label, group_mask in (("oracle", oracle_mask), ("random", ~oracle_mask)):
        if bool(group_mask.any()):
            frame_mask = masks & group_mask[:, None]
            trans_mask = transition_mask & group_mask[:, None]
            metrics[f"eval_{label}_prediction_mse"] = _masked_mean(transition_error, trans_mask)
            metrics[f"eval_{label}_value_mse"] = _masked_mean(value_error, frame_mask)
            metrics[f"eval_{label}_goal_distance_mean"] = _masked_mean(output.goal_distances, frame_mask)
    return metrics


def _masked_window_mean(values: torch.Tensor, mask: torch.Tensor, start_frac: float, end_frac: float) -> float:
    if values.shape[1] == 0:
        return 0.0
    valid_counts = mask.sum(dim=1).clamp_min(1)
    step_indices = torch.arange(values.shape[1], device=values.device).unsqueeze(0)
    start = torch.floor(valid_counts.float() * start_frac).long().unsqueeze(1)
    end = torch.ceil(valid_counts.float() * end_frac).long().clamp_min(1).unsqueeze(1)
    window = (step_indices >= start) & (step_indices < end) & mask
    return _masked_mean(values, window)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    mask_f = mask.to(dtype=values.dtype)
    denom = mask_f.sum().clamp_min(1.0)
    return float(((values * mask_f).sum() / denom).detach().cpu())


def _masked_std(values: torch.Tensor, mask: torch.Tensor) -> float:
    mask_f = mask.to(dtype=values.dtype)
    denom = mask_f.sum().clamp_min(1.0)
    mean = (values * mask_f).sum() / denom
    var = ((values - mean).square() * mask_f).sum() / denom
    return float(torch.sqrt(var.clamp_min(0.0)).detach().cpu())


def _masked_corr(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> float:
    valid_a = a[mask].float()
    valid_b = b[mask].float()
    if valid_a.numel() < 2:
        return 0.0
    a_std = valid_a.std(unbiased=False)
    b_std = valid_b.std(unbiased=False)
    if float(a_std.cpu()) == 0.0 or float(b_std.cpu()) == 0.0:
        return 0.0
    corr = torch.mean((valid_a - valid_a.mean()) * (valid_b - valid_b.mean())) / (a_std * b_std)
    return float(corr.detach().cpu())


def _parse_int_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, str):
        return tuple(int(item) for item in value.split(",") if item)
    if isinstance(value, int):
        return (int(value),)
    return tuple(int(item) for item in value)


def _save_checkpoint(
    path: Path,
    model: LeWMSudokuModel,
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


def _param_count(model: torch.nn.Module, *, trainable_only: bool = False) -> int:
    params = model.parameters()
    if trainable_only:
        return sum(param.numel() for param in params if param.requires_grad)
    return sum(param.numel() for param in params)


@hydra.main(version_base=None, config_path="../../configs/puzzle", config_name="lewm_sudoku")
def main(cfg: DictConfig) -> None:
    config = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(config, dict):
        raise TypeError("Hydra config must resolve to a mapping.")
    print(json.dumps(run_lewm_sudoku(config), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
