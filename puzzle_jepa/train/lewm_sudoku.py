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
from puzzle_jepa.eval.lewm_planner_matrix import (
    latent_statistics,
    local_action_rank_diagnostics,
    oracle_distance_diagnostics,
    run_planner_matrix,
)
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

    model = LeWMSudokuModel(**dict(config["model"])).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 5.0e-5)),
        weight_decay=float(train_cfg.get("weight_decay", 1.0e-3)),
    )
    max_steps = int(train_cfg["max_steps"])
    batch_size = int(train_cfg.get("batch_size", 128))
    num_frames = int(train_cfg.get("num_frames", 8))
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
            output = model(batch.boards, batch.actions, batch.goals)
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
                    "learning_rate": float(train_cfg.get("learning_rate", 5.0e-5)),
                    "batch_size": batch_size,
                    "num_frames": num_frames,
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


def _sample_batch(
    examples: list[PuzzleExample],
    rng: np.random.Generator,
    *,
    batch_size: int,
    num_frames: int,
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
    num_frames: int,
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
        output = model(batch.boards, batch.actions, batch.goals)
    return {
        "eval_loss": float(output.loss.detach().cpu()),
        "eval_prediction_loss": float(output.prediction_loss.cpu()),
        "eval_sigreg_loss": float(output.sigreg_loss.cpu()),
        "eval_value_loss": float(output.value_loss.cpu()),
    }


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
    diagnostics: dict[str, Any] = {}
    diagnostics.update(
        latent_statistics(
            model,
            examples,
            device=device,
            max_examples=int(eval_cfg.get("latent_examples", 128)),
        )
    )
    diagnostics.update(
        oracle_distance_diagnostics(
            model,
            examples,
            device=device,
            max_examples=int(eval_cfg.get("trajectory_examples", 32)),
        )
    )
    diagnostics.update(
        local_action_rank_diagnostics(
            model,
            examples,
            device=device,
            max_examples=int(eval_cfg.get("rank_examples", 32)),
        )
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
