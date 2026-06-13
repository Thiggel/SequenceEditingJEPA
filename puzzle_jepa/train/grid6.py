from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW

from puzzle_jepa.data import collate_rollouts, sample_curriculum_rollout_transition
from puzzle_jepa.models import CausalTrajectoryJEPA
from puzzle_jepa.train.grid0 import _build_world, _load_examples, _param_count


def run_grid6(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", 0))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_cfg = dict(config["task"])
    train_cfg = dict(config["training"])
    eval_cfg = dict(config.get("eval", {}))
    output_dir = Path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    world = _build_world(task_cfg)
    train_examples = _load_examples(task_cfg, "train")
    eval_examples = _load_examples(task_cfg, "eval")
    model = CausalTrajectoryJEPA(vocab_size=world.vocab_size, **dict(config["model"])).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1.0e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.05)),
    )
    use_amp = bool(train_cfg.get("bf16", True)) and device.type == "cuda"
    max_steps = int(train_cfg["max_steps"])
    batch_size = int(train_cfg["batch_size"])
    rollout_steps = int(train_cfg.get("rollout_steps", 32))
    oracle_probability = float(train_cfg.get("rollout_oracle_probability", 0.5))
    goal_energy_weight = float(train_cfg.get("goal_energy_weight", 1.0))
    eval_every = int(train_cfg.get("eval_every_steps", max_steps))
    save_every = int(train_cfg.get("save_every_steps", eval_every))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    metrics_path = output_dir / "metrics.jsonl"
    latest_metrics: dict[str, Any] = {}
    with (output_dir / "config.json").open("w") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)

    for step in range(1, max_steps + 1):
        model.train()
        batch = _sample_grid6_rollout_batch(
            world,
            train_examples,
            rng,
            batch_size,
            rollout_steps,
            device,
            oracle_probability,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            output = model.rollout_loss(
                batch.states,
                batch.actions,
                batch.target_states,
                batch.goals,
                clue_masks=batch.clue_masks,
                goal_energy_weight=goal_energy_weight,
            )
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        model.update_target_encoder()

        if step == 1 or step % eval_every == 0 or step == max_steps:
            latest_metrics = _eval_grid6(model, world, eval_examples, rng, train_cfg, eval_cfg, device)
            latest_metrics.update(
                {
                    "step": step,
                    "train_loss": float(output.loss.detach().cpu()),
                    "train_prediction_loss": float(output.prediction_loss.cpu()),
                    "train_sigreg_loss": float(output.sigreg_loss.cpu()),
                    "train_stabilizer_loss": float(output.sigreg_loss.cpu()),
                    "train_goal_energy_loss": float(output.goal_energy_loss.cpu()),
                    "rollout_steps": rollout_steps,
                    "rollout_oracle_probability": oracle_probability,
                    "goal_energy_weight": goal_energy_weight,
                    "horizons": list(model.horizons),
                    "d_model": int(config["model"]["d_model"]),
                    "encoder_layers": int(config["model"]["encoder_layers"]),
                    "predictor_layers": int(config["model"]["predictor_layers"]),
                    "action_chunk_layers": int(config["model"]["action_chunk_layers"]),
                    "action_dim": int(config["model"]["action_dim"]),
                    "stabilizer_type": str(config["model"].get("stabilizer_type", "sigreg")),
                    "target_encoder_momentum": float(config["model"].get("target_encoder_momentum", 0.99)),
                    "param_count": _param_count(model),
                    "trainable_param_count": _param_count(model, trainable_only=True),
                }
            )
            for horizon, loss in output.horizon_losses.items():
                latest_metrics[f"train_h{horizon}_loss"] = float(loss.cpu())
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(latest_metrics, sort_keys=True) + "\n")
            print(json.dumps(latest_metrics, sort_keys=True), flush=True)

        if step % save_every == 0 or step == max_steps:
            _save_grid6_checkpoint(output_dir / f"checkpoint-{step}.pt", model, optimizer, step, latest_metrics, config)
            _save_grid6_checkpoint(output_dir / "checkpoint.pt", model, optimizer, step, latest_metrics, config)

    (output_dir / "metrics.json").write_text(json.dumps(latest_metrics, indent=2, sort_keys=True))
    return latest_metrics


def _sample_grid6_rollout_batch(world, examples, rng, batch_size, steps, device, oracle_probability):
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


@torch.no_grad()
def _eval_grid6(model, world, examples, rng, train_cfg, eval_cfg, device) -> dict[str, Any]:
    batch_size = int(eval_cfg.get("batch_size", min(64, int(train_cfg["batch_size"]))))
    rollout_steps = int(train_cfg.get("rollout_steps", 32))
    oracle_probability = float(eval_cfg.get("rollout_oracle_probability", 1.0))
    batch = _sample_grid6_rollout_batch(
        world,
        examples,
        rng,
        batch_size,
        rollout_steps,
        device,
        oracle_probability,
    )
    output = model.rollout_loss(
        batch.states,
        batch.actions,
        batch.target_states,
        batch.goals,
        clue_masks=batch.clue_masks,
        goal_energy_weight=float(train_cfg.get("goal_energy_weight", 1.0)),
    )
    metrics = {
        "eval_loss": float(output.loss.detach().cpu()),
        "eval_prediction_loss": float(output.prediction_loss.cpu()),
        "eval_sigreg_loss": float(output.sigreg_loss.cpu()),
        "eval_stabilizer_loss": float(output.sigreg_loss.cpu()),
        "eval_goal_energy_loss": float(output.goal_energy_loss.cpu()),
    }
    for horizon, loss in output.horizon_losses.items():
        metrics[f"eval_h{horizon}_loss"] = float(loss.cpu())
    return metrics


def _save_grid6_checkpoint(
    path: Path,
    model: CausalTrajectoryJEPA,
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


@hydra.main(version_base=None, config_path="../../configs/puzzle", config_name="grid6_sudoku_trajectory")
def main(cfg: DictConfig) -> None:
    config = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(config, dict):
        raise TypeError("Hydra config must resolve to a mapping.")
    print(json.dumps(run_grid6(config), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
