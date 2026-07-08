from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW

from puzzle_jepa.object_dynamics.batching import sample_object_dynamics_batch
from puzzle_jepa.object_dynamics.generator import ObjectDynamicsGenerator, ObjectDynamicsSpec
from puzzle_jepa.object_dynamics.model import ObjectDynamicsJEPA
from puzzle_jepa.object_dynamics.probes import run_object_dynamics_probes


def run_object_dynamics_training(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", 0))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = _resolve_device(str(config.get("device", "auto")))
    output_dir = Path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = _without_name(dict(config["data"]))
    generator = ObjectDynamicsGenerator(ObjectDynamicsSpec(**data_cfg))
    model_cfg = _without_name(dict(config["model"]))
    objective_cfg = _without_name(dict(config["objective"]))
    model = ObjectDynamicsJEPA(
        grid_size=generator.spec.grid_size,
        num_colors=generator.spec.num_colors,
        **model_cfg,
        **objective_cfg,
    ).to(device)
    config = {**config, "param_count": _param_count(model)}
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))

    training_cfg = dict(config["training"])
    eval_cfg = dict(config["eval"])
    optimizer = AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 3.0e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1.0e-4)),
    )
    max_steps = int(training_cfg.get("max_steps", 1000))
    batch_size = int(training_cfg.get("batch_size", 64))
    eval_every = int(training_cfg.get("eval_every_steps", 250))
    save_every = int(training_cfg.get("save_every_steps", max_steps))
    grad_clip = float(training_cfg.get("grad_clip", 1.0))
    use_amp = bool(training_cfg.get("bf16", True)) and device.type == "cuda"
    horizon = max(int(model_cfg.get("rollout_horizon", 1)), int(model_cfg.get("hierarchy_horizon", 0)), 1)
    metrics_path = output_dir / "metrics.jsonl"
    latest: dict[str, Any] = {}

    for step in range(1, max_steps + 1):
        model.train()
        batch = sample_object_dynamics_batch(generator, rng, batch_size=batch_size, horizon=horizon, device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            output = model(batch)
        if not torch.isfinite(output.loss.detach()):
            raise FloatingPointError(f"Non-finite object-dynamics loss at step {step}: {float(output.loss.detach().cpu())}")
        output.loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        model.update_target_encoder()

        if step == 1 or step % eval_every == 0 or step == max_steps:
            latest = {
                "step": step,
                "data": config["data"]["name"],
                "model": config["model"]["name"],
                "objective": config["objective"]["name"],
                "train_loss": float(output.loss.detach().cpu()),
                "train_rollout_loss": float(output.rollout_loss.detach().cpu()),
                "train_hierarchy_loss": float(output.hierarchy_loss.detach().cpu()),
                "train_ldad_loss": float(output.ldad_loss.detach().cpu()),
                "train_regularizer_loss": float(output.regularizer_loss.detach().cpu()),
                "grad_norm_pre_clip": float(grad_norm.detach().cpu()),
                "batch_semantic_rate": float(batch.valid_state.mean().detach().cpu()),
                "batch_mean_object_count": float(batch.object_count.float().mean().detach().cpu()),
                "device": device.type,
                "param_count": _param_count(model),
                **run_object_dynamics_probes(
                    model,
                    generator,
                    rng,
                    train_samples=int(eval_cfg.get("probe_train_samples", 256)),
                    eval_samples=int(eval_cfg.get("probe_eval_samples", 128)),
                    batch_size=int(eval_cfg.get("probe_batch_size", min(batch_size, 64))),
                    horizon=horizon,
                    device=device,
                    steps=int(eval_cfg.get("probe_steps", 150)),
                    learning_rate=float(eval_cfg.get("probe_learning_rate", 1.0e-2)),
                ),
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(latest, sort_keys=True) + "\n")
            print(json.dumps(latest, sort_keys=True), flush=True)

        if step % save_every == 0 or step == max_steps:
            _save_checkpoint(output_dir / "checkpoint.pt", model, optimizer, step, latest, config)

    (output_dir / "metrics.json").write_text(json.dumps(latest, indent=2, sort_keys=True))
    return latest


def _without_name(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != "name"}


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _param_count(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "metrics": metrics,
            "config": config,
        },
        path,
    )


@hydra.main(config_path="../../configs/object_dynamics", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    run_object_dynamics_training(OmegaConf.to_container(cfg, resolve=True))


if __name__ == "__main__":
    main()
