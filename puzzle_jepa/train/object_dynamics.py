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
from puzzle_jepa.object_dynamics.initialization import initialize_low_level_from_checkpoint
from puzzle_jepa.object_dynamics.model import ObjectDynamicsJEPA
from puzzle_jepa.object_dynamics.probes import run_object_dynamics_probes


def run_object_dynamics_training(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", 0))
    train_rng = np.random.default_rng(seed)
    probe_seed = seed + 100_003
    torch.manual_seed(seed)
    device = _resolve_device(str(config.get("device", "auto")))
    output_dir = Path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = _without_name(dict(config["data"]))
    generator = ObjectDynamicsGenerator(ObjectDynamicsSpec(**data_cfg))
    eval_cfg = dict(config["eval"])
    probe_data_cfg = {
        **data_cfg,
        "trajectory_kind": str(eval_cfg.get("probe_trajectory_kind", data_cfg["trajectory_kind"])),
    }
    probe_generator = ObjectDynamicsGenerator(ObjectDynamicsSpec(**probe_data_cfg))
    model_cfg = _without_name(dict(config["model"]))
    objective_cfg = _without_name(dict(config["objective"]))
    training_cfg = dict(config["training"])
    model = ObjectDynamicsJEPA(
        grid_size=generator.spec.grid_size,
        num_colors=generator.spec.num_colors,
        **model_cfg,
        **objective_cfg,
    ).to(device)
    loaded_keys = initialize_low_level_from_checkpoint(
        model,
        training_cfg.get("initial_checkpoint"),
        device=device,
    )
    _set_trainable_components(model, str(training_cfg.get("trainable_components", "all")))
    config = {
        **config,
        "param_count": _param_count(model),
        "initial_checkpoint_loaded_keys": loaded_keys,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))

    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(training_cfg.get("learning_rate", 3.0e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1.0e-4)),
    )
    max_steps = int(training_cfg.get("max_steps", 1000))
    batch_size = int(training_cfg.get("batch_size", 64))
    eval_every = int(training_cfg.get("eval_every_steps", 250))
    save_every = int(training_cfg.get("save_every_steps", max_steps))
    grad_clip = float(training_cfg.get("grad_clip", 1.0))
    use_amp = bool(training_cfg.get("bf16", True)) and device.type == "cuda"
    horizon = model.training_horizon
    metrics_path = output_dir / "metrics.jsonl"
    latest: dict[str, Any] = {}

    def evaluate_probes() -> dict[str, Any]:
        with torch.random.fork_rng(devices=_fork_rng_devices(device)):
            torch.manual_seed(probe_seed)
            return run_object_dynamics_probes(
                model,
                probe_generator,
                np.random.default_rng(probe_seed),
                train_samples=int(eval_cfg.get("probe_train_samples", 256)),
                eval_samples=int(eval_cfg.get("probe_eval_samples", 128)),
                batch_size=int(eval_cfg.get("probe_batch_size", min(batch_size, 64))),
                horizon=horizon,
                device=device,
                steps=int(eval_cfg.get("probe_steps", 150)),
                learning_rate=float(eval_cfg.get("probe_learning_rate", 1.0e-2)),
            )

    if bool(eval_cfg.get("run_initial_probes", True)):
        initial = {
            "step": 0,
            "data": config["data"]["name"],
            "model": config["model"]["name"],
            "objective": config["objective"]["name"],
            "device": device.type,
            "param_count": _param_count(model),
            "probe_seed": probe_seed,
            **evaluate_probes(),
        }
        _append_metrics(metrics_path, initial)
        print(json.dumps(initial, sort_keys=True), flush=True)

    for step in range(1, max_steps + 1):
        model.train()
        batch = sample_object_dynamics_batch(generator, train_rng, batch_size=batch_size, horizon=horizon, device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            output = model(batch)
        if not torch.isfinite(output.loss.detach()):
            raise FloatingPointError(f"Non-finite object-dynamics loss at step {step}: {float(output.loss.detach().cpu())}")
        output.loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        model.update_target_encoder()

        if step % eval_every == 0 or step == max_steps:
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
                "train_reconstruction_loss": float(output.reconstruction_loss.detach().cpu()),
                "grad_norm_pre_clip": float(grad_norm.detach().cpu()),
                "batch_semantic_rate": float((batch.trajectory_category == 0).float().mean().detach().cpu()),
                "batch_counterfactual_rate": float((batch.trajectory_category == 1).float().mean().detach().cpu()),
                "batch_wrong_rate": float((batch.trajectory_category == 2).float().mean().detach().cpu()),
                "batch_valid_state_rate": float(batch.valid_state.mean().detach().cpu()),
                "batch_mean_object_count": float(batch.object_count.float().mean().detach().cpu()),
                "batch_mean_scene_object_count": float(batch.scene_object_count.float().mean().detach().cpu()),
                "device": device.type,
                "param_count": _param_count(model),
                "probe_seed": probe_seed,
            }
            if bool(eval_cfg.get("run_probes_during_training", True)):
                latest.update(evaluate_probes())
            _append_metrics(metrics_path, latest)
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
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def _initialize_from_checkpoint(
    model: ObjectDynamicsJEPA,
    checkpoint: Any,
    *,
    device: torch.device,
) -> list[str]:
    return initialize_low_level_from_checkpoint(model, checkpoint, device=device)


def _fork_rng_devices(device: torch.device) -> list[int]:
    if device.type != "cuda":
        return []
    return [device.index if device.index is not None else torch.cuda.current_device()]


def _set_trainable_components(model: ObjectDynamicsJEPA, selection: str) -> None:
    if selection == "all":
        return
    if selection != "hierarchy_only":
        raise ValueError(f"Unknown trainable_components selection {selection!r}.")
    if not model.hierarchy_planning:
        raise ValueError("hierarchy_only training requires hierarchy_planning=true.")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (model.chunk_encoder, model.hierarchy_predictor):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    model.rollout_weight = 0.0
    model.ldad_weight = 0.0
    model.regularizer_weight = 0.0
    model.reconstruction_weight = 0.0


def _append_metrics(path: Path, metrics: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics, sort_keys=True) + "\n")


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
