from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW

from puzzle_jepa.controlled_objects.batching import build_controlled_dataset
from puzzle_jepa.controlled_objects.evaluation import evaluate_controlled_model
from puzzle_jepa.controlled_objects.generator import (
    ControlledObjectGenerator,
    ControlledObjectSpec,
)
from puzzle_jepa.controlled_objects.model import ControlledObjectJEPA
from puzzle_jepa.moving_objects.reproducibility import configure_reproducibility


def run_controlled_object_training(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    train_rng = np.random.default_rng(seed)
    device = _resolve_device(str(config.get("device", "auto")))
    output_dir = Path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = _without_name(dict(config["data"]))
    model_cfg = _without_name(dict(config["model"]))
    objective_cfg = _without_name(dict(config["objective"]))
    training_cfg = dict(config["training"])
    eval_cfg = dict(config["eval"])
    configure_reproducibility(bool(training_cfg.get("deterministic", True)))

    generator = ControlledObjectGenerator(ControlledObjectSpec(**data_cfg))
    model = ControlledObjectJEPA(
        grid_size=generator.spec.grid_size,
        num_colors=generator.spec.num_colors,
        **model_cfg,
        **objective_cfg,
    ).to(device)
    init_checkpoint = training_cfg.get("init_checkpoint")
    if init_checkpoint:
        _initialize_low_level(model, Path(str(init_checkpoint)))
    train_from_level = int(training_cfg.get("train_from_level", 0))
    model.freeze_below_level(train_from_level)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("No trainable parameters remain after applying the stage freeze.")

    trajectory_length = generator.spec.trajectory_length
    if model.required_horizon > trajectory_length:
        raise ValueError(
            f"Model requires horizon {model.required_horizon}, but data has {trajectory_length}."
        )
    data_seed = int(config.get("data_seed", 8801))
    train_dataset = build_controlled_dataset(
        generator,
        trajectory_count=int(training_cfg.get("train_trajectories", 1024)),
        seed=data_seed,
    )
    eval_dataset = build_controlled_dataset(
        generator,
        trajectory_count=int(eval_cfg.get("trajectories", 256)),
        seed=data_seed + 1,
    )

    param_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_param_count = sum(parameter.numel() for parameter in trainable)
    config = {
        **config,
        "param_count": param_count,
        "trainable_param_count": trainable_param_count,
        "required_horizon": model.required_horizon,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )
    optimizer = AdamW(
        trainable,
        lr=float(training_cfg.get("learning_rate", 3.0e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1.0e-4)),
    )
    max_steps = int(training_cfg.get("max_steps", 20_000))
    batch_size = int(training_cfg.get("batch_size", 64))
    eval_every = int(training_cfg.get("eval_every_steps", max_steps))
    grad_clip = float(training_cfg.get("grad_clip", 1.0))
    warmup_steps = int(training_cfg.get("warmup_steps", 500))
    use_amp = bool(training_cfg.get("bf16", True)) and device.type == "cuda"
    metrics_path = output_dir / "metrics.jsonl"

    def evaluate(step: int, *, planning: bool) -> dict[str, Any]:
        metrics = evaluate_controlled_model(
            model,
            eval_dataset,
            generator,
            seed=seed + 100_003,
            batch_size=int(eval_cfg.get("batch_size", 64)),
            device=device,
            planning_episodes=(int(eval_cfg.get("planning_episodes", 4)) if planning else 0),
            planning_candidates=int(eval_cfg.get("planning_candidates", 16)),
        )
        return {
            "step": step,
            "run_name": config["run_name"],
            "data": config["data"]["name"],
            "model": config["model"]["name"],
            "objective": config["objective"]["name"],
            "seed": seed,
            "device": device.type,
            "param_count": param_count,
            "trainable_param_count": trainable_param_count,
            "hierarchy_depth": model.hierarchy_depth,
            "hierarchy_stride": model.hierarchy_stride,
            "token_dim": model.token_dim,
            "latent_dim": model.latent_dim,
            "ldad_horizon": model.ldad_horizon,
            "ldad_weight": model.ldad_weight,
            "rollout_steps": model.rollout_steps,
            "rollout_all_levels": model.rollout_all_levels,
            "rollout_lambda": model.rollout_lambda,
            "latent_representation": model.latent_representation,
            "target_mode": model.target_mode,
            "stop_gradient_targets": model.stop_gradient_targets,
            "required_horizon": model.required_horizon,
            **metrics,
        }

    initial = evaluate(0, planning=False)
    _append_jsonl(metrics_path, initial)
    print(json.dumps(initial, sort_keys=True), flush=True)
    latest_train: dict[str, Any] = {}
    latest = initial
    for step in range(1, max_steps + 1):
        model.train()
        batch = train_dataset.sample_batch(
            train_rng,
            batch_size=batch_size,
            horizon=model.required_horizon,
            device=device,
        )
        _set_learning_rate(
            optimizer,
            step=step,
            max_steps=max_steps,
            warmup_steps=warmup_steps,
            peak=float(training_cfg.get("learning_rate", 3.0e-4)),
            floor=float(training_cfg.get("min_learning_rate", 3.0e-5)),
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            output = model(batch)
        if not torch.isfinite(output.loss.detach()):
            raise FloatingPointError(f"Non-finite controlled-object loss at step {step}.")
        output.loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        optimizer.step()
        model.update_target_encoder()
        latest_train = {
            "train_loss": float(output.loss.detach().cpu()),
            "train_prediction_loss": float(output.prediction_loss.detach().cpu()),
            "train_vicreg_loss": float(output.vicreg_loss.detach().cpu()),
            "train_ldad_loss": float(output.ldad_loss.detach().cpu()),
            "train_level_losses": [
                float(loss.detach().cpu()) for loss in output.level_losses
            ],
            "grad_norm_pre_clip": float(grad_norm.detach().cpu()),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        if step % eval_every == 0 or step == max_steps:
            latest = {
                **evaluate(step, planning=step == max_steps),
                **latest_train,
            }
            _append_jsonl(metrics_path, latest)
            print(json.dumps(latest, sort_keys=True), flush=True)

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": max_steps,
        "metrics": latest,
        "config": config,
    }
    torch.save(checkpoint, output_dir / "checkpoint.pt")
    (output_dir / "metrics.json").write_text(
        json.dumps(latest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return latest


def _initialize_low_level(model: ControlledObjectJEPA, checkpoint_path: Path) -> None:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Low-level checkpoint does not exist: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = payload["model"]
    target = model.state_dict()
    required_prefixes = ("encoder.", "action_encoders.0.", "dynamics.0.")
    if model.target_encoder is not None:
        required_prefixes += ("target_encoder.",)
    compatible = {
        key: value
        for key, value in source.items()
        if key in target and target[key].shape == value.shape
    }
    missing_required = [
        key
        for key in target
        if key.startswith(required_prefixes) and key not in compatible
    ]
    if missing_required:
        raise ValueError(
            "Low-level checkpoint is incompatible with the hierarchy model: "
            + ", ".join(missing_required[:5])
        )
    model.load_state_dict(compatible, strict=False)


def _set_learning_rate(
    optimizer: AdamW,
    *,
    step: int,
    max_steps: int,
    warmup_steps: int,
    peak: float,
    floor: float,
) -> None:
    if warmup_steps > 0 and step <= warmup_steps:
        value = peak * step / warmup_steps
    else:
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        value = floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = value


def _without_name(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != "name"}


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


@hydra.main(
    config_path="../../configs/controlled_objects",
    config_name="train",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    run_controlled_object_training(OmegaConf.to_container(cfg, resolve=True))


if __name__ == "__main__":
    main()
