from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW

from puzzle_jepa.moving_objects.batching import sample_moving_object_batch
from puzzle_jepa.moving_objects.generator import MovingObjectGenerator, MovingObjectSpec
from puzzle_jepa.moving_objects.model import MovingObjectJEPA
from puzzle_jepa.moving_objects.probes import run_moving_object_probes


def run_moving_object_training(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    train_rng = np.random.default_rng(seed)
    probe_seed = seed + 100_003
    device = _resolve_device(str(config.get("device", "auto")))
    output_dir = Path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = _without_name(dict(config["data"]))
    model_cfg = _without_name(dict(config["model"]))
    objective_cfg = _without_name(dict(config["objective"]))
    training_cfg = dict(config["training"])
    _configure_reproducibility(bool(training_cfg.get("deterministic", True)))
    eval_cfg = dict(config["eval"])
    generator = MovingObjectGenerator(MovingObjectSpec(**data_cfg))
    model = MovingObjectJEPA(
        grid_size=generator.spec.grid_size,
        num_colors=generator.spec.num_colors,
        **model_cfg,
        **objective_cfg,
    ).to(device)
    config = {**config, "param_count": sum(parameter.numel() for parameter in model.parameters())}
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))

    optimizer = AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 3.0e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1.0e-4)),
    )
    max_steps = int(training_cfg.get("max_steps", 5000))
    batch_size = int(training_cfg.get("batch_size", 128))
    eval_every = int(training_cfg.get("eval_every_steps", max_steps))
    grad_clip = float(training_cfg.get("grad_clip", 1.0))
    use_amp = bool(training_cfg.get("bf16", True)) and device.type == "cuda"
    metrics_path = output_dir / "metrics.jsonl"

    def evaluate(step: int) -> dict[str, Any]:
        with torch.random.fork_rng(devices=_fork_rng_devices(device)):
            torch.manual_seed(probe_seed)
            metrics = run_moving_object_probes(
                model,
                generator,
                np.random.default_rng(probe_seed),
                train_samples=int(eval_cfg.get("probe_train_samples", 512)),
                eval_samples=int(eval_cfg.get("probe_eval_samples", 256)),
                batch_size=int(eval_cfg.get("probe_batch_size", 64)),
                device=device,
                steps=int(eval_cfg.get("probe_steps", 100)),
                learning_rate=float(eval_cfg.get("probe_learning_rate", 1.0e-2)),
            )
        return {
            "step": step,
            "data": config["data"]["name"],
            "model": config["model"]["name"],
            "objective": config["objective"]["name"],
            "latent_dim": model.latent_dim,
            "min_objects": generator.spec.min_objects,
            "max_objects": generator.spec.max_objects,
            "seed": seed,
            "device": device.type,
            "param_count": config["param_count"],
            **metrics,
        }

    initial = evaluate(0)
    _append_jsonl(metrics_path, initial)
    print(json.dumps(initial, sort_keys=True), flush=True)
    latest_train: dict[str, float] = {}
    for step in range(1, max_steps + 1):
        model.train()
        batch = sample_moving_object_batch(
            generator,
            train_rng,
            batch_size=batch_size,
            horizon=model.rollout_horizon,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            output = model(batch)
        if not torch.isfinite(output.loss.detach()):
            raise FloatingPointError(f"Non-finite moving-object loss at step {step}.")
        output.loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        model.update_target_encoder()
        latest_train = {
            "train_loss": float(output.loss.detach().cpu()),
            "train_prediction_loss": float(output.prediction_loss.detach().cpu()),
            "train_regularizer_loss": float(output.regularizer_loss.detach().cpu()),
            "train_temporal_delta_loss": float(output.temporal_delta_loss.detach().cpu()),
            "grad_norm_pre_clip": float(grad_norm.detach().cpu()),
        }
        if step % eval_every == 0 or step == max_steps:
            latest = {**evaluate(step), **latest_train}
            _append_jsonl(metrics_path, latest)
            print(json.dumps(latest, sort_keys=True), flush=True)
        if step == max_steps:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "metrics": latest,
                    "config": config,
                },
                output_dir / "checkpoint.pt",
            )
    (output_dir / "metrics.json").write_text(json.dumps(latest, indent=2, sort_keys=True))
    return latest


def _without_name(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != "name"}


def _resolve_device(requested: str) -> torch.device:
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else ("cpu" if requested == "auto" else requested))


def _fork_rng_devices(device: torch.device) -> list[int]:
    if device.type != "cuda":
        return []
    return [device.index if device.index is not None else torch.cuda.current_device()]


def _configure_reproducibility(deterministic: bool) -> None:
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(not deterministic)
        torch.backends.cuda.enable_mem_efficient_sdp(not deterministic)
        torch.backends.cuda.enable_math_sdp(True)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


@hydra.main(config_path="../../configs/moving_objects", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    run_moving_object_training(OmegaConf.to_container(cfg, resolve=True))


if __name__ == "__main__":
    main()
