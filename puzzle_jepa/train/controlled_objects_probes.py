from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from puzzle_jepa.controlled_objects.generator import (
    ControlledObjectGenerator,
    ControlledObjectSpec,
)
from puzzle_jepa.controlled_objects.model import ControlledObjectJEPA
from puzzle_jepa.controlled_objects.probes import run_controlled_object_probes
from puzzle_jepa.train.controlled_objects import _initialize_low_level, _without_name


def evaluate_checkpoint_probes(
    checkpoint_path: Path,
    *,
    device: torch.device,
    probe_seed: int,
    train_samples: int,
    eval_samples: int,
    batch_size: int,
    steps: int,
    learning_rate: float,
) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = dict(payload["config"])
    generator = ControlledObjectGenerator(
        ControlledObjectSpec(**_without_name(dict(config["data"])))
    )

    initial = _build_model(config, generator, device=device, initialize_stage=True)
    final = _build_model(config, generator, device=device, initialize_stage=False)
    final.load_state_dict(payload["model"])

    probe_args = {
        "seed": probe_seed,
        "train_samples": train_samples,
        "eval_samples": eval_samples,
        "batch_size": batch_size,
        "device": device,
        "steps": steps,
        "learning_rate": learning_rate,
    }
    initial_metrics = run_controlled_object_probes(initial, generator, **probe_args)
    final_metrics = run_controlled_object_probes(final, generator, **probe_args)
    del initial, final
    if device.type == "cuda":
        torch.cuda.empty_cache()

    deltas = {
        name: float(final_metrics[name]) - float(initial_metrics[name])
        for name in final_metrics
        if name != "probe_schema"
        and isinstance(final_metrics[name], (float, int))
        and np.isfinite(float(final_metrics[name]))
        and np.isfinite(float(initial_metrics[name]))
    }
    return {
        "probe_schema": "controlled_objects_checkpoint_v2",
        "checkpoint": str(checkpoint_path),
        "step": int(payload["step"]),
        "run_name": str(config["run_name"]),
        "seed": int(config["seed"]),
        "probe_seed": probe_seed,
        "train_samples": train_samples,
        "eval_samples": eval_samples,
        "probe_steps": steps,
        "initial": initial_metrics,
        "final": final_metrics,
        "delta": deltas,
    }


def _build_model(
    config: dict[str, Any],
    generator: ControlledObjectGenerator,
    *,
    device: torch.device,
    initialize_stage: bool,
) -> ControlledObjectJEPA:
    torch.manual_seed(int(config["seed"]))
    model = ControlledObjectJEPA(
        grid_size=generator.spec.grid_size,
        num_colors=generator.spec.num_colors,
        **_without_name(dict(config["model"])),
        **_without_name(dict(config["objective"])),
    ).to(device)
    init_checkpoint = config["training"].get("init_checkpoint")
    if initialize_stage and init_checkpoint:
        _initialize_low_level(model, Path(str(init_checkpoint)))
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--probe-seed", type=int, default=9917)
    parser.add_argument("--train-samples", type=int, default=1024)
    parser.add_argument("--eval-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3.0e-3)
    args = parser.parse_args()
    requested_device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else args.device
    device = torch.device(requested_device)
    result = evaluate_checkpoint_probes(
        args.checkpoint,
        device=device,
        probe_seed=args.probe_seed,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        batch_size=args.batch_size,
        steps=args.steps,
        learning_rate=args.learning_rate,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "run_name": result["run_name"]}))


if __name__ == "__main__":
    main()
