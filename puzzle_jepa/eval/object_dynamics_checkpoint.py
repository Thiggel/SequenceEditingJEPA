from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from puzzle_jepa.object_dynamics.generator import ObjectDynamicsGenerator, ObjectDynamicsSpec
from puzzle_jepa.object_dynamics.initialization import initialize_low_level_from_checkpoint
from puzzle_jepa.object_dynamics.model import ObjectDynamicsJEPA
from puzzle_jepa.object_dynamics.probes import run_object_dynamics_probes


def evaluate_object_dynamics_checkpoint(
    checkpoint_path: Path,
    *,
    output_path: Path,
    device: str = "auto",
    train_samples: int | None = None,
    eval_samples: int | None = None,
    batch_size: int | None = None,
    steps: int | None = None,
    learning_rate: float | None = None,
) -> dict[str, Any]:
    resolved_device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "auto"
        else torch.device(device)
    )
    payload = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    config = payload["config"]
    data_config = {key: value for key, value in dict(config["data"]).items() if key != "name"}
    model_config = {key: value for key, value in dict(config["model"]).items() if key != "name"}
    objective_config = {key: value for key, value in dict(config["objective"]).items() if key != "name"}
    eval_config = dict(config["eval"])
    probe_data_config = {
        **data_config,
        "trajectory_kind": str(eval_config.get("probe_trajectory_kind", data_config["trajectory_kind"])),
    }
    generator = ObjectDynamicsGenerator(ObjectDynamicsSpec(**probe_data_config))
    torch.manual_seed(int(config["seed"]))
    model = ObjectDynamicsJEPA(
        grid_size=generator.spec.grid_size,
        num_colors=generator.spec.num_colors,
        **model_config,
        **objective_config,
    ).to(resolved_device)
    initialize_low_level_from_checkpoint(
        model,
        dict(config.get("training", {})).get("initial_checkpoint"),
        device=resolved_device,
    )
    probe_seed = int(config["seed"]) + 100_003
    torch.manual_seed(probe_seed)
    horizon = model.training_horizon
    probe_kwargs = {
        "train_samples": int(train_samples or eval_config.get("probe_train_samples", 512)),
        "eval_samples": int(eval_samples or eval_config.get("probe_eval_samples", 256)),
        "batch_size": int(batch_size or eval_config.get("probe_batch_size", 64)),
        "horizon": horizon,
        "device": resolved_device,
        "steps": int(steps or eval_config.get("probe_steps", 150)),
        "learning_rate": float(learning_rate or eval_config.get("probe_learning_rate", 1.0e-2)),
    }
    initial_metrics = run_object_dynamics_probes(
        model,
        generator,
        np.random.default_rng(probe_seed),
        **probe_kwargs,
    )
    model.load_state_dict(payload["model"])
    torch.manual_seed(probe_seed)
    metrics = run_object_dynamics_probes(
        model,
        generator,
        np.random.default_rng(probe_seed),
        **probe_kwargs,
    )
    initial_values = {f"initial_{key}": value for key, value in initial_metrics.items()}
    delta_values = {
        f"delta_{key}": float(value) - float(initial_metrics[key])
        for key, value in metrics.items()
        if key in initial_metrics and _is_finite_number(value) and _is_finite_number(initial_metrics[key])
    }
    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(payload["step"]),
        "data": str(config["data"]["name"]),
        "model": str(config["model"]["name"]),
        "objective": str(config["objective"]["name"]),
        "seed": int(config["seed"]),
        "probe_seed": probe_seed,
        "device": resolved_device.type,
        **initial_values,
        **metrics,
        **delta_values,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _is_finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)):
        return False
    return bool(np.isfinite(float(value)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run class-balanced frozen probes on an object-dynamics checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-samples", type=int)
    parser.add_argument("--eval-samples", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    args = parser.parse_args()
    result = evaluate_object_dynamics_checkpoint(
        args.checkpoint,
        output_path=args.output,
        device=args.device,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        batch_size=args.batch_size,
        steps=args.steps,
        learning_rate=args.learning_rate,
    )
    print(json.dumps({"checkpoint_step": result["checkpoint_step"], "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
