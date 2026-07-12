from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from puzzle_jepa.moving_objects.generator import MovingObjectGenerator, MovingObjectSpec
from puzzle_jepa.moving_objects.model import MovingObjectJEPA
from puzzle_jepa.moving_objects.probes import run_moving_object_probes
from puzzle_jepa.moving_objects.reproducibility import configure_reproducibility


def evaluate_checkpoint_probes(
    checkpoint_path: Path,
    *,
    output_path: Path,
    train_samples: int,
    eval_samples: int,
    batch_size: int,
    steps: int,
    learning_rate: float,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    seed = int(config["seed"])
    probe_seed = seed + 100_003
    data_cfg = _without_name(dict(config["data"]))
    model_cfg = _without_name(dict(config["model"]))
    objective_cfg = _without_name(dict(config["objective"]))
    generator = MovingObjectGenerator(MovingObjectSpec(**data_cfg))
    configure_reproducibility(bool(config["training"].get("deterministic", True)))

    torch.manual_seed(seed)
    model = MovingObjectJEPA(
        grid_size=generator.spec.grid_size,
        num_colors=generator.spec.num_colors,
        **model_cfg,
        **objective_cfg,
    ).to(device)

    def probe() -> dict[str, float | int | str | None]:
        torch.manual_seed(probe_seed)
        return run_moving_object_probes(
            model,
            generator,
            np.random.default_rng(probe_seed),
            train_samples=train_samples,
            eval_samples=eval_samples,
            batch_size=batch_size,
            device=device,
            steps=steps,
            learning_rate=learning_rate,
        )

    initial = probe()
    model.load_state_dict(checkpoint["model"])
    final = probe()
    payload = {
        "schema": "moving_objects_probe_eval_v6",
        "run_name": checkpoint_path.parent.name,
        "step": int(checkpoint["step"]),
        "seed": seed,
        "latent_dim": int(model.latent_dim),
        "latent_quantization_levels": int(model.latent_quantization_levels),
        "latent_capacity_bits": model.latent_capacity_bits,
        "max_objects": int(generator.spec.max_objects),
        "probe_seed": probe_seed,
        "initial": initial,
        "final": final,
        "delta": {
            key: float(final[key]) - float(initial[key])
            for key in final
            if isinstance(final[key], (int, float)) and isinstance(initial.get(key), (int, float))
        },
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _without_name(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != "name"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1.0e-2)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    payload = evaluate_checkpoint_probes(
        args.checkpoint,
        output_path=args.output,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        batch_size=args.batch_size,
        steps=args.steps,
        learning_rate=args.learning_rate,
        device=device,
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
