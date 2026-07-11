from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from puzzle_jepa.moving_objects.generator import MovingObjectGenerator, MovingObjectSpec
from puzzle_jepa.moving_objects.model import MovingObjectJEPA
from puzzle_jepa.moving_objects.probes import run_moving_object_dynamics_diagnostics


def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    output_path: Path,
    samples: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    seed = int(config["seed"])
    probe_seed = seed + 200_003
    data_cfg = _without_name(dict(config["data"]))
    model_cfg = _without_name(dict(config["model"]))
    objective_cfg = _without_name(dict(config["objective"]))
    generator = MovingObjectGenerator(MovingObjectSpec(**data_cfg))

    torch.manual_seed(seed)
    model = MovingObjectJEPA(
        grid_size=generator.spec.grid_size,
        num_colors=generator.spec.num_colors,
        **model_cfg,
        **objective_cfg,
    ).to(device)
    initial = run_moving_object_dynamics_diagnostics(
        model,
        generator,
        np.random.default_rng(probe_seed),
        samples=samples,
        batch_size=batch_size,
        device=device,
    )
    model.load_state_dict(checkpoint["model"])
    final = run_moving_object_dynamics_diagnostics(
        model,
        generator,
        np.random.default_rng(probe_seed),
        samples=samples,
        batch_size=batch_size,
        device=device,
    )
    payload = {
        "schema": "moving_objects_dynamics_v1",
        "run_name": checkpoint_path.parent.name,
        "step": int(checkpoint["step"]),
        "seed": seed,
        "latent_dim": int(model.latent_dim),
        "max_objects": int(generator.spec.max_objects),
        "samples": int(samples),
        "initial": initial,
        "final": final,
        "delta": {key: final[key] - initial[key] for key in final},
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _without_name(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != "name"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    payload = evaluate_checkpoint(
        args.checkpoint,
        output_path=args.output,
        samples=args.samples,
        batch_size=args.batch_size,
        device=device,
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
