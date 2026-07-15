from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from puzzle_jepa.controlled_objects.batching import build_controlled_dataset
from puzzle_jepa.controlled_objects.evaluation import evaluate_planner_interface
from puzzle_jepa.controlled_objects.generator import (
    ControlledObjectGenerator,
    ControlledObjectSpec,
)
from puzzle_jepa.train.controlled_objects import _without_name
from puzzle_jepa.train.controlled_objects_probes import _build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=12017)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--candidates", type=int, default=512)
    parser.add_argument("--hard-project-support", action="store_true")
    parser.add_argument("--reachability-weight", type=float, default=0.0)
    parser.add_argument("--reachability-candidates", type=int, default=64)
    args = parser.parse_args()

    requested_device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else args.device
    device = torch.device(requested_device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = dict(payload["config"])
    generator = ControlledObjectGenerator(
        ControlledObjectSpec(**_without_name(dict(config["data"])))
    )
    model = _build_model(config, generator, device=device, initialize_stage=False)
    model.load_state_dict(payload["model"])
    dataset = build_controlled_dataset(
        generator,
        trajectory_count=int(config["eval"].get("trajectories", 128)),
        seed=int(config.get("data_seed", 8801)) + 1,
    )
    metrics = evaluate_planner_interface(
        model,
        dataset,
        generator,
        seed=args.seed,
        episodes=args.episodes,
        candidates=args.candidates,
        device=device,
        hard_project_support=args.hard_project_support,
        reachability_weight=args.reachability_weight,
        reachability_candidates=args.reachability_candidates,
    )
    result = {
        "schema": "controlled_objects_planner_interface_v1",
        "checkpoint": str(args.checkpoint),
        "run_name": str(config["run_name"]),
        "seed": int(config["seed"]),
        **metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
