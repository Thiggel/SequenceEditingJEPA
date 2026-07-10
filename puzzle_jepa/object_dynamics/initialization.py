from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from puzzle_jepa.object_dynamics.model import ObjectDynamicsJEPA


def initialize_low_level_from_checkpoint(
    model: ObjectDynamicsJEPA,
    checkpoint: Any,
    *,
    device: torch.device,
) -> list[str]:
    if checkpoint is None or str(checkpoint) in {"", "null"}:
        return []
    checkpoint_path = Path(str(checkpoint))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Initial checkpoint does not exist: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    source = payload["model"]
    target = model.state_dict()
    low_level_prefixes = ("encoder.", "target_encoder.", "actions.", "predictor.")
    compatible = {
        name: value
        for name, value in source.items()
        if name.startswith(low_level_prefixes) and name in target and target[name].shape == value.shape
    }
    if not any(name.startswith("encoder.") for name in compatible):
        raise RuntimeError(f"Checkpoint {checkpoint_path} has no compatible encoder parameters.")
    model.load_state_dict(compatible, strict=False)
    return sorted(compatible)
