from __future__ import annotations

import torch

from seq_edit_jepa.losses.dyn_loss import masked_mse


def rollout_mse(predictions: list[torch.Tensor], targets: list[torch.Tensor], masks: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    if not predictions:
        raise ValueError("rollout_mse requires at least one prediction.")
    total = predictions[0].sum() * 0.0
    for pred, target, mask, weight in zip(predictions, targets, masks, weights):
        total = total + float(weight) * masked_mse(pred, target, mask)
    return total
