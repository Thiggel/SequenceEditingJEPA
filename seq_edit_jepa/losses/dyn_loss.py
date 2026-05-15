from __future__ import annotations

import torch


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.bool().unsqueeze(-1)
    if not valid.any():
        return pred.sum() * 0.0
    diff = (pred - target).pow(2)
    return diff.masked_select(valid.expand_as(diff)).mean()
