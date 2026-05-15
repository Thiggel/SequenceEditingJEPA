from __future__ import annotations

import torch
import torch.nn.functional as F


def value_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if target.dtype.is_floating_point:
        return F.mse_loss(pred.float(), target.float())
    return F.binary_cross_entropy_with_logits(pred.float(), target.float())
