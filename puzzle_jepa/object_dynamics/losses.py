from __future__ import annotations

import torch
from torch.nn import functional as F


def variance_loss(z: torch.Tensor, *, eps: float = 1.0e-4, target_std: float = 1.0) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=0) + eps)
    return F.relu(target_std - std).mean()


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    if z.shape[0] <= 1:
        return z.sum() * 0.0
    centered = z - z.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / (z.shape[0] - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return off_diag.pow(2).sum() / z.shape[1]


def vicreg_regularizer(z: torch.Tensor, *, variance_weight: float = 1.0, covariance_weight: float = 0.04) -> torch.Tensor:
    return variance_weight * variance_loss(z) + covariance_weight * covariance_loss(z)


def sigreg_regularizer(z: torch.Tensor, *, target_std: float = 1.0) -> torch.Tensor:
    centered = z - z.mean(dim=0, keepdim=True)
    normed = centered / centered.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    gram = normed @ normed.T
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return (gram - eye).pow(2).mean() + variance_loss(z, target_std=target_std)
