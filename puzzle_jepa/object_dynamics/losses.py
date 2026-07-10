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


def sigreg_regularizer(
    z: torch.Tensor,
    *,
    num_slices: int = 1024,
    t_max: float = 3.0,
    num_points: int = 17,
) -> torch.Tensor:
    if z.ndim != 2:
        raise ValueError(f"SIGReg expects [samples, features], got {tuple(z.shape)}.")
    if z.shape[0] <= 1:
        return z.sum() * 0.0
    if num_slices <= 0 or num_points < 2:
        raise ValueError("SIGReg requires positive slices and at least two integration points.")

    values = z.float()
    with torch.no_grad():
        directions = torch.randn(values.shape[1], num_slices, device=values.device, dtype=values.dtype)
        directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1.0e-12)
        frequencies = torch.linspace(0.0, t_max, num_points, device=values.device, dtype=values.dtype)
        step = t_max / (num_points - 1)
        weights = torch.full((num_points,), 2.0 * step, device=values.device, dtype=values.dtype)
        weights[[0, -1]] = step
        gaussian_cf = torch.exp(-0.5 * frequencies.square())
        weights = weights * gaussian_cf

    projected = values @ directions
    phases = projected.unsqueeze(-1) * frequencies
    empirical_real = torch.cos(phases).mean(dim=0)
    empirical_imag = torch.sin(phases).mean(dim=0)
    error = (empirical_real - gaussian_cf).square() + empirical_imag.square()
    return (values.shape[0] * (error * weights).sum(dim=-1)).mean()
