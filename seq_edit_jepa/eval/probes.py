from __future__ import annotations

import torch


def covariance_rank(latents: torch.Tensor, mask: torch.Tensor, tol: float = 1e-5) -> int:
    z = latents[mask.bool()]
    if z.numel() == 0:
        return 0
    z = z - z.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(z)
    return int((singular_values > tol).sum().item())


def rollout_degradation(one_step_mse: float, k_step_mse: float) -> float:
    return float(k_step_mse / max(1e-12, one_step_mse))
