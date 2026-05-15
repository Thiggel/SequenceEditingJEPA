from __future__ import annotations

import torch


def sigreg_loss(latents: torch.Tensor, mask: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    valid = mask.bool()
    if valid.sum() < 2:
        return latents.sum() * 0.0
    z = latents[valid]
    z = z - z.mean(dim=0, keepdim=True)
    cov = z.T @ z / max(1, z.shape[0] - 1)
    diag = torch.diag(cov)
    offdiag = cov - torch.diag_embed(diag)
    mean_term = latents[valid].mean(dim=0).pow(2).mean()
    diag_term = (diag - 1.0).pow(2).mean()
    offdiag_term = offdiag.pow(2).sum() / max(1, offdiag.numel() - diag.numel())
    return mean_term + diag_term + offdiag_term + eps * 0.0
