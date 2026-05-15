from __future__ import annotations

import copy

import torch
from torch import nn


def clone_target_encoder(encoder: nn.Module) -> nn.Module:
    target = copy.deepcopy(encoder)
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    target.eval()
    return target


@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    for target_param, online_param in zip(target.parameters(), online.parameters()):
        target_param.data.mul_(tau).add_(online_param.data, alpha=1.0 - tau)
    for target_buffer, online_buffer in zip(target.buffers(), online.buffers()):
        target_buffer.copy_(online_buffer)
