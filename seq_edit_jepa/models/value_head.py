from __future__ import annotations

import torch
from torch import nn


class ValueHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.token_value = nn.Linear(hidden_size, 1)
        self.pooled_value = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        token_values = self.token_value(hidden_states).squeeze(-1)
        mask = attention_mask.bool().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1)
        pooled = (hidden_states * mask).sum(dim=1) / denom
        return token_values, self.pooled_value(pooled).squeeze(-1)
