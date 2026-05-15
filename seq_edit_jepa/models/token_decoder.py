from __future__ import annotations

import torch
from torch import nn


class TokenDecoder(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states)
