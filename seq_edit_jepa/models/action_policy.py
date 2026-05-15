from __future__ import annotations

import torch
from torch import nn

from seq_edit_jepa.models.layers import BidirectionalTransformerStack, RMSNorm


class ActionPolicy(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_ops: int,
        num_layers: int = 0,
        intermediate_size: int | None = None,
        num_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        qk_norm: bool = True,
    ):
        super().__init__()
        intermediate_size = int(intermediate_size or hidden_size * 4)
        self.layers = (
            BidirectionalTransformerStack(
                num_layers=num_layers,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_heads=num_heads,
                dropout=dropout,
                attention_dropout=attention_dropout,
                norm_eps=norm_eps,
                rope_theta=rope_theta,
                qk_norm=qk_norm,
            )
            if int(num_layers) > 0
            else nn.Identity()
        )
        self.norm = RMSNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.SiLU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.op_head = nn.Linear(hidden_size, num_ops)
        self.token_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(self.layers, BidirectionalTransformerStack):
            hidden_states = self.layers(hidden_states, attention_mask=attention_mask)
        features = hidden_states + self.mlp(self.norm(hidden_states))
        return self.op_head(features), self.token_head(features)
