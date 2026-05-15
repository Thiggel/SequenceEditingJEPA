from __future__ import annotations

import torch
from torch import nn

from seq_edit_jepa.models.layers import BidirectionalTransformerStack, SinusoidalTimestepEmbedding


class BidirectionalSequenceEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_layers: int,
        num_heads: int,
        max_length: int,
        num_steps: int,
        pad_token_id: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        qk_norm: bool = True,
        timestep_embedding_size: int | None = None,
    ):
        super().__init__()
        self.pad_token_id = int(pad_token_id)
        self.max_length = int(max_length)
        self.num_steps = int(num_steps)
        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.segment_embedding = nn.Embedding(2, hidden_size)
        self.time_embedding = SinusoidalTimestepEmbedding(hidden_size, timestep_embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.layers = BidirectionalTransformerStack(
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

    def forward(
        self,
        input_ids: torch.Tensor,
        n: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        segment_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length = input_ids.shape
        if length > self.max_length:
            raise ValueError(f"Input length {length} exceeds max_length={self.max_length}.")
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).long()
        if segment_ids is None:
            segment_ids = torch.zeros_like(input_ids)
        if n.ndim == 0:
            n = n.expand(batch)
        n = n.clamp(min=0, max=self.num_steps)
        x = (
            self.token_embedding(input_ids)
            + self.segment_embedding(segment_ids.clamp(0, 1))
            + self.time_embedding(n).unsqueeze(1)
        )
        x = self.dropout(x)
        return self.layers(x, attention_mask=attention_mask)
