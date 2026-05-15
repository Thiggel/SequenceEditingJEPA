from __future__ import annotations

import torch
from torch import nn

from seq_edit_jepa.models.layers import BidirectionalTransformerStack, RMSNorm, SinusoidalTimestepEmbedding


class LatentPredictor(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_layers: int,
        num_heads: int,
        max_length: int,
        num_steps: int,
        num_ops: int,
        pad_token_id: int,
        dropout: float,
        attention_dropout: float = 0.0,
        norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        qk_norm: bool = True,
        timestep_embedding_size: int | None = None,
        predictor_type: str = "transformer",
        action_conditioned: bool = True,
        action_conditioning: str = "concat",
    ):
        super().__init__()
        self.action_conditioned = bool(action_conditioned)
        self.action_conditioning = str(action_conditioning)
        self.predictor_type = predictor_type
        self.max_length = int(max_length)
        self.num_steps = int(num_steps)
        self.op_embedding = nn.Embedding(num_ops, hidden_size)
        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.time_embedding = SinusoidalTimestepEmbedding(hidden_size, timestep_embedding_size)
        if self.action_conditioning == "concat":
            self.condition_projection = nn.Linear(hidden_size * 2, hidden_size)
        elif self.action_conditioning == "add":
            self.condition_projection = nn.Identity()
        else:
            raise ValueError(f"Unknown action_conditioning={self.action_conditioning}")
        self.input_norm = RMSNorm(hidden_size, eps=norm_eps)
        if predictor_type == "mlp":
            self.predictor = nn.Sequential(
                nn.Linear(hidden_size, intermediate_size),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(intermediate_size, hidden_size),
            )
            self.output_norm = RMSNorm(hidden_size, eps=norm_eps)
        elif predictor_type == "transformer":
            self.predictor = BidirectionalTransformerStack(
                num_layers=max(1, num_layers),
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_heads=num_heads,
                dropout=dropout,
                attention_dropout=attention_dropout,
                norm_eps=norm_eps,
                rope_theta=rope_theta,
                qk_norm=qk_norm,
            )
            self.output_norm = nn.Identity()
        else:
            raise ValueError(f"Unknown predictor_type={predictor_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        action_ops: torch.Tensor,
        action_tokens: torch.Tensor,
        n: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length, _ = hidden_states.shape
        if length > self.max_length:
            raise ValueError(f"Input length {length} exceeds max_length={self.max_length}.")
        if n.ndim == 0:
            n = n.expand(batch)
        update = self.time_embedding(n.clamp(0, self.num_steps)).unsqueeze(1)
        if self.action_conditioned:
            update = update + self.op_embedding(action_ops.clamp_min(0)) + self.token_embedding(action_tokens.clamp_min(0))
        if self.action_conditioning == "concat":
            x = self.condition_projection(torch.cat([hidden_states, update.expand_as(hidden_states)], dim=-1))
        else:
            x = hidden_states + update
        x = self.input_norm(x)
        if self.predictor_type == "mlp":
            x = x + self.predictor(x)
            return self.output_norm(x)
        return self.predictor(x, attention_mask=attention_mask)
