from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return self.weight * x * torch.rsqrt(variance + self.eps)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.up = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * value)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads.")
        self.attn_norm = RMSNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
            bias=False,
        )
        self.mlp_norm = RMSNorm(hidden_size)
        self.mlp = SwiGLU(hidden_size, intermediate_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.attn_norm(x)
        attn, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + self.dropout(attn)
        x = x + self.dropout(self.mlp(self.mlp_norm(x)))
        return x


class TransformerStack(nn.Module):
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class GridEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_layers: int,
        num_heads: int,
        max_height: int,
        max_width: int,
        task_vocab_size: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.max_height = int(max_height)
        self.max_width = int(max_width)
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.row_embedding = nn.Embedding(max_height, hidden_size)
        self.col_embedding = nn.Embedding(max_width, hidden_size)
        self.task_embedding = nn.Embedding(task_vocab_size, hidden_size)
        self.layers = TransformerStack(num_layers, hidden_size, intermediate_size, num_heads, dropout)

    def forward(self, tokens: torch.Tensor, task_ids: torch.Tensor | None = None) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"GridEncoder expects [batch, height, width], got {tuple(tokens.shape)}.")
        batch, height, width = tokens.shape
        if height > self.max_height or width > self.max_width:
            raise ValueError(f"Grid shape {(height, width)} exceeds max {(self.max_height, self.max_width)}.")
        if tokens.min() < 0 or tokens.max() >= self.vocab_size:
            raise ValueError("Grid contains token outside the encoder vocabulary.")
        rows = torch.arange(height, device=tokens.device).view(1, height, 1).expand(batch, height, width)
        cols = torch.arange(width, device=tokens.device).view(1, 1, width).expand(batch, height, width)
        if task_ids is None:
            task_ids = torch.zeros(batch, dtype=torch.long, device=tokens.device)
        task = self.task_embedding(task_ids).view(batch, 1, 1, -1)
        x = self.token_embedding(tokens) + self.row_embedding(rows) + self.col_embedding(cols) + task
        return self.layers(x.reshape(batch, height * width, -1))
