from __future__ import annotations

import math

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


class SinusoidalTimestepEmbedding(nn.Module):
    """Diffusion-style sinusoidal timestep embedding followed by an MLP."""

    def __init__(self, hidden_size: int, embedding_size: int | None = None):
        super().__init__()
        embedding_size = int(embedding_size or hidden_size)
        self.embedding_size = embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim == 0:
            timesteps = timesteps.unsqueeze(0)
        timesteps = timesteps.float()
        half = self.embedding_size // 2
        frequencies = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(1, half - 1)
        )
        args = timesteps[:, None] * frequencies[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.embedding_size:
            emb = F.pad(emb, (0, self.embedding_size - emb.shape[-1]))
        return self.mlp(emb)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even attention head dimension.")
        inv_freq = 1.0 / (float(theta) ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(length, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq.to(device))
        return freqs.cos()[None, None, :, :], freqs.sin()[None, None, :, :]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1)
    return rotated.flatten(-2)


class QKNormSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float,
        attention_dropout: float,
        norm_eps: float,
        rope_theta: float,
        qk_norm: bool = True,
        causal: bool = False,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim, theta=rope_theta)
        self.q_norm = RMSNorm(self.head_dim, eps=norm_eps) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim, eps=norm_eps) if qk_norm else nn.Identity()
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.causal = bool(causal)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, length, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        cos, sin = self.rotary(length, x.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if self.causal:
            causal_mask = torch.triu(torch.ones(length, length, device=x.device, dtype=torch.bool), diagonal=1)
            attn = attn.masked_fill(causal_mask[None, None, :, :], torch.finfo(attn.dtype).min)
        if attention_mask is not None:
            key_mask = ~attention_mask.bool()
            attn = attn.masked_fill(key_mask[:, None, None, :], torch.finfo(attn.dtype).min)
        attn_probs = self.attn_dropout(torch.softmax(attn, dim=-1))
        context = torch.matmul(attn_probs, v)
        context = context.transpose(1, 2).contiguous().view(batch, length, self.hidden_size)
        return self.resid_dropout(self.out_proj(context))


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float):
        super().__init__()
        self.gate_up = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.gate_up(x).chunk(2, dim=-1)
        return self.dropout(self.down(F.silu(gate) * value))


class BidirectionalTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        dropout: float,
        attention_dropout: float,
        norm_eps: float,
        rope_theta: float,
        qk_norm: bool,
        causal: bool = False,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(hidden_size, eps=norm_eps)
        self.attn = QKNormSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            norm_eps=norm_eps,
            rope_theta=rope_theta,
            qk_norm=qk_norm,
            causal=causal,
        )
        self.mlp_norm = RMSNorm(hidden_size, eps=norm_eps)
        self.mlp = SwiGLU(hidden_size, intermediate_size, dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), attention_mask=attention_mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class BidirectionalTransformerStack(nn.Module):
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        dropout: float,
        attention_dropout: float,
        norm_eps: float,
        rope_theta: float,
        qk_norm: bool,
        causal: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                BidirectionalTransformerBlock(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    norm_eps=norm_eps,
                    rope_theta=rope_theta,
                    qk_norm=qk_norm,
                    causal=causal,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(hidden_size, eps=norm_eps)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        return self.final_norm(x)
