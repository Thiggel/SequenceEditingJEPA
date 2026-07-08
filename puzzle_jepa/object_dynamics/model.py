from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from puzzle_jepa.object_dynamics.batching import ObjectDynamicsBatch
from puzzle_jepa.object_dynamics.losses import sigreg_regularizer, vicreg_regularizer


@dataclass(frozen=True, slots=True)
class ObjectDynamicsOutput:
    loss: torch.Tensor
    rollout_loss: torch.Tensor
    hierarchy_loss: torch.Tensor
    ldad_loss: torch.Tensor
    regularizer_loss: torch.Tensor
    predicted: torch.Tensor
    targets: torch.Tensor


class CLSGridEncoder(nn.Module):
    def __init__(
        self,
        *,
        grid_size: int,
        num_colors: int,
        d_model: int,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.grid_size = int(grid_size)
        self.num_colors = int(num_colors)
        self.d_model = int(d_model)
        self.color = nn.Embedding(num_colors, d_model)
        self.row = nn.Embedding(grid_size, d_model)
        self.col = nn.Embedding(grid_size, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError(f"CLSGridEncoder expects [B,H,W], got {tuple(values.shape)}.")
        batch, height, width = values.shape
        if height != self.grid_size or width != self.grid_size:
            raise ValueError(f"Expected {self.grid_size}x{self.grid_size}, got {(height, width)}.")
        rows = torch.arange(height, device=values.device).view(1, height, 1).expand(batch, height, width)
        cols = torch.arange(width, device=values.device).view(1, 1, width).expand(batch, height, width)
        tokens = self.color(values.clamp(0, self.num_colors - 1)) + self.row(rows) + self.col(cols)
        tokens = tokens.reshape(batch, height * width, self.d_model)
        cls = self.cls.expand(batch, -1, -1)
        encoded = self.encoder(torch.cat([cls, tokens], dim=1))
        return self.norm(encoded[:, 0])


class ActionEncoder(nn.Module):
    def __init__(self, *, grid_size: int, num_colors: int, d_model: int):
        super().__init__()
        self.op = nn.Embedding(3, d_model)
        self.row = nn.Embedding(grid_size, d_model)
        self.col = nn.Embedding(grid_size, d_model)
        self.color = nn.Embedding(num_colors, d_model)
        self.net = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model))

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        op = actions[..., 0].clamp(0, 2)
        row = actions[..., 1].clamp(0, self.row.num_embeddings - 1)
        col = actions[..., 2].clamp(0, self.col.num_embeddings - 1)
        color = actions[..., 3].clamp(0, self.color.num_embeddings - 1)
        return self.net(self.op(op) + self.row(row) + self.col(col) + self.color(color))


class ObjectDynamicsJEPA(nn.Module):
    def __init__(
        self,
        *,
        grid_size: int = 16,
        num_colors: int = 10,
        d_model: int = 128,
        encoder_layers: int = 3,
        encoder_heads: int = 4,
        rollout_horizon: int = 4,
        hierarchy_horizon: int = 0,
        target_ema: bool = False,
        ema_decay: float = 0.99,
        ldad_weight: float = 0.0,
        regularizer: str = "none",
        regularizer_weight: float = 0.0,
    ):
        super().__init__()
        self.rollout_horizon = int(rollout_horizon)
        self.hierarchy_horizon = int(hierarchy_horizon)
        self.target_ema = bool(target_ema)
        self.ema_decay = float(ema_decay)
        self.ldad_weight = float(ldad_weight)
        self.regularizer = str(regularizer)
        self.regularizer_weight = float(regularizer_weight)
        self.encoder = CLSGridEncoder(
            grid_size=grid_size,
            num_colors=num_colors,
            d_model=d_model,
            num_layers=encoder_layers,
            num_heads=encoder_heads,
        )
        self.target_encoder = CLSGridEncoder(
            grid_size=grid_size,
            num_colors=num_colors,
            d_model=d_model,
            num_layers=encoder_layers,
            num_heads=encoder_heads,
        )
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        for param in self.target_encoder.parameters():
            param.requires_grad_(False)
        self.actions = ActionEncoder(grid_size=grid_size, num_colors=num_colors, d_model=d_model)
        self.predictor = nn.Sequential(
            nn.Linear(2 * d_model, 2 * d_model),
            nn.GELU(),
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
        )
        self.chunk_encoder = nn.GRU(input_size=d_model, hidden_size=d_model, batch_first=True)
        self.hierarchy_predictor = nn.Sequential(
            nn.Linear(2 * d_model, 2 * d_model),
            nn.GELU(),
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
        )
        self.delta_op = nn.Linear(d_model, 3)
        self.delta_row = nn.Linear(d_model, grid_size)
        self.delta_col = nn.Linear(d_model, grid_size)
        self.delta_color = nn.Linear(d_model, num_colors)

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        if not self.target_ema:
            self.target_encoder.load_state_dict(self.encoder.state_dict())
            return
        for target, online in zip(self.target_encoder.parameters(), self.encoder.parameters(), strict=True):
            target.mul_(self.ema_decay).add_(online, alpha=1.0 - self.ema_decay)

    def encode(self, states: torch.Tensor, *, target: bool = False) -> torch.Tensor:
        encoder = self.target_encoder if target else self.encoder
        return encoder(states)

    def forward(self, batch: ObjectDynamicsBatch) -> ObjectDynamicsOutput:
        horizon = min(max(1, self.rollout_horizon), batch.actions.shape[1], batch.futures.shape[1])
        target_horizon = min(
            max(horizon, self.hierarchy_horizon, 1),
            batch.actions.shape[1],
            batch.futures.shape[1],
        )
        current = self.encoder(batch.states)
        action_embeddings = self.actions(batch.actions[:, :target_horizon])
        targets = self._encode_targets(batch.futures[:, :target_horizon])
        predicted = []
        rollout_losses = []
        ldad_losses = []
        state = current
        for step in range(horizon):
            next_state = self.predictor(torch.cat([state, action_embeddings[:, step]], dim=-1))
            predicted.append(next_state)
            rollout_losses.append(F.mse_loss(next_state, targets[:, step]))
            if self.ldad_weight > 0.0:
                ldad_losses.append(self._ldad_loss(next_state - state, batch.actions[:, step]))
            state = next_state
        predicted_tensor = torch.stack(predicted, dim=1)
        rollout_loss = torch.stack(rollout_losses).mean()
        ldad_loss = torch.stack(ldad_losses).mean() if ldad_losses else rollout_loss.detach() * 0.0
        hierarchy_loss = self._hierarchy_loss(current, action_embeddings, targets)
        regularizer_loss = self._regularizer_loss(torch.cat([current, targets.reshape(-1, targets.shape[-1])], dim=0))
        loss = rollout_loss + hierarchy_loss + self.ldad_weight * ldad_loss + self.regularizer_weight * regularizer_loss
        return ObjectDynamicsOutput(
            loss=loss,
            rollout_loss=rollout_loss,
            hierarchy_loss=hierarchy_loss,
            ldad_loss=ldad_loss,
            regularizer_loss=regularizer_loss,
            predicted=predicted_tensor,
            targets=targets,
        )

    def _encode_targets(self, futures: torch.Tensor) -> torch.Tensor:
        batch, horizon, height, width = futures.shape
        flat = futures.reshape(batch * horizon, height, width)
        with torch.no_grad():
            encoded = self.target_encoder(flat).reshape(batch, horizon, -1)
        return encoded.detach()

    def _hierarchy_loss(self, current: torch.Tensor, action_embeddings: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.hierarchy_horizon <= 1 or self.hierarchy_horizon > action_embeddings.shape[1]:
            return current.sum() * 0.0
        horizon = min(self.hierarchy_horizon, targets.shape[1], action_embeddings.shape[1])
        _, hidden = self.chunk_encoder(action_embeddings[:, :horizon])
        chunk = hidden[-1]
        predicted = self.hierarchy_predictor(torch.cat([current, chunk], dim=-1))
        return F.mse_loss(predicted, targets[:, horizon - 1])

    def _ldad_loss(self, delta: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return (
            F.cross_entropy(self.delta_op(delta), actions[:, 0].clamp(0, 2))
            + F.cross_entropy(self.delta_row(delta), actions[:, 1].clamp(0, self.delta_row.out_features - 1))
            + F.cross_entropy(self.delta_col(delta), actions[:, 2].clamp(0, self.delta_col.out_features - 1))
            + F.cross_entropy(self.delta_color(delta), actions[:, 3].clamp(0, self.delta_color.out_features - 1))
        )

    def _regularizer_loss(self, z: torch.Tensor) -> torch.Tensor:
        if self.regularizer == "none" or self.regularizer_weight <= 0.0:
            return z.sum() * 0.0
        if self.regularizer == "vicreg":
            return vicreg_regularizer(z)
        if self.regularizer == "sigreg":
            return sigreg_regularizer(z)
        raise ValueError(f"Unknown regularizer {self.regularizer!r}.")
