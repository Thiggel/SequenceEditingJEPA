from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from puzzle_jepa.moving_objects.batching import MovingObjectBatch
from puzzle_jepa.object_dynamics.losses import covariance_loss, variance_loss, vicreg_regularizer


def balanced_reconstruction_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    per_pixel = F.cross_entropy(logits, targets, reduction="none")
    foreground = targets != 0
    background = ~foreground
    if foreground.any() and background.any():
        return 0.5 * (per_pixel[foreground].mean() + per_pixel[background].mean())
    return per_pixel.mean()


@dataclass(slots=True)
class MovingObjectOutput:
    loss: torch.Tensor
    prediction_loss: torch.Tensor
    regularizer_loss: torch.Tensor
    temporal_delta_loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    predictions: torch.Tensor
    targets: torch.Tensor


class MotionContextEncoder(nn.Module):
    def __init__(
        self,
        *,
        grid_size: int,
        num_colors: int,
        token_dim: int,
        latent_dim: int,
        num_layers: int,
        num_heads: int,
    ):
        super().__init__()
        self.grid_size = int(grid_size)
        self.token_dim = int(token_dim)
        self.color = nn.Embedding(num_colors, token_dim)
        self.row = nn.Embedding(grid_size, token_dim)
        self.col = nn.Embedding(grid_size, token_dim)
        self.time = nn.Embedding(2, token_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, token_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=4 * token_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.project = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, latent_dim), nn.Tanh())

    def forward(self, contexts: torch.Tensor) -> torch.Tensor:
        if contexts.ndim != 4 or contexts.shape[1] != 2:
            raise ValueError("MotionContextEncoder expects [B,2,H,W].")
        batch, _, height, width = contexts.shape
        if (height, width) != (self.grid_size, self.grid_size):
            raise ValueError(f"Expected {self.grid_size}x{self.grid_size} frames.")
        rows = torch.arange(height, device=contexts.device).view(1, 1, height, 1)
        cols = torch.arange(width, device=contexts.device).view(1, 1, 1, width)
        times = torch.arange(2, device=contexts.device).view(1, 2, 1, 1)
        tokens = self.color(contexts) + self.row(rows) + self.col(cols) + self.time(times)
        tokens = tokens.reshape(batch, 2 * height * width, self.token_dim)
        encoded = self.transformer(torch.cat([self.cls.expand(batch, -1, -1), tokens], dim=1))
        return self.project(encoded[:, 0])


class MovingObjectJEPA(nn.Module):
    def __init__(
        self,
        *,
        grid_size: int = 16,
        num_colors: int = 10,
        token_dim: int = 64,
        latent_dim: int = 16,
        encoder_layers: int = 2,
        encoder_heads: int = 4,
        rollout_horizon: int = 4,
        ema_decay: float = 0.99,
        regularizer_weight: float = 0.05,
        temporal_delta_weight: float = 0.0,
        temporal_delta_target_std: float = 0.1,
        prediction_weight: float = 1.0,
        reconstruction_weight: float = 0.0,
    ):
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        self.latent_dim = int(latent_dim)
        self.rollout_horizon = int(rollout_horizon)
        self.ema_decay = float(ema_decay)
        self.regularizer_weight = float(regularizer_weight)
        self.temporal_delta_weight = float(temporal_delta_weight)
        self.temporal_delta_target_std = float(temporal_delta_target_std)
        self.prediction_weight = float(prediction_weight)
        self.reconstruction_weight = float(reconstruction_weight)
        if min(self.temporal_delta_weight, self.prediction_weight, self.reconstruction_weight) < 0.0:
            raise ValueError("Objective weights must be nonnegative.")
        if self.temporal_delta_target_std <= 0.0:
            raise ValueError("Temporal-delta target std must be positive.")
        encoder_args = dict(
            grid_size=grid_size,
            num_colors=num_colors,
            token_dim=token_dim,
            latent_dim=latent_dim,
            num_layers=encoder_layers,
            num_heads=encoder_heads,
        )
        self.encoder = MotionContextEncoder(**encoder_args)
        self.target_encoder = MotionContextEncoder(**encoder_args)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 4 * latent_dim),
            nn.GELU(),
            nn.Linear(4 * latent_dim, latent_dim),
        )
        self.reconstruction_decoder = (
            nn.Linear(latent_dim, grid_size * grid_size * num_colors)
            if self.reconstruction_weight > 0.0
            else None
        )
        self.grid_size = int(grid_size)
        self.num_colors = int(num_colors)

    def encode(self, contexts: torch.Tensor, *, target: bool = False) -> torch.Tensor:
        return (self.target_encoder if target else self.encoder)(contexts)

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        for target, online in zip(self.target_encoder.parameters(), self.encoder.parameters(), strict=True):
            target.mul_(self.ema_decay).add_(online, alpha=1.0 - self.ema_decay)

    def forward(self, batch: MovingObjectBatch) -> MovingObjectOutput:
        horizon = min(self.rollout_horizon, batch.future_contexts.shape[1])
        current = self.encoder(batch.contexts)
        state = current
        with torch.no_grad():
            flat = batch.future_contexts[:, :horizon].flatten(0, 1)
            targets = self.target_encoder(flat).reshape(len(state), horizon, self.latent_dim)
        predictions = []
        for _ in range(horizon):
            state = torch.tanh(state + self.predictor(state))
            predictions.append(state)
        predicted = torch.stack(predictions, dim=1)
        prediction_loss = F.mse_loss(predicted, targets)
        regularizer_loss = vicreg_regularizer(torch.cat([current, predicted[:, 0]], dim=0))
        if self.temporal_delta_weight > 0.0:
            online_future = self.encoder(batch.future_contexts[:, 0])
            temporal_delta = online_future - current
            temporal_delta_loss = variance_loss(
                temporal_delta, target_std=self.temporal_delta_target_std
            ) + 0.04 * covariance_loss(temporal_delta)
        else:
            temporal_delta_loss = prediction_loss.detach() * 0.0
        if self.reconstruction_decoder is not None:
            reconstruction_logits = self.reconstruction_decoder(current).reshape(
                len(current), self.grid_size * self.grid_size, self.num_colors
            )
            reconstruction_loss = balanced_reconstruction_loss(
                reconstruction_logits.flatten(0, 1), batch.current_grid.flatten()
            )
        else:
            reconstruction_loss = prediction_loss.detach() * 0.0
        return MovingObjectOutput(
            loss=(
                self.prediction_weight * prediction_loss
                + self.regularizer_weight * regularizer_loss
                + self.temporal_delta_weight * temporal_delta_loss
                + self.reconstruction_weight * reconstruction_loss
            ),
            prediction_loss=prediction_loss,
            regularizer_loss=regularizer_loss,
            temporal_delta_loss=temporal_delta_loss,
            reconstruction_loss=reconstruction_loss,
            predictions=predicted,
            targets=targets,
        )
