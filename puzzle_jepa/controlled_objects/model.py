from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from puzzle_jepa.controlled_objects.batching import ControlledObjectBatch
from puzzle_jepa.controlled_objects.generator import TRANSFORM_NAMES
from puzzle_jepa.object_dynamics.losses import vicreg_regularizer


@dataclass(slots=True)
class ControlledObjectOutput:
    loss: torch.Tensor
    prediction_loss: torch.Tensor
    vicreg_loss: torch.Tensor
    ldad_loss: torch.Tensor
    level_losses: tuple[torch.Tensor, ...]
    predictions: tuple[torch.Tensor, ...]
    targets: tuple[torch.Tensor, ...]
    rollout_weights: tuple[torch.Tensor, ...]
    ldad_logits: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None


class ControlledStateEncoder(nn.Module):
    def __init__(
        self,
        *,
        grid_size: int,
        num_colors: int,
        token_dim: int,
        latent_dim: int,
        num_layers: int,
        num_heads: int,
        latent_representation: str,
    ):
        super().__init__()
        if latent_representation not in {"cls", "grid"}:
            raise ValueError("latent_representation must be 'cls' or 'grid'.")
        self.grid_size = int(grid_size)
        self.token_dim = int(token_dim)
        self.latent_representation = latent_representation
        self.color = nn.Embedding(num_colors, token_dim)
        self.row = nn.Embedding(grid_size, token_dim)
        self.col = nn.Embedding(grid_size, token_dim)
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
        self.project = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, latent_dim))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim != 3:
            raise ValueError("ControlledStateEncoder expects [B,H,W].")
        batch, height, width = states.shape
        if (height, width) != (self.grid_size, self.grid_size):
            raise ValueError(f"Expected {self.grid_size}x{self.grid_size} states.")
        rows = torch.arange(height, device=states.device).view(1, height, 1)
        cols = torch.arange(width, device=states.device).view(1, 1, width)
        tokens = self.color(states) + self.row(rows) + self.col(cols)
        tokens = tokens.reshape(batch, height * width, self.token_dim)
        encoded = self.transformer(
            torch.cat([self.cls.expand(batch, -1, -1), tokens], dim=1)
        )
        selected = encoded[:, 0] if self.latent_representation == "cls" else encoded[:, 1:]
        return self.project(selected)


class ActionChunkEncoder(nn.Module):
    def __init__(
        self,
        *,
        grid_size: int,
        chunk_length: int,
        token_dim: int,
        macro_dim: int,
        num_heads: int,
    ):
        super().__init__()
        self.chunk_length = int(chunk_length)
        self.row = nn.Embedding(grid_size, token_dim)
        self.col = nn.Embedding(grid_size, token_dim)
        self.transform = nn.Embedding(len(TRANSFORM_NAMES), token_dim)
        self.position = nn.Embedding(chunk_length, token_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, token_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=2 * token_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.project = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, macro_dim))

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim != 3 or actions.shape[1:] != (self.chunk_length, 3):
            raise ValueError(
                f"Expected action chunks [B,{self.chunk_length},3], got {tuple(actions.shape)}."
            )
        positions = torch.arange(self.chunk_length, device=actions.device).view(1, -1)
        tokens = (
            self.row(actions[..., 0])
            + self.col(actions[..., 1])
            + self.transform(actions[..., 2])
            + self.position(positions)
        )
        encoded = self.transformer(
            torch.cat([self.cls.expand(len(actions), -1, -1), tokens], dim=1)
        )
        return self.project(encoded[:, 0])


class LatentDynamics(nn.Module):
    def __init__(self, *, latent_dim: int, macro_dim: int):
        super().__init__()
        self.condition = nn.Linear(macro_dim, latent_dim)
        self.predictor = nn.Sequential(
            nn.LayerNorm(2 * latent_dim),
            nn.Linear(2 * latent_dim, 4 * latent_dim),
            nn.GELU(),
            nn.Linear(4 * latent_dim, latent_dim),
        )

    def forward(self, latent: torch.Tensor, macro: torch.Tensor) -> torch.Tensor:
        condition = self.condition(macro)
        while condition.ndim < latent.ndim:
            condition = condition.unsqueeze(1)
        condition = condition.expand(*latent.shape[:-1], condition.shape[-1])
        delta = self.predictor(torch.cat([latent, condition], dim=-1))
        return latent + delta


class CategoricalLDAD(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        grid_size: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.trunk = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.row = nn.Linear(hidden_dim, grid_size)
        self.col = nn.Linear(hidden_dim, grid_size)
        self.transform = nn.Linear(hidden_dim, len(TRANSFORM_NAMES))

    def forward(
        self, delta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flattened = delta.flatten(1)
        if flattened.shape[1] != self.input_dim:
            raise ValueError(
                f"LDAD expected displacement dim {self.input_dim}, got {flattened.shape[1]}."
            )
        hidden = self.trunk(self.input_projection(flattened))
        return self.row(hidden), self.col(hidden), self.transform(hidden)


class MultiStepCategoricalLDAD(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        grid_size: int,
        horizon: int,
        hidden_dim: int,
        num_heads: int = 4,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.horizon = int(horizon)
        self.queries = nn.Parameter(torch.zeros(1, horizon, hidden_dim))
        self.condition = nn.Linear(input_dim, 2 * hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=2 * hidden_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.row = nn.Linear(hidden_dim, grid_size)
        self.col = nn.Linear(hidden_dim, grid_size)
        self.transform = nn.Linear(hidden_dim, len(TRANSFORM_NAMES))

    def forward(
        self, delta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flattened = delta.flatten(1)
        if flattened.shape[1] != self.input_dim:
            raise ValueError(
                f"LDAD expected displacement dim {self.input_dim}, got {flattened.shape[1]}."
            )
        scale, shift = self.condition(flattened).chunk(2, dim=-1)
        queries = self.queries.expand(len(delta), -1, -1)
        normalized = F.layer_norm(queries, (queries.shape[-1],))
        hidden = self.transformer(
            normalized * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        )
        return self.row(hidden), self.col(hidden), self.transform(hidden)


class ControlledObjectJEPA(nn.Module):
    def __init__(
        self,
        *,
        grid_size: int = 16,
        num_colors: int = 10,
        token_dim: int = 64,
        latent_dim: int = 32,
        encoder_layers: int = 2,
        encoder_heads: int = 4,
        latent_representation: str = "cls",
        hierarchy_depth: int = 1,
        hierarchy_stride: int = 4,
        macro_dim: int = 8,
        action_token_dim: int = 32,
        action_heads: int = 4,
        rollout_steps: int = 1,
        rollout_all_levels: bool = False,
        rollout_lambda: float = 1.0,
        target_mode: str = "ema",
        stop_gradient_targets: bool = True,
        ema_decay: float = 0.99,
        prediction_weight: float = 1.0,
        vicreg_weight: float = 0.05,
        ldad_weight: float = 0.0,
        ldad_horizon: int = 1,
    ):
        super().__init__()
        if hierarchy_depth not in {1, 2, 3, 4}:
            raise ValueError("hierarchy_depth must be in {1,2,3,4}.")
        if hierarchy_stride not in {2, 4, 8}:
            raise ValueError("hierarchy_stride must be in {2,4,8}.")
        if rollout_steps not in {1, 2, 4, 8}:
            raise ValueError("rollout_steps must be in {1,2,4,8}.")
        if not (0.0 < rollout_lambda <= 1.0):
            raise ValueError("rollout_lambda must lie in (0,1].")
        if ldad_horizon not in {1, 2, 4, 8}:
            raise ValueError("ldad_horizon must be in {1,2,4,8}.")
        if target_mode not in {"shared", "ema"}:
            raise ValueError("target_mode must be 'shared' or 'ema'.")
        if target_mode == "ema" and not stop_gradient_targets:
            raise ValueError("EMA targets are necessarily stop-gradient targets.")
        if latent_representation == "grid" and hierarchy_depth != 1:
            raise ValueError("The paired full-grid control is defined only for flat LDAD rows.")
        self.grid_size = int(grid_size)
        self.latent_dim = int(latent_dim)
        self.latent_representation = latent_representation
        self.hierarchy_depth = int(hierarchy_depth)
        self.hierarchy_stride = int(hierarchy_stride)
        self.rollout_steps = int(rollout_steps)
        self.rollout_all_levels = bool(rollout_all_levels)
        self.rollout_lambda = float(rollout_lambda)
        self.target_mode = target_mode
        self.stop_gradient_targets = bool(stop_gradient_targets)
        self.ema_decay = float(ema_decay)
        self.prediction_weight = float(prediction_weight)
        self.vicreg_weight = float(vicreg_weight)
        self.ldad_weight = float(ldad_weight)
        self.ldad_horizon = int(ldad_horizon)
        encoder_args = dict(
            grid_size=grid_size,
            num_colors=num_colors,
            token_dim=token_dim,
            latent_dim=latent_dim,
            num_layers=encoder_layers,
            num_heads=encoder_heads,
            latent_representation=latent_representation,
        )
        self.encoder = ControlledStateEncoder(**encoder_args)
        if target_mode == "ema":
            self.target_encoder: ControlledStateEncoder | None = ControlledStateEncoder(
                **encoder_args
            )
            self.target_encoder.load_state_dict(self.encoder.state_dict())
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad_(False)
        else:
            self.target_encoder = None
        self.level_spans = tuple(hierarchy_stride**level for level in range(hierarchy_depth))
        self.action_encoders = nn.ModuleList(
            [
                ActionChunkEncoder(
                    grid_size=grid_size,
                    chunk_length=span,
                    token_dim=action_token_dim,
                    macro_dim=macro_dim,
                    num_heads=action_heads,
                )
                for span in self.level_spans
            ]
        )
        self.dynamics = nn.ModuleList(
            [LatentDynamics(latent_dim=latent_dim, macro_dim=macro_dim) for _ in self.level_spans]
        )
        if ldad_weight > 0.0:
            ldad_input_dim = latent_dim * (
                grid_size * grid_size if latent_representation == "grid" else 1
            )
            decoder_args = dict(
                input_dim=ldad_input_dim,
                grid_size=grid_size,
                hidden_dim=max(64, 2 * latent_dim),
            )
            if self.ldad_horizon == 1:
                self.ldad_decoder: nn.Module | None = CategoricalLDAD(**decoder_args)
            else:
                self.ldad_decoder = MultiStepCategoricalLDAD(
                    **decoder_args,
                    horizon=self.ldad_horizon,
                )
        else:
            self.ldad_decoder = None

    @property
    def required_horizon(self) -> int:
        rollout_horizon = max(
            span * self.level_rollout_steps(level)
            for level, span in enumerate(self.level_spans)
        )
        return max(rollout_horizon, self.ldad_horizon if self.ldad_decoder else 1)

    def level_rollout_steps(self, level: int) -> int:
        return self.rollout_steps if level == 0 or self.rollout_all_levels else 1

    def encode(self, states: torch.Tensor, *, target: bool = False) -> torch.Tensor:
        if target and self.target_encoder is not None:
            with torch.no_grad():
                return self.target_encoder(states)
        latent = self.encoder(states)
        return latent.detach() if target and self.stop_gradient_targets else latent

    def encode_action_chunk(self, level: int, actions: torch.Tensor) -> torch.Tensor:
        return self.action_encoders[level](actions)

    def predict_from_macro(
        self, level: int, latent: torch.Tensor, macro: torch.Tensor
    ) -> torch.Tensor:
        return self.dynamics[level](latent, macro)

    def predict_chunk(
        self, level: int, latent: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        return self.predict_from_macro(level, latent, self.encode_action_chunk(level, actions))

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        if self.target_encoder is None:
            return
        for target, online in zip(
            self.target_encoder.parameters(), self.encoder.parameters(), strict=True
        ):
            target.mul_(self.ema_decay).add_(online, alpha=1.0 - self.ema_decay)

    def freeze_below_level(self, level: int) -> None:
        if level <= 0:
            return
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        if self.target_encoder is not None:
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad_(False)
        for index in range(min(level, self.hierarchy_depth)):
            for parameter in self.action_encoders[index].parameters():
                parameter.requires_grad_(False)
            for parameter in self.dynamics[index].parameters():
                parameter.requires_grad_(False)

    def forward(self, batch: ControlledObjectBatch) -> ControlledObjectOutput:
        if batch.actions.shape[1] < self.required_horizon:
            raise ValueError(
                f"Model requires {self.required_horizon} actions, got {batch.actions.shape[1]}."
            )
        current = self.encode(batch.states[:, 0])
        level_losses = []
        all_predictions = []
        all_targets = []
        all_weights = []
        for level, span in enumerate(self.level_spans):
            state = current
            predictions = []
            targets = []
            rollout_count = self.level_rollout_steps(level)
            weights = current.new_tensor(
                [self.rollout_lambda**index for index in range(rollout_count)]
            )
            weights = weights / weights.sum()
            for rollout_index in range(rollout_count):
                action_start = rollout_index * span
                action_stop = action_start + span
                state = self.predict_chunk(
                    level, state, batch.actions[:, action_start:action_stop]
                )
                target = self.encode(
                    batch.states[:, action_stop],
                    target=True,
                )
                predictions.append(state)
                targets.append(target)
            predicted = torch.stack(predictions, dim=1)
            target_stack = torch.stack(targets, dim=1)
            reduce_dims = tuple(range(2, predicted.ndim))
            per_step = (predicted - target_stack).square().mean(dim=reduce_dims)
            level_loss = (per_step * weights.view(1, -1)).sum(dim=1).mean()
            level_losses.append(level_loss)
            all_predictions.append(predicted)
            all_targets.append(target_stack)
            all_weights.append(weights)
        prediction_loss = torch.stack(level_losses).mean()

        if self.vicreg_weight > 0.0:
            online_future = self.encoder(batch.states[:, self.required_horizon])
            vicreg_loss = 0.5 * (
                vicreg_regularizer(_vicreg_samples(current))
                + vicreg_regularizer(_vicreg_samples(online_future))
            )
        else:
            vicreg_loss = prediction_loss.detach() * 0.0

        ldad_logits = None
        if self.ldad_decoder is not None:
            endpoint = self.encode(batch.states[:, self.ldad_horizon], target=True)
            ldad_logits = self.ldad_decoder(endpoint - current)
            changed = (
                batch.states[:, 1 : self.ldad_horizon + 1]
                != batch.states[:, : self.ldad_horizon]
            )
            changed = changed.flatten(2).any(dim=2).all(dim=1)
            effective = (
                batch.action_validity[:, : self.ldad_horizon].all(dim=1) & changed
            )
            if bool(effective.any()):
                action_targets = batch.actions[effective, : self.ldad_horizon]
                field_losses = []
                for index, logits in enumerate(ldad_logits):
                    selected = logits[effective]
                    targets = action_targets[..., index]
                    if self.ldad_horizon == 1:
                        targets = targets[:, 0]
                    else:
                        selected = selected.flatten(0, 1)
                        targets = targets.flatten()
                    field_losses.append(F.cross_entropy(selected, targets))
                ldad_loss = sum(field_losses) / len(field_losses)
            else:
                ldad_loss = sum(logits.sum() for logits in ldad_logits) * 0.0
        else:
            ldad_loss = prediction_loss.detach() * 0.0
        return ControlledObjectOutput(
            loss=(
                self.prediction_weight * prediction_loss
                + self.vicreg_weight * vicreg_loss
                + self.ldad_weight * ldad_loss
            ),
            prediction_loss=prediction_loss,
            vicreg_loss=vicreg_loss,
            ldad_loss=ldad_loss,
            level_losses=tuple(level_losses),
            predictions=tuple(all_predictions),
            targets=tuple(all_targets),
            rollout_weights=tuple(all_weights),
            ldad_logits=ldad_logits,
        )


def _vicreg_samples(latent: torch.Tensor) -> torch.Tensor:
    return latent if latent.ndim == 2 else latent.flatten(0, 1)
