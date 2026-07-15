from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from puzzle_jepa.controlled_objects.batching import ControlledObjectBatch
from puzzle_jepa.object_dynamics.losses import (
    covariance_loss,
    sigreg_regularizer,
    variance_loss,
)


RGB_PALETTE = torch.tensor(
    [
        [0.00, 0.00, 0.00],
        [0.90, 0.12, 0.12],
        [0.10, 0.72, 0.24],
        [0.12, 0.34, 0.92],
        [0.95, 0.78, 0.08],
        [0.76, 0.16, 0.82],
        [0.05, 0.76, 0.82],
        [0.95, 0.42, 0.08],
        [0.42, 0.20, 0.88],
        [0.92, 0.92, 0.92],
    ],
    dtype=torch.float32,
)


@dataclass(slots=True)
class ControlledObjectOutput:
    loss: torch.Tensor
    prediction_loss: torch.Tensor
    teacher_forcing_loss: torch.Tensor
    rollout_loss: torch.Tensor
    vicreg_loss: torch.Tensor
    vicreg_variance_loss: torch.Tensor
    vicreg_covariance_loss: torch.Tensor
    sigreg_loss: torch.Tensor
    ldad_loss: torch.Tensor
    cross_level_consistency_loss: torch.Tensor
    level_consistency_losses: tuple[torch.Tensor, ...]
    level_losses: tuple[torch.Tensor, ...]
    predictions: tuple[torch.Tensor, ...]
    teacher_forced_predictions: tuple[torch.Tensor, ...]
    targets: tuple[torch.Tensor, ...]
    rollout_initials: tuple[torch.Tensor, ...]
    rollout_weights: tuple[torch.Tensor, ...]
    ldad_logits: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None


class ControlledStateEncoder(nn.Module):
    """A single-hidden-state MLP over a fixed 16x16 RGB rendering."""

    def __init__(
        self,
        *,
        grid_size: int,
        num_colors: int,
        hidden_dim: int,
    ):
        super().__init__()
        if num_colors > len(RGB_PALETTE):
            raise ValueError(f"RGB palette supports at most {len(RGB_PALETTE)} colors.")
        self.grid_size = int(grid_size)
        self.num_colors = int(num_colors)
        self.hidden_dim = int(hidden_dim)
        self.input_dim = self.grid_size * self.grid_size * 3
        self.register_buffer("palette", RGB_PALETTE[:num_colors].clone())
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.GELU(),
        )

    def render_rgb(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim != 3:
            raise ValueError("ControlledStateEncoder expects [B,H,W].")
        if tuple(states.shape[1:]) != (self.grid_size, self.grid_size):
            raise ValueError(f"Expected {self.grid_size}x{self.grid_size} states.")
        return self.palette[states]

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        rgb = self.render_rgb(states)
        return self.mlp(rgb.flatten(1))


class ActionChunkEncoder(nn.Module):
    """Nonlinear bottleneck over an ordered chunk of rigid pixel-delta commands."""

    def __init__(
        self,
        *,
        grid_size: int,
        num_action_types: int,
        chunk_length: int,
        macro_dim: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.grid_size = int(grid_size)
        self.num_action_types = int(num_action_types)
        self.chunk_length = int(chunk_length)
        self.action_dim = 2 * self.grid_size + self.num_action_types
        self.project = nn.Sequential(
            nn.Linear(self.chunk_length * self.action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, macro_dim),
        )

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim != 3 or actions.shape[1:] != (self.chunk_length, 3):
            raise ValueError(
                f"Expected action chunks [B,{self.chunk_length},3], got {tuple(actions.shape)}."
            )
        encoded = torch.cat(
            (
                F.one_hot(actions[..., 0], self.grid_size),
                F.one_hot(actions[..., 1], self.grid_size),
                F.one_hot(actions[..., 2], self.num_action_types),
            ),
            dim=-1,
        ).to(dtype=self.project[0].weight.dtype)
        return self.project(encoded.flatten(1))


class _GatedDeltaBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("Gated DeltaNet hidden_dim must be divisible by num_heads.")
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = hidden_dim // num_heads
        self.norm = nn.LayerNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.alpha_proj = nn.Linear(hidden_dim, num_heads, bias=True)
        self.beta_proj = nn.Linear(hidden_dim, num_heads, bias=True)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.is_cuda:
            with torch.autocast(device_type="cuda", enabled=False):
                return self._forward_impl(inputs.float()).to(inputs.dtype)
        return self._forward_impl(inputs)

    def _forward_impl(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(inputs)
        shape = (*normalized.shape[:2], self.num_heads, self.head_dim)
        queries = F.normalize(self.q_proj(normalized).reshape(shape).float(), dim=-1).to(
            normalized.dtype
        )
        keys = F.normalize(self.k_proj(normalized).reshape(shape).float(), dim=-1).to(
            normalized.dtype
        )
        values = F.silu(self.v_proj(normalized)).reshape(shape)
        log_decay = F.logsigmoid(self.alpha_proj(normalized))
        beta = torch.sigmoid(self.beta_proj(normalized))
        if inputs.is_cuda:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule

            mixed, _ = chunk_gated_delta_rule(
                q=queries,
                k=keys,
                v=values,
                g=log_decay,
                beta=beta,
                use_qk_l2norm_in_kernel=False,
            )
        else:
            mixed = _reference_gated_delta_rule(
                queries, keys, values, log_decay.exp(), beta
            )
        residual = inputs + self.out_proj(mixed.flatten(2))
        return residual + self.ffn(self.ffn_norm(residual))


def _reference_gated_delta_rule(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    batch, steps, heads, key_dim = keys.shape
    value_dim = values.shape[-1]
    memory = values.new_zeros(batch, heads, key_dim, value_dim)
    outputs = []
    for step in range(steps):
        key = keys[:, step]
        value = values[:, step]
        memory = memory * decay[:, step, :, None, None]
        retrieved = torch.einsum("bhkv,bhk->bhv", memory, key)
        update = torch.einsum("bhk,bhv->bhkv", key, value - retrieved)
        memory = memory + beta[:, step, :, None, None] * update
        outputs.append(torch.einsum("bhkv,bhk->bhv", memory, queries[:, step]))
    return torch.stack(outputs, dim=1)


class CausalLatentPredictor(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        macro_dim: int,
        architecture: str,
        num_layers: int,
        num_heads: int,
        max_context: int,
    ):
        super().__init__()
        if architecture not in {"transformer", "gated_deltanet", "lstm"}:
            raise ValueError(
                "predictor_architecture must be transformer, gated_deltanet, or lstm."
            )
        if latent_dim % num_heads:
            raise ValueError("latent_dim must be divisible by predictor_heads.")
        self.architecture = architecture
        self.max_context = int(max_context)
        self.state_projection = nn.Linear(latent_dim, latent_dim)
        self.macro_projection = nn.Linear(macro_dim, latent_dim)
        self.position = nn.Embedding(2 * max_context, latent_dim)
        if architecture == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=num_heads,
                dim_feedforward=4 * latent_dim,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.sequence_model: nn.Module = nn.TransformerEncoder(
                layer, num_layers=num_layers
            )
        elif architecture == "gated_deltanet":
            self.sequence_model = nn.ModuleList(
                [_GatedDeltaBlock(latent_dim, num_heads) for _ in range(num_layers)]
            )
        else:
            self.sequence_model = nn.LSTM(
                input_size=latent_dim,
                hidden_size=latent_dim,
                num_layers=num_layers,
                batch_first=True,
            )
        self.output = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 2 * latent_dim),
            nn.GELU(),
            nn.Linear(2 * latent_dim, latent_dim),
        )

    def forward(self, states: torch.Tensor, macros: torch.Tensor) -> torch.Tensor:
        if states.shape[:2] != macros.shape[:2]:
            raise ValueError("Causal predictor states and macro-actions must align.")
        steps = states.shape[1]
        if steps > self.max_context:
            raise ValueError(f"Predictor context {steps} exceeds {self.max_context}.")
        state_tokens = self.state_projection(states)
        macro_tokens = self.macro_projection(macros)
        hidden = torch.stack((state_tokens, macro_tokens), dim=2).flatten(1, 2)
        token_count = hidden.shape[1]
        positions = torch.arange(token_count, device=states.device).view(1, -1)
        hidden = hidden + self.position(positions)
        if self.architecture == "transformer":
            causal_mask = torch.ones(
                token_count, token_count, dtype=torch.bool, device=states.device
            ).triu(1)
            hidden = self.sequence_model(hidden, mask=causal_mask)
        elif self.architecture == "gated_deltanet":
            for block in self.sequence_model:
                hidden = block(hidden)
        else:
            hidden, _ = self.sequence_model(hidden)
        return states + self.output(hidden[:, 1::2])

    def rollout(self, initial: torch.Tensor, macros: torch.Tensor) -> torch.Tensor:
        predicted = []
        for step in range(macros.shape[1]):
            history = torch.stack([initial, *predicted], dim=1)
            next_state = self(history, macros[:, : step + 1])[:, -1]
            predicted.append(next_state)
        return torch.stack(predicted, dim=1)


class CategoricalLDAD(nn.Module):
    def __init__(self, *, input_dim: int, grid_size: int, num_action_types: int):
        super().__init__()
        hidden_dim = max(64, 2 * input_dim)
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.row = nn.Linear(hidden_dim, grid_size)
        self.col = nn.Linear(hidden_dim, grid_size)
        self.action_type = nn.Linear(hidden_dim, num_action_types)

    def forward(
        self, delta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.trunk(delta)
        return self.row(hidden), self.col(hidden), self.action_type(hidden)


class ControlledObjectJEPA(nn.Module):
    def __init__(
        self,
        *,
        grid_size: int = 16,
        num_colors: int = 10,
        hidden_dim: int | None = None,
        token_dim: int = 256,
        latent_dim: int = 256,
        encoder_layers: int = 1,
        encoder_heads: int = 1,
        latent_representation: str = "cls",
        level_spans: tuple[int, ...] | list[int] | None = None,
        hierarchy_depth: int = 1,
        hierarchy_stride: int = 4,
        macro_dim: int = 16,
        action_token_dim: int = 32,
        action_heads: int = 4,
        predictor_architecture: str = "transformer",
        predictor_layers: int = 2,
        predictor_heads: int = 4,
        predictor_max_context: int = 64,
        rollout_steps: int = 1,
        rollout_steps_by_level: tuple[int, ...] | list[int] | None = None,
        rollout_all_levels: bool = True,
        rollout_lambda: float = 1.0,
        dense_trajectory_training: bool = False,
        cross_level_consistency_weight: float = 0.0,
        teacher_forcing_weight: float = 1.0,
        autonomous_rollout_weight: float = 1.0,
        target_mode: str = "ema",
        stop_gradient_targets: bool = True,
        ema_decay: float = 0.99,
        prediction_weight: float = 1.0,
        vicreg_weight: float = 0.05,
        vicreg_variance_weight: float = 1.0,
        vicreg_covariance_weight: float = 0.04,
        vicreg_adjust_cov: bool = False,
        sigreg_weight: float = 0.0,
        sigreg_num_slices: int = 1024,
        sigreg_t_max: float = 3.0,
        sigreg_num_points: int = 17,
        ldad_weight: float = 0.0,
        ldad_horizon: int = 1,
    ):
        super().__init__()
        del encoder_layers, encoder_heads, action_token_dim, action_heads
        if latent_representation != "cls":
            raise ValueError("Controlled experiments support only one learned latent state.")
        if rollout_steps < 1:
            raise ValueError("rollout_steps must be positive.")
        if not (0.0 < rollout_lambda <= 1.0):
            raise ValueError("rollout_lambda must lie in (0,1].")
        if target_mode not in {"shared", "ema"}:
            raise ValueError("target_mode must be 'shared' or 'ema'.")
        if target_mode == "ema" and not stop_gradient_targets:
            raise ValueError("EMA targets are necessarily stop-gradient targets.")
        if min(
            vicreg_weight,
            vicreg_variance_weight,
            vicreg_covariance_weight,
            sigreg_weight,
        ) < 0.0:
            raise ValueError("Representation regularizer weights must be non-negative.")
        if sigreg_num_slices < 1 or sigreg_num_points < 2 or sigreg_t_max <= 0.0:
            raise ValueError("SIGReg sampling parameters must be positive.")
        resolved_spans = (
            tuple(int(span) for span in level_spans)
            if level_spans is not None
            else tuple(hierarchy_stride**level for level in range(hierarchy_depth))
        )
        if not resolved_spans or resolved_spans[0] != 1:
            raise ValueError("level_spans must start at primitive span 1.")
        if any(left >= right for left, right in zip(resolved_spans, resolved_spans[1:])):
            raise ValueError("level_spans must be strictly increasing.")
        resolved_rollouts = (
            tuple(int(steps) for steps in rollout_steps_by_level)
            if rollout_steps_by_level is not None
            else None
        )
        if resolved_rollouts is not None and (
            len(resolved_rollouts) != len(resolved_spans)
            or any(steps < 1 for steps in resolved_rollouts)
        ):
            raise ValueError(
                "rollout_steps_by_level must provide one positive value per level."
            )
        if cross_level_consistency_weight < 0.0:
            raise ValueError("cross_level_consistency_weight must be non-negative.")
        state_dim = int(hidden_dim if hidden_dim is not None else latent_dim)
        self.grid_size = int(grid_size)
        self.num_colors = int(num_colors)
        self.token_dim = state_dim
        self.latent_dim = state_dim
        self.latent_representation = "cls"
        self.level_spans = resolved_spans
        self.hierarchy_depth = len(resolved_spans)
        self.hierarchy_stride = resolved_spans[1] if len(resolved_spans) > 1 else 1
        self.rollout_steps = int(rollout_steps)
        self.rollout_steps_by_level = resolved_rollouts
        self.rollout_all_levels = bool(rollout_all_levels)
        self.rollout_lambda = float(rollout_lambda)
        self.dense_trajectory_training = bool(dense_trajectory_training)
        self.cross_level_consistency_weight = float(cross_level_consistency_weight)
        self.teacher_forcing_weight = float(teacher_forcing_weight)
        self.autonomous_rollout_weight = float(autonomous_rollout_weight)
        self.target_mode = target_mode
        self.stop_gradient_targets = bool(stop_gradient_targets)
        self.ema_decay = float(ema_decay)
        self.prediction_weight = float(prediction_weight)
        self.vicreg_weight = float(vicreg_weight)
        self.vicreg_variance_weight = float(vicreg_variance_weight)
        self.vicreg_covariance_weight = float(vicreg_covariance_weight)
        self.vicreg_adjust_cov = bool(vicreg_adjust_cov)
        self.sigreg_weight = float(sigreg_weight)
        self.sigreg_num_slices = int(sigreg_num_slices)
        self.sigreg_t_max = float(sigreg_t_max)
        self.sigreg_num_points = int(sigreg_num_points)
        self.ldad_weight = float(ldad_weight)
        self.ldad_horizon = int(ldad_horizon)
        self.predictor_architecture = predictor_architecture
        self.train_from_level = 0
        self.encoder = ControlledStateEncoder(
            grid_size=grid_size,
            num_colors=num_colors,
            hidden_dim=state_dim,
        )
        if target_mode == "ema":
            self.target_encoder: ControlledStateEncoder | None = ControlledStateEncoder(
                grid_size=grid_size,
                num_colors=num_colors,
                hidden_dim=state_dim,
            )
            self.target_encoder.load_state_dict(self.encoder.state_dict())
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad_(False)
        else:
            self.target_encoder = None
        self.action_encoders = nn.ModuleList(
            [
                ActionChunkEncoder(
                    grid_size=grid_size,
                    num_action_types=7,
                    chunk_length=span,
                    macro_dim=macro_dim,
                )
                for span in self.level_spans
            ]
        )
        self.dynamics = nn.ModuleList(
            [
                CausalLatentPredictor(
                    latent_dim=state_dim,
                    macro_dim=macro_dim,
                    architecture=predictor_architecture,
                    num_layers=predictor_layers,
                    num_heads=predictor_heads,
                    max_context=predictor_max_context,
                )
                for _ in self.level_spans
            ]
        )
        self.ldad_decoder = (
            CategoricalLDAD(
                input_dim=state_dim,
                grid_size=grid_size,
                num_action_types=7,
            )
            if ldad_weight > 0.0
            else None
        )

    @property
    def required_horizon(self) -> int:
        if self.dense_trajectory_training:
            return max(self.level_spans)
        return max(
            span * self.level_rollout_steps(level)
            for level, span in enumerate(self.level_spans)
        )

    def level_rollout_steps(self, level: int) -> int:
        if self.rollout_steps_by_level is not None:
            return self.rollout_steps_by_level[level]
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
        return self.dynamics[level](latent[:, None], macro[:, None])[:, 0]

    def predict_chunk(
        self, level: int, latent: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        return self.predict_from_macro(level, latent, self.encode_action_chunk(level, actions))

    def rollout_level(
        self, level: int, initial: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        span = self.level_spans[level]
        if actions.ndim != 3 or actions.shape[1] % span:
            raise ValueError("Rollout actions must contain complete level chunks.")
        steps = actions.shape[1] // span
        flat = actions.reshape(len(actions) * steps, span, 3)
        macros = self.encode_action_chunk(level, flat).reshape(len(actions), steps, -1)
        return self.dynamics[level].rollout(initial, macros)

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        if self.target_encoder is None:
            return
        for target, online in zip(
            self.target_encoder.parameters(), self.encoder.parameters(), strict=True
        ):
            target.mul_(self.ema_decay).add_(online, alpha=1.0 - self.ema_decay)

    def freeze_below_level(self, level: int) -> None:
        if not (0 <= level < self.hierarchy_depth):
            raise ValueError(
                f"train_from_level must be in [0,{self.hierarchy_depth - 1}]."
            )
        self.train_from_level = int(level)
        if level == 0:
            return
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        if self.target_encoder is not None:
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad_(False)
        for index in range(level):
            for parameter in self.action_encoders[index].parameters():
                parameter.requires_grad_(False)
            for parameter in self.dynamics[index].parameters():
                parameter.requires_grad_(False)

    def forward(self, batch: ControlledObjectBatch) -> ControlledObjectOutput:
        if self.dense_trajectory_training:
            return self._forward_dense_trajectory(batch)
        return self._forward_sparse(batch)

    def _forward_sparse(self, batch: ControlledObjectBatch) -> ControlledObjectOutput:
        if batch.actions.shape[1] < self.required_horizon:
            raise ValueError(
                f"Model requires {self.required_horizon} actions, got {batch.actions.shape[1]}."
            )
        current = self.encode(batch.states[:, 0])
        level_losses = []
        teacher_losses = []
        rollout_losses = []
        all_predictions = []
        all_teacher_predictions = []
        all_targets = []
        all_weights = []
        for level, span in enumerate(self.level_spans):
            rollout_count = self.level_rollout_steps(level)
            endpoint_indices = torch.arange(
                span, span * rollout_count + 1, span, device=batch.states.device
            )
            endpoint_states = batch.states.index_select(1, endpoint_indices)
            flat_endpoints = endpoint_states.flatten(0, 1)
            targets = self.encode(flat_endpoints, target=True).reshape(
                len(batch.states), rollout_count, self.latent_dim
            )
            action_window = batch.actions[:, : span * rollout_count]
            flat_actions = action_window.reshape(
                len(batch.actions) * rollout_count, span, 3
            )
            macros = self.encode_action_chunk(level, flat_actions).reshape(
                len(batch.actions), rollout_count, -1
            )
            teacher_inputs = torch.cat((current[:, None], targets[:, :-1]), dim=1)
            teacher_predictions = self.dynamics[level](teacher_inputs, macros)
            predictions = self.dynamics[level].rollout(current, macros)
            weights = current.new_tensor(
                [self.rollout_lambda**index for index in range(rollout_count)]
            )
            weights = weights / weights.sum()
            teacher_per_step = (teacher_predictions - targets).square().mean(dim=-1)
            rollout_per_step = (predictions - targets).square().mean(dim=-1)
            teacher_loss = (teacher_per_step * weights).sum(dim=1).mean()
            rollout_loss = (rollout_per_step * weights).sum(dim=1).mean()
            level_loss = (
                self.teacher_forcing_weight * teacher_loss
                + self.autonomous_rollout_weight * rollout_loss
            ) / max(
                1.0, self.teacher_forcing_weight + self.autonomous_rollout_weight
            )
            level_losses.append(level_loss)
            teacher_losses.append(teacher_loss)
            rollout_losses.append(rollout_loss)
            all_predictions.append(predictions)
            all_teacher_predictions.append(teacher_predictions)
            all_targets.append(targets)
            all_weights.append(weights)
        selected = slice(self.train_from_level, None)
        prediction_loss = torch.stack(level_losses[selected]).mean()
        teacher_forcing_loss = torch.stack(teacher_losses[selected]).mean()
        autonomous_rollout_loss = torch.stack(rollout_losses[selected]).mean()

        if self.vicreg_weight > 0.0:
            vicreg_variance = variance_loss(current)
            vicreg_covariance = covariance_loss(
                current, adjust_cov=self.vicreg_adjust_cov
            )
            vicreg_loss = (
                self.vicreg_variance_weight * vicreg_variance
                + self.vicreg_covariance_weight * vicreg_covariance
            )
        else:
            vicreg_loss = prediction_loss.detach() * 0.0
            vicreg_variance = vicreg_loss
            vicreg_covariance = vicreg_loss

        if self.sigreg_weight > 0.0:
            sigreg_loss = sigreg_regularizer(
                current,
                num_slices=self.sigreg_num_slices,
                t_max=self.sigreg_t_max,
                num_points=self.sigreg_num_points,
            )
        else:
            sigreg_loss = prediction_loss.detach() * 0.0

        ldad_logits = None
        if self.ldad_decoder is not None:
            endpoint = self.encode(batch.states[:, 1], target=True)
            ldad_logits = self.ldad_decoder(endpoint - current)
            field_losses = [
                F.cross_entropy(logits, batch.actions[:, 0, index])
                for index, logits in enumerate(ldad_logits)
            ]
            ldad_loss = sum(field_losses) / len(field_losses)
        else:
            ldad_loss = prediction_loss.detach() * 0.0
        cross_level_consistency_loss = prediction_loss.detach() * 0.0
        return ControlledObjectOutput(
            loss=(
                self.prediction_weight * prediction_loss
                + self.vicreg_weight * vicreg_loss
                + self.sigreg_weight * sigreg_loss
                + self.ldad_weight * ldad_loss
            ),
            prediction_loss=prediction_loss,
            teacher_forcing_loss=teacher_forcing_loss,
            rollout_loss=autonomous_rollout_loss,
            vicreg_loss=vicreg_loss,
            vicreg_variance_loss=vicreg_variance,
            vicreg_covariance_loss=vicreg_covariance,
            sigreg_loss=sigreg_loss,
            ldad_loss=ldad_loss,
            cross_level_consistency_loss=cross_level_consistency_loss,
            level_consistency_losses=(),
            level_losses=tuple(level_losses),
            predictions=tuple(all_predictions),
            teacher_forced_predictions=tuple(all_teacher_predictions),
            targets=tuple(all_targets),
            rollout_initials=tuple(current for _ in self.level_spans),
            rollout_weights=tuple(all_weights),
            ldad_logits=ldad_logits,
        )

    def _forward_dense_trajectory(
        self, batch: ControlledObjectBatch
    ) -> ControlledObjectOutput:
        horizon = int(batch.actions.shape[1])
        if horizon < self.required_horizon:
            raise ValueError(
                f"Model requires at least {self.required_horizon} actions, got {horizon}."
            )
        batch_size = len(batch.states)
        online_states = self.encode(batch.states.flatten(0, 1)).reshape(
            batch_size, horizon + 1, self.latent_dim
        )
        target_states = self.encode(
            batch.states.flatten(0, 1), target=True
        ).reshape(batch_size, horizon + 1, self.latent_dim)

        level_losses = []
        teacher_losses = []
        rollout_losses = []
        all_predictions = []
        all_teacher_predictions = []
        all_targets = []
        all_rollout_initials = []
        all_weights = []
        level_online_states = []
        level_macros = []
        for level, span in enumerate(self.level_spans):
            segment_count = horizon // span
            endpoint_indices = torch.arange(
                0,
                (segment_count + 1) * span,
                span,
                device=batch.states.device,
            )
            current_endpoints = online_states.index_select(1, endpoint_indices[:-1])
            target_endpoints = target_states.index_select(1, endpoint_indices[1:])
            action_segments = batch.actions[:, : segment_count * span].reshape(
                batch_size * segment_count, span, 3
            )
            macros = self.encode_action_chunk(level, action_segments).reshape(
                batch_size, segment_count, -1
            )
            teacher_predictions = self.dynamics[level](current_endpoints, macros)
            teacher_loss = F.mse_loss(teacher_predictions, target_endpoints)

            rollout_count = min(self.level_rollout_steps(level), segment_count)
            anchor_count = segment_count - rollout_count + 1
            macro_windows = torch.stack(
                [
                    macros[:, offset : offset + anchor_count]
                    for offset in range(rollout_count)
                ],
                dim=2,
            )
            target_windows = torch.stack(
                [
                    target_endpoints[:, offset : offset + anchor_count]
                    for offset in range(rollout_count)
                ],
                dim=2,
            )
            rollout_initial = current_endpoints[:, :anchor_count].reshape(
                batch_size * anchor_count, self.latent_dim
            )
            flat_macro_windows = macro_windows.reshape(
                batch_size * anchor_count, rollout_count, -1
            )
            flat_targets = target_windows.reshape(
                batch_size * anchor_count, rollout_count, self.latent_dim
            )
            predictions = self.dynamics[level].rollout(
                rollout_initial, flat_macro_windows
            )
            weights = predictions.new_tensor(
                [self.rollout_lambda**index for index in range(rollout_count)]
            )
            weights = weights / weights.sum()
            rollout_per_step = (predictions - flat_targets).square().mean(dim=-1)
            rollout_loss = (rollout_per_step * weights).sum(dim=1).mean()
            level_loss = (
                self.teacher_forcing_weight * teacher_loss
                + self.autonomous_rollout_weight * rollout_loss
            ) / max(
                1.0, self.teacher_forcing_weight + self.autonomous_rollout_weight
            )

            level_losses.append(level_loss)
            teacher_losses.append(teacher_loss)
            rollout_losses.append(rollout_loss)
            all_predictions.append(predictions)
            all_teacher_predictions.append(teacher_predictions)
            all_targets.append(flat_targets)
            all_rollout_initials.append(rollout_initial)
            all_weights.append(weights)
            level_online_states.append(current_endpoints)
            level_macros.append(macros)

        selected = slice(self.train_from_level, None)
        prediction_loss = torch.stack(level_losses[selected]).mean()
        teacher_forcing_loss = torch.stack(teacher_losses[selected]).mean()
        autonomous_rollout_loss = torch.stack(rollout_losses[selected]).mean()

        level_consistency_losses = []
        for level in range(1, self.hierarchy_depth):
            ratio, remainder = divmod(
                self.level_spans[level], self.level_spans[level - 1]
            )
            if remainder:
                raise ValueError("Adjacent hierarchy spans must divide exactly.")
            high_macros = level_macros[level]
            high_initial = level_online_states[level]
            high_segments = high_macros.shape[1]
            lower_macros = level_macros[level - 1][
                :, : high_segments * ratio
            ].reshape(batch_size * high_segments, ratio, -1)
            flat_initial = high_initial.reshape(
                batch_size * high_segments, self.latent_dim
            )
            high_prediction = self.predict_from_macro(
                level,
                flat_initial,
                high_macros.reshape(batch_size * high_segments, -1),
            )
            lower_prediction = self.dynamics[level - 1].rollout(
                flat_initial, lower_macros
            )[:, -1]
            level_consistency_losses.append(
                F.mse_loss(high_prediction, lower_prediction)
            )
        cross_level_consistency_loss = (
            torch.stack(level_consistency_losses).mean()
            if level_consistency_losses
            else prediction_loss.detach() * 0.0
        )

        regularization_latents = online_states[:, :-1].flatten(0, 1)
        if self.vicreg_weight > 0.0:
            vicreg_variance = variance_loss(regularization_latents)
            vicreg_covariance = covariance_loss(
                regularization_latents, adjust_cov=self.vicreg_adjust_cov
            )
            vicreg_loss = (
                self.vicreg_variance_weight * vicreg_variance
                + self.vicreg_covariance_weight * vicreg_covariance
            )
        else:
            vicreg_loss = prediction_loss.detach() * 0.0
            vicreg_variance = vicreg_loss
            vicreg_covariance = vicreg_loss

        if self.sigreg_weight > 0.0:
            sigreg_loss = sigreg_regularizer(
                regularization_latents,
                num_slices=self.sigreg_num_slices,
                t_max=self.sigreg_t_max,
                num_points=self.sigreg_num_points,
            )
        else:
            sigreg_loss = prediction_loss.detach() * 0.0

        ldad_logits = None
        if self.ldad_decoder is not None:
            current = online_states[:, 0]
            endpoint = target_states[:, 1]
            ldad_logits = self.ldad_decoder(endpoint - current)
            ldad_loss = sum(
                F.cross_entropy(logits, batch.actions[:, 0, index])
                for index, logits in enumerate(ldad_logits)
            ) / len(ldad_logits)
        else:
            ldad_loss = prediction_loss.detach() * 0.0

        total_loss = (
            self.prediction_weight * prediction_loss
            + self.vicreg_weight * vicreg_loss
            + self.sigreg_weight * sigreg_loss
            + self.ldad_weight * ldad_loss
            + self.cross_level_consistency_weight * cross_level_consistency_loss
        )
        return ControlledObjectOutput(
            loss=total_loss,
            prediction_loss=prediction_loss,
            teacher_forcing_loss=teacher_forcing_loss,
            rollout_loss=autonomous_rollout_loss,
            vicreg_loss=vicreg_loss,
            vicreg_variance_loss=vicreg_variance,
            vicreg_covariance_loss=vicreg_covariance,
            sigreg_loss=sigreg_loss,
            ldad_loss=ldad_loss,
            cross_level_consistency_loss=cross_level_consistency_loss,
            level_consistency_losses=tuple(level_consistency_losses),
            level_losses=tuple(level_losses),
            predictions=tuple(all_predictions),
            teacher_forced_predictions=tuple(all_teacher_predictions),
            targets=tuple(all_targets),
            rollout_initials=tuple(all_rollout_initials),
            rollout_weights=tuple(all_weights),
            ldad_logits=ldad_logits,
        )
