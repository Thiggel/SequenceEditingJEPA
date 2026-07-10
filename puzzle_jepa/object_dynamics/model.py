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
    reconstruction_loss: torch.Tensor
    predicted: torch.Tensor
    targets: torch.Tensor


@dataclass(frozen=True, slots=True)
class HierarchicalPlan:
    high_level_indices: torch.Tensor
    low_level_indices: torch.Tensor
    high_level_scores: torch.Tensor
    low_level_scores: torch.Tensor
    subgoals: torch.Tensor


@dataclass(frozen=True, slots=True)
class ContinuousMacroPlan:
    macro_actions: torch.Tensor
    predicted_states: torch.Tensor
    goal_scores: torch.Tensor


@dataclass(frozen=True, slots=True)
class PrimitiveActionPlan:
    actions: torch.Tensor
    predicted_endpoints: torch.Tensor
    subgoal_scores: torch.Tensor


class GridStateEncoder(nn.Module):
    def __init__(
        self,
        *,
        grid_size: int,
        num_colors: int,
        d_model: int,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.0,
        latent_representation: str = "cls",
    ):
        super().__init__()
        self.grid_size = int(grid_size)
        self.num_colors = int(num_colors)
        self.d_model = int(d_model)
        if latent_representation not in {"cls", "grid"}:
            raise ValueError("latent_representation must be 'cls' or 'grid'.")
        self.latent_representation = str(latent_representation)
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
        encoded = self.encoder(self._input_tokens(values))
        if self.latent_representation == "cls":
            return self.norm(encoded[:, 0])
        return self.norm(encoded[:, 1:])

    def cls_attention(self, values: torch.Tensor) -> torch.Tensor:
        encoded = self._input_tokens(values)
        final_attention = None
        for layer in self.encoder.layers:
            if not layer.norm_first:
                raise RuntimeError("CLS attention extraction requires norm-first transformer layers.")
            normalized = layer.norm1(encoded)
            attention_output, final_attention = layer.self_attn(
                normalized,
                normalized,
                normalized,
                need_weights=True,
                average_attn_weights=False,
            )
            encoded = encoded + layer.dropout1(attention_output)
            feedforward_input = layer.norm2(encoded)
            feedforward = layer.linear2(layer.dropout(layer.activation(layer.linear1(feedforward_input))))
            encoded = encoded + layer.dropout2(feedforward)
        if final_attention is None:
            raise RuntimeError("CLS attention requires at least one transformer layer.")
        cell_attention = final_attention[:, :, 0, 1:]
        cell_attention = cell_attention / cell_attention.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        return cell_attention.reshape(values.shape[0], -1, self.grid_size, self.grid_size)

    def _input_tokens(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError(f"GridStateEncoder expects [B,H,W], got {tuple(values.shape)}.")
        batch, height, width = values.shape
        if height != self.grid_size or width != self.grid_size:
            raise ValueError(f"Expected {self.grid_size}x{self.grid_size}, got {(height, width)}.")
        rows = torch.arange(height, device=values.device).view(1, height, 1).expand(batch, height, width)
        cols = torch.arange(width, device=values.device).view(1, 1, width).expand(batch, height, width)
        tokens = self.color(values.clamp(0, self.num_colors - 1)) + self.row(rows) + self.col(cols)
        tokens = tokens.reshape(batch, height * width, self.d_model)
        cls = self.cls.expand(batch, -1, -1)
        return torch.cat([cls, tokens], dim=1)


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


class MacroActionEncoder(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        macro_action_dim: int,
        max_horizon: int,
        num_heads: int,
        num_layers: int,
    ):
        super().__init__()
        self.max_horizon = int(max_horizon)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, std=0.02)
        self.position = nn.Embedding(self.max_horizon + 1, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.to_macro = nn.Sequential(
            nn.Linear(d_model, macro_action_dim),
            nn.LayerNorm(macro_action_dim),
        )
        self.from_macro = nn.Linear(macro_action_dim, d_model)

    def forward(self, action_embeddings: torch.Tensor) -> torch.Tensor:
        if action_embeddings.ndim != 3:
            raise ValueError(f"Macro actions must be [B,K,D], got {tuple(action_embeddings.shape)}.")
        horizon = action_embeddings.shape[1]
        if horizon <= 0 or horizon > self.max_horizon:
            raise ValueError(f"Macro horizon must lie in [1, {self.max_horizon}], got {horizon}.")
        tokens = torch.cat([self.cls.expand(action_embeddings.shape[0], -1, -1), action_embeddings], dim=1)
        positions = torch.arange(horizon + 1, device=action_embeddings.device)
        encoded = self.encoder(tokens + self.position(positions).unsqueeze(0))
        return self.to_macro(self.norm(encoded[:, 0]))

    def project(self, macro_action: torch.Tensor) -> torch.Tensor:
        return self.from_macro(macro_action)


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
        target_mode: str | None = None,
        ema_decay: float = 0.99,
        ldad_weight: float = 0.0,
        regularizer: str = "none",
        regularizer_weight: float = 0.0,
        latent_representation: str = "cls",
        hierarchy_planning: bool = False,
        hierarchy_rollout_steps: int = 1,
        macro_encoder_layers: int = 1,
        macro_action_dim: int | None = None,
        dynamics_weight: float = 1.0,
        reconstruction_weight: float = 0.0,
        rollout_weight: float = 1.0,
        hierarchy_weight: float = 1.0,
    ):
        super().__init__()
        self.rollout_horizon = int(rollout_horizon)
        self.hierarchy_horizon = int(hierarchy_horizon)
        self.target_mode = target_mode or ("ema" if target_ema else "stop_gradient")
        if self.target_mode not in {"stop_gradient", "shared", "ema"}:
            raise ValueError(f"Unknown target_mode {self.target_mode!r}.")
        if target_ema and self.target_mode != "ema":
            raise ValueError("target_ema=True requires target_mode='ema'.")
        self.target_ema = self.target_mode == "ema"
        self.ema_decay = float(ema_decay)
        self.ldad_weight = float(ldad_weight)
        self.regularizer = str(regularizer)
        self.regularizer_weight = float(regularizer_weight)
        self.hierarchy_planning = bool(hierarchy_planning)
        self.hierarchy_rollout_steps = int(hierarchy_rollout_steps)
        requested_macro_dim = min(16, max(1, d_model // 2)) if macro_action_dim is None else int(macro_action_dim)
        self.macro_action_dim = requested_macro_dim if self.hierarchy_planning else int(d_model)
        self.dynamics_weight = float(dynamics_weight)
        self.reconstruction_weight = float(reconstruction_weight)
        self.rollout_weight = float(rollout_weight)
        self.hierarchy_weight = float(hierarchy_weight)
        self.grid_size = int(grid_size)
        self.num_colors = int(num_colors)
        self.d_model = int(d_model)
        if self.hierarchy_planning and self.hierarchy_horizon <= 1:
            raise ValueError("hierarchy_planning requires hierarchy_horizon > 1.")
        if self.hierarchy_rollout_steps <= 0:
            raise ValueError("hierarchy_rollout_steps must be positive.")
        if self.hierarchy_planning and not (0 < self.macro_action_dim < d_model):
            raise ValueError("hierarchy_planning requires 0 < macro_action_dim < d_model.")
        if min(
            self.dynamics_weight,
            self.reconstruction_weight,
            self.rollout_weight,
            self.hierarchy_weight,
        ) < 0.0:
            raise ValueError("Objective weights must be non-negative.")
        if latent_representation not in {"cls", "grid"}:
            raise ValueError("latent_representation must be 'cls' or 'grid'.")
        self.latent_representation = str(latent_representation)
        self.encoder = GridStateEncoder(
            grid_size=grid_size,
            num_colors=num_colors,
            d_model=d_model,
            num_layers=encoder_layers,
            num_heads=encoder_heads,
            latent_representation=self.latent_representation,
        )
        self.target_encoder = GridStateEncoder(
            grid_size=grid_size,
            num_colors=num_colors,
            d_model=d_model,
            num_layers=encoder_layers,
            num_heads=encoder_heads,
            latent_representation=self.latent_representation,
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
        if self.hierarchy_planning:
            self.chunk_encoder = MacroActionEncoder(
                d_model=d_model,
                macro_action_dim=self.macro_action_dim,
                max_horizon=self.hierarchy_horizon,
                num_heads=encoder_heads,
                num_layers=int(macro_encoder_layers),
            )
        else:
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
        if self.latent_representation == "grid":
            self.delta_pool = nn.Linear(d_model, 1)
        if self.reconstruction_weight > 0.0:
            if self.latent_representation == "grid":
                self.reconstruction_decoder = nn.Linear(d_model, num_colors)
            else:
                self.reconstruction_decoder = nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, 2 * d_model),
                    nn.GELU(),
                    nn.Linear(2 * d_model, grid_size * grid_size * num_colors),
                )

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        if self.target_mode != "ema":
            return
        for target, online in zip(self.target_encoder.parameters(), self.encoder.parameters(), strict=True):
            target.mul_(self.ema_decay).add_(online, alpha=1.0 - self.ema_decay)

    def encode(self, states: torch.Tensor, *, target: bool = False) -> torch.Tensor:
        encoder = self.target_encoder if target and self.target_mode == "ema" else self.encoder
        return encoder(states)

    def attention_maps(self, states: torch.Tensor) -> torch.Tensor:
        return self.encoder.cls_attention(states)

    @property
    def training_horizon(self) -> int:
        hierarchy_horizon = self.hierarchy_horizon * self.hierarchy_rollout_steps
        return max(1, self.rollout_horizon, hierarchy_horizon)

    def predict_latents(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        state = self.encoder(states)
        action_embeddings = self.actions(actions)
        predictions = []
        for step in range(action_embeddings.shape[1]):
            state = self._predict_step(state, action_embeddings[:, step])
            predictions.append(state)
        return torch.stack(predictions, dim=1)

    def pool_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return latents.mean(dim=-2) if self.latent_representation == "grid" else latents

    def delta_probe_features(self, delta: torch.Tensor) -> torch.Tensor:
        return delta.flatten(1) if self.latent_representation == "grid" else delta

    def encode_action_chunk(self, actions: torch.Tensor) -> torch.Tensor:
        action_embeddings = self.actions(actions)
        return self._encode_action_embeddings(action_embeddings)

    def predict_high_level_latents(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        self._require_hierarchy_planning()
        state = self.encoder(states)
        if actions.shape[1] % self.hierarchy_horizon != 0:
            raise ValueError("High-level prediction actions must contain complete macro chunks.")
        for start in range(0, actions.shape[1], self.hierarchy_horizon):
            chunk = self.encode_action_chunk(actions[:, start : start + self.hierarchy_horizon])
            state = self._predict_hierarchy(state, chunk)
        return state

    def rollout_high_level(self, states: torch.Tensor, action_chunks: torch.Tensor) -> torch.Tensor:
        self._require_hierarchy_planning()
        if action_chunks.ndim != 4 or action_chunks.shape[0] != states.shape[0]:
            raise ValueError("High-level rollout expects action chunks [B,L,K,4].")
        state = self.encoder(states)
        predictions = []
        for level_step in range(action_chunks.shape[1]):
            chunk = self.encode_action_chunk(action_chunks[:, level_step])
            state = self._predict_hierarchy(state, chunk)
            predictions.append(state)
        return torch.stack(predictions, dim=1)

    @torch.no_grad()
    def plan_macro_actions(
        self,
        states: torch.Tensor,
        candidate_action_chunks: torch.Tensor,
        goal_states: torch.Tensor,
    ) -> HierarchicalPlan:
        self._require_hierarchy_planning()
        if candidate_action_chunks.ndim != 4 or candidate_action_chunks.shape[0] != states.shape[0]:
            raise ValueError("Macro candidates must have shape [B,C,K,4].")
        batch, candidates, horizon = candidate_action_chunks.shape[:3]
        if horizon != self.hierarchy_horizon:
            raise ValueError(f"Expected macro chunks of length {self.hierarchy_horizon}, got {horizon}.")
        state = self.encoder(states)
        goal = self.encoder(goal_states)
        expanded_state = state[:, None].expand(batch, candidates, *state.shape[1:]).reshape(
            batch * candidates, *state.shape[1:]
        )
        flat_actions = candidate_action_chunks.reshape(batch * candidates, horizon, 4)
        macro = self.encode_action_chunk(flat_actions)
        high_endpoints = self._predict_hierarchy(expanded_state, macro).reshape(
            batch, candidates, *state.shape[1:]
        )
        high_scores = (high_endpoints - goal[:, None]).square().flatten(2).mean(dim=-1)
        high_indices = high_scores.argmin(dim=1)
        batch_indices = torch.arange(batch, device=states.device)
        subgoals = high_endpoints[batch_indices, high_indices]

        low_endpoints = self._rollout_from_latent(expanded_state, flat_actions)[:, -1].reshape(
            batch, candidates, *state.shape[1:]
        )
        low_scores = (low_endpoints - subgoals[:, None]).square().flatten(2).mean(dim=-1)
        low_indices = low_scores.argmin(dim=1)
        return HierarchicalPlan(
            high_level_indices=high_indices,
            low_level_indices=low_indices,
            high_level_scores=high_scores,
            low_level_scores=low_scores,
            subgoals=subgoals,
        )

    @torch.no_grad()
    def optimize_macro_actions(
        self,
        states: torch.Tensor,
        goal_states: torch.Tensor,
        *,
        high_level_steps: int = 1,
        num_samples: int = 64,
        num_elites: int = 8,
        num_iterations: int = 4,
        momentum: float = 0.1,
    ) -> ContinuousMacroPlan:
        self._require_hierarchy_planning()
        if high_level_steps <= 0 or num_iterations <= 0:
            raise ValueError("CEM requires positive planning steps and iterations.")
        if not (1 <= num_elites <= num_samples):
            raise ValueError("CEM requires 1 <= num_elites <= num_samples.")
        if not (0.0 <= momentum < 1.0):
            raise ValueError("CEM momentum must lie in [0, 1).")
        state = self.encoder(states)
        goal = self.encoder(goal_states)
        batch = states.shape[0]
        mean = torch.zeros(
            batch,
            high_level_steps,
            self.macro_action_dim,
            device=states.device,
            dtype=state.dtype,
        )
        std = torch.ones_like(mean)
        for _ in range(num_iterations):
            noise = torch.randn(
                batch,
                num_samples,
                high_level_steps,
                self.macro_action_dim,
                device=states.device,
                dtype=state.dtype,
            )
            samples = mean[:, None] + std[:, None] * noise
            expanded_state = state[:, None].expand(batch, num_samples, *state.shape[1:]).reshape(
                batch * num_samples, *state.shape[1:]
            )
            predicted = self._rollout_high_from_macro(
                expanded_state,
                samples.reshape(batch * num_samples, high_level_steps, self.macro_action_dim),
            )
            endpoints = predicted[:, -1].reshape(batch, num_samples, *state.shape[1:])
            scores = (endpoints - goal[:, None]).abs().flatten(2).mean(dim=-1)
            elite_indices = scores.topk(num_elites, largest=False, dim=1).indices
            gather_shape = (batch, num_elites, high_level_steps, self.macro_action_dim)
            elite_actions = samples.gather(
                1,
                elite_indices[:, :, None, None].expand(gather_shape),
            )
            elite_mean = elite_actions.mean(dim=1)
            elite_std = elite_actions.std(dim=1, unbiased=False).clamp_min(0.05)
            mean = momentum * mean + (1.0 - momentum) * elite_mean
            std = momentum * std + (1.0 - momentum) * elite_std

        predicted_states = self._rollout_high_from_macro(state, mean)
        goal_scores = (predicted_states[:, -1] - goal).abs().flatten(1).mean(dim=-1)
        return ContinuousMacroPlan(
            macro_actions=mean,
            predicted_states=predicted_states,
            goal_scores=goal_scores,
        )

    @torch.no_grad()
    def optimize_primitive_actions(
        self,
        states: torch.Tensor,
        subgoals: torch.Tensor,
        *,
        horizon: int | None = None,
        num_samples: int = 128,
        num_elites: int = 16,
        num_iterations: int = 4,
        momentum: float = 0.1,
    ) -> PrimitiveActionPlan:
        self._require_hierarchy_planning()
        horizon = self.hierarchy_horizon if horizon is None else int(horizon)
        if horizon <= 0 or num_iterations <= 0:
            raise ValueError("Primitive CEM requires positive horizon and iterations.")
        if not (1 <= num_elites <= num_samples):
            raise ValueError("Primitive CEM requires 1 <= num_elites <= num_samples.")
        if not (0.0 <= momentum < 1.0):
            raise ValueError("Primitive CEM momentum must lie in [0, 1).")
        batch = states.shape[0]
        state = self.encoder(states)
        cardinalities = (3, self.grid_size, self.grid_size, self.num_colors - 1)
        probabilities = [
            torch.full(
                (batch, horizon, cardinality),
                1.0 / cardinality,
                device=states.device,
                dtype=state.dtype,
            )
            for cardinality in cardinalities
        ]

        for _ in range(num_iterations):
            sampled_fields = []
            for field_probabilities in probabilities:
                expanded = field_probabilities[:, None].expand(batch, num_samples, horizon, -1)
                sampled = torch.multinomial(
                    expanded.reshape(batch * num_samples * horizon, -1).float(),
                    1,
                ).reshape(batch, num_samples, horizon)
                sampled_fields.append(sampled)
            sampled_parameters = torch.stack(sampled_fields, dim=-1)
            sampled_actions = sampled_parameters.clone()
            sampled_actions[..., 3] += 1
            sampled_actions = self._normalize_planner_actions(sampled_actions, states=states)
            expanded_state = state[:, None].expand(batch, num_samples, *state.shape[1:]).reshape(
                batch * num_samples, *state.shape[1:]
            )
            endpoints = self._rollout_from_latent(
                expanded_state,
                sampled_actions.reshape(batch * num_samples, horizon, 4),
            )[:, -1].reshape(batch, num_samples, *state.shape[1:])
            scores = (endpoints - subgoals[:, None]).abs().flatten(2).mean(dim=-1)
            elite_indices = scores.topk(num_elites, largest=False, dim=1).indices
            elite_actions = sampled_actions.gather(
                1,
                elite_indices[:, :, None, None].expand(batch, num_elites, horizon, 4),
            )
            elite_parameters = elite_actions.clone()
            elite_parameters[..., 3] = sampled_parameters[..., 3].gather(
                1,
                elite_indices[:, :, None].expand(batch, num_elites, horizon),
            )
            updated_probabilities = []
            for field, cardinality in enumerate(cardinalities):
                counts = F.one_hot(elite_parameters[..., field], num_classes=cardinality).float().mean(dim=1)
                smoothed = (counts + 1.0e-3) / (counts.sum(dim=-1, keepdim=True) + cardinality * 1.0e-3)
                updated_probabilities.append(
                    momentum * probabilities[field] + (1.0 - momentum) * smoothed.to(state.dtype)
                )
            probabilities = updated_probabilities

        actions = torch.stack([field.argmax(dim=-1) for field in probabilities], dim=-1)
        actions[..., 3] += 1
        actions = self._normalize_planner_actions(actions, states=states)
        predicted_endpoints = self._rollout_from_latent(state, actions)[:, -1]
        scores = (predicted_endpoints - subgoals).abs().flatten(1).mean(dim=-1)
        return PrimitiveActionPlan(
            actions=actions,
            predicted_endpoints=predicted_endpoints,
            subgoal_scores=scores,
        )

    @torch.no_grad()
    def track_subgoal(
        self,
        states: torch.Tensor,
        candidate_action_chunks: torch.Tensor,
        subgoals: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._require_hierarchy_planning()
        if candidate_action_chunks.ndim != 4 or candidate_action_chunks.shape[0] != states.shape[0]:
            raise ValueError("Low-level candidates must have shape [B,C,K,4].")
        batch, candidates, horizon = candidate_action_chunks.shape[:3]
        state = self.encoder(states)
        expanded_state = state[:, None].expand(batch, candidates, *state.shape[1:]).reshape(
            batch * candidates, *state.shape[1:]
        )
        endpoints = self._rollout_from_latent(
            expanded_state,
            candidate_action_chunks.reshape(batch * candidates, horizon, 4),
        )[:, -1].reshape(batch, candidates, *state.shape[1:])
        scores = (endpoints - subgoals[:, None]).abs().flatten(2).mean(dim=-1)
        return scores.argmin(dim=1), scores, endpoints

    def forward(self, batch: ObjectDynamicsBatch) -> ObjectDynamicsOutput:
        horizon = min(max(1, self.rollout_horizon), batch.actions.shape[1], batch.futures.shape[1])
        target_horizon = min(
            self.training_horizon,
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
        encoded_previous = current
        for step in range(horizon):
            next_state = self._predict_step(state, action_embeddings[:, step])
            predicted.append(next_state)
            rollout_losses.append(F.mse_loss(next_state, targets[:, step]))
            if self.ldad_weight > 0.0:
                delta = self._pool_delta(targets[:, step] - encoded_previous)
                ldad_losses.append(self._ldad_loss(delta, batch.actions[:, step]))
            encoded_previous = targets[:, step]
            state = next_state
        predicted_tensor = torch.stack(predicted, dim=1)
        rollout_loss = torch.stack(rollout_losses).mean()
        ldad_loss = torch.stack(ldad_losses).mean() if ldad_losses else rollout_loss.detach() * 0.0
        hierarchy_loss = self._hierarchy_loss(current, action_embeddings, targets)
        regularizer_loss = self._regularizer_loss(torch.cat([current.unsqueeze(1), targets], dim=1))
        reconstruction_loss = self._reconstruction_loss(
            torch.cat([current.unsqueeze(1), targets], dim=1),
            torch.cat([batch.states.unsqueeze(1), batch.futures[:, :target_horizon]], dim=1),
        )
        loss = reconstruction_loss * self.reconstruction_weight
        if self.dynamics_weight > 0.0:
            loss = loss + self.dynamics_weight * (
                self.rollout_weight * rollout_loss
                + self.hierarchy_weight * hierarchy_loss
                + self.ldad_weight * ldad_loss
            )
        if self.regularizer_weight > 0.0:
            loss = loss + self.regularizer_weight * regularizer_loss
        return ObjectDynamicsOutput(
            loss=loss,
            rollout_loss=rollout_loss,
            hierarchy_loss=hierarchy_loss,
            ldad_loss=ldad_loss,
            regularizer_loss=regularizer_loss,
            reconstruction_loss=reconstruction_loss,
            predicted=predicted_tensor,
            targets=targets,
        )

    def _encode_targets(self, futures: torch.Tensor) -> torch.Tensor:
        batch, horizon, height, width = futures.shape
        flat = futures.reshape(batch * horizon, height, width)
        if self.target_mode == "ema":
            with torch.no_grad():
                encoded_flat = self.target_encoder(flat)
                return encoded_flat.reshape(batch, horizon, *encoded_flat.shape[1:]).detach()
        encoded_flat = self.encoder(flat)
        encoded = encoded_flat.reshape(batch, horizon, *encoded_flat.shape[1:])
        return encoded.detach() if self.target_mode == "stop_gradient" else encoded

    def _hierarchy_loss(self, current: torch.Tensor, action_embeddings: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.hierarchy_horizon <= 1 or self.hierarchy_horizon > action_embeddings.shape[1]:
            return current.sum() * 0.0
        macro_steps = min(
            self.hierarchy_rollout_steps,
            action_embeddings.shape[1] // self.hierarchy_horizon,
            targets.shape[1] // self.hierarchy_horizon,
        )
        losses = []
        for macro_step in range(macro_steps):
            start = macro_step * self.hierarchy_horizon
            stop = start + self.hierarchy_horizon
            state = current if macro_step == 0 else targets[:, start - 1]
            chunk = self._encode_action_embeddings(action_embeddings[:, start:stop])
            predicted = self._predict_hierarchy(state, chunk)
            losses.append(F.mse_loss(predicted, targets[:, stop - 1]))
        return torch.stack(losses).mean()

    def _encode_action_embeddings(self, action_embeddings: torch.Tensor) -> torch.Tensor:
        if self.hierarchy_planning:
            return self.chunk_encoder(action_embeddings)
        _, hidden = self.chunk_encoder(action_embeddings)
        return hidden[-1]

    def _rollout_from_latent(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        action_embeddings = self.actions(actions)
        predictions = []
        for step in range(action_embeddings.shape[1]):
            state = self._predict_step(state, action_embeddings[:, step])
            predictions.append(state)
        return torch.stack(predictions, dim=1)

    def _predict_step(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        conditioned_action = self._expand_condition(action, state)
        return self.predictor(torch.cat([state, conditioned_action], dim=-1))

    def _predict_hierarchy(self, state: torch.Tensor, chunk: torch.Tensor) -> torch.Tensor:
        if self.hierarchy_planning:
            chunk = self.chunk_encoder.project(chunk)
        conditioned_chunk = self._expand_condition(chunk, state)
        return self.hierarchy_predictor(torch.cat([state, conditioned_chunk], dim=-1))

    def _rollout_high_from_macro(self, state: torch.Tensor, macro_actions: torch.Tensor) -> torch.Tensor:
        predictions = []
        for step in range(macro_actions.shape[1]):
            state = self._predict_hierarchy(state, macro_actions[:, step])
            predictions.append(state)
        return torch.stack(predictions, dim=1)

    @staticmethod
    def _expand_condition(condition: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        while condition.ndim < state.ndim:
            condition = condition.unsqueeze(-2)
        return condition.expand(*state.shape[:-1], condition.shape[-1])

    def _pool_delta(self, delta: torch.Tensor) -> torch.Tensor:
        if delta.ndim == 2:
            return delta
        weights = torch.softmax(self.delta_pool(delta).squeeze(-1), dim=-1)
        return (delta * weights.unsqueeze(-1)).sum(dim=-2)

    def _ldad_loss(self, delta: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return (
            F.cross_entropy(self.delta_op(delta), actions[:, 0].clamp(0, 2))
            + F.cross_entropy(self.delta_row(delta), actions[:, 1].clamp(0, self.delta_row.out_features - 1))
            + F.cross_entropy(self.delta_col(delta), actions[:, 2].clamp(0, self.delta_col.out_features - 1))
            + F.cross_entropy(self.delta_color(delta), actions[:, 3].clamp(0, self.delta_color.out_features - 1))
        )

    def _reconstruction_loss(self, latents: torch.Tensor, grids: torch.Tensor) -> torch.Tensor:
        if self.reconstruction_weight <= 0.0:
            return latents.sum() * 0.0
        logits = self.reconstruction_decoder(latents)
        if self.latent_representation == "grid":
            logits = logits.reshape(*grids.shape[:2], self.grid_size, self.grid_size, self.num_colors)
        else:
            logits = logits.reshape(*grids.shape[:2], self.grid_size, self.grid_size, self.num_colors)
        logits = logits.permute(0, 1, 4, 2, 3).flatten(0, 1)
        return F.cross_entropy(logits, grids.flatten(0, 1))

    def _require_hierarchy_planning(self) -> None:
        if not self.hierarchy_planning:
            raise RuntimeError("Hierarchical planning is disabled for this model.")

    def _normalize_planner_actions(
        self,
        actions: torch.Tensor,
        *,
        states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized = actions.clone()
        if states is None:
            erase = normalized[..., 0] == 1
            normalized[..., 3] = torch.where(
                erase,
                torch.zeros_like(normalized[..., 3]),
                normalized[..., 3].clamp(1, self.num_colors - 1),
            )
            return normalized

        expanded_states = states
        while expanded_states.ndim < normalized.ndim:
            expanded_states = expanded_states.unsqueeze(1)
        expanded_states = expanded_states.expand(*normalized.shape[:-2], *states.shape[-2:]).clone()
        flat_states = expanded_states.reshape(-1, *states.shape[-2:])
        flat_actions = normalized.reshape(-1, normalized.shape[-2], 4)
        batch_indices = torch.arange(len(flat_states), device=states.device)
        for step in range(flat_actions.shape[1]):
            row = flat_actions[:, step, 1].clamp(0, self.grid_size - 1)
            col = flat_actions[:, step, 2].clamp(0, self.grid_size - 1)
            occupied = flat_states[batch_indices, row, col] != 0
            operation = flat_actions[:, step, 0]
            operation = torch.where(
                occupied & (operation == 0),
                torch.full_like(operation, 2),
                operation,
            )
            operation = torch.where(~occupied & (operation != 0), torch.zeros_like(operation), operation)
            color = torch.where(
                operation == 1,
                torch.zeros_like(flat_actions[:, step, 3]),
                flat_actions[:, step, 3].clamp(1, self.num_colors - 1),
            )
            flat_actions[:, step, 0] = operation
            flat_actions[:, step, 3] = color
            flat_states[batch_indices, row, col] = color
        return normalized

    def _regularizer_loss(self, z: torch.Tensor) -> torch.Tensor:
        if self.regularizer == "none" or self.regularizer_weight <= 0.0:
            return z.sum() * 0.0
        if z.ndim == 2:
            z = z.unsqueeze(1)
        if self.regularizer == "vicreg":
            return torch.stack(
                [vicreg_regularizer(z[:, step].reshape(-1, z.shape[-1])) for step in range(z.shape[1])]
            ).mean()
        if self.regularizer == "sigreg":
            return torch.stack(
                [sigreg_regularizer(z[:, step].reshape(-1, z.shape[-1])) for step in range(z.shape[1])]
            ).mean()
        raise ValueError(f"Unknown regularizer {self.regularizer!r}.")
