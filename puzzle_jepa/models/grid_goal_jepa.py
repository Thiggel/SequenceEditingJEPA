from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


ROLE_CONTEXT = 0
ROLE_STATE = 1
ROLE_GOAL_QUERY = 2


@dataclass(frozen=True, slots=True)
class GridGoalJEPAOutput:
    loss: torch.Tensor
    dynamics_loss: torch.Tensor
    dense_future_loss: torch.Tensor
    hierarchy_loss: torch.Tensor
    sigreg_loss: torch.Tensor
    goal_mse_loss: torch.Tensor
    goal_nce_loss: torch.Tensor
    progress_rank_loss: torch.Tensor
    action_rank_loss: torch.Tensor
    policy_prior_loss: torch.Tensor
    temporal_straightening_loss: torch.Tensor
    terminal_corrupt_loss: torch.Tensor
    state_latents: torch.Tensor
    predicted_next_latents: torch.Tensor
    predicted_goal_latents: torch.Tensor
    goal_target_latents: torch.Tensor
    distances: torch.Tensor


def tokenwise_distance(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor, projector: nn.Module, eps: float = 1e-6) -> torch.Tensor:
    if a.shape != b.shape:
        raise ValueError(f"Distance inputs must have matching shapes, got {tuple(a.shape)} and {tuple(b.shape)}.")
    if mask.shape != a.shape[:-1]:
        raise ValueError(f"Distance mask must have shape {tuple(a.shape[:-1])}, got {tuple(mask.shape)}.")
    a_proj = F.normalize(projector(a), dim=-1, eps=eps)
    b_proj = F.normalize(projector(b), dim=-1, eps=eps)
    per_token = (a_proj - b_proj).square().sum(dim=-1)
    denom = mask.float().sum(dim=-1).clamp_min(1.0)
    return (per_token * mask.float()).sum(dim=-1) / denom


def covariance_sigreg(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if tokens.shape[:-1] != mask.shape:
        raise ValueError(f"SIGReg mask must match token prefix shape, got {tuple(mask.shape)} for {tuple(tokens.shape)}.")
    valid = tokens[mask].float()
    if valid.shape[0] < 2:
        return tokens.sum() * 0.0
    valid = valid - valid.mean(dim=0, keepdim=True)
    cov = valid.T @ valid / max(1, valid.shape[0] - 1)
    eye = torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
    mean_loss = tokens[mask].float().mean(dim=0).square().mean()
    cov_loss = (cov - eye).square().mean()
    return mean_loss + cov_loss


def vicreg_regularizer(tokens: torch.Tensor, mask: torch.Tensor, *, eps: float = 1.0e-4) -> torch.Tensor:
    if tokens.shape[:-1] != mask.shape:
        raise ValueError(f"VICReg mask must match token prefix shape, got {tuple(mask.shape)} for {tuple(tokens.shape)}.")
    valid = tokens[mask].float()
    if valid.shape[0] < 2:
        return tokens.sum() * 0.0
    centered = valid - valid.mean(dim=0, keepdim=True)
    std = torch.sqrt(centered.var(dim=0, unbiased=False) + eps)
    var_loss = F.relu(1.0 - std).mean()
    cov = centered.T @ centered / max(1, valid.shape[0] - 1)
    cov_loss = _off_diagonal(cov).square().mean()
    mean_loss = valid.mean(dim=0).square().mean()
    return mean_loss + var_loss + cov_loss


class GridTokenEmbedder(nn.Module):
    def __init__(self, *, d_model: int = 256, value_vocab: int = 10, max_rows: int = 9, max_cols: int = 9):
        super().__init__()
        self.d_model = int(d_model)
        self.max_rows = int(max_rows)
        self.max_cols = int(max_cols)
        self.value = nn.Embedding(value_vocab, d_model)
        self.row = nn.Embedding(max_rows, d_model)
        self.col = nn.Embedding(max_cols, d_model)
        self.role = nn.Embedding(8, d_model)
        self.known = nn.Embedding(2, d_model)
        self.editable = nn.Embedding(2, d_model)
        self.active = nn.Embedding(2, d_model)

    def forward(
        self,
        values: torch.Tensor,
        *,
        role: int,
        known_mask: torch.Tensor,
        editable_mask: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError(f"GridTokenEmbedder expects [batch, rows, cols], got {tuple(values.shape)}.")
        batch, rows, cols = values.shape
        if rows > self.max_rows or cols > self.max_cols:
            raise ValueError(f"Grid shape {(rows, cols)} exceeds configured max {(self.max_rows, self.max_cols)}.")
        row_ids = torch.arange(rows, device=values.device).view(1, rows, 1).expand(batch, rows, cols)
        col_ids = torch.arange(cols, device=values.device).view(1, 1, cols).expand(batch, rows, cols)
        role_ids = torch.full_like(values, int(role))
        x = (
            self.value(values)
            + self.row(row_ids)
            + self.col(col_ids)
            + self.role(role_ids)
            + self.known(known_mask.long())
            + self.editable(editable_mask.long())
            + self.active(active_mask.long())
        )
        return x.reshape(batch, rows * cols, self.d_model)

    def query_tokens(self, active_mask: torch.Tensor) -> torch.Tensor:
        values = torch.zeros_like(active_mask, dtype=torch.long)
        zeros = torch.zeros_like(active_mask, dtype=torch.bool)
        return self(values, role=ROLE_GOAL_QUERY, known_mask=zeros, editable_mask=zeros, active_mask=active_mask)


class CrossAttentionBlock(nn.Module):
    def __init__(self, *, d_model: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(d_model * mlp_ratio), d_model),
            nn.Dropout(dropout),
        )
        self.norm_self = nn.LayerNorm(d_model)
        self.norm_cross = nn.LayerNorm(d_model)
        self.norm_mlp = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.self_attn(self.norm_self(x), self.norm_self(x), self.norm_self(x), need_weights=False)[0]
        if context is not None:
            x = x + self.cross_attn(self.norm_cross(x), context, context, need_weights=False)[0]
        x = x + self.mlp(self.norm_mlp(x))
        return x


class BidirectionalTransformer(nn.Module):
    def __init__(self, *, num_layers: int, d_model: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                CrossAttentionBlock(d_model=d_model, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, context)
        return self.norm(x)


class SudokuActionToken(nn.Module):
    def __init__(self, *, d_model: int = 256):
        super().__init__()
        self.action_type = nn.Embedding(2, d_model)
        self.row = nn.Embedding(9, d_model)
        self.col = nn.Embedding(9, d_model)
        self.digit = nn.Embedding(10, d_model)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.shape[-1] != 3:
            raise ValueError(f"Sudoku action tensor must end in 3 values, got {tuple(actions.shape)}.")
        rows = actions[..., 0].clamp(0, 8)
        cols = actions[..., 1].clamp(0, 8)
        digits = actions[..., 2].clamp(0, 9)
        types = torch.ones_like(rows)
        return self.action_type(types) + self.row(rows) + self.col(cols) + self.digit(digits)


class MacroActionEncoder(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        max_steps: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ):
        super().__init__()
        self.max_steps = int(max_steps)
        self.position = nn.Embedding(max_steps, d_model)
        self.encoder = BidirectionalTransformer(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

    def forward(self, action_tokens: torch.Tensor) -> torch.Tensor:
        if action_tokens.ndim != 3:
            raise ValueError(f"MacroActionEncoder expects [batch, steps, dim], got {tuple(action_tokens.shape)}.")
        steps = action_tokens.shape[1]
        if steps > self.max_steps:
            raise ValueError(f"Macro action length {steps} exceeds configured max {self.max_steps}.")
        pos = torch.arange(steps, device=action_tokens.device).view(1, steps)
        tokens = action_tokens + self.position(pos)
        return self.encoder(tokens).mean(dim=1)


class GridTokenGoalJEPA(nn.Module):
    def __init__(
        self,
        *,
        d_model: int = 256,
        distance_dim: int = 128,
        context_layers: int = 4,
        state_layers: int = 6,
        predictor_layers: int = 4,
        goal_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        sigreg_weight: float = 0.1,
        goal_nce_weight: float = 0.1,
        progress_rank_weight: float = 1.0,
        action_rank_weight: float = 1.0,
        temporal_straightening_weight: float = 0.1,
        terminal_corrupt_weight: float = 1.0,
        progress_margin: float = 0.1,
        rank_margin: float = 0.1,
        rank_temperature: float = 0.1,
        multi_step_horizons: tuple[int, ...] = (1, 4, 8, 16),
        dense_future_weight: float = 0.0,
        dense_rollout_all_steps: bool = False,
        rollout_detach_interval: int = 0,
        hierarchy_levels: tuple[int, ...] = (),
        hierarchy_loss_weight: float = 0.0,
        hierarchy_dense_future_weight: float = 0.0,
        macro_action_encoder_layers: int = 1,
        shared_hierarchy_predictor: bool = False,
        goal_conditioning: str = "initial_current",
        progress_rank_target: str = "predicted",
        action_rank_mode: str = "pairwise",
        action_rank_target: str = "predicted",
        listwise_action_rank_max_actions: int = 729,
        policy_prior_weight: float = 0.0,
        policy_prior_mode: str = "pairwise",
        policy_prior_planning_weight: float = 0.0,
        distance_mode: str = "tokenwise",
        action_conditioning: str = "action_token",
        predict_delta: bool = False,
        dynamics_weighting: str = "uniform",
        affected_dynamics_weight: float = 32.0,
        regularizer: str = "sigreg",
        use_ema_target_encoder: bool = False,
        ema_decay: float = 0.995,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.max_rows = 9
        self.max_cols = 9
        self.embedder = GridTokenEmbedder(d_model=d_model)
        self.context_encoder = BidirectionalTransformer(
            num_layers=context_layers, d_model=d_model, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout
        )
        self.state_encoder = BidirectionalTransformer(
            num_layers=state_layers, d_model=d_model, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout
        )
        self.action_token = SudokuActionToken(d_model=d_model)
        self.affected_marker = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.affected_marker, std=0.02)
        self.local_action_type = nn.Embedding(2, d_model)
        self.local_action_digit = nn.Embedding(10, d_model)
        self.action_film = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))
        self.old_local_concat = nn.Linear(2 * d_model, d_model) if action_conditioning == "old_local_concat" else None
        self.predictor = BidirectionalTransformer(
            num_layers=predictor_layers, d_model=d_model, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout
        )
        self.predictor_out = nn.Linear(d_model, d_model)
        self.hierarchy_levels = tuple(sorted({int(level) for level in hierarchy_levels if int(level) > 1}))
        self.hierarchy_loss_weight = float(hierarchy_loss_weight)
        self.hierarchy_dense_future_weight = float(hierarchy_dense_future_weight)
        self.dense_future_weight = float(dense_future_weight)
        self.dense_rollout_all_steps = bool(dense_rollout_all_steps)
        self.rollout_detach_interval = int(rollout_detach_interval)
        if self.rollout_detach_interval < 0:
            raise ValueError("rollout_detach_interval must be non-negative.")
        self.shared_hierarchy_predictor = bool(shared_hierarchy_predictor)
        if self.hierarchy_levels:
            max_level = max(self.hierarchy_levels)
            self.macro_action_encoder = MacroActionEncoder(
                d_model=d_model,
                max_steps=max_level,
                num_layers=macro_action_encoder_layers,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            if self.shared_hierarchy_predictor:
                self.hierarchy_level_embed = nn.Embedding(max_level + 1, d_model)
                self.shared_high_level_predictor = BidirectionalTransformer(
                    num_layers=predictor_layers,
                    d_model=d_model,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                self.shared_high_level_predictor_out = nn.Linear(d_model, d_model)
                self.high_level_predictors = nn.ModuleDict()
                self.high_level_predictor_out = nn.ModuleDict()
            else:
                self.hierarchy_level_embed = None
                self.shared_high_level_predictor = None
                self.shared_high_level_predictor_out = None
                self.high_level_predictors = nn.ModuleDict(
                    {
                        str(level): BidirectionalTransformer(
                            num_layers=predictor_layers,
                            d_model=d_model,
                            num_heads=num_heads,
                            mlp_ratio=mlp_ratio,
                            dropout=dropout,
                        )
                        for level in self.hierarchy_levels
                    }
                )
                self.high_level_predictor_out = nn.ModuleDict(
                    {str(level): nn.Linear(d_model, d_model) for level in self.hierarchy_levels}
                )
        else:
            self.macro_action_encoder = None
            self.hierarchy_level_embed = None
            self.shared_high_level_predictor = None
            self.shared_high_level_predictor_out = None
            self.high_level_predictors = nn.ModuleDict()
            self.high_level_predictor_out = nn.ModuleDict()
        allowed_goal_conditioning = {"context", "initial_current"}
        if goal_conditioning not in allowed_goal_conditioning:
            raise ValueError(f"goal_conditioning must be one of {sorted(allowed_goal_conditioning)}.")
        self.goal_conditioning = str(goal_conditioning)
        if self.goal_conditioning == "initial_current":
            self.goal_state_role = nn.Embedding(2, d_model)
        else:
            self.goal_state_role = None
        self.goal_decoder = BidirectionalTransformer(
            num_layers=goal_layers, d_model=d_model, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout
        )
        allowed_progress_targets = {"predicted", "oracle", "both", "none"}
        if progress_rank_target not in allowed_progress_targets:
            raise ValueError(f"progress_rank_target must be one of {sorted(allowed_progress_targets)}.")
        allowed_action_rank_modes = {"pairwise", "listwise", "none"}
        if action_rank_mode not in allowed_action_rank_modes:
            raise ValueError(f"action_rank_mode must be one of {sorted(allowed_action_rank_modes)}.")
        allowed_policy_prior_modes = {"pairwise", "listwise", "none"}
        if policy_prior_mode not in allowed_policy_prior_modes:
            raise ValueError(f"policy_prior_mode must be one of {sorted(allowed_policy_prior_modes)}.")
        allowed_action_rank_targets = {"predicted", "oracle", "both"}
        if action_rank_target not in allowed_action_rank_targets:
            raise ValueError(f"action_rank_target must be one of {sorted(allowed_action_rank_targets)}.")
        self.progress_rank_target = str(progress_rank_target)
        self.action_rank_mode = str(action_rank_mode)
        self.action_rank_target = str(action_rank_target)
        self.listwise_action_rank_max_actions = int(listwise_action_rank_max_actions)
        if self.listwise_action_rank_max_actions <= 0:
            raise ValueError("listwise_action_rank_max_actions must be positive.")
        self.policy_prior_weight = float(policy_prior_weight)
        self.policy_prior_mode = str(policy_prior_mode)
        self.policy_prior_planning_weight = float(policy_prior_planning_weight)
        self.policy_query = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.policy_action = nn.Linear(d_model, d_model, bias=False)
        self.distance_projector = nn.Linear(d_model, distance_dim)
        self.sigreg_weight = float(sigreg_weight)
        self.goal_nce_weight = float(goal_nce_weight)
        self.progress_rank_weight = float(progress_rank_weight)
        self.action_rank_weight = float(action_rank_weight)
        self.temporal_straightening_weight = float(temporal_straightening_weight)
        self.terminal_corrupt_weight = float(terminal_corrupt_weight)
        self.progress_margin = float(progress_margin)
        self.rank_margin = float(rank_margin)
        self.rank_temperature = float(rank_temperature)
        self.multi_step_horizons = tuple(sorted({int(k) for k in multi_step_horizons if int(k) > 0}))
        if distance_mode not in {"tokenwise", "mean_pooled"}:
            raise ValueError("distance_mode must be 'tokenwise' or 'mean_pooled'.")
        self.distance_mode = str(distance_mode)
        allowed_action_conditioning = {
            "action_token",
            "affected_marker",
            "local_action_feature",
            "old_local_value",
            "old_local_concat",
            "action_cross_attention",
            "adaln_action",
        }
        if action_conditioning not in allowed_action_conditioning:
            raise ValueError(f"action_conditioning must be one of {sorted(allowed_action_conditioning)}.")
        if dynamics_weighting not in {"uniform", "affected"}:
            raise ValueError("dynamics_weighting must be 'uniform' or 'affected'.")
        if regularizer not in {"sigreg", "vicreg", "none"}:
            raise ValueError("regularizer must be 'sigreg', 'vicreg', or 'none'.")
        self.action_conditioning = str(action_conditioning)
        self.predict_delta = bool(predict_delta)
        self.dynamics_weighting = str(dynamics_weighting)
        self.affected_dynamics_weight = float(affected_dynamics_weight)
        self.regularizer = str(regularizer)
        self.use_ema_target_encoder = bool(use_ema_target_encoder)
        self.ema_decay = float(ema_decay)
        if self.use_ema_target_encoder:
            self.target_embedder = GridTokenEmbedder(d_model=d_model)
            self.target_state_encoder = BidirectionalTransformer(
                num_layers=state_layers, d_model=d_model, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout
            )
            self._reset_ema_target_encoder()

    def encode_context(
        self,
        context: torch.Tensor,
        clue_mask: torch.Tensor,
        editable_mask: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not bool(clue_mask.any()) and bool(editable_mask.all()):
            context = torch.zeros_like(context)
        tokens = self.embedder(
            context, role=ROLE_CONTEXT, known_mask=clue_mask, editable_mask=editable_mask, active_mask=active_mask
        )
        return self.context_encoder(tokens)

    def encode_state(
        self,
        state: torch.Tensor,
        context_latents: torch.Tensor,
        clue_mask: torch.Tensor,
        editable_mask: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.embedder(
            state, role=ROLE_STATE, known_mask=clue_mask, editable_mask=editable_mask, active_mask=active_mask
        )
        return self.state_encoder(tokens, context_latents)

    def encode_state_target(
        self,
        state: torch.Tensor,
        context_latents: torch.Tensor,
        clue_mask: torch.Tensor,
        editable_mask: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_ema_target_encoder:
            return self.encode_state(state, context_latents, clue_mask, editable_mask, active_mask)
        was_training = self.target_state_encoder.training
        self.target_state_encoder.eval()
        try:
            tokens = self.target_embedder(
                state, role=ROLE_STATE, known_mask=clue_mask, editable_mask=editable_mask, active_mask=active_mask
            )
            return self.target_state_encoder(tokens, context_latents)
        finally:
            self.target_state_encoder.train(was_training)

    def predict_next(self, state_latent: torch.Tensor, action: torch.Tensor, context_latents: torch.Tensor) -> torch.Tensor:
        if self.action_conditioning in {"old_local_value", "old_local_concat"}:
            state_input = self._condition_state_latents(state_latent, action, None)
            y = self.predictor(state_input, context_latents)
            predicted = self.predictor_out(y)
        elif self.action_conditioning == "action_cross_attention":
            action_token = self.action_token(action).unsqueeze(-2)
            state_input = self._condition_state_latents(state_latent, action, action_token.squeeze(-2))
            action_context = torch.cat([action_token, context_latents], dim=-2)
            y = self.predictor(state_input, action_context)
            predicted = self.predictor_out(y)
        else:
            action_token = self.action_token(action).unsqueeze(-2)
            state_input = self._condition_state_latents(state_latent, action, action_token.squeeze(-2))
            y = torch.cat([action_token, state_input], dim=-2)
            y = self.predictor(y, context_latents)
            predicted = self.predictor_out(y[..., 1:, :])
        if self.predict_delta:
            predicted = state_latent + predicted
        return predicted

    def encode_macro_action(self, actions: torch.Tensor) -> torch.Tensor:
        if self.macro_action_encoder is None:
            raise RuntimeError("This model was not configured with hierarchy_levels.")
        if actions.ndim != 3 or actions.shape[-1] != 3:
            raise ValueError(f"Macro actions must have shape [batch, steps, 3], got {tuple(actions.shape)}.")
        batch, steps = actions.shape[:2]
        tokens = self.action_token(actions.reshape(batch * steps, 3)).reshape(batch, steps, self.d_model)
        return self.macro_action_encoder(tokens)

    def predict_high_level(
        self,
        state_latent: torch.Tensor,
        actions: torch.Tensor,
        context_latents: torch.Tensor,
        *,
        level: int,
    ) -> torch.Tensor:
        macro_action = self.encode_macro_action(actions)
        return self.predict_high_level_from_macro(state_latent, macro_action, context_latents, level=level, actions=actions)

    def predict_high_level_from_macro(
        self,
        state_latent: torch.Tensor,
        macro_action: torch.Tensor,
        context_latents: torch.Tensor,
        *,
        level: int,
        actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        key = str(int(level))
        if self.shared_hierarchy_predictor:
            if self.shared_high_level_predictor is None or self.shared_high_level_predictor_out is None or self.hierarchy_level_embed is None:
                raise RuntimeError("Shared hierarchy predictor was not initialized.")
        elif key not in self.high_level_predictors:
            raise ValueError(f"No high-level predictor configured for level {level}.")
        if macro_action.ndim != 2 or macro_action.shape[-1] != self.d_model:
            raise ValueError(f"Macro action must have shape [batch, {self.d_model}], got {tuple(macro_action.shape)}.")
        if self.shared_hierarchy_predictor:
            level_ids = torch.full(
                (macro_action.shape[0],),
                min(max(0, int(level)), self.hierarchy_level_embed.num_embeddings - 1),
                dtype=torch.long,
                device=macro_action.device,
            )
            macro_action = macro_action + self.hierarchy_level_embed(level_ids).to(dtype=macro_action.dtype)
        action_token = macro_action.unsqueeze(-2)
        state_input = state_latent
        if actions is not None and self.action_conditioning in {"old_local_value", "old_local_concat"}:
            state_input = self._condition_state_latents_with_macro_actions(state_input, actions)
        y = torch.cat([action_token, state_input], dim=-2)
        if self.shared_hierarchy_predictor:
            y = self.shared_high_level_predictor(y, context_latents)
            predicted = self.shared_high_level_predictor_out(y[..., 1:, :])
        else:
            y = self.high_level_predictors[key](y, context_latents)
            predicted = self.high_level_predictor_out[key](y[..., 1:, :])
        if self.predict_delta:
            predicted = state_latent + predicted
        return predicted

    @torch.no_grad()
    def update_ema_target_encoder(self, decay: float | None = None) -> None:
        if not self.use_ema_target_encoder:
            return
        decay = self.ema_decay if decay is None else float(decay)
        _ema_update(self.target_embedder, self.embedder, decay)
        _ema_update(self.target_state_encoder, self.state_encoder, decay)

    def predict_goal(
        self,
        context_latents: torch.Tensor,
        active_mask: torch.Tensor,
        *,
        initial_latents: torch.Tensor | None = None,
        current_latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        queries = self.embedder.query_tokens(active_mask)
        memory = context_latents
        if self.goal_conditioning == "initial_current":
            if initial_latents is None or current_latents is None:
                raise ValueError("goal_conditioning='initial_current' requires initial_latents and current_latents.")
            if initial_latents.shape != current_latents.shape:
                raise ValueError(
                    f"initial_latents and current_latents must have matching shapes, got "
                    f"{tuple(initial_latents.shape)} and {tuple(current_latents.shape)}."
                )
            if initial_latents.shape != context_latents.shape:
                raise ValueError(
                    f"Goal state latents must match context token shape {tuple(context_latents.shape)}, "
                    f"got {tuple(initial_latents.shape)}."
                )
            if self.goal_state_role is None:
                raise RuntimeError("Conditional goal role embeddings were not initialized.")
            role_ids = torch.arange(2, device=context_latents.device)
            role = self.goal_state_role(role_ids).to(dtype=context_latents.dtype)
            initial = initial_latents + role[0].view(1, 1, -1)
            current = current_latents + role[1].view(1, 1, -1)
            memory = torch.cat([context_latents, initial, current], dim=-2)
        return self.goal_decoder(queries, memory)

    def distance(self, state_latents: torch.Tensor, goal_latents: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        if active_mask.ndim == 3:
            mask = active_mask.reshape(active_mask.shape[0], -1)
        else:
            mask = active_mask
        if self.distance_mode == "mean_pooled":
            state_latents = _masked_summary(state_latents, mask).unsqueeze(1)
            goal_latents = _masked_summary(goal_latents, mask).unsqueeze(1)
            mask = torch.ones((state_latents.shape[0], 1), dtype=torch.bool, device=state_latents.device)
        return tokenwise_distance(state_latents, goal_latents, mask, self.distance_projector)

    def score_action_prior(
        self,
        state_latents: torch.Tensor,
        goal_latents: torch.Tensor,
        context_latents: torch.Tensor,
        active_mask: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        if active_mask.ndim == 3:
            mask = active_mask.reshape(active_mask.shape[0], -1)
        else:
            mask = active_mask
        squeeze = False
        if actions.ndim == 2:
            actions = actions[:, None, :]
            squeeze = True
        if actions.ndim != 3 or actions.shape[-1] != 3:
            raise ValueError(f"Action prior expects actions [batch, actions, 3], got {tuple(actions.shape)}.")
        goal_latents = _expand_batch_tokens(goal_latents, state_latents)
        context_latents = _expand_batch_tokens(context_latents, state_latents)
        summary = (
            _masked_summary(state_latents, mask)
            + _masked_summary(goal_latents, mask)
            + context_latents.mean(dim=1)
        )
        query = F.normalize(self.policy_query(summary), dim=-1, eps=1.0e-6)
        batch, action_count = actions.shape[:2]
        action_tokens = self.action_token(actions.reshape(batch * action_count, 3)).reshape(batch, action_count, self.d_model)
        action_tokens = F.normalize(self.policy_action(action_tokens), dim=-1, eps=1.0e-6)
        logits = (action_tokens * query[:, None]).sum(dim=-1) * (self.d_model**0.5)
        return logits[:, 0] if squeeze else logits

    def score_macro_action_prior(
        self,
        state_latents: torch.Tensor,
        goal_latents: torch.Tensor,
        context_latents: torch.Tensor,
        active_mask: torch.Tensor,
        macro_actions: torch.Tensor,
        *,
        level: int,
    ) -> torch.Tensor:
        if macro_actions.ndim == 3:
            macro_actions = macro_actions[:, None]
            squeeze = True
        else:
            squeeze = False
        if macro_actions.ndim != 4 or macro_actions.shape[-1] != 3:
            raise ValueError(f"Macro action prior expects [batch, actions, level, 3], got {tuple(macro_actions.shape)}.")
        batch, action_count, steps = macro_actions.shape[:3]
        if steps != int(level):
            raise ValueError(f"Macro action length {steps} does not match level {level}.")
        if active_mask.ndim == 3:
            mask = active_mask.reshape(active_mask.shape[0], -1)
        else:
            mask = active_mask
        goal_latents = _expand_batch_tokens(goal_latents, state_latents)
        context_latents = _expand_batch_tokens(context_latents, state_latents)
        summary = (
            _masked_summary(state_latents, mask)
            + _masked_summary(goal_latents, mask)
            + context_latents.mean(dim=1)
        )
        query = F.normalize(self.policy_query(summary), dim=-1, eps=1.0e-6)
        flat_actions = macro_actions.reshape(batch * action_count, steps, 3)
        macro_tokens = self.encode_macro_action(flat_actions).reshape(batch, action_count, self.d_model)
        macro_tokens = F.normalize(self.policy_action(macro_tokens), dim=-1, eps=1.0e-6)
        logits = (macro_tokens * query[:, None]).sum(dim=-1) * (self.d_model**0.5)
        return logits[:, 0] if squeeze else logits

    def forward(
        self,
        boards: torch.Tensor,
        actions: torch.Tensor,
        context: torch.Tensor,
        clue_mask: torch.Tensor,
        editable_mask: torch.Tensor,
        active_mask: torch.Tensor,
        goals: torch.Tensor,
        *,
        masks: torch.Tensor,
        oracle_mask: torch.Tensor | None = None,
        action_rank_states: torch.Tensor | None = None,
        positive_actions: torch.Tensor | None = None,
        negative_actions: torch.Tensor | None = None,
        corrupt_goals: torch.Tensor | None = None,
    ) -> GridGoalJEPAOutput:
        batch, frames = boards.shape[:2]
        rows, cols = boards.shape[-2:]
        token_count = rows * cols
        context_latents = self.encode_context(context, clue_mask, editable_mask, active_mask)
        flat_boards = boards.reshape(batch * frames, *boards.shape[-2:])
        flat_context = context_latents[:, None].expand(batch, frames, token_count, self.d_model).reshape(
            batch * frames, token_count, self.d_model
        )
        flat_clue = clue_mask[:, None].expand(batch, frames, rows, cols).reshape(batch * frames, rows, cols)
        flat_edit = editable_mask[:, None].expand(batch, frames, rows, cols).reshape(batch * frames, rows, cols)
        flat_active = active_mask[:, None].expand(batch, frames, rows, cols).reshape(batch * frames, rows, cols)
        state_latents = self.encode_state(flat_boards, flat_context, flat_clue, flat_edit, flat_active).reshape(
            batch, frames, token_count, self.d_model
        )
        with torch.no_grad():
            target_state_latents = self.encode_state_target(flat_boards, flat_context, flat_clue, flat_edit, flat_active).reshape(
                batch, frames, token_count, self.d_model
            )
            goal_target = self.encode_state_target(goals, context_latents, clue_mask, editable_mask, active_mask)
        active_flat = active_mask.reshape(batch, token_count)
        if self.goal_conditioning == "initial_current":
            initial_for_goal = state_latents[:, :1].expand(batch, frames, token_count, self.d_model).reshape(
                batch * frames, token_count, self.d_model
            )
            predicted_goal_sequence = self.predict_goal(
                flat_context,
                flat_active,
                initial_latents=initial_for_goal,
                current_latents=state_latents.reshape(batch * frames, token_count, self.d_model),
            ).reshape(batch, frames, token_count, self.d_model)
            predicted_goal = predicted_goal_sequence[:, 0]
        else:
            predicted_goal = self.predict_goal(context_latents, active_mask)
            predicted_goal_sequence = predicted_goal[:, None].expand(batch, frames, token_count, self.d_model)

        transition_mask = masks[:, :-1] & masks[:, 1:]
        dynamics_terms = []
        dense_future_terms = []
        if frames > 1:
            predicted_next = self.predict_next(
                state_latents[:, :-1].reshape(-1, token_count, self.d_model),
                actions[:, :-1].reshape(-1, 3),
                context_latents[:, None].expand(batch, frames - 1, token_count, self.d_model).reshape(
                    -1, token_count, self.d_model
                ),
            ).reshape(batch, frames - 1, token_count, self.d_model)
            one_step_error = self._dynamics_error(
                predicted_next,
                target_state_latents[:, 1:],
                actions[:, :-1],
                rows=rows,
                cols=cols,
            )
            dynamics_terms.append(_masked_mean(one_step_error, transition_mask))
        else:
            predicted_next = state_latents[:, :0]
            dynamics_terms.append(state_latents.sum() * 0.0)
        if self.dense_rollout_all_steps and self.multi_step_horizons:
            max_horizon = min(max(self.multi_step_horizons), frames - 1)
            start_count = frames - max_horizon
            if start_count > 0:
                rollout = state_latents[:, :start_count].reshape(batch * start_count, token_count, self.d_model)
                ctx = context_latents[:, None].expand(batch, start_count, token_count, self.d_model).reshape(
                    -1, token_count, self.d_model
                )
                for offset in range(max_horizon):
                    act = actions[:, offset : offset + start_count].reshape(batch * start_count, 3)
                    rollout = self.predict_next(rollout, act, ctx)
                    dense_horizon = offset + 1
                    if dense_horizon > 1:
                        target = target_state_latents[:, dense_horizon : dense_horizon + start_count]
                        valid = masks[:, :start_count] & masks[:, dense_horizon : dense_horizon + start_count]
                        rollout_actions = actions[:, : dense_horizon + start_count - 1]
                        dense_error = self._dynamics_error(
                            rollout.reshape(batch, start_count, token_count, self.d_model),
                            target,
                            rollout_actions,
                            rows=rows,
                            cols=cols,
                            horizon=dense_horizon,
                        )
                        dense_future_terms.append(_masked_mean(dense_error, valid) / (dense_horizon**0.5))
                    if (
                        self.rollout_detach_interval > 0
                        and (offset + 1) % self.rollout_detach_interval == 0
                        and offset + 1 < max_horizon
                    ):
                        rollout = rollout.detach()
        else:
            for horizon in self.multi_step_horizons:
                if horizon <= 1 or frames <= horizon:
                    continue
                start_count = frames - horizon
                rollout = state_latents[:, :start_count].reshape(batch * start_count, token_count, self.d_model)
                ctx = context_latents[:, None].expand(batch, start_count, token_count, self.d_model).reshape(
                    -1, token_count, self.d_model
                )
                for offset in range(horizon):
                    act = actions[:, offset : offset + start_count].reshape(batch * start_count, 3)
                    rollout = self.predict_next(rollout, act, ctx)
                    if self.dense_future_weight > 0.0:
                        dense_horizon = offset + 1
                        dense_target = target_state_latents[:, dense_horizon : dense_horizon + start_count]
                        dense_valid = masks[:, :start_count] & masks[:, dense_horizon : dense_horizon + start_count]
                        dense_actions = actions[:, : dense_horizon + start_count - 1]
                        dense_error = self._dynamics_error(
                            rollout.reshape(batch, start_count, token_count, self.d_model),
                            dense_target,
                            dense_actions,
                            rows=rows,
                            cols=cols,
                            horizon=dense_horizon,
                        )
                        dense_future_terms.append(_masked_mean(dense_error, dense_valid) / (dense_horizon**0.5))
                    if (
                        self.rollout_detach_interval > 0
                        and (offset + 1) % self.rollout_detach_interval == 0
                        and offset + 1 < horizon
                    ):
                        rollout = rollout.detach()
                target = target_state_latents[:, horizon : horizon + start_count]
                valid = masks[:, :start_count] & masks[:, horizon : horizon + start_count]
                rollout_actions = actions[:, :horizon + start_count - 1]
                error = self._dynamics_error(
                    rollout.reshape(batch, start_count, token_count, self.d_model),
                    target,
                    rollout_actions,
                    rows=rows,
                    cols=cols,
                    horizon=horizon,
                )
                dynamics_terms.append(_masked_mean(error, valid) / (horizon**0.5))
        dynamics_loss = torch.stack(dynamics_terms).sum()
        if dense_future_terms:
            dense_future_loss = torch.stack(dense_future_terms).mean()
        else:
            dense_future_loss = state_latents.sum() * 0.0

        hierarchy_terms = []
        hierarchy_dense_terms = []
        for level in self.hierarchy_levels:
            if frames <= level:
                continue
            start_count = frames - level
            starts = state_latents[:, :start_count].reshape(batch * start_count, token_count, self.d_model)
            ctx = context_latents[:, None].expand(batch, start_count, token_count, self.d_model).reshape(
                -1, token_count, self.d_model
            )
            chunks = []
            for offset in range(level):
                chunks.append(actions[:, offset : offset + start_count])
            chunk_actions = torch.stack(chunks, dim=2).reshape(batch * start_count, level, 3)
            predicted_waypoint = self.predict_high_level(starts, chunk_actions, ctx, level=level).reshape(
                batch, start_count, token_count, self.d_model
            )
            target_waypoint = target_state_latents[:, level : level + start_count]
            valid = masks[:, :start_count] & masks[:, level : level + start_count]
            rollout_actions = actions[:, : level + start_count - 1]
            error = self._dynamics_error(
                predicted_waypoint,
                target_waypoint,
                rollout_actions,
                rows=rows,
                cols=cols,
                horizon=level,
            )
            hierarchy_terms.append(_masked_mean(error, valid) / (level**0.5))
            if self.hierarchy_dense_future_weight > 0.0 and self.multi_step_horizons:
                max_dense_horizon = max(self.multi_step_horizons)
                for dense_horizon in range(2 * level, max_dense_horizon + 1, level):
                    if frames <= dense_horizon:
                        continue
                    dense_start_count = frames - dense_horizon
                    rollout = state_latents[:, :dense_start_count].reshape(
                        batch * dense_start_count, token_count, self.d_model
                    )
                    dense_ctx = context_latents[:, None].expand(
                        batch, dense_start_count, token_count, self.d_model
                    ).reshape(-1, token_count, self.d_model)
                    for chunk_start in range(0, dense_horizon, level):
                        dense_chunks = []
                        for offset in range(level):
                            dense_chunks.append(actions[:, chunk_start + offset : chunk_start + offset + dense_start_count])
                        dense_actions = torch.stack(dense_chunks, dim=2).reshape(batch * dense_start_count, level, 3)
                        rollout = self.predict_high_level(rollout, dense_actions, dense_ctx, level=level)
                    dense_target = target_state_latents[:, dense_horizon : dense_horizon + dense_start_count]
                    dense_valid = masks[:, :dense_start_count] & masks[:, dense_horizon : dense_horizon + dense_start_count]
                    dense_rollout_actions = actions[:, : dense_horizon + dense_start_count - 1]
                    dense_error = self._dynamics_error(
                        rollout.reshape(batch, dense_start_count, token_count, self.d_model),
                        dense_target,
                        dense_rollout_actions,
                        rows=rows,
                        cols=cols,
                        horizon=dense_horizon,
                    )
                    hierarchy_dense_terms.append(_masked_mean(dense_error, dense_valid) / (dense_horizon**0.5))
        if hierarchy_terms:
            hierarchy_loss = torch.stack(hierarchy_terms).sum()
        else:
            hierarchy_loss = state_latents.sum() * 0.0
        if hierarchy_dense_terms:
            hierarchy_loss = hierarchy_loss + self.hierarchy_dense_future_weight * torch.stack(hierarchy_dense_terms).mean()

        goal_target_sequence = goal_target[:, None].expand(batch, frames, token_count, self.d_model)
        goal_token_error = (predicted_goal_sequence - goal_target_sequence.detach()).square().mean(dim=-1)
        goal_token_weights = active_flat[:, None].expand(batch, frames, token_count) & masks[:, :, None]
        goal_mse_loss = _masked_mean(goal_token_error, goal_token_weights)
        pred_summary = _masked_summary(
            predicted_goal_sequence.reshape(batch * frames, token_count, self.d_model),
            active_flat[:, None].expand(batch, frames, token_count).reshape(batch * frames, token_count),
        ).reshape(batch, frames, self.d_model)
        target_summary = _masked_summary(goal_target.detach(), active_flat)
        valid_summary = masks.reshape(batch * frames)
        if bool(valid_summary.any()):
            logits = (
                F.normalize(pred_summary.reshape(batch * frames, self.d_model)[valid_summary], dim=-1)
                @ F.normalize(target_summary, dim=-1).T
                / 0.1
            )
            labels = torch.arange(batch, device=boards.device)[:, None].expand(batch, frames).reshape(batch * frames)[valid_summary]
            goal_nce_loss = F.cross_entropy(logits, labels)
        else:
            goal_nce_loss = state_latents.sum() * 0.0

        predicted_distances = self.distance(
            state_latents.reshape(batch * frames, token_count, self.d_model),
            predicted_goal_sequence.reshape(batch * frames, token_count, self.d_model),
            active_mask[:, None].expand(batch, frames, rows, cols).reshape(batch * frames, rows, cols),
        ).reshape(batch, frames)
        oracle_distances = self.distance(
            state_latents.reshape(batch * frames, token_count, self.d_model),
            goal_target[:, None].expand(batch, frames, token_count, self.d_model).reshape(
                batch * frames, token_count, self.d_model
            ),
            active_mask[:, None].expand(batch, frames, rows, cols).reshape(batch * frames, rows, cols),
        ).reshape(batch, frames)
        oracle_rows = torch.zeros_like(masks[:, 0], dtype=torch.bool) if oracle_mask is None else oracle_mask
        progress_masks = masks & oracle_rows[:, None]
        progress_rank_loss = self._progress_rank_objective(predicted_distances, oracle_distances, progress_masks)
        temporal_straightening_loss = _temporal_straightening_loss(
            state_latents,
            predicted_goal,
            masks=masks,
            active_mask=active_flat,
        )

        action_rank_loss = state_latents.sum() * 0.0
        policy_prior_loss = state_latents.sum() * 0.0
        policy_prior_terms = []
        needs_rank_state = (
            (self.action_rank_mode != "none" and negative_actions is not None)
            or (self.policy_prior_weight > 0.0 and self.policy_prior_mode != "none")
        )
        if needs_rank_state:
            rank_states = boards[:, 0] if action_rank_states is None else action_rank_states
            rank_state_latents = self.encode_state(rank_states, context_latents, clue_mask, editable_mask, active_mask)
            rank_goal = self._goal_for_current_state(context_latents, active_mask, state_latents[:, 0], rank_state_latents)
            if positive_actions is None:
                positive_actions = actions[:, 0]
            if self.action_rank_mode != "none" and negative_actions is not None:
                if self.action_rank_mode == "listwise":
                    action_rank_loss = self._listwise_action_rank_loss(
                        rank_states,
                        positive_actions,
                        context_latents,
                        clue_mask,
                        editable_mask,
                        active_mask,
                        rank_goal,
                        goal_target,
                    )
                else:
                    pos_boards = _apply_set_cell_actions(rank_states, positive_actions)
                    neg_boards = _apply_set_cell_actions(rank_states, negative_actions)
                    pos_latents = self.encode_state(pos_boards, context_latents, clue_mask, editable_mask, active_mask)
                    neg_latents = self.encode_state(neg_boards, context_latents, clue_mask, editable_mask, active_mask)
                    pos_d = self._rank_target_distance(pos_latents, rank_goal, goal_target, active_mask)
                    neg_d = self._rank_target_distance(neg_latents, rank_goal, goal_target, active_mask)
                    action_rank_loss = F.softplus((pos_d - neg_d + self.rank_margin) / self.rank_temperature).mean()
            if self.policy_prior_weight > 0.0 and self.policy_prior_mode != "none":
                if self.policy_prior_mode == "listwise":
                    policy_prior_terms.append(
                        self._listwise_policy_prior_loss(
                            rank_states,
                            positive_actions,
                            rank_state_latents,
                            rank_goal,
                            context_latents,
                            active_mask,
                        )
                    )
                elif negative_actions is not None:
                    pair_actions = torch.stack([positive_actions, negative_actions], dim=1)
                    logits = self.score_action_prior(
                        rank_state_latents,
                        rank_goal,
                        context_latents,
                        active_mask,
                        pair_actions,
                    )
                    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
                    policy_prior_terms.append(F.cross_entropy(logits / self.rank_temperature, labels))
        if self.policy_prior_weight > 0.0 and self.policy_prior_mode != "none" and self.hierarchy_levels:
            for level in self.hierarchy_levels:
                if frames <= level:
                    continue
                valid = masks[:, : level + 1].all(dim=1) & oracle_rows
                if not bool(valid.any()):
                    continue
                positive_macro = actions[:, :level]
                negative_macro = positive_macro.clone()
                negative_macro[:, -1, 2] = (negative_macro[:, -1, 2] % 9) + 1
                macro_actions = torch.stack([positive_macro, negative_macro], dim=1)
                logits = self.score_macro_action_prior(
                    state_latents[:, 0],
                    predicted_goal,
                    context_latents,
                    active_mask,
                    macro_actions,
                    level=level,
                )
                labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
                macro_loss = F.cross_entropy(logits[valid] / self.rank_temperature, labels[valid])
                policy_prior_terms.append(macro_loss)
        if policy_prior_terms:
            policy_prior_loss = torch.stack(policy_prior_terms).mean()

        terminal_corrupt_loss = state_latents.sum() * 0.0
        if corrupt_goals is not None:
            corrupt_latents = self.encode_state(corrupt_goals, context_latents, clue_mask, editable_mask, active_mask)
            good_d = self.distance(goal_target.detach(), predicted_goal, active_mask)
            bad_d = self.distance(corrupt_latents, predicted_goal, active_mask)
            terminal_corrupt_loss = F.softplus((good_d - bad_d + self.rank_margin) / self.rank_temperature).mean()

        sigreg_loss = self._regularizer_loss(state_latents, masks[:, :, None].expand(batch, frames, token_count))
        loss = (
            dynamics_loss
            + self.dense_future_weight * dense_future_loss
            + self.hierarchy_loss_weight * hierarchy_loss
            + self.sigreg_weight * sigreg_loss
            + goal_mse_loss
            + self.goal_nce_weight * goal_nce_loss
            + self.progress_rank_weight * progress_rank_loss
            + self.action_rank_weight * action_rank_loss
            + self.policy_prior_weight * policy_prior_loss
            + self.temporal_straightening_weight * temporal_straightening_loss
            + self.terminal_corrupt_weight * terminal_corrupt_loss
        )
        return GridGoalJEPAOutput(
            loss=loss,
            dynamics_loss=dynamics_loss.detach(),
            dense_future_loss=dense_future_loss.detach(),
            hierarchy_loss=hierarchy_loss.detach(),
            sigreg_loss=sigreg_loss.detach(),
            goal_mse_loss=goal_mse_loss.detach(),
            goal_nce_loss=goal_nce_loss.detach(),
            progress_rank_loss=progress_rank_loss.detach(),
            action_rank_loss=action_rank_loss.detach(),
            policy_prior_loss=policy_prior_loss.detach(),
            temporal_straightening_loss=temporal_straightening_loss.detach(),
            terminal_corrupt_loss=terminal_corrupt_loss.detach(),
            state_latents=state_latents,
            predicted_next_latents=predicted_next,
            predicted_goal_latents=predicted_goal,
            goal_target_latents=goal_target,
            distances=predicted_distances.detach(),
        )

    def _goal_for_current_state(
        self,
        context_latents: torch.Tensor,
        active_mask: torch.Tensor,
        initial_latents: torch.Tensor,
        current_latents: torch.Tensor,
    ) -> torch.Tensor:
        if self.goal_conditioning == "initial_current":
            return self.predict_goal(
                context_latents,
                active_mask,
                initial_latents=initial_latents,
                current_latents=current_latents,
            )
        return self.predict_goal(context_latents, active_mask)

    def _progress_rank_objective(
        self,
        predicted_distances: torch.Tensor,
        oracle_distances: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        terms = []
        if self.progress_rank_target in {"predicted", "both"}:
            terms.append(
                _progress_rank_loss(
                    predicted_distances,
                    masks,
                    margin=self.progress_margin,
                    temperature=self.rank_temperature,
                )
            )
        if self.progress_rank_target in {"oracle", "both"}:
            terms.append(
                _progress_rank_loss(
                    oracle_distances,
                    masks,
                    margin=self.progress_margin,
                    temperature=self.rank_temperature,
                )
            )
        if not terms:
            return predicted_distances.sum() * 0.0
        return torch.stack(terms).mean()

    def _rank_target_distance(
        self,
        latents: torch.Tensor,
        predicted_goal: torch.Tensor,
        oracle_goal: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        terms = []
        if self.action_rank_target in {"predicted", "both"}:
            terms.append(self.distance(latents, predicted_goal, active_mask))
        if self.action_rank_target in {"oracle", "both"}:
            terms.append(self.distance(latents, oracle_goal.detach(), active_mask))
        return torch.stack(terms).mean(dim=0)

    def _listwise_action_rank_loss(
        self,
        rank_states: torch.Tensor,
        positive_actions: torch.Tensor,
        context_latents: torch.Tensor,
        clue_mask: torch.Tensor,
        editable_mask: torch.Tensor,
        active_mask: torch.Tensor,
        predicted_goal: torch.Tensor,
        oracle_goal: torch.Tensor,
    ) -> torch.Tensor:
        losses = []
        batch, rows, cols = rank_states.shape
        for index in range(batch):
            empties = torch.nonzero(rank_states[index] == 0, as_tuple=False)
            if empties.numel() == 0:
                continue
            candidates = []
            labels = []
            pos = positive_actions[index]
            for row_col in empties:
                row = int(row_col[0].item())
                col = int(row_col[1].item())
                for value in range(1, 10):
                    candidates.append([row, col, value])
                    labels.append(row == int(pos[0].item()) and col == int(pos[1].item()) and value == int(pos[2].item()))
            if not any(labels):
                candidates.append([int(pos[0].item()), int(pos[1].item()), int(pos[2].item())])
                labels.append(True)
            if len(candidates) > self.listwise_action_rank_max_actions:
                positive = [candidate for candidate, label in zip(candidates, labels, strict=True) if label][0]
                negatives = [candidate for candidate, label in zip(candidates, labels, strict=True) if not label]
                keep = max(0, self.listwise_action_rank_max_actions - 1)
                candidates = [positive, *negatives[:keep]]
                labels = [True, *([False] * keep)]
            action_t = torch.as_tensor(candidates, dtype=torch.long, device=rank_states.device)
            board_t = rank_states[index : index + 1].expand(action_t.shape[0], rows, cols)
            succ = _apply_set_cell_actions(board_t, action_t)
            context = context_latents[index : index + 1].expand(action_t.shape[0], -1, -1)
            clue = clue_mask[index : index + 1].expand(action_t.shape[0], rows, cols)
            edit = editable_mask[index : index + 1].expand(action_t.shape[0], rows, cols)
            active = active_mask[index : index + 1].expand(action_t.shape[0], rows, cols)
            succ_latents = self.encode_state(succ, context, clue, edit, active)
            pred_goal = predicted_goal[index : index + 1].expand(action_t.shape[0], -1, -1)
            oracle = oracle_goal[index : index + 1].expand(action_t.shape[0], -1, -1)
            distances = self._rank_target_distance(succ_latents, pred_goal, oracle, active)
            label_index = labels.index(True)
            losses.append(F.cross_entropy((-distances / self.rank_temperature).unsqueeze(0), torch.tensor([label_index], device=rank_states.device)))
        if not losses:
            return rank_states.sum() * 0.0
        return torch.stack(losses).mean()

    def _listwise_policy_prior_loss(
        self,
        rank_states: torch.Tensor,
        positive_actions: torch.Tensor,
        state_latents: torch.Tensor,
        goal_latents: torch.Tensor,
        context_latents: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        losses = []
        batch = rank_states.shape[0]
        for index in range(batch):
            empties = torch.nonzero(rank_states[index] == 0, as_tuple=False)
            if empties.numel() == 0:
                continue
            candidates = []
            labels = []
            pos = positive_actions[index]
            for row_col in empties:
                row = int(row_col[0].item())
                col = int(row_col[1].item())
                for value in range(1, 10):
                    candidates.append([row, col, value])
                    labels.append(row == int(pos[0].item()) and col == int(pos[1].item()) and value == int(pos[2].item()))
            if not any(labels):
                candidates.append([int(pos[0].item()), int(pos[1].item()), int(pos[2].item())])
                labels.append(True)
            if len(candidates) > self.listwise_action_rank_max_actions:
                positive = [candidate for candidate, label in zip(candidates, labels, strict=True) if label][0]
                negatives = [candidate for candidate, label in zip(candidates, labels, strict=True) if not label]
                keep = max(0, self.listwise_action_rank_max_actions - 1)
                candidates = [positive, *negatives[:keep]]
                labels = [True, *([False] * keep)]
            action_t = torch.as_tensor(candidates, dtype=torch.long, device=rank_states.device)[None]
            logits = self.score_action_prior(
                state_latents[index : index + 1],
                goal_latents[index : index + 1],
                context_latents[index : index + 1],
                active_mask[index : index + 1],
                action_t,
            )
            label_index = labels.index(True)
            losses.append(F.cross_entropy(logits / self.rank_temperature, torch.tensor([label_index], device=rank_states.device)))
        if not losses:
            return rank_states.sum() * 0.0
        return torch.stack(losses).mean()

    def _condition_state_latents(
        self,
        state_latent: torch.Tensor,
        action: torch.Tensor,
        action_embedding: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.action_conditioning == "action_token" or self.action_conditioning == "action_cross_attention":
            return state_latent
        if self.action_conditioning == "adaln_action":
            if action_embedding is None:
                raise ValueError("AdaLN action conditioning requires an action embedding.")
            scale, shift = self.action_film(action_embedding).chunk(2, dim=-1)
            return state_latent * (1.0 + scale.unsqueeze(-2)) + shift.unsqueeze(-2)
        if self.action_conditioning == "old_local_value":
            values = self._old_local_action_values(action, state_latent.dtype)
            return _add_at_action_positions(state_latent, action, values, rows=self.max_rows, cols=self.max_cols)
        if self.action_conditioning == "old_local_concat":
            values = self._old_local_action_values(action, state_latent.dtype)
            return self._replace_action_cells_with_concat(state_latent, action, values)
        values = self.affected_marker.to(dtype=state_latent.dtype, device=state_latent.device).expand_as(state_latent[:, 0])
        if self.action_conditioning == "local_action_feature":
            action_type = torch.ones_like(action[..., 0])
            values = (
                values
                + self.local_action_type(action_type).to(dtype=state_latent.dtype)
                + self.local_action_digit(action[..., 2].clamp(0, 9)).to(dtype=state_latent.dtype)
            )
        return _add_at_action_positions(state_latent, action, values, rows=self.max_rows, cols=self.max_cols)

    def _old_local_action_values(self, action: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        values = self.local_action_digit(action[..., 2].clamp(0, 9)).to(dtype=dtype)
        return F.layer_norm(values, (self.d_model,)).to(dtype=dtype)

    def _replace_action_cells_with_concat(
        self,
        state_latent: torch.Tensor,
        action: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        if self.old_local_concat is None:
            raise RuntimeError("old_local_concat projection was not initialized.")
        if state_latent.ndim != 3 or action.ndim != 2:
            raise ValueError("Expected state_latent [batch, tokens, dim] and actions [batch, 3].")
        batch_ids = torch.arange(state_latent.shape[0], device=state_latent.device)
        positions = _action_positions(action, rows=self.max_rows, cols=self.max_cols)
        updated = self.old_local_concat(torch.cat([state_latent[batch_ids, positions], values], dim=-1))
        conditioned = state_latent.clone()
        conditioned[batch_ids, positions] = updated
        return conditioned

    def _condition_state_latents_with_macro_actions(self, state_latent: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim != 3 or actions.shape[-1] != 3:
            raise ValueError(f"Macro local conditioning expects actions [batch, steps, 3], got {tuple(actions.shape)}.")
        conditioned = state_latent
        for offset in range(actions.shape[1]):
            action = actions[:, offset]
            values = self._old_local_action_values(action, state_latent.dtype)
            if self.action_conditioning == "old_local_concat":
                conditioned = self._replace_action_cells_with_concat(conditioned, action, values)
            else:
                conditioned = _add_at_action_positions(conditioned, action, values, rows=self.max_rows, cols=self.max_cols)
        return conditioned

    def _dynamics_error(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        actions: torch.Tensor,
        *,
        rows: int,
        cols: int,
        horizon: int = 1,
    ) -> torch.Tensor:
        per_token = (predicted - target).square().mean(dim=-1)
        if self.dynamics_weighting == "uniform":
            return per_token.mean(dim=-1)
        weights = _affected_token_weights(
            actions,
            token_count=rows * cols,
            rows=rows,
            cols=cols,
            affected_weight=self.affected_dynamics_weight,
            horizon=horizon,
        ).to(dtype=per_token.dtype, device=per_token.device)
        return (per_token * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)

    def _regularizer_loss(self, state_latents: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.regularizer == "sigreg":
            return covariance_sigreg(state_latents, mask)
        if self.regularizer == "vicreg":
            return vicreg_regularizer(state_latents, mask)
        return state_latents.sum() * 0.0

    def _reset_ema_target_encoder(self) -> None:
        self.target_embedder.load_state_dict(self.embedder.state_dict())
        self.target_state_encoder.load_state_dict(self.state_encoder.state_dict())
        for module in (self.target_embedder, self.target_state_encoder):
            module.requires_grad_(False)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.float()
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _expand_batch_tokens(tokens: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if tokens.shape == reference.shape:
        return tokens
    if tokens.shape[0] == 1 and tokens.shape[1:] == reference.shape[1:]:
        return tokens.expand(reference.shape[0], -1, -1)
    return tokens


def _masked_summary(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.float().unsqueeze(-1)
    return (tokens * weights).sum(dim=-2) / weights.sum(dim=-2).clamp_min(1.0)


def _off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    rows, cols = matrix.shape
    if rows != cols:
        raise ValueError(f"Expected square matrix, got {tuple(matrix.shape)}.")
    return matrix.flatten()[:-1].view(rows - 1, rows + 1)[:, 1:].flatten()


@torch.no_grad()
def _ema_update(target: nn.Module, source: nn.Module, decay: float) -> None:
    for target_param, source_param in zip(target.parameters(), source.parameters(), strict=True):
        target_param.data.mul_(decay).add_(source_param.data, alpha=1.0 - decay)


def _action_positions(actions: torch.Tensor, *, rows: int, cols: int) -> torch.Tensor:
    row = actions[..., 0].clamp(0, rows - 1)
    col = actions[..., 1].clamp(0, cols - 1)
    return row * cols + col


def _add_at_action_positions(
    state_latent: torch.Tensor,
    actions: torch.Tensor,
    values: torch.Tensor,
    *,
    rows: int,
    cols: int,
) -> torch.Tensor:
    if state_latent.ndim != 3 or actions.ndim != 2:
        raise ValueError("Expected state_latent [batch, tokens, dim] and actions [batch, 3].")
    conditioned = state_latent.clone()
    batch_ids = torch.arange(state_latent.shape[0], device=state_latent.device)
    positions = _action_positions(actions, rows=rows, cols=cols)
    conditioned[batch_ids, positions] = conditioned[batch_ids, positions] + values
    return conditioned


def _affected_token_weights(
    actions: torch.Tensor,
    *,
    token_count: int,
    rows: int,
    cols: int,
    affected_weight: float,
    horizon: int,
) -> torch.Tensor:
    if actions.ndim == 2:
        weights = torch.ones((actions.shape[0], token_count), device=actions.device)
        batch_ids = torch.arange(actions.shape[0], device=actions.device)
        weights[batch_ids, _action_positions(actions, rows=rows, cols=cols)] = float(affected_weight)
        return weights
    if actions.ndim != 3:
        raise ValueError(f"Expected actions [batch, 3] or [batch, frames, 3], got {tuple(actions.shape)}.")
    batch, starts = actions.shape[:2]
    start_count = max(1, starts - horizon + 1)
    weights = torch.ones((batch, start_count, token_count), device=actions.device)
    batch_ids = torch.arange(batch, device=actions.device)[:, None].expand(batch, start_count)
    start_ids = torch.arange(start_count, device=actions.device)[None, :].expand(batch, start_count)
    for offset in range(horizon):
        positions = _action_positions(actions[:, offset : offset + start_count], rows=rows, cols=cols)
        weights[batch_ids, start_ids, positions] = float(affected_weight)
    return weights


def _progress_rank_loss(distances: torch.Tensor, masks: torch.Tensor, *, margin: float, temperature: float) -> torch.Tensor:
    losses = []
    _, frames = distances.shape
    for gap in (1, 4, 8, 16):
        if frames <= gap:
            continue
        valid = masks[:, :-gap] & masks[:, gap:]
        scaled_margin = margin * gap / max(1, frames - 1)
        losses.append(_masked_mean(F.softplus((distances[:, gap:] - distances[:, :-gap] + scaled_margin) / temperature), valid))
    if not losses:
        return distances.sum() * 0.0
    return torch.stack(losses).mean()


def _temporal_straightening_loss(
    state_latents: torch.Tensor,
    goal_latents: torch.Tensor,
    *,
    masks: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    del goal_latents
    if state_latents.shape[1] < 3:
        return state_latents.sum() * 0.0
    if active_mask.ndim == 3:
        active_mask = active_mask.reshape(active_mask.shape[0], -1)
    if active_mask.shape != state_latents.shape[:1] + state_latents.shape[2:3]:
        raise ValueError(
            f"Temporal straightening active mask must have shape "
            f"{tuple(state_latents.shape[:1] + state_latents.shape[2:3])}, got {tuple(active_mask.shape)}."
        )
    if masks.shape != state_latents.shape[:2]:
        raise ValueError(f"Temporal straightening masks must have shape {tuple(state_latents.shape[:2])}, got {tuple(masks.shape)}.")
    valid = masks[:, :-2] & masks[:, 1:-1] & masks[:, 2:]
    if not bool(valid.any()):
        return state_latents.sum() * 0.0
    velocity_a = state_latents[:, 1:-1] - state_latents[:, :-2]
    velocity_b = state_latents[:, 2:] - state_latents[:, 1:-1]
    weights = active_mask[:, None, :, None].to(dtype=state_latents.dtype)
    velocity_a = velocity_a * weights
    velocity_b = velocity_b * weights
    numerator = (velocity_a * velocity_b).sum(dim=(-1, -2))
    denom = velocity_a.square().sum(dim=(-1, -2)).sqrt() * velocity_b.square().sum(dim=(-1, -2)).sqrt()
    cosine = numerator / denom.clamp_min(1.0e-6)
    return _masked_mean(1.0 - cosine, valid)


def _apply_set_cell_actions(boards: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    if boards.ndim != 3 or actions.ndim != 2 or actions.shape[-1] != 3:
        raise ValueError("Expected boards [batch, rows, cols] and actions [batch, 3].")
    out = boards.clone()
    batch, rows, cols = boards.shape
    batch_ids = torch.arange(batch, device=boards.device)
    row = actions[:, 0].clamp(0, rows - 1)
    col = actions[:, 1].clamp(0, cols - 1)
    value = actions[:, 2].clamp_min(0)
    out[batch_ids, row, col] = value
    return out
