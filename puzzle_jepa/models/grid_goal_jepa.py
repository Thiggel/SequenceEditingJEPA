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
    sigreg_loss: torch.Tensor
    goal_mse_loss: torch.Tensor
    goal_nce_loss: torch.Tensor
    progress_rank_loss: torch.Tensor
    action_rank_loss: torch.Tensor
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
        distance_mode: str = "tokenwise",
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.embedder = GridTokenEmbedder(d_model=d_model)
        self.context_encoder = BidirectionalTransformer(
            num_layers=context_layers, d_model=d_model, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout
        )
        self.state_encoder = BidirectionalTransformer(
            num_layers=state_layers, d_model=d_model, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout
        )
        self.action_token = SudokuActionToken(d_model=d_model)
        self.predictor = BidirectionalTransformer(
            num_layers=predictor_layers, d_model=d_model, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout
        )
        self.predictor_out = nn.Linear(d_model, d_model)
        self.goal_decoder = BidirectionalTransformer(
            num_layers=goal_layers, d_model=d_model, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout
        )
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
        self.multi_step_horizons = tuple(int(k) for k in multi_step_horizons)
        if distance_mode not in {"tokenwise", "mean_pooled"}:
            raise ValueError("distance_mode must be 'tokenwise' or 'mean_pooled'.")
        self.distance_mode = str(distance_mode)

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

    def predict_next(self, state_latent: torch.Tensor, action: torch.Tensor, context_latents: torch.Tensor) -> torch.Tensor:
        action_token = self.action_token(action).unsqueeze(-2)
        y = torch.cat([action_token, state_latent], dim=-2)
        y = self.predictor(y, context_latents)
        return self.predictor_out(y[..., 1:, :])

    def predict_goal(self, context_latents: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        queries = self.embedder.query_tokens(active_mask)
        return self.goal_decoder(queries, context_latents)

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
        goal_target = self.encode_state(goals, context_latents, clue_mask, editable_mask, active_mask)
        predicted_goal = self.predict_goal(context_latents, active_mask)

        transition_mask = masks[:, :-1] & masks[:, 1:]
        dynamics_terms = []
        if frames > 1:
            predicted_next = self.predict_next(
                state_latents[:, :-1].reshape(-1, token_count, self.d_model),
                actions[:, :-1].reshape(-1, 3),
                context_latents[:, None].expand(batch, frames - 1, token_count, self.d_model).reshape(
                    -1, token_count, self.d_model
                ),
            ).reshape(batch, frames - 1, token_count, self.d_model)
            one_step_error = (predicted_next - state_latents[:, 1:].detach()).square().mean(dim=(-1, -2))
            dynamics_terms.append(_masked_mean(one_step_error, transition_mask))
        else:
            predicted_next = state_latents[:, :0]
            dynamics_terms.append(state_latents.sum() * 0.0)
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
            target = state_latents[:, horizon : horizon + start_count].detach()
            valid = masks[:, :start_count] & masks[:, horizon : horizon + start_count]
            error = (rollout.reshape(batch, start_count, token_count, self.d_model) - target).square().mean(dim=(-1, -2))
            dynamics_terms.append(_masked_mean(error, valid) / (horizon**0.5))
        dynamics_loss = torch.stack(dynamics_terms).sum()

        active_flat = active_mask.reshape(batch, token_count)
        goal_mse_loss = ((predicted_goal - goal_target.detach()).square().mean(dim=-1) * active_flat.float()).sum() / active_flat.float().sum().clamp_min(1.0)
        pred_summary = _masked_summary(predicted_goal, active_flat)
        target_summary = _masked_summary(goal_target.detach(), active_flat)
        logits = F.normalize(pred_summary, dim=-1) @ F.normalize(target_summary, dim=-1).T / 0.1
        goal_nce_loss = F.cross_entropy(logits, torch.arange(batch, device=boards.device))

        distances = self.distance(
            state_latents.reshape(batch * frames, token_count, self.d_model),
            predicted_goal[:, None].expand(batch, frames, token_count, self.d_model).reshape(
                batch * frames, token_count, self.d_model
            ),
            active_mask[:, None].expand(batch, frames, rows, cols).reshape(batch * frames, rows, cols),
        ).reshape(batch, frames)
        oracle_rows = torch.zeros_like(masks[:, 0], dtype=torch.bool) if oracle_mask is None else oracle_mask
        progress_masks = masks & oracle_rows[:, None]
        progress_rank_loss = _progress_rank_loss(
            distances, progress_masks, margin=self.progress_margin, temperature=self.rank_temperature
        )
        temporal_straightening_loss = _temporal_straightening_loss(
            state_latents,
            predicted_goal,
            masks=progress_masks,
            active_mask=active_flat,
        )

        action_rank_loss = state_latents.sum() * 0.0
        if negative_actions is not None:
            rank_states = boards[:, 0] if action_rank_states is None else action_rank_states
            if positive_actions is None:
                positive_actions = actions[:, 0]
            else:
                positive_actions = positive_actions
            pos_boards = _apply_set_cell_actions(rank_states, positive_actions)
            neg_boards = _apply_set_cell_actions(rank_states, negative_actions)
            pos_latents = self.encode_state(pos_boards, context_latents, clue_mask, editable_mask, active_mask)
            neg_latents = self.encode_state(neg_boards, context_latents, clue_mask, editable_mask, active_mask)
            pos_d = self.distance(pos_latents, predicted_goal, active_mask)
            neg_d = self.distance(neg_latents, predicted_goal, active_mask)
            action_rank_loss = F.softplus((pos_d - neg_d + self.rank_margin) / self.rank_temperature).mean()

        terminal_corrupt_loss = state_latents.sum() * 0.0
        if corrupt_goals is not None:
            corrupt_latents = self.encode_state(corrupt_goals, context_latents, clue_mask, editable_mask, active_mask)
            good_d = self.distance(goal_target.detach(), predicted_goal, active_mask)
            bad_d = self.distance(corrupt_latents, predicted_goal, active_mask)
            terminal_corrupt_loss = F.softplus((good_d - bad_d + self.rank_margin) / self.rank_temperature).mean()

        sigreg_loss = covariance_sigreg(state_latents, masks[:, :, None].expand(batch, frames, token_count))
        loss = (
            dynamics_loss
            + self.sigreg_weight * sigreg_loss
            + goal_mse_loss
            + self.goal_nce_weight * goal_nce_loss
            + self.progress_rank_weight * progress_rank_loss
            + self.action_rank_weight * action_rank_loss
            + self.temporal_straightening_weight * temporal_straightening_loss
            + self.terminal_corrupt_weight * terminal_corrupt_loss
        )
        return GridGoalJEPAOutput(
            loss=loss,
            dynamics_loss=dynamics_loss.detach(),
            sigreg_loss=sigreg_loss.detach(),
            goal_mse_loss=goal_mse_loss.detach(),
            goal_nce_loss=goal_nce_loss.detach(),
            progress_rank_loss=progress_rank_loss.detach(),
            action_rank_loss=action_rank_loss.detach(),
            temporal_straightening_loss=temporal_straightening_loss.detach(),
            terminal_corrupt_loss=terminal_corrupt_loss.detach(),
            state_latents=state_latents,
            predicted_next_latents=predicted_next,
            predicted_goal_latents=predicted_goal,
            goal_target_latents=goal_target,
            distances=distances.detach(),
        )


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.float()
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _masked_summary(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.float().unsqueeze(-1)
    return (tokens * weights).sum(dim=-2) / weights.sum(dim=-2).clamp_min(1.0)


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
    if state_latents.shape[1] < 2:
        return state_latents.sum() * 0.0
    summaries = _masked_summary(
        state_latents.reshape(-1, state_latents.shape[-2], state_latents.shape[-1]),
        active_mask[:, None].expand(state_latents.shape[0], state_latents.shape[1], -1).reshape(
            -1, active_mask.shape[-1]
        ),
    ).reshape(state_latents.shape[0], state_latents.shape[1], state_latents.shape[-1])
    goal_summary = _masked_summary(goal_latents, active_mask)
    step_vec = summaries[:, 1:] - summaries[:, :-1]
    goal_vec = goal_summary[:, None, :] - summaries[:, :-1]
    valid = masks[:, 1:] & masks[:, :-1]
    if not bool(valid.any()):
        return state_latents.sum() * 0.0
    cosine = F.cosine_similarity(step_vec, goal_vec, dim=-1, eps=1.0e-6)
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
