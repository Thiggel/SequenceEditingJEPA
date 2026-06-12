from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from puzzle_jepa.models.layers import TransformerStack


@dataclass(slots=True)
class SigRegJEPAOutput:
    loss: torch.Tensor
    prediction_loss: torch.Tensor
    teacher_forced_loss: torch.Tensor
    recursive_loss: torch.Tensor
    sigreg_loss: torch.Tensor
    goal_energy_loss: torch.Tensor
    pred_latents: torch.Tensor
    target_latents: torch.Tensor


def _make_one_hidden_mlp(input_size: int, hidden_size: int, output_size: int, dropout: float = 0.0) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(int(input_size), int(hidden_size)),
        nn.SiLU(),
        nn.Dropout(float(dropout)),
        nn.Linear(int(hidden_size), int(output_size)),
    )


class BoardMLPEncoder(nn.Module):
    """Encode a fixed Sudoku board into one global latent vector."""

    def __init__(
        self,
        vocab_size: int,
        latent_size: int,
        hidden_size: int,
        max_height: int,
        max_width: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.max_height = int(max_height)
        self.max_width = int(max_width)
        self.net = _make_one_hidden_mlp(
            self.max_height * self.max_width * self.vocab_size,
            hidden_size,
            latent_size,
            dropout,
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim != 3:
            raise ValueError(f"BoardMLPEncoder expects [batch,height,width], got {tuple(states.shape)}.")
        batch, height, width = states.shape
        if (height, width) != (self.max_height, self.max_width):
            raise ValueError(f"Expected board shape {(self.max_height, self.max_width)}, got {(height, width)}.")
        if states.min() < 0 or states.max() >= self.vocab_size:
            raise ValueError("state token outside encoder vocabulary.")
        one_hot = F.one_hot(states, num_classes=self.vocab_size).to(dtype=self.net[0].weight.dtype)
        return self.net(one_hot.reshape(batch, -1))


class BoardCLSTransformerEncoder(nn.Module):
    """Bidirectional token encoder that returns only the CLS state vector."""

    def __init__(
        self,
        vocab_size: int,
        latent_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_layers: int,
        num_heads: int,
        max_height: int,
        max_width: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_size != latent_size:
            raise ValueError("Grid 5 keeps transformer hidden_size equal to latent_size for a single state vector.")
        self.vocab_size = int(vocab_size)
        self.max_height = int(max_height)
        self.max_width = int(max_width)
        self.token_embedding = nn.Embedding(vocab_size, latent_size)
        self.row_embedding = nn.Embedding(max_height, latent_size)
        self.col_embedding = nn.Embedding(max_width, latent_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, latent_size))
        self.stack = TransformerStack(num_layers, latent_size, intermediate_size, num_heads, dropout)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim != 3:
            raise ValueError(f"BoardCLSTransformerEncoder expects [batch,height,width], got {tuple(states.shape)}.")
        batch, height, width = states.shape
        if height > self.max_height or width > self.max_width:
            raise ValueError(f"Board shape {(height, width)} exceeds max {(self.max_height, self.max_width)}.")
        if states.min() < 0 or states.max() >= self.vocab_size:
            raise ValueError("state token outside encoder vocabulary.")
        rows = torch.arange(height, device=states.device).view(1, height, 1).expand(batch, height, width)
        cols = torch.arange(width, device=states.device).view(1, 1, width).expand(batch, height, width)
        tokens = self.token_embedding(states) + self.row_embedding(rows) + self.col_embedding(cols)
        tokens = tokens.reshape(batch, height * width, -1)
        cls = self.cls_token.expand(batch, -1, -1)
        return self.stack(torch.cat([cls, tokens], dim=1))[:, 0]


class ActionEncoder(nn.Module):
    def __init__(self, max_height: int, max_width: int, action_value_vocab_size: int, action_size: int):
        super().__init__()
        self.row_embedding = nn.Embedding(int(max_height), int(action_size))
        self.col_embedding = nn.Embedding(int(max_width), int(action_size))
        self.value_embedding = nn.Embedding(int(action_value_vocab_size), int(action_size))
        self.norm = nn.LayerNorm(int(action_size))

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim < 2 or actions.shape[-1] != 4:
            raise ValueError("actions must end with [task,row,col,value].")
        rows = actions[..., 1].clamp(0, self.row_embedding.num_embeddings - 1)
        cols = actions[..., 2].clamp(0, self.col_embedding.num_embeddings - 1)
        values = actions[..., 3].clamp(0, self.value_embedding.num_embeddings - 1)
        return self.norm(self.row_embedding(rows) + self.col_embedding(cols) + self.value_embedding(values))


class CausalTransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads.")
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(hidden_size)
        self.mlp = _make_one_hidden_mlp(hidden_size, intermediate_size, hidden_size, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        normed = self.attn_norm(x)
        attn, _ = self.attn(normed, normed, normed, attn_mask=causal_mask, need_weights=False)
        x = x + self.dropout(attn)
        return x + self.dropout(self.mlp(self.mlp_norm(x)))


class CausalTransformerPredictor(nn.Module):
    def __init__(
        self,
        latent_size: int,
        action_size: int,
        intermediate_size: int,
        num_layers: int,
        num_heads: int,
        max_steps: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.max_steps = int(max_steps)
        self.action_proj = nn.Linear(action_size, latent_size)
        self.position_embedding = nn.Embedding(max_steps, latent_size)
        self.blocks = nn.ModuleList(
            [
                CausalTransformerBlock(latent_size, intermediate_size, num_heads, dropout)
                for _ in range(int(num_layers))
            ]
        )
        self.norm = nn.LayerNorm(latent_size)
        self.out = nn.Linear(latent_size, latent_size)

    def forward(self, latents: torch.Tensor, action_embeddings: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 3 or action_embeddings.ndim != 3:
            raise ValueError("AR predictor expects latents/actions with shape [batch,steps,features].")
        batch, steps, hidden = latents.shape
        if action_embeddings.shape[:2] != (batch, steps):
            raise ValueError("latent and action sequence dimensions must match.")
        if steps > self.max_steps:
            raise ValueError(f"sequence length {steps} exceeds predictor max_steps {self.max_steps}.")
        positions = torch.arange(steps, device=latents.device)
        x = latents + self.action_proj(action_embeddings) + self.position_embedding(positions).unsqueeze(0)
        causal_mask = torch.triu(torch.ones(steps, steps, dtype=torch.bool, device=latents.device), diagonal=1)
        for block in self.blocks:
            x = block(x, causal_mask)
        return self.out(self.norm(x))


class SigRegActionJEPA(nn.Module):
    """Single-state action-conditioned JEPA with SIGReg anti-collapse loss."""

    def __init__(
        self,
        vocab_size: int,
        latent_size: int = 64,
        encoder_type: str = "mlp",
        predictor_type: str = "mlp",
        predict_delta: bool = False,
        encoder_hidden_size: int | None = None,
        predictor_hidden_size: int | None = None,
        transformer_layers: int = 1,
        predictor_layers: int = 1,
        num_heads: int = 4,
        max_height: int = 9,
        max_width: int = 9,
        action_value_vocab_size: int = 10,
        action_size: int = 16,
        max_rollout_steps: int = 8,
        dropout: float = 0.0,
        sigreg_weight: float = 1.0,
        sigreg_projections: int = 64,
        sigreg_knots: int = 16,
        sigreg_knot_max: float = 5.0,
    ):
        super().__init__()
        if encoder_type not in {"mlp", "cls_transformer"}:
            raise ValueError("encoder_type must be 'mlp' or 'cls_transformer'.")
        if predictor_type not in {"mlp", "ar_transformer"}:
            raise ValueError("predictor_type must be 'mlp' or 'ar_transformer'.")
        self.latent_size = int(latent_size)
        self.encoder_type = encoder_type
        self.predictor_type = predictor_type
        self.predict_delta = bool(predict_delta)
        self.max_rollout_steps = int(max_rollout_steps)
        self.sigreg_weight = float(sigreg_weight)
        self.sigreg_projections = int(sigreg_projections)
        self.sigreg_knots = int(sigreg_knots)
        self.sigreg_knot_max = float(sigreg_knot_max)
        encoder_hidden_size = int(encoder_hidden_size or max(64, 4 * latent_size))
        predictor_hidden_size = int(predictor_hidden_size or max(64, 4 * latent_size))
        if encoder_type == "mlp":
            self.encoder = BoardMLPEncoder(
                vocab_size=vocab_size,
                latent_size=latent_size,
                hidden_size=encoder_hidden_size,
                max_height=max_height,
                max_width=max_width,
                dropout=dropout,
            )
        else:
            self.encoder = BoardCLSTransformerEncoder(
                vocab_size=vocab_size,
                latent_size=latent_size,
                hidden_size=latent_size,
                intermediate_size=encoder_hidden_size,
                num_layers=transformer_layers,
                num_heads=num_heads,
                max_height=max_height,
                max_width=max_width,
                dropout=dropout,
            )
        self.action_encoder = ActionEncoder(max_height, max_width, action_value_vocab_size, action_size)
        if predictor_type == "mlp":
            self.predictor = _make_one_hidden_mlp(latent_size + action_size, predictor_hidden_size, latent_size, dropout)
        else:
            self.predictor = CausalTransformerPredictor(
                latent_size=latent_size,
                action_size=action_size,
                intermediate_size=predictor_hidden_size,
                num_layers=predictor_layers,
                num_heads=num_heads,
                max_steps=max_rollout_steps,
                dropout=dropout,
            )
        self.goal_energy_head = nn.Sequential(
            nn.LayerNorm(2 * latent_size),
            nn.Linear(2 * latent_size, predictor_hidden_size),
            nn.SiLU(),
            nn.Linear(predictor_hidden_size, 1),
        )

    def encode(self, states: torch.Tensor) -> torch.Tensor:
        return self.encoder(states)

    def predict_sequence(self, latents: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 3:
            raise ValueError("latents must have shape [batch,steps,latent].")
        if actions.ndim != 3:
            raise ValueError("actions must have shape [batch,steps,4].")
        action_embeddings = self.action_encoder(actions)
        if self.predictor_type == "mlp":
            pred = self.predictor(torch.cat([latents, action_embeddings], dim=-1))
        else:
            pred = self.predictor(latents, action_embeddings)
        if self.predict_delta:
            pred = latents + pred
        return pred

    def predict_next(self, state_latents: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if state_latents.ndim != 2:
            raise ValueError("state_latents must have shape [batch,latent].")
        if actions.ndim != 2:
            raise ValueError("actions must have shape [batch,4].")
        return self.predict_sequence(state_latents[:, None, :], actions[:, None, :])[:, -1]

    def predict_goal_energy_from_latents(self, current_latents: torch.Tensor, initial_latents: torch.Tensor) -> torch.Tensor:
        if current_latents.shape != initial_latents.shape:
            raise ValueError("current and initial latents must have the same shape.")
        return self.goal_energy_head(torch.cat([current_latents, initial_latents], dim=-1)).squeeze(-1)

    def predict_goal_energy(self, states: torch.Tensor, initial_states: torch.Tensor) -> torch.Tensor:
        if states.ndim == 2:
            states = states.unsqueeze(0)
        if initial_states.ndim == 2:
            initial_states = initial_states.unsqueeze(0)
        if initial_states.shape[0] == 1 and states.shape[0] != 1:
            initial_states = initial_states.expand(states.shape[0], -1, -1)
        current_latents = self.encode(states.to(next(self.parameters()).device))
        initial_latents = self.encode(initial_states.to(next(self.parameters()).device))
        return self.predict_goal_energy_from_latents(current_latents, initial_latents)

    def rollout_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        target_states: torch.Tensor,
        goals: torch.Tensor,
        *,
        goal_energy_weight: float = 1.0,
        recursive_steps: int = 1,
        recursive_weight: float = 0.0,
    ) -> SigRegJEPAOutput:
        if actions.ndim != 3 or actions.shape[-1] != 4:
            raise ValueError("actions must have shape [batch,steps,4].")
        if target_states.ndim != 4:
            raise ValueError("target_states must have shape [batch,steps,height,width].")
        if states.shape[0] != actions.shape[0] or actions.shape[:2] != target_states.shape[:2]:
            raise ValueError("states/actions/target_states dimensions are inconsistent.")
        if goals.ndim != 3 or goals.shape[0] != states.shape[0]:
            raise ValueError("goals must have shape [batch,height,width].")
        state_sequence = torch.cat([states[:, None], target_states], dim=1)
        batch, sequence_length, height, width = state_sequence.shape
        flat_states = state_sequence.reshape(batch * sequence_length, height, width)
        latents = self.encode(flat_states).reshape(batch, sequence_length, self.latent_size)
        inputs = latents[:, :-1]
        targets = latents[:, 1:]
        pred = self.predict_sequence(inputs, actions)
        teacher_forced_loss = F.mse_loss(pred, targets)
        recursive = self.recursive_rollout_loss(latents, actions, recursive_steps)
        prediction_loss = teacher_forced_loss + float(recursive_weight) * recursive
        sigreg = sigreg_loss(
            latents.reshape(batch * sequence_length, self.latent_size),
            projections=self.sigreg_projections,
            knots=self.sigreg_knots,
            knot_max=self.sigreg_knot_max,
        )
        initial_latents = latents[:, :1].expand(batch, sequence_length, self.latent_size)
        pred_energy = self.predict_goal_energy_from_latents(
            latents.reshape(batch * sequence_length, self.latent_size),
            initial_latents.reshape(batch * sequence_length, self.latent_size),
        )
        with torch.no_grad():
            goal_latents = self.encode(goals).unsqueeze(1).expand(batch, sequence_length, self.latent_size)
            target_energy = F.mse_loss(latents, goal_latents, reduction="none").mean(dim=-1).reshape(
                batch * sequence_length
            )
        goal_energy_loss = F.mse_loss(pred_energy, target_energy)
        loss = prediction_loss + self.sigreg_weight * sigreg + float(goal_energy_weight) * goal_energy_loss
        return SigRegJEPAOutput(
            loss=loss,
            prediction_loss=prediction_loss.detach(),
            teacher_forced_loss=teacher_forced_loss.detach(),
            recursive_loss=recursive.detach(),
            sigreg_loss=sigreg.detach(),
            goal_energy_loss=goal_energy_loss.detach(),
            pred_latents=pred,
            target_latents=targets.detach(),
        )

    def recursive_rollout_loss(self, latents: torch.Tensor, actions: torch.Tensor, recursive_steps: int) -> torch.Tensor:
        if recursive_steps <= 1:
            return latents.new_zeros(())
        if latents.ndim != 3 or actions.ndim != 3:
            raise ValueError("recursive rollout expects latents [batch,steps+1,latent] and actions [batch,steps,4].")
        batch, sequence_length, latent_size = latents.shape
        action_steps = actions.shape[1]
        if sequence_length != action_steps + 1:
            raise ValueError("latents must contain exactly one more item than actions.")
        max_horizon = min(int(recursive_steps), action_steps, int(self.max_rollout_steps))
        if max_horizon <= 1:
            return latents.new_zeros(())
        predictions: list[torch.Tensor] = []
        losses = []
        for horizon in range(1, max_horizon + 1):
            valid_starts = action_steps - horizon + 1
            latent_window_parts = [latents[:, :valid_starts]]
            for offset in range(1, horizon):
                latent_window_parts.append(predictions[offset - 1][:, :valid_starts])
            latent_window = torch.stack(latent_window_parts, dim=2).reshape(
                batch * valid_starts,
                horizon,
                latent_size,
            )
            action_window = torch.stack(
                [actions[:, offset : offset + valid_starts] for offset in range(horizon)],
                dim=2,
            ).reshape(batch * valid_starts, horizon, actions.shape[-1])
            pred = self.predict_sequence(latent_window, action_window)[:, -1].reshape(
                batch,
                valid_starts,
                latent_size,
            )
            target = latents[:, horizon : horizon + valid_starts]
            losses.append(F.mse_loss(pred, target))
            predictions.append(pred)
        return torch.stack(losses).mean()

    def score_states_to_goal(self, states: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        if states.ndim == 2:
            states = states.unsqueeze(0)
        if goal.ndim == 2:
            goal = goal.unsqueeze(0)
        if goal.shape[0] == 1 and states.shape[0] != 1:
            goal = goal.expand(states.shape[0], -1, -1)
        latents = self.encode(states.to(next(self.parameters()).device))
        goal_latents = self.encode(goal.to(next(self.parameters()).device))
        return F.mse_loss(latents, goal_latents, reduction="none").mean(dim=-1)


def sigreg_loss(
    embeddings: torch.Tensor,
    *,
    projections: int = 64,
    knots: int = 16,
    knot_max: float = 5.0,
) -> torch.Tensor:
    """Sketched isotropic Gaussian regularizer via random 1D characteristic tests."""

    if embeddings.ndim != 2:
        raise ValueError("SIGReg embeddings must have shape [items,features].")
    items, features = embeddings.shape
    if items < 2:
        return embeddings.new_zeros(())
    directions = torch.randn(int(projections), features, device=embeddings.device, dtype=embeddings.dtype)
    directions = F.normalize(directions, dim=-1)
    projected = embeddings @ directions.t()
    t = torch.linspace(
        float(knot_max) / float(knots),
        float(knot_max),
        int(knots),
        device=embeddings.device,
        dtype=embeddings.dtype,
    )
    values = projected.unsqueeze(-1) * t.view(1, 1, -1)
    empirical_real = torch.cos(values).mean(dim=0)
    empirical_imag = torch.sin(values).mean(dim=0)
    target_real = torch.exp(-0.5 * t.pow(2)).view(1, -1)
    return (empirical_real - target_real).pow(2).mean() + empirical_imag.pow(2).mean()
