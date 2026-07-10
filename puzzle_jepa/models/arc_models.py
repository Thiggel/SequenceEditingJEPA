from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from puzzle_jepa.data.arc_training import ARC_FEATURE_DIM, ARCBatch


@dataclass(frozen=True, slots=True)
class ARCModelOutput:
    loss: torch.Tensor
    energy_loss: torch.Tensor
    dynamics_loss: torch.Tensor
    logits: torch.Tensor


class ARCGridEncoder(nn.Module):
    def __init__(self, *, d_model: int = 128):
        super().__init__()
        self.color = nn.Embedding(11, d_model)
        self.row = nn.Embedding(30, d_model)
        self.col = nn.Embedding(30, d_model)
        self.active = nn.Embedding(2, d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def forward(self, values: torch.Tensor, active: torch.Tensor | None = None) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError(f"ARCGridEncoder expects values [B,H,W], got {tuple(values.shape)}.")
        batch, height, width = values.shape
        if height != 30 or width != 30:
            raise ValueError(f"ARCGridEncoder expects padded 30x30 values, got {(height, width)}.")
        if active is None:
            active = torch.ones_like(values, dtype=torch.bool)
        rows = torch.arange(height, device=values.device).view(1, height, 1).expand(batch, height, width)
        cols = torch.arange(width, device=values.device).view(1, 1, width).expand(batch, height, width)
        x = (
            self.color(values.clamp(0, 10))
            + self.row(rows)
            + self.col(cols)
            + self.active(active.long())
        )
        x = self.net(x)
        weights = active.float().unsqueeze(-1)
        denom = weights.sum(dim=(1, 2)).clamp_min(1.0)
        return (x * weights).sum(dim=(1, 2)) / denom


class ARCContextEncoder(nn.Module):
    def __init__(self, *, d_model: int = 128):
        super().__init__()
        self.grid = ARCGridEncoder(d_model=d_model)
        self.pair = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        context_inputs: torch.Tensor,
        context_outputs: torch.Tensor,
        context_input_active: torch.Tensor,
        context_output_active: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, max_context, height, width = context_inputs.shape
        inputs = context_inputs.reshape(batch * max_context, height, width)
        outputs = context_outputs.reshape(batch * max_context, height, width)
        input_active = context_input_active.reshape(batch * max_context, height, width)
        output_active = context_output_active.reshape(batch * max_context, height, width)
        in_vec = self.grid(inputs, input_active)
        out_vec = self.grid(outputs, output_active)
        pair_vec = self.pair(torch.cat([in_vec, out_vec], dim=-1)).reshape(batch, max_context, -1)
        weights = context_mask.float().unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (pair_vec * weights).sum(dim=1) / denom


class ARCCandidateScorer(nn.Module):
    def __init__(
        self,
        *,
        d_model: int = 128,
        use_action_features: bool = False,
        use_jepa: bool = False,
        dynamics_weight: float = 0.25,
    ):
        super().__init__()
        self.use_action_features = bool(use_action_features)
        self.use_jepa = bool(use_jepa)
        self.dynamics_weight = float(dynamics_weight)
        self.context = ARCContextEncoder(d_model=d_model)
        self.grid = self.context.grid
        feature_dim = ARC_FEATURE_DIM if self.use_action_features else 0
        self.action_proj = nn.Sequential(
            nn.Linear(ARC_FEATURE_DIM, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.energy = nn.Sequential(
            nn.Linear(3 * d_model + feature_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )
        self.state_proj = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.predictor = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

    def forward(self, batch: ARCBatch) -> ARCModelOutput:
        context = self.context(
            batch.context_inputs,
            batch.context_outputs,
            batch.context_input_active,
            batch.context_output_active,
            batch.context_mask,
        )
        query = self.grid(batch.query, batch.query_active)
        candidate = self.grid(batch.candidate, batch.candidate_active)
        energy_parts = [context, query, candidate]
        if self.use_action_features:
            energy_parts.append(batch.action_features)
        logits = self.energy(torch.cat(energy_parts, dim=-1)).squeeze(-1)
        energy_loss = F.binary_cross_entropy_with_logits(logits, batch.labels)
        dynamics_loss = logits.sum() * 0.0
        if self.use_jepa:
            current = self.grid(batch.current, batch.current_active)
            current_state = self.state_proj(torch.cat([context, query, current], dim=-1))
            target_state = self.state_proj(torch.cat([context, query, candidate], dim=-1)).detach()
            action = self.action_proj(batch.action_features)
            predicted = self.predictor(torch.cat([current_state, action], dim=-1))
            per_record = (predicted - target_state).square().mean(dim=-1)
            if bool(batch.dynamics_valid.any()):
                dynamics_loss = per_record[batch.dynamics_valid].mean()
            else:
                dynamics_loss = predicted.sum() * 0.0
        loss = energy_loss + self.dynamics_weight * dynamics_loss
        return ARCModelOutput(loss=loss, energy_loss=energy_loss, dynamics_loss=dynamics_loss, logits=logits)
