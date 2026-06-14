from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale) + shift


class LeWMSIGReg(nn.Module):
    """Step-wise Sketched Isotropic Gaussian Regularizer from LeWorldModel."""

    def __init__(self, *, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        if knots < 2:
            raise ValueError("SIGReg requires at least two integration knots.")
        if num_proj <= 0:
            raise ValueError("num_proj must be positive.")
        self.num_proj = int(num_proj)
        t = torch.linspace(0.0, 3.0, int(knots), dtype=torch.float32)
        dt = 3.0 / float(knots - 1)
        weights = torch.full((knots,), 2.0 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, emb_tbd: torch.Tensor, mask_tb: torch.Tensor | None = None) -> torch.Tensor:
        if emb_tbd.ndim != 3:
            raise ValueError(f"SIGReg expects [time, batch, dim], got {tuple(emb_tbd.shape)}.")
        emb = emb_tbd.float()
        if mask_tb is None:
            if emb_tbd.shape[1] < 2:
                raise ValueError("SIGReg needs batch size >= 2 for a meaningful batch statistic.")
            return self._statistic(emb)
        if mask_tb.shape != emb_tbd.shape[:2]:
            raise ValueError(f"SIGReg mask must have shape {tuple(emb_tbd.shape[:2])}, got {tuple(mask_tb.shape)}.")
        losses = []
        for step in range(emb.shape[0]):
            valid = emb[step, mask_tb[step]]
            if valid.shape[0] >= 2:
                losses.append(self._statistic(valid.unsqueeze(0)))
        if not losses:
            return emb.sum() * 0.0
        return torch.stack(losses).mean()

    def _statistic(self, emb: torch.Tensor) -> torch.Tensor:
        projections = torch.randn(emb.shape[-1], self.num_proj, device=emb.device, dtype=emb.dtype)
        projections = projections / projections.norm(p=2, dim=0, keepdim=True).clamp_min(1.0e-12)
        x_t = (emb @ projections).unsqueeze(-1) * self.t.to(device=emb.device, dtype=emb.dtype)
        phi = self.phi.to(device=emb.device, dtype=emb.dtype)
        weights = self.weights.to(device=emb.device, dtype=emb.dtype)
        err = (x_t.cos().mean(dim=1) - phi).square() + x_t.sin().mean(dim=1).square()
        statistic = (err @ weights) * emb.shape[1]
        return statistic.mean()


class BatchNormProjector(nn.Module):
    """LeWM-style projector: Linear -> BatchNorm -> GELU -> Linear."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 2048):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_flat = x.reshape(-1, original_shape[-1])
        if self.training and x_flat.shape[0] == 1:
            x_flat = self.fc1(x_flat)
        else:
            x_flat = self.bn(self.fc1(x_flat))
        x_flat = self.fc2(self.act(x_flat))
        return x_flat.reshape(*original_shape[:-1], -1)


class SudokuBoardEncoder(nn.Module):
    """6-layer bidirectional transformer encoder over the current Sudoku board only."""

    def __init__(
        self,
        *,
        d_model: int = 128,
        latent_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        projector_hidden_dim: int = 2048,
    ):
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads.")
        self.d_model = int(d_model)
        self.latent_dim = int(latent_dim)
        self.digit_embedding = nn.Embedding(10, d_model)
        self.row_embedding = nn.Embedding(9, d_model)
        self.col_embedding = nn.Embedding(9, d_model)
        self.box_embedding = nn.Embedding(9, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=int(d_model * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.projector = BatchNormProjector(d_model, latent_dim, hidden_dim=projector_hidden_dim)

    def forward(self, boards: torch.Tensor) -> torch.Tensor:
        if boards.ndim != 3 or boards.shape[-2:] != (9, 9):
            raise ValueError(f"SudokuBoardEncoder expects [batch, 9, 9], got {tuple(boards.shape)}.")
        if boards.min() < 0 or boards.max() > 9:
            raise ValueError("Sudoku board tokens must be in [0, 9].")
        batch = boards.shape[0]
        rows = torch.arange(9, device=boards.device).view(1, 9, 1).expand(batch, 9, 9)
        cols = torch.arange(9, device=boards.device).view(1, 1, 9).expand(batch, 9, 9)
        boxes = (rows // 3) * 3 + cols // 3
        x = (
            self.digit_embedding(boards)
            + self.row_embedding(rows)
            + self.col_embedding(cols)
            + self.box_embedding(boxes)
        )
        x = x.reshape(batch, 81, self.d_model)
        cls = self.cls_token.expand(batch, -1, -1)
        x = torch.cat([cls, x], dim=1)
        encoded = self.norm(self.transformer(x))[:, 0]
        return self.projector(encoded)


class SudokuActionEncoder(nn.Module):
    """Small row/column/digit action embedding projected to an AdaLN condition."""

    def __init__(self, *, component_dim: int = 8, action_dim: int = 32):
        super().__init__()
        if component_dim <= 0 or action_dim <= 0:
            raise ValueError("component_dim and action_dim must be positive.")
        self.row_embedding = nn.Embedding(9, component_dim)
        self.col_embedding = nn.Embedding(9, component_dim)
        self.digit_embedding = nn.Embedding(10, component_dim)
        self.project = nn.Sequential(
            nn.Linear(3 * component_dim, action_dim),
            nn.SiLU(),
            nn.Linear(action_dim, action_dim),
        )

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim != 3 or actions.shape[-1] != 3:
            raise ValueError(f"SudokuActionEncoder expects [batch, time, 3], got {tuple(actions.shape)}.")
        rows = actions[..., 0].clamp(0, 8)
        cols = actions[..., 1].clamp(0, 8)
        digits = actions[..., 2].clamp(0, 9)
        x = torch.cat(
            [self.row_embedding(rows), self.col_embedding(cols), self.digit_embedding(digits)],
            dim=-1,
        )
        return self.project(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, *, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1.0e-6)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time = x.shape[1]
        mask = torch.ones(time, time, dtype=torch.bool, device=x.device).triu(1)
        x_norm = self.norm(x)
        out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=mask, need_weights=False)
        return self.dropout(out)


class FeedForward(nn.Module):
    def __init__(self, *, d_model: int, mlp_ratio: float, dropout: float):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1.0e-6)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x))


class AdaLNCausalBlock(nn.Module):
    """Causal transformer block with AdaLN-zero action conditioning."""

    def __init__(self, *, d_model: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.attn = CausalSelfAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.mlp = FeedForward(d_model=d_model, mlp_ratio=mlp_ratio, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1.0e-6)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1.0e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0.0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0.0)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(condition).chunk(
            6, dim=-1
        )
        attn_in = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.attn(attn_in)
        mlp_in = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.mlp(mlp_in)
        return x


class LeWMPredictor(nn.Module):
    """6-layer causal autoregressive transformer over encoded board states."""

    def __init__(
        self,
        *,
        latent_dim: int = 128,
        d_model: int = 128,
        action_dim: int = 32,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_history: int = 81,
        projector_hidden_dim: int = 2048,
    ):
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads.")
        self.input_proj = nn.Linear(latent_dim, d_model) if latent_dim != d_model else nn.Identity()
        self.action_proj = nn.Linear(action_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_history, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                AdaLNCausalBlock(d_model=d_model, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.projector = BatchNormProjector(d_model, latent_dim, hidden_dim=projector_hidden_dim)

    def forward(self, embeddings: torch.Tensor, action_embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 3:
            raise ValueError(f"Predictor embeddings must be [batch, time, dim], got {tuple(embeddings.shape)}.")
        if action_embeddings.shape[:2] != embeddings.shape[:2]:
            raise ValueError("Action embeddings must share batch/time dimensions with embeddings.")
        time = embeddings.shape[1]
        if time > self.pos_embedding.shape[1]:
            raise ValueError(f"Sequence length {time} exceeds max_history {self.pos_embedding.shape[1]}.")
        x = self.input_proj(embeddings) + self.pos_embedding[:, :time]
        x = self.dropout(x)
        condition = self.action_proj(action_embeddings)
        for layer in self.layers:
            x = layer(x, condition)
        return self.projector(self.norm(x))


@dataclass(frozen=True, slots=True)
class LeWMLossOutput:
    loss: torch.Tensor
    prediction_loss: torch.Tensor
    sigreg_loss: torch.Tensor
    value_loss: torch.Tensor
    embeddings: torch.Tensor
    predicted_embeddings: torch.Tensor
    goal_distances: torch.Tensor
    predicted_goal_distances: torch.Tensor


class LeWMSudokuModel(nn.Module):
    def __init__(
        self,
        *,
        d_model: int = 128,
        latent_dim: int = 128,
        encoder_layers: int = 6,
        predictor_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        action_component_dim: int = 8,
        action_dim: int = 32,
        max_history: int = 81,
        projector_hidden_dim: int = 2048,
        sigreg_knots: int = 17,
        sigreg_projections: int = 1024,
        sigreg_weight: float = 0.1,
        value_weight: float = 1.0,
        stop_gradient_target: bool = False,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.sigreg_weight = float(sigreg_weight)
        self.value_weight = float(value_weight)
        self.stop_gradient_target = bool(stop_gradient_target)
        self.encoder = SudokuBoardEncoder(
            d_model=d_model,
            latent_dim=latent_dim,
            num_layers=encoder_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            projector_hidden_dim=projector_hidden_dim,
        )
        self.action_encoder = SudokuActionEncoder(component_dim=action_component_dim, action_dim=action_dim)
        self.predictor = LeWMPredictor(
            latent_dim=latent_dim,
            d_model=d_model,
            action_dim=action_dim,
            num_layers=predictor_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            max_history=max_history,
            projector_hidden_dim=projector_hidden_dim,
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 2 * latent_dim),
            nn.GELU(),
            nn.Linear(2 * latent_dim, 1),
        )
        self.sigreg = LeWMSIGReg(knots=sigreg_knots, num_proj=sigreg_projections)

    def encode_board(self, boards: torch.Tensor) -> torch.Tensor:
        return self.encoder(boards)

    def encode_sequence(self, boards: torch.Tensor) -> torch.Tensor:
        if boards.ndim != 4 or boards.shape[-2:] != (9, 9):
            raise ValueError(f"Expected board sequence [batch, time, 9, 9], got {tuple(boards.shape)}.")
        batch, time = boards.shape[:2]
        flat = boards.reshape(batch * time, 9, 9)
        emb = self.encode_board(flat)
        return emb.reshape(batch, time, -1)

    def predict_sequence(self, embeddings: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        action_embeddings = self.action_encoder(actions)
        return self.predictor(embeddings, action_embeddings)

    def forward(
        self,
        boards: torch.Tensor,
        actions: torch.Tensor,
        goals: torch.Tensor,
        masks: torch.Tensor | None = None,
    ) -> LeWMLossOutput:
        embeddings = self.encode_sequence(boards)
        predicted = self.predict_sequence(embeddings, actions)
        target = embeddings[:, 1:]
        if self.stop_gradient_target:
            target = target.detach()
        if masks is None:
            masks = torch.ones(boards.shape[:2], dtype=torch.bool, device=boards.device)
        if masks.shape != boards.shape[:2]:
            raise ValueError(f"masks must have shape {tuple(boards.shape[:2])}, got {tuple(masks.shape)}.")
        transition_mask = masks[:, :-1] & masks[:, 1:]
        prediction_loss = _masked_mse(predicted[:, :-1], target, transition_mask)
        sigreg_loss = self.sigreg(embeddings.transpose(0, 1), masks.transpose(0, 1))

        goal_embeddings = self.encode_board(goals)
        goal_distances = torch.linalg.vector_norm(
            embeddings.detach() - goal_embeddings.detach().unsqueeze(1),
            dim=-1,
        )
        predicted_goal_distances = self.value_head(embeddings).squeeze(-1)
        value_loss = _masked_scalar_mse(predicted_goal_distances, goal_distances, masks)
        loss = prediction_loss + self.sigreg_weight * sigreg_loss + self.value_weight * value_loss
        return LeWMLossOutput(
            loss=loss,
            prediction_loss=prediction_loss.detach(),
            sigreg_loss=sigreg_loss.detach(),
            value_loss=value_loss.detach(),
            embeddings=embeddings,
            predicted_embeddings=predicted,
            goal_distances=goal_distances.detach(),
            predicted_goal_distances=predicted_goal_distances,
        )

    @torch.no_grad()
    def score_value(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.value_head(embeddings).squeeze(-1)

    def rollout_latent(self, start_embedding: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Roll out a batch of action sequences from a current latent state.

        Args:
            start_embedding: `[batch, latent_dim]`
            actions: `[batch, horizon, 3]`
        Returns:
            Latent trajectory `[batch, horizon + 1, latent_dim]`.
        """

        if start_embedding.ndim != 2:
            raise ValueError("start_embedding must have shape [batch, latent_dim].")
        if actions.ndim != 3 or actions.shape[-1] != 3:
            raise ValueError("actions must have shape [batch, horizon, 3].")
        embeddings = start_embedding[:, None]
        used_actions = actions[:, :1]
        for index in range(actions.shape[1]):
            if index > 0:
                used_actions = torch.cat([used_actions, actions[:, index : index + 1]], dim=1)
            pred = self.predict_sequence(embeddings, used_actions)[:, -1:]
            embeddings = torch.cat([embeddings, pred], dim=1)
        return embeddings


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per_item = F.mse_loss(pred, target, reduction="none").mean(dim=-1)
    return _masked_mean(per_item, mask)


def _masked_scalar_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return _masked_mean((pred - target).square(), mask)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(dtype=values.dtype)
    denom = mask_f.sum().clamp_min(1.0)
    return (values * mask_f).sum() / denom
