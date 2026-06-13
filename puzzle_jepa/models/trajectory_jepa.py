from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from puzzle_jepa.models.sigreg_jepa import CausalTransformerBlock, sigreg_loss, vicreg_loss


@dataclass(slots=True)
class TrajectoryJEPAOutput:
    loss: torch.Tensor
    prediction_loss: torch.Tensor
    sigreg_loss: torch.Tensor
    goal_energy_loss: torch.Tensor
    horizon_losses: dict[int, torch.Tensor]
    pred_latents: torch.Tensor
    target_latents: torch.Tensor


def _make_mlp(input_size: int, hidden_size: int, output_size: int, dropout: float = 0.0) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(int(input_size), int(hidden_size)),
        nn.SiLU(),
        nn.Dropout(float(dropout)),
        nn.Linear(int(hidden_size), int(output_size)),
    )


class BoardContextStem(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        height: int,
        width: int,
        d_model: int,
        hidden_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.height = int(height)
        self.width = int(width)
        input_size = self.height * self.width * (2 * self.vocab_size + 1)
        self.net = nn.Sequential(
            nn.Linear(input_size, int(hidden_size)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_size), int(d_model)),
            nn.LayerNorm(int(d_model)),
        )

    def forward(self, states: torch.Tensor, initial_boards: torch.Tensor, clue_masks: torch.Tensor) -> torch.Tensor:
        if states.ndim != 4:
            raise ValueError(f"states must have shape [batch,seq,height,width], got {tuple(states.shape)}.")
        batch, seq_len, height, width = states.shape
        if (height, width) != (self.height, self.width):
            raise ValueError(f"expected board shape {(self.height, self.width)}, got {(height, width)}.")
        if initial_boards.shape != (batch, height, width):
            raise ValueError("initial_boards must have shape [batch,height,width].")
        if clue_masks.shape != (batch, height, width):
            raise ValueError("clue_masks must have shape [batch,height,width].")
        if states.min() < 0 or states.max() >= self.vocab_size:
            raise ValueError("state token outside board vocabulary.")
        state_onehot = F.one_hot(states, num_classes=self.vocab_size).to(dtype=self.net[0].weight.dtype)
        initial_onehot = F.one_hot(initial_boards, num_classes=self.vocab_size).to(dtype=state_onehot.dtype)
        initial_onehot = initial_onehot[:, None].expand(batch, seq_len, height, width, self.vocab_size)
        mask = clue_masks.to(dtype=state_onehot.dtype)[:, None, :, :, None].expand(batch, seq_len, height, width, 1)
        flat = torch.cat([state_onehot, initial_onehot, mask], dim=-1).reshape(batch * seq_len, -1)
        return self.net(flat).reshape(batch, seq_len, -1)


class SudokuActionEmbedding(nn.Module):
    def __init__(self, *, height: int, width: int, value_vocab_size: int, action_dim: int):
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        self.value_vocab_size = int(value_vocab_size)
        self.row_embedding = nn.Embedding(self.height + 1, int(action_dim))
        self.col_embedding = nn.Embedding(self.width + 1, int(action_dim))
        self.value_embedding = nn.Embedding(self.value_vocab_size + 1, int(action_dim))
        self.norm = nn.LayerNorm(int(action_dim))

    @property
    def null_row(self) -> int:
        return self.height

    @property
    def null_col(self) -> int:
        return self.width

    @property
    def null_value(self) -> int:
        return self.value_vocab_size

    def null_actions(self, batch: int, device: torch.device) -> torch.Tensor:
        actions = torch.zeros(batch, 4, dtype=torch.long, device=device)
        actions[:, 1] = self.null_row
        actions[:, 2] = self.null_col
        actions[:, 3] = self.null_value
        return actions

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim < 2 or actions.shape[-1] != 4:
            raise ValueError("actions must end with [task,row,col,value].")
        rows = actions[..., 1].clamp(0, self.null_row)
        cols = actions[..., 2].clamp(0, self.null_col)
        values = actions[..., 3].clamp(0, self.null_value)
        return self.norm(self.row_embedding(rows) + self.col_embedding(cols) + self.value_embedding(values))


class CausalTrajectoryEncoder(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        height: int,
        width: int,
        value_vocab_size: int,
        d_model: int,
        board_hidden_size: int,
        action_dim: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
        max_sequence_steps: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.max_sequence_steps = int(max_sequence_steps)
        self.board_stem = BoardContextStem(
            vocab_size=vocab_size,
            height=height,
            width=width,
            d_model=d_model,
            hidden_size=board_hidden_size,
            dropout=dropout,
        )
        self.action_embedding = SudokuActionEmbedding(
            height=height,
            width=width,
            value_vocab_size=value_vocab_size,
            action_dim=action_dim,
        )
        self.action_proj = nn.Linear(int(action_dim), int(d_model))
        self.position_embedding = nn.Embedding(self.max_sequence_steps + 1, int(d_model))
        self.blocks = nn.ModuleList(
            [
                CausalTransformerBlock(int(d_model), int(intermediate_size), int(num_heads), dropout)
                for _ in range(int(num_layers))
            ]
        )
        self.norm = nn.LayerNorm(int(d_model))

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        initial_boards: torch.Tensor,
        clue_masks: torch.Tensor,
    ) -> torch.Tensor:
        if states.ndim != 4:
            raise ValueError("states must have shape [batch,seq,height,width].")
        if actions.ndim != 3 or actions.shape[-1] != 4:
            raise ValueError("actions must have shape [batch,seq-1,4].")
        batch, seq_len, _height, _width = states.shape
        if actions.shape[:2] != (batch, max(0, seq_len - 1)):
            raise ValueError("actions must contain exactly one fewer item than states.")
        if seq_len > self.max_sequence_steps:
            states = states[:, -self.max_sequence_steps :]
            seq_len = states.shape[1]
            actions = actions[:, -(seq_len - 1) :] if seq_len > 1 else actions[:, :0]
        board_tokens = self.board_stem(states, initial_boards, clue_masks)
        if seq_len == 1:
            prev_actions = self.action_embedding.null_actions(batch, states.device)[:, None, :]
        else:
            null = self.action_embedding.null_actions(batch, states.device)[:, None, :]
            prev_actions = torch.cat([null, actions], dim=1)
        action_tokens = self.action_proj(self.action_embedding(prev_actions))
        positions = torch.arange(seq_len, device=states.device).clamp_max(self.position_embedding.num_embeddings - 1)
        x = board_tokens + action_tokens + self.position_embedding(positions).unsqueeze(0)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=states.device), diagonal=1)
        for block in self.blocks:
            x = block(x, causal_mask)
        return self.norm(x)


class ActionChunkEncoder(nn.Module):
    def __init__(
        self,
        *,
        action_embedding: SudokuActionEmbedding,
        action_dim: int,
        d_model: int,
        max_horizon: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.action_embedding = action_embedding
        self.max_horizon = int(max_horizon)
        self.input_proj = nn.Linear(int(action_dim), int(d_model))
        self.relative_position_embedding = nn.Embedding(self.max_horizon, int(d_model))
        self.horizon_embedding = nn.Embedding(self.max_horizon + 1, int(d_model))
        self.blocks = nn.ModuleList(
            [
                CausalTransformerBlock(int(d_model), int(intermediate_size), int(num_heads), dropout)
                for _ in range(int(num_layers))
            ]
        )
        self.norm = nn.LayerNorm(int(d_model))

    def forward(self, action_chunks: torch.Tensor, horizon: int) -> torch.Tensor:
        if action_chunks.ndim != 4 or action_chunks.shape[-1] != 4:
            raise ValueError("action_chunks must have shape [batch,starts,horizon,4].")
        batch, starts, chunk_len, _ = action_chunks.shape
        if chunk_len != int(horizon):
            raise ValueError("action chunk length must match horizon.")
        if chunk_len <= 0 or chunk_len > self.max_horizon:
            raise ValueError(f"horizon must be in [1,{self.max_horizon}], got {chunk_len}.")
        flat = action_chunks.reshape(batch * starts, chunk_len, 4)
        x = self.input_proj(self.action_embedding(flat))
        positions = torch.arange(chunk_len, device=action_chunks.device)
        x = x + self.relative_position_embedding(positions).unsqueeze(0)
        causal_mask = torch.triu(torch.ones(chunk_len, chunk_len, dtype=torch.bool, device=action_chunks.device), diagonal=1)
        for block in self.blocks:
            x = block(x, causal_mask)
        pooled = self.norm(x[:, -1])
        pooled = pooled + self.horizon_embedding(torch.full((batch * starts,), chunk_len, device=action_chunks.device))
        return pooled.reshape(batch, starts, -1)


class HorizonPredictor(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        max_sequence_steps: int,
        max_horizon: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.max_sequence_steps = int(max_sequence_steps)
        self.state_proj = nn.Linear(int(d_model), int(d_model))
        self.chunk_proj = nn.Linear(int(d_model), int(d_model))
        self.horizon_embedding = nn.Embedding(int(max_horizon) + 1, int(d_model))
        self.position_embedding = nn.Embedding(self.max_sequence_steps, int(d_model))
        self.blocks = nn.ModuleList(
            [
                CausalTransformerBlock(int(d_model), int(intermediate_size), int(num_heads), dropout)
                for _ in range(int(num_layers))
            ]
        )
        self.norm = nn.LayerNorm(int(d_model))
        self.out = nn.Linear(int(d_model), int(d_model))

    def forward(self, start_latents: torch.Tensor, chunk_latents: torch.Tensor, horizon: int) -> torch.Tensor:
        if start_latents.shape != chunk_latents.shape:
            raise ValueError("start_latents and chunk_latents must have the same shape.")
        if start_latents.ndim != 3:
            raise ValueError("predictor inputs must have shape [batch,starts,d_model].")
        batch, starts, _ = start_latents.shape
        if starts > self.max_sequence_steps:
            start_latents = start_latents[:, -self.max_sequence_steps :]
            chunk_latents = chunk_latents[:, -self.max_sequence_steps :]
            starts = start_latents.shape[1]
        horizon_ids = torch.full((batch, starts), int(horizon), dtype=torch.long, device=start_latents.device)
        positions = torch.arange(starts, device=start_latents.device)
        x = (
            self.state_proj(start_latents)
            + self.chunk_proj(chunk_latents)
            + self.horizon_embedding(horizon_ids)
            + self.position_embedding(positions).unsqueeze(0)
        )
        causal_mask = torch.triu(torch.ones(starts, starts, dtype=torch.bool, device=start_latents.device), diagonal=1)
        for block in self.blocks:
            x = block(x, causal_mask)
        return self.out(self.norm(x))


class CausalTrajectoryJEPA(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int = 320,
        board_hidden_size: int = 1280,
        action_dim: int = 32,
        encoder_layers: int = 4,
        predictor_layers: int = 4,
        action_chunk_layers: int = 2,
        num_heads: int = 8,
        intermediate_size: int = 1280,
        max_height: int = 9,
        max_width: int = 9,
        action_value_vocab_size: int = 10,
        max_sequence_steps: int = 33,
        max_horizon: int = 16,
        horizons: Iterable[int] = (1,),
        dropout: float = 0.0,
        stabilizer_type: str = "sigreg",
        sigreg_weight: float = 1.0,
        sigreg_projections: int = 64,
        sigreg_knots: int = 16,
        sigreg_knot_max: float = 5.0,
        vicreg_variance_weight: float = 1.0,
        vicreg_covariance_weight: float = 0.04,
        target_encoder_momentum: float = 0.99,
    ):
        super().__init__()
        if stabilizer_type not in {"sigreg", "vicreg"}:
            raise ValueError("stabilizer_type must be 'sigreg' or 'vicreg'.")
        self.d_model = int(d_model)
        self.max_horizon = int(max_horizon)
        self.horizons = tuple(sorted({int(k) for k in horizons if int(k) > 0}))
        if not self.horizons:
            raise ValueError("at least one positive horizon is required.")
        if max(self.horizons) > self.max_horizon:
            raise ValueError("all horizons must be <= max_horizon.")
        self.stabilizer_type = stabilizer_type
        self.sigreg_weight = float(sigreg_weight)
        self.sigreg_projections = int(sigreg_projections)
        self.sigreg_knots = int(sigreg_knots)
        self.sigreg_knot_max = float(sigreg_knot_max)
        self.vicreg_variance_weight = float(vicreg_variance_weight)
        self.vicreg_covariance_weight = float(vicreg_covariance_weight)
        self.target_encoder_momentum = float(target_encoder_momentum)
        self.encoder = CausalTrajectoryEncoder(
            vocab_size=vocab_size,
            height=max_height,
            width=max_width,
            value_vocab_size=action_value_vocab_size,
            d_model=d_model,
            board_hidden_size=board_hidden_size,
            action_dim=action_dim,
            num_layers=encoder_layers,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            max_sequence_steps=max_sequence_steps,
            dropout=dropout,
        )
        self.target_encoder = copy.deepcopy(self.encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad_(False)
        self.chunk_encoder = ActionChunkEncoder(
            action_embedding=self.encoder.action_embedding,
            action_dim=action_dim,
            d_model=d_model,
            max_horizon=max_horizon,
            num_layers=action_chunk_layers,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            dropout=dropout,
        )
        self.predictor = HorizonPredictor(
            d_model=d_model,
            max_sequence_steps=max_sequence_steps,
            max_horizon=max_horizon,
            num_layers=predictor_layers,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            dropout=dropout,
        )
        self.goal_energy_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, intermediate_size),
            nn.SiLU(),
            nn.Linear(intermediate_size, 1),
        )

    def _default_clue_masks(self, initial_boards: torch.Tensor) -> torch.Tensor:
        return initial_boards.ne(0)

    def _puzzle_boards(self, states: torch.Tensor, clue_masks: torch.Tensor) -> torch.Tensor:
        start = states[:, 0] if states.ndim == 4 else states
        return torch.where(clue_masks, start, torch.zeros_like(start))

    def encode_context(
        self,
        states: torch.Tensor,
        actions: torch.Tensor | None = None,
        *,
        initial_boards: torch.Tensor | None = None,
        clue_masks: torch.Tensor | None = None,
        target: bool = False,
    ) -> torch.Tensor:
        if states.ndim == 3:
            states = states[:, None]
        if states.ndim != 4:
            raise ValueError("states must have shape [batch,seq,height,width] or [batch,height,width].")
        batch, seq_len, height, width = states.shape
        if actions is None:
            actions = torch.zeros(batch, max(0, seq_len - 1), 4, dtype=torch.long, device=states.device)
        if initial_boards is None:
            initial_boards = states[:, 0]
        if clue_masks is None:
            clue_masks = self._default_clue_masks(initial_boards)
        encoder = self.target_encoder if target else self.encoder
        return encoder(states, actions, initial_boards, clue_masks)

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        momentum = float(self.target_encoder_momentum)
        for online, target in zip(self.encoder.parameters(), self.target_encoder.parameters(), strict=True):
            target.data.mul_(momentum).add_(online.data, alpha=1.0 - momentum)
        for online_buffer, target_buffer in zip(self.encoder.buffers(), self.target_encoder.buffers(), strict=True):
            target_buffer.copy_(online_buffer)

    def predict_horizon(self, start_latents: torch.Tensor, action_chunks: torch.Tensor, horizon: int) -> torch.Tensor:
        if action_chunks.ndim != 4:
            raise ValueError("action_chunks must have shape [batch,starts,horizon,4].")
        chunk_latents = self.chunk_encoder(action_chunks, int(horizon))
        return self.predictor(start_latents, chunk_latents, int(horizon))

    def predict_goal_energy_from_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return self.goal_energy_head(latents).squeeze(-1)

    def rollout_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        target_states: torch.Tensor,
        goals: torch.Tensor,
        *,
        clue_masks: torch.Tensor | None = None,
        goal_energy_weight: float = 1.0,
    ) -> TrajectoryJEPAOutput:
        if target_states.ndim != 4:
            raise ValueError("target_states must have shape [batch,steps,height,width].")
        if actions.ndim != 3 or actions.shape[-1] != 4:
            raise ValueError("actions must have shape [batch,steps,4].")
        state_sequence = torch.cat([states[:, None], target_states], dim=1)
        batch, sequence_length, _height, _width = state_sequence.shape
        if actions.shape[:2] != (batch, sequence_length - 1):
            raise ValueError("actions and target_states sequence lengths are inconsistent.")
        if clue_masks is None:
            clue_masks = self._default_clue_masks(states)
        initial_boards = self._puzzle_boards(state_sequence, clue_masks)
        online = self.encode_context(
            state_sequence,
            actions,
            initial_boards=initial_boards,
            clue_masks=clue_masks,
            target=False,
        )
        with torch.no_grad():
            target_latents = self.encode_context(
                state_sequence,
                actions,
                initial_boards=initial_boards,
                clue_masks=clue_masks,
                target=True,
            )
        losses = []
        horizon_losses: dict[int, torch.Tensor] = {}
        first_pred = online.new_zeros((batch, 0, self.d_model))
        first_target = online.new_zeros((batch, 0, self.d_model))
        action_steps = actions.shape[1]
        for horizon in self.horizons:
            if horizon > action_steps:
                continue
            starts = action_steps - horizon + 1
            start_latents = online[:, :starts]
            chunks = torch.stack([actions[:, offset : offset + starts] for offset in range(horizon)], dim=2)
            pred = self.predict_horizon(start_latents, chunks, horizon)
            target = target_latents[:, horizon : horizon + starts]
            loss = F.mse_loss(pred, target)
            horizon_losses[int(horizon)] = loss.detach()
            losses.append(loss)
            if horizon == self.horizons[0]:
                first_pred = pred
                first_target = target
        if not losses:
            raise ValueError("no configured horizon fits inside the sampled rollout.")
        prediction_loss = torch.stack(losses).mean()
        stabilizer = self.stabilizer_loss(online.reshape(batch * sequence_length, self.d_model))
        with torch.no_grad():
            goal_sequence = goals[:, None]
            empty_actions = actions[:, :0]
            goal_latents = self.encode_context(
                goal_sequence,
                empty_actions,
                initial_boards=initial_boards,
                clue_masks=clue_masks,
                target=True,
            )[:, 0]
            target_energy = F.mse_loss(online, goal_latents[:, None].expand_as(online), reduction="none").mean(dim=-1)
        pred_energy = self.predict_goal_energy_from_latents(online).reshape_as(target_energy)
        goal_energy_loss = F.mse_loss(pred_energy, target_energy)
        loss = prediction_loss + self.sigreg_weight * stabilizer + float(goal_energy_weight) * goal_energy_loss
        return TrajectoryJEPAOutput(
            loss=loss,
            prediction_loss=prediction_loss.detach(),
            sigreg_loss=stabilizer.detach(),
            goal_energy_loss=goal_energy_loss.detach(),
            horizon_losses=horizon_losses,
            pred_latents=first_pred,
            target_latents=first_target.detach(),
        )

    def stabilizer_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.stabilizer_type == "sigreg":
            return sigreg_loss(
                embeddings,
                projections=self.sigreg_projections,
                knots=self.sigreg_knots,
                knot_max=self.sigreg_knot_max,
            )
        if self.stabilizer_type == "vicreg":
            return vicreg_loss(
                embeddings,
                variance_weight=self.vicreg_variance_weight,
                covariance_weight=self.vicreg_covariance_weight,
            )
        raise ValueError(f"unknown stabilizer_type {self.stabilizer_type!r}.")

