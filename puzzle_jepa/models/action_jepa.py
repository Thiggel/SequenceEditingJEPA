from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from puzzle_jepa.data.worlds import WorldAction
from puzzle_jepa.models.layers import GridEncoder, TransformerStack


@dataclass(slots=True)
class ActionConditionedJEPAOutput:
    loss: torch.Tensor
    pred_latents: torch.Tensor
    target_latents: torch.Tensor
    components: dict[str, torch.Tensor]


class ActionConditionedWorldModel(nn.Module):
    """JEPA-style latent world model: encode state, condition on action, predict next-state latent."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 128,
        intermediate_size: int = 256,
        encoder_layers: int = 2,
        predictor_layers: int = 2,
        num_heads: int = 4,
        max_height: int = 30,
        max_width: int = 30,
        task_vocab_size: int = 4,
        action_value_vocab_size: int = 16,
        dropout: float = 0.0,
        target_momentum: float = 0.99,
    ):
        super().__init__()
        self.max_width = int(max_width)
        self.target_momentum = float(target_momentum)
        self.encoder = GridEncoder(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_layers=encoder_layers,
            num_heads=num_heads,
            max_height=max_height,
            max_width=max_width,
            task_vocab_size=task_vocab_size,
            dropout=dropout,
        )
        self.target_encoder = GridEncoder(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_layers=encoder_layers,
            num_heads=num_heads,
            max_height=max_height,
            max_width=max_width,
            task_vocab_size=task_vocab_size,
            dropout=0.0,
        )
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        for param in self.target_encoder.parameters():
            param.requires_grad_(False)

        self.task_embedding = nn.Embedding(task_vocab_size, hidden_size)
        self.row_embedding = nn.Embedding(max_height, hidden_size)
        self.col_embedding = nn.Embedding(max_width, hidden_size)
        self.value_embedding = nn.Embedding(action_value_vocab_size, hidden_size)
        self.selected_cell = nn.Parameter(torch.zeros(hidden_size))
        self.action_norm = nn.LayerNorm(hidden_size)
        self.predictor = TransformerStack(predictor_layers, hidden_size, intermediate_size, num_heads, dropout)

    @torch.no_grad()
    def sync_target(self) -> None:
        momentum = self.target_momentum
        for target, online in zip(self.target_encoder.parameters(), self.encoder.parameters(), strict=True):
            target.data.mul_(momentum).add_(online.data, alpha=1.0 - momentum)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
    ) -> ActionConditionedJEPAOutput:
        task_ids = actions[:, 0]
        pred_latents = self.predict_latent(states, actions)
        with torch.no_grad():
            target_latents = self.target_encoder(next_states, task_ids=task_ids)
        if loss_mask is None:
            mask = torch.ones(pred_latents.shape[:2], dtype=torch.bool, device=pred_latents.device)
        else:
            mask = self._flatten_mask(loss_mask, pred_latents.shape[1])
        per_token = F.mse_loss(pred_latents, target_latents, reduction="none").mean(dim=-1)
        loss = per_token[mask].mean()
        return ActionConditionedJEPAOutput(
            loss=loss,
            pred_latents=pred_latents,
            target_latents=target_latents,
            components={"loss/world_model_mse": loss.detach()},
        )

    def predict_latent(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim != 2 or actions.shape[-1] != 4:
            raise ValueError("actions must have shape [batch, 4] = task_id,row,col,value.")
        task_ids = actions[:, 0]
        latents = self.encoder(states, task_ids=task_ids)
        _batch, height, width = states.shape
        return self.predict_latent_from_latent(latents, actions, height=height, width=width)

    def predict_latent_from_latent(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
        *,
        height: int,
        width: int,
    ) -> torch.Tensor:
        if actions.ndim != 2 or actions.shape[-1] != 4:
            raise ValueError("actions must have shape [batch, 4] = task_id,row,col,value.")
        if latents.ndim != 3:
            raise ValueError("latents must have shape [batch, tokens, hidden].")
        batch = latents.shape[0]
        if actions.shape[0] != batch:
            raise ValueError("actions batch size must match latents batch size.")
        if latents.shape[1] != int(height) * int(width):
            raise ValueError("height*width must match latent token length.")
        action_context = self._action_embedding(actions).unsqueeze(1)
        marker = torch.zeros_like(latents)
        positions = actions[:, 1].clamp(0, int(height) - 1) * int(width) + actions[:, 2].clamp(0, int(width) - 1)
        marker[torch.arange(batch, device=latents.device), positions] = self.selected_cell
        return self.predictor(latents + action_context + marker)

    @torch.no_grad()
    def score_actions_to_goal(
        self,
        state: torch.Tensor,
        actions: list[WorldAction],
        goal: torch.Tensor,
        task_id: int,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        state = state.to(device)
        goal = goal.to(device)
        if not actions:
            return torch.empty(0, device=device)
        action_tensor = torch.as_tensor(
            [[task_id, action.row, action.col, action.value] for action in actions],
            dtype=torch.long,
            device=device,
        )
        states = state.unsqueeze(0).expand(len(actions), -1, -1)
        goals = goal.unsqueeze(0).expand(len(actions), -1, -1)
        pred = self.predict_latent(states, action_tensor)
        target = self.target_encoder(goals, task_ids=action_tensor[:, 0])
        return -F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2))

    @torch.no_grad()
    def score_states_to_goal(
        self,
        states: torch.Tensor,
        goals: torch.Tensor,
        task_ids: int | torch.Tensor,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        states = states.to(device)
        goals = goals.to(device)
        if states.ndim == 2:
            states = states.unsqueeze(0)
            goals = goals.unsqueeze(0)
        if isinstance(task_ids, int):
            task_tensor = torch.full((states.shape[0],), task_ids, dtype=torch.long, device=device)
        else:
            task_tensor = task_ids.to(device)
        latents = self.encoder(states, task_ids=task_tensor)
        target = self.target_encoder(goals, task_ids=task_tensor)
        return -F.mse_loss(latents, target, reduction="none").mean(dim=(1, 2))

    def _action_embedding(self, actions: torch.Tensor) -> torch.Tensor:
        task, row, col, value = actions.unbind(dim=-1)
        return self.action_norm(
            self.task_embedding(task)
            + self.row_embedding(row)
            + self.col_embedding(col)
            + self.value_embedding(value)
        )

    @staticmethod
    def _flatten_mask(mask: torch.Tensor, length: int) -> torch.Tensor:
        if mask.ndim == 3:
            mask = mask.reshape(mask.shape[0], -1)
        if mask.shape[-1] != length:
            raise ValueError(f"loss_mask length {mask.shape[-1]} does not match latent length {length}.")
        return mask.bool()
