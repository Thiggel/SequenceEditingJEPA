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
        use_task_embedding: bool = True,
        use_selected_cell_marker: bool = True,
        action_injection: str = "global",
        predict_residual: bool = False,
        use_cls_token: bool = False,
        use_goal_energy_head: bool = False,
        hierarchy_levels: int = 1,
        hierarchy_stride: int = 2,
    ):
        super().__init__()
        self.max_width = int(max_width)
        self.target_momentum = float(target_momentum)
        self.use_task_embedding = bool(use_task_embedding)
        self.use_selected_cell_marker = bool(use_selected_cell_marker)
        if action_injection not in {"global", "local_value"}:
            raise ValueError("action_injection must be 'global' or 'local_value'.")
        self.action_injection = action_injection
        self.predict_residual = bool(predict_residual)
        self.use_cls_token = bool(use_cls_token)
        self.use_goal_energy_head = bool(use_goal_energy_head)
        self.hierarchy_levels = max(1, int(hierarchy_levels))
        self.hierarchy_stride = max(1, int(hierarchy_stride))
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
            use_cls_token=self.use_cls_token,
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
            use_cls_token=self.use_cls_token,
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
        self.higher_predictors = nn.ModuleList(
            [
                TransformerStack(predictor_layers, hidden_size, intermediate_size, num_heads, dropout)
                for _ in range(self.hierarchy_levels - 1)
            ]
        )
        if self.use_goal_energy_head:
            self.goal_energy_head = nn.Sequential(
                nn.LayerNorm(2 * hidden_size),
                nn.Linear(2 * hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 1),
            )

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
        loss_weights: torch.Tensor | None = None,
        goals: torch.Tensor | None = None,
        initial_states: torch.Tensor | None = None,
        goal_energy_weight: float = 0.0,
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
        if loss_weights is None:
            loss = per_token[mask].mean()
        else:
            weights = self._flatten_weights(loss_weights, pred_latents.shape[1])
            weights = weights * mask.to(dtype=weights.dtype)
            loss = (per_token * weights).sum() / weights.sum().clamp_min(1.0e-12)
        components = {"loss/world_model_mse": loss.detach()}
        if self.use_goal_energy_head and float(goal_energy_weight) > 0.0:
            if goals is None:
                raise ValueError("goals are required when goal_energy_weight is positive.")
            energy_output = self.goal_energy_loss(
                states,
                goals,
                task_ids=task_ids,
                initial_states=initial_states,
            )
            loss = loss + float(goal_energy_weight) * energy_output.loss
            components.update(energy_output.components)
        return ActionConditionedJEPAOutput(
            loss=loss,
            pred_latents=pred_latents,
            target_latents=target_latents,
            components=components,
        )

    def goal_energy_loss(
        self,
        states: torch.Tensor,
        goals: torch.Tensor,
        *,
        task_ids: torch.Tensor,
        initial_states: torch.Tensor | None = None,
    ) -> ActionConditionedJEPAOutput:
        if not self.use_goal_energy_head:
            raise ValueError("goal energy head is disabled for this model.")
        if initial_states is None:
            initial_states = states
        pred_energy = self.predict_goal_energy(states, initial_states, task_ids)
        with torch.no_grad():
            state_latents = self.target_encoder(states, task_ids=task_ids)
            goal_latents = self.target_encoder(goals, task_ids=task_ids)
            target_energy = F.mse_loss(state_latents, goal_latents, reduction="none").mean(dim=(1, 2))
        loss = F.mse_loss(pred_energy, target_energy)
        return ActionConditionedJEPAOutput(
            loss=loss,
            pred_latents=pred_energy[:, None, None],
            target_latents=target_energy[:, None, None],
            components={
                "loss/goal_energy_mse": loss.detach(),
                "metric/goal_energy_pred_mean": pred_energy.detach().mean(),
                "metric/goal_energy_target_mean": target_energy.detach().mean(),
            },
        )

    def rollout_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        target_states: torch.Tensor,
    ) -> ActionConditionedJEPAOutput:
        if actions.ndim != 3 or actions.shape[-1] != 4:
            raise ValueError("rollout actions must have shape [batch, steps, 4].")
        if target_states.ndim != 4:
            raise ValueError("target_states must have shape [batch, steps, height, width].")
        if actions.shape[:2] != target_states.shape[:2]:
            raise ValueError("actions and target_states must agree on batch and step dimensions.")
        if states.shape[0] != actions.shape[0]:
            raise ValueError("states and actions must agree on batch size.")
        batch, steps = actions.shape[:2]
        height, width = int(states.shape[-2]), int(states.shape[-1])
        task_ids = actions[:, 0, 0]
        latent = self.encoder(states, task_ids=task_ids)
        losses = []
        final_target = None
        for step in range(steps):
            step_actions = actions[:, step]
            latent = self.predict_latent_from_latent(latent, step_actions, height=height, width=width)
            with torch.no_grad():
                target = self.target_encoder(target_states[:, step], task_ids=step_actions[:, 0])
            final_target = target
            losses.append(F.mse_loss(latent, target, reduction="none").mean(dim=-1).mean())
        loss = torch.stack(losses).mean()
        components = {"loss/rollout_mse": loss.detach()}
        for index, step_loss in enumerate(losses, start=1):
            components[f"loss/rollout_step_{index}_mse"] = step_loss.detach()
        if final_target is None:
            raise ValueError("rollout actions must contain at least one step.")
        return ActionConditionedJEPAOutput(
            loss=loss,
            pred_latents=latent,
            target_latents=final_target,
            components=components,
        )

    def hierarchy_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        target_states: torch.Tensor,
    ) -> ActionConditionedJEPAOutput:
        if actions.ndim != 3 or actions.shape[-1] != 4:
            raise ValueError("hierarchy actions must have shape [batch, steps, 4].")
        if target_states.ndim != 4:
            raise ValueError("target_states must have shape [batch, steps, height, width].")
        if actions.shape[:2] != target_states.shape[:2]:
            raise ValueError("actions and target_states must agree on batch and step dimensions.")
        batch, steps = actions.shape[:2]
        if states.shape[0] != batch:
            raise ValueError("states and actions must agree on batch size.")
        height, width = int(states.shape[-2]), int(states.shape[-1])
        task_ids = actions[:, 0, 0]
        latents = self.encoder(states, task_ids=task_ids)
        losses = []
        components: dict[str, torch.Tensor] = {}
        final_pred = None
        final_target = None
        for level in range(self.hierarchy_levels):
            horizon = self.hierarchy_stride**level
            if horizon > steps:
                continue
            pred = self.predict_latent_sequence_from_latent(
                latents,
                actions[:, :horizon],
                height=height,
                width=width,
                level=level,
            )
            with torch.no_grad():
                target = self.target_encoder(target_states[:, horizon - 1], task_ids=task_ids)
            step_loss = F.mse_loss(pred, target, reduction="none").mean(dim=-1).mean()
            losses.append(step_loss)
            components[f"loss/hierarchy_level_{level}_h{horizon}_mse"] = step_loss.detach()
            final_pred = pred
            final_target = target
        if not losses or final_pred is None or final_target is None:
            raise ValueError("No hierarchy levels fit inside the provided rollout.")
        loss = torch.stack(losses).mean()
        components["loss/hierarchy_mse"] = loss.detach()
        return ActionConditionedJEPAOutput(
            loss=loss,
            pred_latents=final_pred,
            target_latents=final_target,
            components=components,
        )

    def predict_latent(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim != 2 or actions.shape[-1] != 4:
            raise ValueError("actions must have shape [batch, 4] = task_id,row,col,value.")
        task_ids = actions[:, 0]
        latents = self.encoder(states, task_ids=task_ids)
        _batch, height, width = states.shape
        return self.predict_latent_from_latent(latents, actions, height=height, width=width)

    def predict_latent_sequence_from_latent(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
        *,
        height: int,
        width: int,
        level: int,
    ) -> torch.Tensor:
        if actions.ndim != 3 or actions.shape[-1] != 4:
            raise ValueError("action sequences must have shape [batch, steps, 4].")
        if int(level) < 0 or int(level) >= self.hierarchy_levels:
            raise ValueError(f"level must be in [0, {self.hierarchy_levels}).")
        conditioned = self._condition_latents(latents, actions, height=height, width=width)
        predicted = self._predictor_for_level(level)(conditioned)
        if self.predict_residual:
            return latents + predicted
        return predicted

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
        expected_tokens = int(height) * int(width) + self._token_offset()
        if latents.shape[1] != expected_tokens:
            raise ValueError("height*width plus optional CLS token must match latent token length.")
        conditioned = self._condition_latents(latents, actions[:, None, :], height=height, width=width)
        predicted = self.predictor(conditioned)
        if self.predict_residual:
            return latents + predicted
        return predicted

    def _condition_latents(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
        *,
        height: int,
        width: int,
    ) -> torch.Tensor:
        if actions.ndim != 3 or actions.shape[-1] != 4:
            raise ValueError("actions must have shape [batch, steps, 4].")
        batch = latents.shape[0]
        if actions.shape[0] != batch:
            raise ValueError("actions batch size must match latents batch size.")
        flat_actions = actions.reshape(-1, actions.shape[-1])
        if self.action_injection == "global":
            action_context = self._action_embedding(flat_actions).reshape(batch, actions.shape[1], -1).mean(dim=1)
            conditioned = latents + action_context.unsqueeze(1)
            if self.use_selected_cell_marker:
                marker = torch.zeros_like(latents)
                for step in range(actions.shape[1]):
                    positions = self._action_positions(actions[:, step], height=height, width=width)
                    marker[torch.arange(batch, device=latents.device), positions] = self.selected_cell
                conditioned = conditioned + marker
        else:
            conditioned = latents.clone()
            for step in range(actions.shape[1]):
                step_actions = actions[:, step]
                positions = self._action_positions(step_actions, height=height, width=width)
                value_context = self.action_norm(self.value_embedding(step_actions[:, 3]))
                conditioned[torch.arange(batch, device=latents.device), positions] = (
                    conditioned[torch.arange(batch, device=latents.device), positions] + value_context
                )
        return conditioned

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
        *,
        initial_states: torch.Tensor | None = None,
        use_goal_energy_head: bool = False,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        states = states.to(device)
        goals = goals.to(device)
        if states.ndim == 2:
            states = states.unsqueeze(0)
            goals = goals.unsqueeze(0)
        if initial_states is not None:
            initial_states = initial_states.to(device)
            if initial_states.ndim == 2:
                initial_states = initial_states.unsqueeze(0)
        if isinstance(task_ids, int):
            task_tensor = torch.full((states.shape[0],), task_ids, dtype=torch.long, device=device)
        else:
            task_tensor = task_ids.to(device)
        if use_goal_energy_head:
            if not self.use_goal_energy_head:
                raise ValueError("goal energy head is disabled for this model.")
            if initial_states is None:
                initial_states = states
            return -self.predict_goal_energy(states, initial_states, task_tensor)
        latents = self.encoder(states, task_ids=task_tensor)
        target = self.target_encoder(goals, task_ids=task_tensor)
        return -F.mse_loss(latents, target, reduction="none").mean(dim=(1, 2))

    def predict_goal_energy(
        self,
        states: torch.Tensor,
        initial_states: torch.Tensor,
        task_ids: int | torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_goal_energy_head:
            raise ValueError("goal energy head is disabled for this model.")
        device = next(self.parameters()).device
        states = states.to(device)
        initial_states = initial_states.to(device)
        if states.ndim == 2:
            states = states.unsqueeze(0)
        if initial_states.ndim == 2:
            initial_states = initial_states.unsqueeze(0)
        if isinstance(task_ids, int):
            task_tensor = torch.full((states.shape[0],), task_ids, dtype=torch.long, device=device)
        else:
            task_tensor = task_ids.to(device)
        state_latents = self.encoder(states, task_ids=task_tensor)
        initial_latents = self.encoder(initial_states, task_ids=task_tensor)
        features = torch.cat([self._state_summary(state_latents), self._state_summary(initial_latents)], dim=-1)
        return self.goal_energy_head(features).squeeze(-1)

    def _action_embedding(self, actions: torch.Tensor) -> torch.Tensor:
        task, row, col, value = actions.unbind(dim=-1)
        action = self.row_embedding(row) + self.col_embedding(col) + self.value_embedding(value)
        if self.use_task_embedding:
            action = action + self.task_embedding(task)
        return self.action_norm(action)

    def _action_positions(self, actions: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
        board_position = actions[:, 1].clamp(0, int(height) - 1) * int(width) + actions[:, 2].clamp(0, int(width) - 1)
        return board_position + self._token_offset()

    def _predictor_for_level(self, level: int) -> TransformerStack:
        if int(level) == 0:
            return self.predictor
        return self.higher_predictors[int(level) - 1]

    def _state_summary(self, latents: torch.Tensor) -> torch.Tensor:
        if self.use_cls_token:
            return latents[:, 0]
        return latents.mean(dim=1)

    def _token_offset(self) -> int:
        return 1 if self.use_cls_token else 0

    def _flatten_mask(self, mask: torch.Tensor, length: int) -> torch.Tensor:
        if mask.ndim == 3:
            mask = mask.reshape(mask.shape[0], -1)
        if mask.shape[-1] == length - self._token_offset():
            prefix = torch.ones(mask.shape[0], self._token_offset(), dtype=torch.bool, device=mask.device)
            mask = torch.cat([prefix, mask.bool()], dim=-1)
        if mask.shape[-1] != length:
            raise ValueError(f"loss_mask length {mask.shape[-1]} does not match latent length {length}.")
        return mask.bool()

    def _flatten_weights(self, weights: torch.Tensor, length: int) -> torch.Tensor:
        if weights.ndim == 3:
            weights = weights.reshape(weights.shape[0], -1)
        if weights.shape[-1] == length - self._token_offset():
            prefix = torch.ones(weights.shape[0], self._token_offset(), dtype=weights.dtype, device=weights.device)
            weights = torch.cat([prefix, weights], dim=-1)
        if weights.shape[-1] != length:
            raise ValueError(f"loss_weights length {weights.shape[-1]} does not match latent length {length}.")
        return weights.float()
