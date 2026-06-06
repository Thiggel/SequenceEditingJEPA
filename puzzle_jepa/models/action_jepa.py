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


class ActionSequenceEncoder(nn.Module):
    """Encode a fixed span of lower-level actions into one abstract action vector."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_layers: int,
        max_span: int,
        dropout: float,
    ):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.position_embedding = nn.Embedding(max_span, hidden_size)
        self.stack = TransformerStack(num_layers, hidden_size, intermediate_size, num_heads, dropout)
        self.norm = nn.LayerNorm(hidden_size)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, lower_actions: torch.Tensor) -> torch.Tensor:
        if lower_actions.ndim != 3:
            raise ValueError("lower_actions must have shape [batch, span, hidden].")
        batch, span, _hidden = lower_actions.shape
        if span > self.position_embedding.num_embeddings:
            raise ValueError(
                f"action span {span} exceeds encoder capacity {self.position_embedding.num_embeddings}."
            )
        positions = torch.arange(span, device=lower_actions.device)
        tokens = lower_actions + self.position_embedding(positions).unsqueeze(0)
        cls = self.cls_token.expand(batch, -1, -1)
        encoded = self.stack(torch.cat([cls, tokens], dim=1))
        return self.norm(encoded[:, 0])


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
        use_action_policy_head: bool = False,
        use_action_value_head: bool = False,
        hierarchy_levels: int = 1,
        hierarchy_stride: int = 2,
        hierarchy_span: int | None = None,
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
        self.use_action_policy_head = bool(use_action_policy_head)
        self.use_action_value_head = bool(use_action_value_head)
        self.hierarchy_levels = max(1, int(hierarchy_levels))
        self.hierarchy_span = max(1, int(hierarchy_span if hierarchy_span is not None else hierarchy_stride))
        self.hierarchy_stride = self.hierarchy_span
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
        self.higher_action_encoders = nn.ModuleList(
            [
                ActionSequenceEncoder(
                    hidden_size,
                    intermediate_size,
                    num_heads,
                    num_layers=1,
                    max_span=self.hierarchy_span,
                    dropout=dropout,
                )
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
        if self.use_action_policy_head:
            self.action_policy_head = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, max_height * max_width * action_value_vocab_size),
            )
        if self.use_action_value_head:
            self.action_value_head = nn.Sequential(
                nn.LayerNorm(3 * hidden_size),
                nn.Linear(3 * hidden_size, hidden_size),
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
        goal_energy_target_scale: float = 1.0,
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
                target_scale=goal_energy_target_scale,
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
        positive_states: torch.Tensor | None = None,
        negative_states: torch.Tensor | None = None,
        contrastive_loss: str = "none",
        contrastive_temperature: float = 0.1,
        contrastive_margin: float = 0.05,
        contrastive_weight: float = 0.0,
        monotonicity_weight: float = 0.0,
        monotonicity_margin: float = 0.0,
        terminal_correctness_weight: float = 0.0,
        terminal_target_mode: str = "binary",
        terminal_discount: float = 0.99,
        regression_weight: float = 1.0,
        target_scale: float = 1.0,
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
            target_energy = target_energy * float(target_scale)
        regression_loss = F.mse_loss(pred_energy, target_energy)
        loss = float(regression_weight) * regression_loss
        components = {
            "loss/goal_energy_mse": regression_loss.detach(),
            "metric/goal_energy_pred_mean": pred_energy.detach().mean(),
            "metric/goal_energy_target_mean": target_energy.detach().mean(),
        }
        if float(terminal_correctness_weight) > 0.0:
            terminal_targets = self._terminal_value_targets(
                states,
                goals,
                initial_states=initial_states,
                mode=terminal_target_mode,
                discount=terminal_discount,
            ).to(dtype=pred_energy.dtype)
            terminal_loss = F.binary_cross_entropy_with_logits(pred_energy, terminal_targets)
            loss = loss + float(terminal_correctness_weight) * terminal_loss
            components["loss/goal_terminal_bce"] = terminal_loss.detach()
            components["metric/goal_terminal_target_mean"] = terminal_targets.detach().mean()
            components["metric/goal_terminal_prob_mean"] = torch.sigmoid(pred_energy.detach()).mean()
        if float(contrastive_weight) > 0.0:
            if positive_states is None or negative_states is None:
                raise ValueError("positive_states and negative_states are required for contrastive goal-energy loss.")
            contrastive = self.goal_energy_contrastive_loss(
                states,
                goals,
                task_ids=task_ids,
                initial_states=initial_states,
                positive_states=positive_states,
                negative_states=negative_states,
                mode=contrastive_loss,
                temperature=contrastive_temperature,
                margin=contrastive_margin,
            )
            loss = loss + float(contrastive_weight) * contrastive
            components[f"loss/goal_energy_{contrastive_loss}"] = contrastive.detach()
        if float(monotonicity_weight) > 0.0:
            if positive_states is None:
                raise ValueError("positive_states are required for goal-energy monotonicity loss.")
            positive_energy = self.predict_goal_energy(positive_states, initial_states, task_ids)
            monotone = F.relu(float(monotonicity_margin) + positive_energy - pred_energy).mean()
            loss = loss + float(monotonicity_weight) * monotone
            components["loss/goal_energy_monotonicity"] = monotone.detach()
        return ActionConditionedJEPAOutput(
            loss=loss,
            pred_latents=pred_energy[:, None, None],
            target_latents=target_energy[:, None, None],
            components=components,
        )

    def _terminal_value_targets(
        self,
        states: torch.Tensor,
        goals: torch.Tensor,
        *,
        initial_states: torch.Tensor,
        mode: str,
        discount: float,
    ) -> torch.Tensor:
        mode = str(mode)
        if mode == "binary":
            return (states == goals).flatten(start_dim=1).all(dim=1).float()
        if mode != "discounted_reachability":
            raise ValueError("terminal target mode must be 'binary' or 'discounted_reachability'.")
        remaining = (states != goals).flatten(start_dim=1).sum(dim=1).float()
        gamma = min(max(float(discount), 0.0), 1.0)
        targets = torch.pow(torch.full_like(remaining, gamma), remaining)
        fixed_mask = initial_states != 0
        invalid_fixed = ((states != initial_states) & fixed_mask).flatten(start_dim=1).any(dim=1)
        return torch.where(invalid_fixed, torch.zeros_like(targets), targets)

    def goal_energy_contrastive_loss(
        self,
        states: torch.Tensor,
        goals: torch.Tensor,
        *,
        task_ids: torch.Tensor,
        initial_states: torch.Tensor,
        positive_states: torch.Tensor,
        negative_states: torch.Tensor,
        mode: str,
        temperature: float,
        margin: float,
    ) -> torch.Tensor:
        mode = str(mode)
        if mode not in {"nce", "infonce", "margin"}:
            raise ValueError("contrastive goal-energy mode must be 'nce', 'infonce', or 'margin'.")
        if negative_states.ndim != 4:
            raise ValueError("negative_states must have shape [batch, negatives, height, width].")
        batch, negatives, height, width = negative_states.shape
        if batch != states.shape[0]:
            raise ValueError("negative_states batch size must match states.")
        if positive_states.ndim == 3:
            positives = 1
            flat_positives = positive_states
            positive_initial = initial_states
            positive_task_ids = task_ids
        elif positive_states.ndim == 4:
            if positive_states.shape[0] != batch:
                raise ValueError("positive_states batch size must match states.")
            positives = positive_states.shape[1]
            flat_positives = positive_states.reshape(batch * positives, height, width)
            positive_initial = initial_states[:, None].expand(batch, positives, height, width).reshape(
                batch * positives,
                height,
                width,
            )
            positive_task_ids = task_ids[:, None].expand(batch, positives).reshape(batch * positives)
        else:
            raise ValueError("positive_states must have shape [batch, height, width] or [batch, positives, height, width].")
        pos_energy = self.predict_goal_energy(flat_positives, positive_initial, positive_task_ids).reshape(
            batch,
            positives,
        )
        flat_negatives = negative_states.reshape(batch * negatives, height, width)
        repeated_initial = initial_states[:, None].expand(batch, negatives, height, width).reshape(
            batch * negatives,
            height,
            width,
        )
        repeated_task_ids = task_ids[:, None].expand(batch, negatives).reshape(batch * negatives)
        neg_energy = self.predict_goal_energy(flat_negatives, repeated_initial, repeated_task_ids).reshape(
            batch,
            negatives,
        )
        if mode == "margin":
            return F.relu(float(margin) + pos_energy[:, :, None] - neg_energy[:, None, :]).mean()
        temperature = max(float(temperature), 1.0e-6)
        if mode == "infonce":
            pos_logits = -pos_energy / temperature
            neg_logits = -neg_energy / temperature
            numerator = torch.logsumexp(pos_logits, dim=1)
            denominator = torch.logsumexp(torch.cat([pos_logits, neg_logits], dim=1), dim=1)
            return -(numerator - denominator).mean()
        pos_logits = -pos_energy / temperature
        neg_logits = -neg_energy / temperature
        return (
            F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
            + F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
        )

    def action_policy_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> ActionConditionedJEPAOutput:
        if not self.use_action_policy_head:
            raise ValueError("action policy head is disabled for this model.")
        if actions.ndim != 2 or actions.shape[-1] != 4:
            raise ValueError("actions must have shape [batch, 4].")
        task_ids = actions[:, 0]
        latents = self.encoder(states, task_ids=task_ids)
        logits = self.action_policy_head(self._state_summary(latents))
        target = (
            actions[:, 1].clamp(0, self.encoder.max_height - 1)
            * self.max_width
            * self.value_embedding.num_embeddings
            + actions[:, 2].clamp(0, self.max_width - 1) * self.value_embedding.num_embeddings
            + actions[:, 3].clamp(0, self.value_embedding.num_embeddings - 1)
        )
        loss = F.cross_entropy(logits, target)
        accuracy = (logits.argmax(dim=-1) == target).float().mean()
        return ActionConditionedJEPAOutput(
            loss=loss,
            pred_latents=logits[:, None],
            target_latents=target[:, None].to(dtype=logits.dtype),
            components={
                "loss/action_policy_ce": loss.detach(),
                "metric/action_policy_accuracy": accuracy.detach(),
            },
        )

    def predict_action_value(
        self,
        states: torch.Tensor,
        initial_states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_action_value_head:
            raise ValueError("action value head is disabled for this model.")
        if actions.ndim != 2 or actions.shape[-1] != 4:
            raise ValueError("actions must have shape [batch, 4].")
        device = next(self.parameters()).device
        states = states.to(device)
        initial_states = initial_states.to(device)
        actions = actions.to(device)
        if states.ndim == 2:
            states = states.unsqueeze(0)
        if initial_states.ndim == 2:
            initial_states = initial_states.unsqueeze(0)
        task_ids = actions[:, 0]
        state_latents = self.encoder(states, task_ids=task_ids)
        initial_latents = self.encoder(initial_states, task_ids=task_ids)
        features = torch.cat(
            [
                self._state_summary(state_latents),
                self._state_summary(initial_latents),
                self._action_embedding(actions),
            ],
            dim=-1,
        )
        return self.action_value_head(features).squeeze(-1)

    def action_value_loss(
        self,
        states: torch.Tensor,
        initial_states: torch.Tensor,
        actions: torch.Tensor,
        targets: torch.Tensor,
    ) -> ActionConditionedJEPAOutput:
        pred = self.predict_action_value(states, initial_states, actions)
        targets = targets.to(device=pred.device, dtype=pred.dtype)
        loss = F.mse_loss(pred, targets)
        return ActionConditionedJEPAOutput(
            loss=loss,
            pred_latents=pred[:, None, None],
            target_latents=targets[:, None, None],
            components={
                "loss/action_value_mse": loss.detach(),
                "metric/action_value_pred_mean": pred.detach().mean(),
                "metric/action_value_target_mean": targets.detach().mean(),
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
            horizon = self.hierarchy_span**level
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
        if int(level) == 0:
            conditioned = self._condition_latents(latents, actions, height=height, width=width)
        else:
            expected = self.hierarchy_span ** int(level)
            if actions.shape[1] != expected:
                raise ValueError(f"level {level} expects exactly {expected} primitive actions.")
            abstract_action = self.encode_hierarchy_action(actions, level=int(level))
            conditioned = latents + abstract_action.unsqueeze(1)
        predicted = self._predictor_for_level(level)(conditioned)
        if self.predict_residual:
            return latents + predicted
        return predicted

    def predict_latent_from_abstract_action(
        self,
        latents: torch.Tensor,
        abstract_actions: torch.Tensor,
        *,
        level: int,
    ) -> torch.Tensor:
        if latents.ndim != 3:
            raise ValueError("latents must have shape [batch, tokens, hidden].")
        if abstract_actions.ndim != 2:
            raise ValueError("abstract_actions must have shape [batch, hidden].")
        if abstract_actions.shape[0] != latents.shape[0]:
            raise ValueError("abstract action batch size must match latents batch size.")
        if abstract_actions.shape[-1] != latents.shape[-1]:
            raise ValueError("abstract action hidden size must match latent hidden size.")
        level = int(level)
        if level <= 0 or level >= self.hierarchy_levels:
            raise ValueError(f"level must be in [1, {self.hierarchy_levels}).")
        conditioned = latents + abstract_actions.unsqueeze(1)
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

    def encode_hierarchy_action(self, actions: torch.Tensor, *, level: int) -> torch.Tensor:
        """Encode K**level primitive actions into one level-action vector."""
        if actions.ndim != 3 or actions.shape[-1] != 4:
            raise ValueError("actions must have shape [batch, steps, 4].")
        level = int(level)
        if level <= 0 or level >= self.hierarchy_levels:
            raise ValueError(f"level must be in [1, {self.hierarchy_levels}).")
        expected = self.hierarchy_span**level
        if actions.shape[1] != expected:
            raise ValueError(f"level {level} expects {expected} primitive actions, got {actions.shape[1]}.")
        if level == 1:
            lower = self._action_embedding(actions.reshape(-1, actions.shape[-1])).reshape(
                actions.shape[0],
                self.hierarchy_span,
                -1,
            )
        else:
            lower_horizon = self.hierarchy_span ** (level - 1)
            grouped = actions.reshape(actions.shape[0] * self.hierarchy_span, lower_horizon, actions.shape[-1])
            lower = self.encode_hierarchy_action(grouped, level=level - 1).reshape(
                actions.shape[0],
                self.hierarchy_span,
                -1,
            )
        return self.higher_action_encoders[level - 1](lower)

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

    @torch.no_grad()
    def score_actions_with_value_head(
        self,
        state: torch.Tensor,
        initial_state: torch.Tensor,
        actions: list[WorldAction],
        task_id: int,
    ) -> torch.Tensor:
        if not self.use_action_value_head:
            raise ValueError("action value head is disabled for this model.")
        device = next(self.parameters()).device
        state = state.to(device)
        initial_state = initial_state.to(device)
        if not actions:
            return torch.empty(0, device=device)
        action_tensor = torch.as_tensor(
            [[task_id, action.row, action.col, action.value] for action in actions],
            dtype=torch.long,
            device=device,
        )
        states = state.unsqueeze(0).expand(len(actions), -1, -1)
        initials = initial_state.unsqueeze(0).expand(len(actions), -1, -1)
        return self.predict_action_value(states, initials, action_tensor)

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

    @torch.no_grad()
    def score_action_sequences_to_goal(
        self,
        state: torch.Tensor,
        action_sequences: torch.Tensor,
        goal: torch.Tensor,
        task_id: int,
        *,
        hierarchy_level: int = 0,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        state = state.to(device)
        goal = goal.to(device)
        action_sequences = action_sequences.to(device)
        if action_sequences.ndim != 3 or action_sequences.shape[-1] != 4:
            raise ValueError("action_sequences must have shape [batch, steps, 4].")
        if state.ndim == 2:
            states = state.unsqueeze(0).expand(action_sequences.shape[0], -1, -1)
        else:
            states = state
        if goal.ndim == 2:
            goals = goal.unsqueeze(0).expand(action_sequences.shape[0], -1, -1)
        else:
            goals = goal
        task_ids = torch.full((action_sequences.shape[0],), task_id, dtype=torch.long, device=device)
        latents = self.encoder(states, task_ids=task_ids)
        height, width = int(states.shape[-2]), int(states.shape[-1])
        level = max(0, int(hierarchy_level))
        if level >= self.hierarchy_levels:
            raise ValueError(f"hierarchy_level must be < {self.hierarchy_levels}.")
        block = self.hierarchy_span**level
        step = 0
        while step < action_sequences.shape[1]:
            remaining = action_sequences.shape[1] - step
            if level > 0 and remaining >= block:
                latents = self.predict_latent_sequence_from_latent(
                    latents,
                    action_sequences[:, step : step + block],
                    height=height,
                    width=width,
                    level=level,
                )
                step += block
            else:
                latents = self.predict_latent_from_latent(
                    latents,
                    action_sequences[:, step],
                    height=height,
                    width=width,
                )
                step += 1
        target = self.target_encoder(goals, task_ids=task_ids)
        return -F.mse_loss(latents, target, reduction="none").mean(dim=(1, 2))

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
