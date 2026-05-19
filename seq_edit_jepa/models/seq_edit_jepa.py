from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from seq_edit_jepa.data.datasets import CorruptionBatch, CorruptionPath
from seq_edit_jepa.losses import action_loss, masked_mse, masked_token_cross_entropy, sigreg_loss
from seq_edit_jepa.models.action_policy import ActionPolicy
from seq_edit_jepa.models.config import SequenceEditJEPAConfig
from seq_edit_jepa.models.encoder import BidirectionalSequenceEncoder
from seq_edit_jepa.models.latent_predictor import LatentPredictor
from seq_edit_jepa.models.layers import RMSNorm
from seq_edit_jepa.models.target_encoder import clone_target_encoder, ema_update
from seq_edit_jepa.models.token_decoder import TokenDecoder
from seq_edit_jepa.models.value_head import ValueHead


@dataclass
class SeqEditJEPAOutput(ModelOutput):
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    op_logits: torch.Tensor | None = None
    token_logits: torch.Tensor | None = None
    hidden_pred: torch.Tensor | None = None
    components: dict[str, torch.Tensor] | None = None


class SequenceEditJEPA(PreTrainedModel):
    config_class = SequenceEditJEPAConfig
    base_model_prefix = "seq_edit_jepa"
    supports_gradient_checkpointing = False

    def __init__(self, config: SequenceEditJEPAConfig):
        super().__init__(config)
        self.encoder = BidirectionalSequenceEncoder(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_layers=config.encoder_layers,
            num_heads=config.num_heads,
            max_length=config.max_position_embeddings,
            num_steps=config.num_steps,
            pad_token_id=config.pad_token_id,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            norm_eps=config.norm_eps,
            rope_theta=config.rope_theta,
            qk_norm=config.qk_norm,
            timestep_embedding_size=config.timestep_embedding_size,
        )
        self.policy = ActionPolicy(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            num_ops=config.num_ops,
            num_layers=config.policy_layers,
            intermediate_size=config.intermediate_size,
            num_heads=config.num_heads,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            norm_eps=config.norm_eps,
            rope_theta=config.rope_theta,
            qk_norm=config.qk_norm,
        )
        self.predictor = LatentPredictor(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_layers=config.predictor_layers,
            num_heads=config.num_heads,
            max_length=config.max_position_embeddings,
            num_steps=config.num_steps,
            num_ops=config.num_ops,
            pad_token_id=config.pad_token_id,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            norm_eps=config.norm_eps,
            rope_theta=config.rope_theta,
            qk_norm=config.qk_norm,
            timestep_embedding_size=config.timestep_embedding_size,
            predictor_type=config.predictor_type,
            action_conditioned=config.action_conditioned,
            action_conditioning=config.action_conditioning,
        )
        self.decoder = TokenDecoder(
            config.hidden_size,
            config.vocab_size,
            num_layers=config.decoder_layers,
            intermediate_size=config.intermediate_size,
            num_heads=config.num_heads,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            norm_eps=config.norm_eps,
            rope_theta=config.rope_theta,
            qk_norm=config.qk_norm,
        )
        self.value_head = ValueHead(config.hidden_size)
        self.online_projector = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.target_projector = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.last_loss_components: dict[str, float] = {}
        self.post_init()
        self.target_encoder = clone_target_encoder(self.encoder)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(self, batch: CorruptionBatch | None = None, **kwargs) -> SeqEditJEPAOutput:
        batch = _coerce_batch(batch, kwargs)
        h_n = self.encoder(batch.input_ids, batch.n, batch.attention_mask, batch.segment_ids)
        op_logits, action_token_logits = self.policy(h_n, attention_mask=batch.attention_mask)
        h_pred = self._predict_next_hidden(h_n, op_logits, action_token_logits, batch)
        target_n = batch.target_n if batch.target_n is not None else torch.clamp(batch.n - 1, min=0)
        with torch.no_grad():
            h_target = self.target_encoder(
                batch.prev_ids,
                target_n,
                batch.prev_attention_mask,
                batch.segment_ids,
            )
        h_pred_proj = self.online_projector(h_pred)
        h_target_proj = self.target_projector(h_target).detach()
        dyn = masked_mse(h_pred_proj, h_target_proj, batch.prev_attention_mask)
        token_source = h_pred.detach() if self.config.detach_token_head else h_pred
        logits = self.decoder(token_source, attention_mask=batch.prev_attention_mask)
        target_mask = batch.target_mask if batch.target_mask is not None else batch.prev_attention_mask.bool()
        tok = masked_token_cross_entropy(logits, batch.prev_ids, target_mask, forbidden_token_ids=[self.config.mask_token_id])
        act_mask = batch.editable_mask & batch.attention_mask.bool()
        act, act_components = action_loss(
            op_logits,
            action_token_logits,
            batch.action_ops,
            batch.action_tokens,
            act_mask,
            forbidden_token_ids=[self.config.mask_token_id],
            op_weight=self.config.lambda_action_op,
            token_weight=self.config.lambda_action_token,
        )
        sig = sigreg_loss(h_n, batch.attention_mask)
        val, val_components = self._value_loss(h_pred, batch)
        loss = (
            self.config.lambda_act * act
            + self.config.lambda_dyn * dyn
            + self.config.lambda_tok * tok
            + self.config.lambda_sig * sig
            + self.config.lambda_val * val
        )
        components = {
            "loss/total": loss.detach(),
            "loss/action": act.detach(),
            "loss/dyn_mse": dyn.detach(),
            "loss/token_ce": tok.detach(),
            "loss/sigreg": sig.detach(),
            "loss/value": val.detach(),
            **act_components,
            **val_components,
        }
        self.last_loss_components = {key: float(value.detach().cpu()) for key, value in components.items()}
        return SeqEditJEPAOutput(
            loss=loss,
            logits=logits,
            op_logits=op_logits,
            token_logits=action_token_logits,
            hidden_pred=h_pred,
            components=components,
        )

    def _predict_next_hidden(
        self,
        h_n: torch.Tensor,
        op_logits: torch.Tensor,
        action_token_logits: torch.Tensor,
        batch: CorruptionBatch,
    ) -> torch.Tensor:
        source = str(getattr(self.config, "predictor_action_source", "gold")).strip().lower()
        if source == "gold":
            return self.predictor(h_n, batch.action_ops, batch.action_tokens, batch.n, batch.attention_mask)
        if source == "predicted_argmax":
            pred_ops = op_logits.argmax(dim=-1)
            token_logits = suppress_action_token_logits(action_token_logits, self.config.mask_token_id)
            pred_tokens = token_logits.argmax(dim=-1)
            pred_tokens = torch.where(
                pred_ops == int(self._replace_op_id()),
                pred_tokens,
                torch.full_like(pred_tokens, int(self.config.pad_token_id)),
            )
            return self.predictor(h_n, pred_ops, pred_tokens, batch.n, batch.attention_mask)
        if source == "predicted_soft":
            temperature = max(float(getattr(self.config, "predictor_action_temperature", 1.0)), 1e-4)
            op_probs = (op_logits / temperature).softmax(dim=-1)
            token_logits = suppress_action_token_logits(action_token_logits, self.config.mask_token_id)
            token_probs = (token_logits / temperature).softmax(dim=-1)
            return self.predictor.forward_soft(
                h_n,
                op_probs,
                token_probs,
                batch.n,
                batch.attention_mask,
                replace_op_id=self._replace_op_id(),
            )
        raise ValueError(
            "predictor_action_source must be 'gold', 'predicted_argmax', or "
            f"'predicted_soft', got {source!r}."
        )

    def _replace_op_id(self) -> int:
        from seq_edit_jepa.actions.action_types import Op

        return int(Op.REPLACE)

    def _value_loss(self, h_pred: torch.Tensor, batch: CorruptionBatch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if float(getattr(self.config, "lambda_val", 0.0)) == 0.0:
            zero = h_pred.sum() * 0.0
            return zero, {
                "loss/value_pooled": zero.detach(),
                "loss/value_token": zero.detach(),
            }
        score_mask = batch.editable_mask.bool() & batch.prev_attention_mask.bool()
        value_source = h_pred.detach() if self.config.detach_value_head else h_pred
        token_value, pooled_value = self.value_head(value_source, batch.prev_attention_mask)
        with torch.no_grad():
            goal_n = torch.zeros((batch.clean_ids.shape[0],), dtype=batch.n.dtype, device=batch.clean_ids.device)
            goal_hidden = self.target_encoder(batch.clean_ids, goal_n, batch.prev_attention_mask, batch.segment_ids)
            token_target, pooled_target = self._oracle_goal_value_targets(h_pred, goal_hidden, score_mask)
        weights = score_mask.float()
        token_val = (((token_value - token_target) ** 2) * weights).sum() / weights.sum().clamp_min(1.0)
        pooled_val = F.mse_loss(pooled_value.float(), pooled_target.float())
        total = pooled_val + float(self.config.lambda_value_token) * token_val
        return total, {
            "loss/value_pooled": pooled_val.detach(),
            "loss/value_token": token_val.detach(),
            "metric/value_target_mean": pooled_target.mean().detach(),
            "metric/value_pred_mean": pooled_value.mean().detach(),
        }

    @torch.no_grad()
    def _oracle_goal_value_targets(
        self,
        hidden: torch.Tensor,
        goal_hidden: torch.Tensor,
        score_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred = self.online_projector(hidden.detach())
        goal = self.target_projector(goal_hidden).detach()
        token_target = -((pred - goal).pow(2).mean(dim=-1))
        weights = score_mask.float()
        pooled_target = (token_target * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)
        return token_target, pooled_target

    def rollout_loss(self, path: CorruptionPath, weights: list[float] | None = None) -> torch.Tensor:
        steps = len(path.action_ops)
        if weights is None:
            weights = [1.0] * steps
        h = self.encoder(path.states[0], path.n_values[0], path.attention_masks[0], path.segment_ids)
        total = h.sum() * 0.0
        for index in range(steps):
            h = self.predictor(h, path.action_ops[index], path.action_tokens[index], path.n_values[index], path.attention_masks[index])
            with torch.no_grad():
                target = self.target_encoder(path.states[index + 1], path.n_values[index + 1], path.attention_masks[index + 1], path.segment_ids)
            total = total + float(weights[index]) * masked_mse(
                self.online_projector(h),
                self.target_projector(target).detach(),
                path.attention_masks[index + 1],
            )
        return total

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        ema_update(self.target_encoder, self.encoder, self.config.ema_tau)


def _coerce_batch(batch: CorruptionBatch | None, kwargs: dict) -> CorruptionBatch:
    if batch is not None:
        return batch
    required = [
        "clean_ids",
        "input_ids",
        "prev_ids",
        "attention_mask",
        "prev_attention_mask",
        "editable_mask",
        "segment_ids",
        "n",
        "action_ops",
        "action_tokens",
    ]
    missing = [key for key in required if key not in kwargs]
    if missing:
        raise TypeError(f"Missing model inputs: {missing}")
    optional = {key: kwargs[key] for key in ("target_mask", "target_n") if key in kwargs}
    return CorruptionBatch(**{key: kwargs[key] for key in required}, **optional)


def suppress_action_token_logits(token_logits: torch.Tensor, mask_token_id: int) -> torch.Tensor:
    logits = token_logits.clone()
    if 0 <= int(mask_token_id) < logits.shape[-1]:
        logits[..., int(mask_token_id)] = torch.finfo(logits.dtype).min
    return logits
