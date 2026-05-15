from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
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
        self.decoder = TokenDecoder(config.hidden_size, config.vocab_size)
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
        h_pred = self.predictor(h_n, batch.action_ops, batch.action_tokens, batch.n, batch.attention_mask)
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
        logits = self.decoder(token_source)
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
        )
        sig = sigreg_loss(h_n, batch.attention_mask)
        loss = (
            self.config.lambda_act * act
            + self.config.lambda_dyn * dyn
            + self.config.lambda_tok * tok
            + self.config.lambda_sig * sig
        )
        components = {
            "loss/total": loss.detach(),
            "loss/action": act.detach(),
            "loss/dyn_mse": dyn.detach(),
            "loss/token_ce": tok.detach(),
            "loss/sigreg": sig.detach(),
            **act_components,
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
