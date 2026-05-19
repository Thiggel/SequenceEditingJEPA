from __future__ import annotations

import torch
from torch import nn
from transformers import PreTrainedModel

from seq_edit_jepa.data.datasets import CorruptionBatch
from seq_edit_jepa.losses import masked_token_cross_entropy
from seq_edit_jepa.models.config import SequenceEditJEPAConfig
from seq_edit_jepa.models.encoder import BidirectionalSequenceEncoder
from seq_edit_jepa.models.seq_edit_jepa import SeqEditJEPAOutput, _coerce_batch
from seq_edit_jepa.models.token_decoder import TokenDecoder


class DenoisingLM(PreTrainedModel):
    """Masked-LM/denoising baseline without JEPA latent transition."""

    config_class = SequenceEditJEPAConfig
    base_model_prefix = "denoising_lm"

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
        self.last_loss_components: dict[str, float] = {}
        self.post_init()

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
        hidden = self.encoder(batch.input_ids, batch.n, batch.attention_mask, batch.segment_ids)
        logits = self.decoder(hidden, attention_mask=batch.attention_mask)
        target_mask = batch.target_mask if batch.target_mask is not None else batch.prev_attention_mask.bool()
        loss = masked_token_cross_entropy(logits, batch.prev_ids, target_mask, forbidden_token_ids=[self.config.mask_token_id])
        components = {"loss/total": loss.detach(), "loss/token_ce": loss.detach()}
        self.last_loss_components = {key: float(value.detach().cpu()) for key, value in components.items()}
        return SeqEditJEPAOutput(loss=loss, logits=logits, components=components)
