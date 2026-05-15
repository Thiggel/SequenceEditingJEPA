from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from seq_edit_jepa.models.config import SequenceEditJEPAConfig
from seq_edit_jepa.models.layers import BidirectionalTransformerStack


class CausalTransformerLM(PreTrainedModel):
    """Decoder-only causal baseline using the same block style as Seq-Edit JEPA."""

    config_class = SequenceEditJEPAConfig
    base_model_prefix = "causal_lm"

    def __init__(self, config: SequenceEditJEPAConfig):
        super().__init__(config)
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = BidirectionalTransformerStack(
            num_layers=config.encoder_layers,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_heads=config.num_heads,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            norm_eps=config.norm_eps,
            rope_theta=config.rope_theta,
            qk_norm=config.qk_norm,
            causal=True,
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
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

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutput:
        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).long()
        hidden = self.dropout(self.token_embedding(input_ids))
        hidden = self.layers(hidden, attention_mask=attention_mask)
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1), ignore_index=-100)
            with torch.no_grad():
                valid = shift_labels != -100
                acc = (shift_logits.argmax(dim=-1)[valid] == shift_labels[valid]).float().mean() if valid.any() else loss.detach() * 0.0
            self.last_loss_components = {"loss/causal_ce": float(loss.detach().cpu()), "metric/causal_token_acc": float(acc.detach().cpu())}
        return CausalLMOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def greedy_generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, max_new_tokens: int, eos_token_id: int | None = None) -> torch.Tensor:
        generated = input_ids.clone()
        mask = attention_mask.clone()
        for _ in range(max_new_tokens):
            logits = self(input_ids=generated, attention_mask=mask).logits
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            mask = torch.cat([mask, torch.ones_like(next_token)], dim=1)
            if eos_token_id is not None and bool((next_token == int(eos_token_id)).all()):
                break
        return generated
