from __future__ import annotations

import torch

from seq_edit_jepa.actions.apply import apply_fixed_actions
from seq_edit_jepa.data.datasets import CorruptionBatch
from seq_edit_jepa.losses import suppress_token_logits


@torch.no_grad()
def greedy_edit_step(model, batch: CorruptionBatch) -> torch.Tensor:
    output = model(batch)
    mask_token_id = int(getattr(model.config, "mask_token_id", 2))
    if output.op_logits is None or output.token_logits is None:
        token_logits = suppress_token_logits(output.logits, [mask_token_id])
        pred = batch.input_ids.clone()
        target_mask = batch.target_mask if batch.target_mask is not None else batch.prev_attention_mask.bool()
        pred[target_mask.bool()] = token_logits.argmax(dim=-1)[target_mask.bool()]
        return pred
    token_logits = suppress_token_logits(output.token_logits, [mask_token_id])
    return apply_fixed_actions(batch.input_ids, output.op_logits.argmax(dim=-1), token_logits.argmax(dim=-1))
