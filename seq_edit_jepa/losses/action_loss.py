from __future__ import annotations

import torch
import torch.nn.functional as F

from seq_edit_jepa.actions.action_types import Op
from seq_edit_jepa.losses.token_ce import suppress_token_logits


def action_loss(
    op_logits: torch.Tensor,
    token_logits: torch.Tensor,
    target_ops: torch.Tensor,
    target_tokens: torch.Tensor,
    mask: torch.Tensor,
    forbidden_token_ids: list[int] | tuple[int, ...] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    valid = mask.bool()
    if valid.any():
        op_loss = F.cross_entropy(op_logits[valid], target_ops[valid])
        op_acc = (op_logits.argmax(dim=-1)[valid] == target_ops[valid]).float().mean()
    else:
        op_loss = op_logits.sum() * 0.0
        op_acc = op_loss.detach()

    replace = valid & (target_ops == int(Op.REPLACE))
    if replace.any():
        replace_logits = suppress_token_logits(token_logits[replace], forbidden_token_ids)
        tok_loss = F.cross_entropy(replace_logits, target_tokens[replace])
        tok_acc = (replace_logits.argmax(dim=-1) == target_tokens[replace]).float().mean()
    else:
        tok_loss = token_logits.sum() * 0.0
        tok_acc = tok_loss.detach()
    return op_loss + tok_loss, {
        "loss/action_op_ce": op_loss.detach(),
        "loss/action_token_ce": tok_loss.detach(),
        "metric/action_op_acc": op_acc.detach(),
        "metric/action_token_acc": tok_acc.detach(),
    }
