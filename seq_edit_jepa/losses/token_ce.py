from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F


def suppress_token_logits(logits: torch.Tensor, forbidden_token_ids: Iterable[int] | None = None) -> torch.Tensor:
    ids = [int(token_id) for token_id in (forbidden_token_ids or []) if 0 <= int(token_id) < logits.shape[-1]]
    if not ids:
        return logits
    output = logits.clone()
    fill = torch.finfo(output.dtype).min
    output[..., ids] = fill
    return output


def masked_token_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    forbidden_token_ids: Iterable[int] | None = None,
) -> torch.Tensor:
    valid = mask.bool()
    if not bool(valid.any()):
        return logits.sum() * 0.0
    selected_logits = suppress_token_logits(logits[valid], forbidden_token_ids)
    return F.cross_entropy(selected_logits, targets[valid])
