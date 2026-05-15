from __future__ import annotations

import torch

from seq_edit_jepa.actions.action_types import Op


def replace_mask_from_ops(ops: torch.Tensor) -> torch.Tensor:
    return ops == int(Op.REPLACE)


def make_keep_actions(shape: torch.Size | tuple[int, ...], token_fill: int, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
    ops = torch.full(shape, int(Op.KEEP), dtype=torch.long, device=device)
    tokens = torch.full(shape, int(token_fill), dtype=torch.long, device=device)
    return ops, tokens
