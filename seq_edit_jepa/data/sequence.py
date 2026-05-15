from __future__ import annotations

from typing import Dict, Iterable, Sequence

import torch


def pad_token_ids(token_ids: Sequence[int], max_length: int, pad_token_id: int) -> Dict[str, torch.Tensor]:
    ids = list(token_ids[:max_length])
    attention = [1] * len(ids)
    if len(ids) < max_length:
        pad = [pad_token_id] * (max_length - len(ids))
        ids.extend(pad)
        attention.extend([0] * len(pad))
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
    }


def causal_lm_labels(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    supervised_positions: Iterable[int] | None = None,
) -> torch.Tensor:
    labels = input_ids.clone()
    labels[attention_mask.eq(0)] = -100
    if supervised_positions is not None:
        keep = torch.zeros_like(labels, dtype=torch.bool)
        for position in supervised_positions:
            if 0 <= int(position) < keep.shape[0]:
                keep[int(position)] = True
        labels[~keep] = -100
    return labels


def tensor_item(item: Dict) -> Dict[str, torch.Tensor]:
    return {key: value for key, value in item.items() if isinstance(value, torch.Tensor)}
