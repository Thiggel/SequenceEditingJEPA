from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch.utils.data import IterableDataset


class CorruptionIterableDataset(IterableDataset):
    """Online dataset that samples clean examples and corrupts them for Trainer."""

    def __init__(self, task, corruptor, seq_len: int, split: str, length: int | None = None):
        super().__init__()
        self.task = task
        self.corruptor = corruptor
        self.seq_len = int(seq_len)
        self.split = str(split)
        self.length = None if length is None else int(length)

    def __iter__(self):
        produced = 0
        while self.length is None or produced < self.length:
            clean = self.task.sample_batch(1, self.seq_len, split=self.split, device="cpu")
            batch = self.corruptor.sample_pair(clean)
            item = {
                "clean_ids": batch.clean_ids.squeeze(0),
                "input_ids": batch.input_ids.squeeze(0),
                "prev_ids": batch.prev_ids.squeeze(0),
                "attention_mask": batch.attention_mask.squeeze(0),
                "prev_attention_mask": batch.prev_attention_mask.squeeze(0),
                "editable_mask": batch.editable_mask.squeeze(0),
                "segment_ids": batch.segment_ids.squeeze(0),
                "n": batch.n.squeeze(0),
                "action_ops": batch.action_ops.squeeze(0),
                "action_tokens": batch.action_tokens.squeeze(0),
            }
            if batch.target_mask is not None:
                item["target_mask"] = batch.target_mask.squeeze(0)
            if batch.target_n is not None:
                item["target_n"] = batch.target_n.squeeze(0)
            yield item
            produced += 1


class CorruptionDataCollator:
    def __call__(self, features: list[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        keys = features[0].keys()
        return {key: torch.stack([feature[key] for feature in features], dim=0) for key in keys}
