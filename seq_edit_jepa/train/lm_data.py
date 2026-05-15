from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch.utils.data import IterableDataset


class CleanLMIterableDataset(IterableDataset):
    def __init__(self, task, seq_len: int, split: str, length: int | None = None):
        super().__init__()
        self.task = task
        self.seq_len = int(seq_len)
        self.split = str(split)
        self.length = None if length is None else int(length)

    def __iter__(self):
        produced = 0
        while self.length is None or produced < self.length:
            clean = self.task.sample_batch(1, self.seq_len, split=self.split, device="cpu")
            input_ids = clean.input_ids.squeeze(0)
            attention_mask = clean.attention_mask.squeeze(0)
            labels = input_ids.masked_fill(~attention_mask.bool(), -100)
            yield {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "editable_mask": clean.editable_mask.squeeze(0),
            }
            produced += 1


class CleanLMDataCollator:
    def __call__(self, features: list[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        return {key: torch.stack([feature[key] for feature in features], dim=0) for key in features[0].keys()}
