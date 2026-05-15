from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

import torch

from seq_edit_jepa.data.datasets import CleanBatch


class SequenceTask(ABC):
    """Open extension point for sequence sources and task-specific probes."""

    def __init__(self, config: dict[str, Any]):
        self.config = dict(config)

    @abstractmethod
    def build_tokenizer(self):
        raise NotImplementedError

    @abstractmethod
    def sample_batch(self, batch_size: int, seq_len: int, split: str, device: torch.device | str) -> CleanBatch:
        raise NotImplementedError

    def evaluate_batch(
        self,
        pred_ids: torch.Tensor,
        target_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        metadata: Sequence[dict[str, Any]] | None = None,
    ) -> dict[str, float]:
        valid = attention_mask.bool()
        token_acc = (pred_ids[valid] == target_ids[valid]).float().mean().item() if valid.any() else 0.0
        exact = []
        for row in range(pred_ids.shape[0]):
            row_mask = valid[row]
            exact.append(bool(torch.equal(pred_ids[row][row_mask], target_ids[row][row_mask])))
        return {"task/token_accuracy": float(token_acc), "task/exact_match": float(sum(exact) / max(1, len(exact)))}
