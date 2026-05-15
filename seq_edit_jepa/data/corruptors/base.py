from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import torch

from seq_edit_jepa.data.datasets import CleanBatch, CorruptionBatch, CorruptionPath


class Corruptor(ABC):
    def __init__(self, config: dict[str, Any], vocab_size: int, mask_token_id: int, pad_token_id: int, special_token_ids: set[int]):
        self.config = dict(config)
        self.num_steps = int(config.get("num_steps", 16))
        self.schedule = str(config.get("schedule", "linear"))
        self.vocab_size = int(vocab_size)
        self.mask_token_id = int(mask_token_id)
        self.pad_token_id = int(pad_token_id)
        self.special_token_ids = set(int(index) for index in special_token_ids)
        self.target_mode = str(config.get("target_mode", "x0"))

    @abstractmethod
    def sample_pair(self, clean: CleanBatch) -> CorruptionBatch:
        raise NotImplementedError

    @abstractmethod
    def sample_path(self, clean: CleanBatch, rollout_steps: int) -> CorruptionPath:
        raise NotImplementedError

    def gamma(self, n: int | torch.Tensor) -> float:
        value = float(n.item()) if isinstance(n, torch.Tensor) else float(n)
        if self.schedule == "cosine":
            return math.sin(math.pi * value / (2.0 * self.num_steps)) ** 2
        if self.schedule != "linear":
            raise ValueError(f"Unknown corruption schedule: {self.schedule}")
        return value / float(self.num_steps)

    def inverse_gamma(self, ratio: float) -> float:
        value = min(1.0, max(0.0, float(ratio)))
        if self.schedule == "cosine":
            return (2.0 * self.num_steps / math.pi) * math.asin(math.sqrt(value))
        if self.schedule != "linear":
            raise ValueError(f"Unknown corruption schedule: {self.schedule}")
        return value * float(self.num_steps)

    def _sample_n(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(1, self.num_steps + 1, (batch_size,), dtype=torch.long, device=device)

    def _sample_start_n(self, batch_size: int, rollout_steps: int, device: torch.device) -> torch.Tensor:
        low = max(rollout_steps, 1)
        return torch.randint(low, self.num_steps + 1, (batch_size,), dtype=torch.long, device=device)

    def _corruption_masks_for_levels(self, clean: CleanBatch, levels: list[torch.Tensor]) -> list[torch.Tensor]:
        batch, _ = clean.input_ids.shape
        masks = [torch.zeros_like(clean.editable_mask, dtype=torch.bool) for _ in levels]
        for row in range(batch):
            editable_positions = torch.where(clean.editable_mask[row])[0]
            if editable_positions.numel() == 0:
                continue
            perm = editable_positions[torch.randperm(editable_positions.numel(), device=clean.input_ids.device)]
            for mask, level in zip(masks, levels):
                level_value = int(level[row].item())
                count = int(math.floor(self.gamma(level_value) * editable_positions.numel()))
                if count > 0:
                    mask[row, perm[:count]] = True
        return masks

    def _sample_uniform_count_mask(self, clean: CleanBatch) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _ = clean.input_ids.shape
        mask = torch.zeros_like(clean.editable_mask, dtype=torch.bool)
        n = torch.zeros((batch,), dtype=torch.float32, device=clean.input_ids.device)
        for row in range(batch):
            editable_positions = torch.where(clean.editable_mask[row])[0]
            editable_count = int(editable_positions.numel())
            if editable_count == 0:
                continue
            count = int(torch.randint(1, editable_count + 1, (), device=clean.input_ids.device).item())
            perm = editable_positions[torch.randperm(editable_count, device=clean.input_ids.device)]
            mask[row, perm[:count]] = True
            n[row] = float(self.inverse_gamma(count / float(editable_count)))
        return mask, n

    def _step_reveal_mask(self, current_mask: torch.Tensor, editable_mask: torch.Tensor) -> torch.Tensor:
        reveal = torch.zeros_like(current_mask, dtype=torch.bool)
        for row in range(current_mask.shape[0]):
            positions = torch.where(current_mask[row])[0]
            if positions.numel() == 0:
                continue
            editable_count = max(1, int(editable_mask[row].sum().item()))
            reveal_count = min(int(positions.numel()), max(1, math.ceil(editable_count / float(max(1, self.num_steps)))))
            reveal[row, positions[:reveal_count]] = True
        return reveal

    def _n_from_current_mask(self, current_mask: torch.Tensor, editable_mask: torch.Tensor) -> torch.Tensor:
        n = torch.zeros((current_mask.shape[0],), dtype=torch.float32, device=current_mask.device)
        for row in range(current_mask.shape[0]):
            editable_count = int(editable_mask[row].sum().item())
            if editable_count <= 0:
                continue
            ratio = float(current_mask[row].sum().item()) / float(editable_count)
            n[row] = float(self.inverse_gamma(ratio))
        return n


def tokenizer_special_ids(tokenizer) -> set[int]:
    if hasattr(tokenizer, "special_token_ids"):
        return set(int(index) for index in tokenizer.special_token_ids())
    ids = set()
    for attr in ("pad_token_id", "unk_token_id", "mask_token_id", "bos_token_id", "eos_token_id", "sep_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            ids.add(int(value))
    return ids
