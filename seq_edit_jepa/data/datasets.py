from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch


@dataclass
class CleanBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    editable_mask: torch.Tensor
    segment_ids: torch.Tensor
    metadata: list[dict[str, Any]]

    def to(self, device: torch.device | str) -> "CleanBatch":
        return replace(
            self,
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            editable_mask=self.editable_mask.to(device),
            segment_ids=self.segment_ids.to(device),
        )


@dataclass
class CorruptionBatch:
    clean_ids: torch.Tensor
    input_ids: torch.Tensor
    prev_ids: torch.Tensor
    attention_mask: torch.Tensor
    prev_attention_mask: torch.Tensor
    editable_mask: torch.Tensor
    segment_ids: torch.Tensor
    n: torch.Tensor
    action_ops: torch.Tensor
    action_tokens: torch.Tensor
    target_mask: torch.Tensor | None = None
    target_n: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> "CorruptionBatch":
        return replace(
            self,
            clean_ids=self.clean_ids.to(device),
            input_ids=self.input_ids.to(device),
            prev_ids=self.prev_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            prev_attention_mask=self.prev_attention_mask.to(device),
            editable_mask=self.editable_mask.to(device),
            segment_ids=self.segment_ids.to(device),
            n=self.n.to(device),
            action_ops=self.action_ops.to(device),
            action_tokens=self.action_tokens.to(device),
            target_mask=None if self.target_mask is None else self.target_mask.to(device),
            target_n=None if self.target_n is None else self.target_n.to(device),
        )


@dataclass
class CorruptionPath:
    states: list[torch.Tensor]
    attention_masks: list[torch.Tensor]
    action_ops: list[torch.Tensor]
    action_tokens: list[torch.Tensor]
    n_values: list[torch.Tensor]
    editable_mask: torch.Tensor
    segment_ids: torch.Tensor

    def to(self, device: torch.device | str) -> "CorruptionPath":
        return CorruptionPath(
            states=[state.to(device) for state in self.states],
            attention_masks=[mask.to(device) for mask in self.attention_masks],
            action_ops=[ops.to(device) for ops in self.action_ops],
            action_tokens=[tokens.to(device) for tokens in self.action_tokens],
            n_values=[n.to(device) for n in self.n_values],
            editable_mask=self.editable_mask.to(device),
            segment_ids=self.segment_ids.to(device),
        )
