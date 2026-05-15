from __future__ import annotations

import torch

from seq_edit_jepa.actions.action_types import Op
from seq_edit_jepa.data.corruptors.base import Corruptor
from seq_edit_jepa.data.datasets import CleanBatch, CorruptionBatch, CorruptionPath


class ReplacementCorruptor(Corruptor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        allowed = [index for index in range(self.vocab_size) if index not in self.special_token_ids]
        if not allowed:
            allowed = [index for index in range(self.vocab_size) if index != self.pad_token_id]
        self.allowed_noise_ids = torch.tensor(allowed, dtype=torch.long)

    def sample_pair(self, clean: CleanBatch) -> CorruptionBatch:
        mask_n, n = self._sample_uniform_count_mask(clean)
        noise = self._sample_noise_like(clean.input_ids)
        input_ids = clean.input_ids.clone()
        input_ids[mask_n] = noise[mask_n]
        if self.target_mode == "step":
            current_mask = mask_n & clean.attention_mask.bool()
            reveal_mask = self._step_reveal_mask(current_mask, clean.editable_mask & clean.attention_mask.bool())
            prev_ids = input_ids.clone()
            prev_ids[reveal_mask] = clean.input_ids[reveal_mask]
            next_mask = current_mask & ~reveal_mask
            action_ops = torch.full_like(clean.input_ids, int(Op.KEEP))
            action_tokens = torch.full_like(clean.input_ids, self.pad_token_id)
            action_ops[reveal_mask] = int(Op.REPLACE)
            action_tokens[reveal_mask] = clean.input_ids[reveal_mask]
            return CorruptionBatch(
                clean_ids=clean.input_ids,
                input_ids=input_ids,
                prev_ids=prev_ids,
                attention_mask=clean.attention_mask,
                prev_attention_mask=clean.attention_mask,
                editable_mask=clean.editable_mask,
                segment_ids=clean.segment_ids,
                n=n,
                action_ops=action_ops,
                action_tokens=action_tokens,
                target_mask=reveal_mask,
                target_n=self._n_from_current_mask(next_mask, clean.editable_mask & clean.attention_mask.bool()),
            )
        if self.target_mode != "x0":
            raise ValueError(f"Unknown replacement target_mode={self.target_mode!r}; expected 'x0' or 'step'.")
        action_ops = torch.full_like(clean.input_ids, int(Op.KEEP))
        action_tokens = torch.full_like(clean.input_ids, self.pad_token_id)
        action_ops[mask_n] = int(Op.REPLACE)
        action_tokens[mask_n] = clean.input_ids[mask_n]
        return CorruptionBatch(
            clean_ids=clean.input_ids,
            input_ids=input_ids,
            prev_ids=clean.input_ids,
            attention_mask=clean.attention_mask,
            prev_attention_mask=clean.attention_mask,
            editable_mask=clean.editable_mask,
            segment_ids=clean.segment_ids,
            n=n,
            action_ops=action_ops,
            action_tokens=action_tokens,
            target_mask=mask_n & clean.attention_mask.bool(),
            target_n=torch.zeros_like(n),
        )

    def sample_path(self, clean: CleanBatch, rollout_steps: int) -> CorruptionPath:
        mask_n, n = self._sample_uniform_count_mask(clean)
        noise = self._sample_noise_like(clean.input_ids)
        start = clean.input_ids.clone()
        start[mask_n] = noise[mask_n]
        if self.target_mode == "step":
            return self._sample_step_path(clean, start, mask_n & clean.attention_mask.bool(), n, rollout_steps)
        if self.target_mode != "x0":
            raise ValueError(f"Unknown replacement target_mode={self.target_mode!r}; expected 'x0' or 'step'.")
        states = [start]
        n_values = [n]
        action_ops = []
        action_tokens = []
        steps = max(0, int(rollout_steps))
        for index in range(steps):
            ops = torch.full_like(clean.input_ids, int(Op.KEEP))
            tokens = torch.full_like(clean.input_ids, self.pad_token_id)
            if index == 0:
                ops[mask_n] = int(Op.REPLACE)
                tokens[mask_n] = clean.input_ids[mask_n]
            action_ops.append(ops)
            action_tokens.append(tokens)
            states.append(clean.input_ids.clone())
            n_values.append(torch.zeros_like(n))
        return CorruptionPath(
            states=states,
            attention_masks=[clean.attention_mask for _ in states],
            action_ops=action_ops,
            action_tokens=action_tokens,
            n_values=n_values,
            editable_mask=clean.editable_mask,
            segment_ids=clean.segment_ids,
        )

    def _sample_noise_like(self, input_ids: torch.Tensor) -> torch.Tensor:
        allowed = self.allowed_noise_ids.to(input_ids.device)
        picks = torch.randint(0, allowed.numel(), input_ids.shape, device=input_ids.device)
        noise = allowed[picks]
        same = noise == input_ids
        if same.any() and allowed.numel() > 1:
            noise[same] = allowed[(picks[same] + 1) % allowed.numel()]
        return noise

    def _sample_step_path(self, clean: CleanBatch, start: torch.Tensor, current_mask: torch.Tensor, n: torch.Tensor, rollout_steps: int) -> CorruptionPath:
        current_ids = start
        current_mask = current_mask.clone()
        states = [current_ids.clone()]
        n_values = [n]
        action_ops = []
        action_tokens = []
        steps = max(0, int(rollout_steps))
        editable_attention = clean.editable_mask & clean.attention_mask.bool()
        for _ in range(steps):
            reveal_mask = self._step_reveal_mask(current_mask, editable_attention)
            ops = torch.full_like(clean.input_ids, int(Op.KEEP))
            tokens = torch.full_like(clean.input_ids, self.pad_token_id)
            ops[reveal_mask] = int(Op.REPLACE)
            tokens[reveal_mask] = clean.input_ids[reveal_mask]
            next_ids = current_ids.clone()
            next_ids[reveal_mask] = clean.input_ids[reveal_mask]
            current_mask = current_mask & ~reveal_mask
            action_ops.append(ops)
            action_tokens.append(tokens)
            states.append(next_ids.clone())
            n_values.append(self._n_from_current_mask(current_mask, editable_attention))
            current_ids = next_ids
        return CorruptionPath(
            states=states,
            attention_masks=[clean.attention_mask for _ in states],
            action_ops=action_ops,
            action_tokens=action_tokens,
            n_values=n_values,
            editable_mask=clean.editable_mask,
            segment_ids=clean.segment_ids,
        )
