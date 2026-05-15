from __future__ import annotations

import random
from typing import Any, Sequence

import torch

from seq_edit_jepa.data.datasets import CleanBatch
from seq_edit_jepa.data.tasks.base import SequenceTask
from seq_edit_jepa.data.tasks.registry import register_task
from seq_edit_jepa.data.tokenize import SimpleTokenizer


OPEN_TO_CLOSE = {"(": ")", "[": "]", "{": "}"}
CLOSE_TO_OPEN = {close: open_ for open_, close in OPEN_TO_CLOSE.items()}
TERMINALS = ["a", "b", "c"]


@register_task("lano")
class LanoTask(SequenceTask):
    """Compact CFG/LANO-style balanced-bracket task with parse-depth probes."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.seed = int(config.get("seed", 0))
        self.max_depth = int(config.get("max_depth", 4))
        self.max_target_tokens = int(config.get("max_target_tokens", config.get("seq_len", 128) - 2))
        self.rngs = {
            "train": random.Random(self.seed),
            "eval": random.Random(self.seed + 10_000),
            "test": random.Random(self.seed + 20_000),
        }
        self.tokenizer = self.build_tokenizer()

    def build_tokenizer(self) -> SimpleTokenizer:
        return SimpleTokenizer([*OPEN_TO_CLOSE.keys(), *OPEN_TO_CLOSE.values(), *TERMINALS])

    def sample_batch(self, batch_size: int, seq_len: int, split: str, device: torch.device | str) -> CleanBatch:
        rng = self.rngs.setdefault(split, random.Random(self.seed + len(self.rngs) * 10_000))
        rows: list[list[int]] = []
        editable: list[list[int]] = []
        metadata = []
        for _ in range(batch_size):
            tokens = self._sample_sentence(rng, max_tokens=max(1, min(self.max_target_tokens, seq_len - 2)))
            ids = [self.tokenizer.bos_token_id, *self.tokenizer.encode(tokens), self.tokenizer.eos_token_id]
            ids = ids[:seq_len]
            row_editable = [0] + [1] * max(0, min(len(tokens), seq_len - 2)) + [0]
            row_editable = row_editable[: len(ids)]
            pad = max(0, seq_len - len(ids))
            rows.append(ids + [self.tokenizer.pad_token_id] * pad)
            editable.append(row_editable + [0] * pad)
            metadata.append({"tokens": tokens, "parse_depths": parse_depths(tokens)})
        input_ids = torch.tensor(rows, dtype=torch.long, device=device)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        editable_mask = torch.tensor(editable, dtype=torch.bool, device=device)
        segment_ids = editable_mask.long()
        return CleanBatch(input_ids, attention_mask, editable_mask, segment_ids, metadata)

    def evaluate_batch(
        self,
        pred_ids: torch.Tensor,
        target_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        metadata: Sequence[dict[str, Any]] | None = None,
    ) -> dict[str, float]:
        base = super().evaluate_batch(pred_ids, target_ids, attention_mask, metadata)
        valid = []
        depth_matches = []
        for row in range(pred_ids.shape[0]):
            pred_tokens = _content_tokens(self.tokenizer, pred_ids[row][attention_mask[row].bool()].tolist())
            target_tokens = _content_tokens(self.tokenizer, target_ids[row][attention_mask[row].bool()].tolist())
            valid.append(is_valid_lano(pred_tokens))
            pred_depth = parse_depths(pred_tokens)
            target_depth = parse_depths(target_tokens)
            overlap = min(len(pred_depth), len(target_depth))
            if overlap:
                depth_matches.extend(int(pred_depth[i] == target_depth[i]) for i in range(overlap))
        base["lano/grammar_validity"] = float(sum(valid) / max(1, len(valid)))
        base["lano/parse_depth_match"] = float(sum(depth_matches) / max(1, len(depth_matches))) if depth_matches else 0.0
        return base

    def _sample_sentence(self, rng: random.Random, max_tokens: int) -> list[str]:
        for _ in range(100):
            tokens = self._expand(rng, depth=0)
            if 0 < len(tokens) <= max_tokens:
                return tokens
        return self._balanced_fallback(rng, max_tokens)

    def _expand(self, rng: random.Random, depth: int) -> list[str]:
        if depth >= self.max_depth or rng.random() < 0.25:
            return [rng.choice(TERMINALS)]
        choice = rng.random()
        if choice < 0.7:
            open_token = rng.choice(list(OPEN_TO_CLOSE))
            return [open_token, *self._expand(rng, depth + 1), OPEN_TO_CLOSE[open_token]]
        return [*self._expand(rng, depth + 1), *self._expand(rng, depth + 1)]

    def _balanced_fallback(self, rng: random.Random, max_tokens: int) -> list[str]:
        depth = max(1, min(self.max_depth, max_tokens // 2))
        opens = [rng.choice(list(OPEN_TO_CLOSE)) for _ in range(depth)]
        closes = [OPEN_TO_CLOSE[token] for token in reversed(opens)]
        return [*opens, rng.choice(TERMINALS), *closes][:max_tokens]


def _content_tokens(tokenizer: SimpleTokenizer, ids: Sequence[int]) -> list[str]:
    specials = {
        tokenizer.pad_token_id,
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.mask_token_id,
        tokenizer.sep_token_id,
    }
    return [str(tokenizer.convert_ids_to_tokens(int(index))) for index in ids if int(index) not in specials]


def is_valid_lano(tokens: Sequence[str]) -> bool:
    stack: list[str] = []
    for token in tokens:
        if token in OPEN_TO_CLOSE:
            stack.append(token)
        elif token in CLOSE_TO_OPEN:
            if not stack or stack[-1] != CLOSE_TO_OPEN[token]:
                return False
            stack.pop()
        elif token not in TERMINALS:
            return False
    return not stack and bool(tokens)


def parse_depths(tokens: Sequence[str]) -> list[int]:
    depth = 0
    depths: list[int] = []
    for token in tokens:
        if token in OPEN_TO_CLOSE:
            depth += 1
            depths.append(depth)
        elif token in CLOSE_TO_OPEN:
            depths.append(depth)
            depth = max(0, depth - 1)
        else:
            depths.append(depth)
    return depths
