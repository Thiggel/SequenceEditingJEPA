from __future__ import annotations

import itertools
import random
from typing import Any

import torch

from seq_edit_jepa.data.datasets import CleanBatch
from seq_edit_jepa.data.tasks.base import SequenceTask


class HFTextTask(SequenceTask):
    """Hugging Face text-window task for LM1B, FineWeb, and reasoning traces."""

    default_dataset_name: str | None = None
    default_dataset_config: str | None = None
    default_text_field = "text"

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.seed = int(config.get("seed", 0))
        self.prompt_tokens = int(config.get("prompt_tokens", 32))
        self.dataset_name = str(config.get("dataset_name", self.default_dataset_name))
        self.dataset_config = config.get("dataset_config", self.default_dataset_config)
        self.text_field = str(config.get("text_field", self.default_text_field))
        self.tokenizer_name = str(config.get("tokenizer_name", "gpt2"))
        self.streaming = bool(config.get("streaming", True))
        self.max_examples = int(config.get("max_examples", 4096))
        self._tokenizer = None
        self._examples: dict[str, list[str]] = {}
        self.rngs = {"train": random.Random(self.seed), "eval": random.Random(self.seed + 10_000)}

    def build_tokenizer(self):
        from transformers import AutoTokenizer

        if self._tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            additions = {}
            if tokenizer.pad_token is None:
                additions["pad_token"] = "<pad>"
            if tokenizer.mask_token is None:
                additions["mask_token"] = "<mask>"
            if additions:
                tokenizer.add_special_tokens(additions)
            self._tokenizer = tokenizer
        return self._tokenizer

    def sample_batch(self, batch_size: int, seq_len: int, split: str, device: torch.device | str) -> CleanBatch:
        tokenizer = self.build_tokenizer()
        texts = self._load_examples(split)
        rng = self.rngs.setdefault(split, random.Random(self.seed + len(self.rngs) * 10_000))
        rows: list[list[int]] = []
        editable: list[list[int]] = []
        metadata = []
        for _ in range(batch_size):
            text = rng.choice(texts)
            ids = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=seq_len)
            ids = ids[:seq_len]
            pad = max(0, seq_len - len(ids))
            rows.append(ids + [tokenizer.pad_token_id] * pad)
            editable_row = [0] * min(self.prompt_tokens, len(ids)) + [1] * max(0, len(ids) - min(self.prompt_tokens, len(ids)))
            editable.append(editable_row[:seq_len] + [0] * pad)
            metadata.append({"text": text[:256]})
        input_ids = torch.tensor(rows, dtype=torch.long, device=device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        editable_mask = torch.tensor(editable, dtype=torch.bool, device=device)
        segment_ids = editable_mask.long()
        return CleanBatch(input_ids, attention_mask, editable_mask, segment_ids, metadata)

    def _load_examples(self, split: str) -> list[str]:
        if split in self._examples:
            return self._examples[split]
        from datasets import load_dataset

        dataset = load_dataset(
            self.dataset_name,
            self.dataset_config,
            split=self.config.get(f"{split}_split", split),
            streaming=self.streaming,
        )
        if self.streaming:
            iterator = iter(dataset)
            rows = list(itertools.islice(iterator, self.max_examples))
        else:
            rows = [dataset[index] for index in range(min(self.max_examples, len(dataset)))]
        texts = [str(row[self.text_field]) for row in rows if row.get(self.text_field)]
        if not texts:
            raise ValueError(f"No text examples loaded for {self.dataset_name}:{split}.")
        self._examples[split] = texts
        return texts
