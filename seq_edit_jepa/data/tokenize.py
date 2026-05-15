from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence


SPECIAL_TOKENS = ["<pad>", "<unk>", "<mask>", "<bos>", "<eos>", "<sep>"]


class SimpleTokenizer:
    """Small deterministic tokenizer for synthetic sequence-edit tasks."""

    def __init__(self, tokens: Iterable[str] = ()):
        ordered = list(dict.fromkeys([*SPECIAL_TOKENS, *tokens]))
        self.token_to_id = {token: index for index, token in enumerate(ordered)}
        self.id_to_token = {index: token for token, index in self.token_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id["<unk>"]

    @property
    def mask_token_id(self) -> int:
        return self.token_to_id["<mask>"]

    @property
    def bos_token_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id["<eos>"]

    @property
    def sep_token_id(self) -> int:
        return self.token_to_id["<sep>"]

    def __len__(self) -> int:
        return self.vocab_size

    def add_tokens(self, tokens: Iterable[str]) -> None:
        for token in tokens:
            if token not in self.token_to_id:
                index = len(self.token_to_id)
                self.token_to_id[token] = index
                self.id_to_token[index] = token

    def convert_tokens_to_ids(self, tokens: str | Sequence[str]) -> int | list[int]:
        if isinstance(tokens, str):
            return self.token_to_id.get(tokens, self.unk_token_id)
        return [int(self.convert_tokens_to_ids(token)) for token in tokens]

    def convert_ids_to_tokens(self, ids: int | Sequence[int]) -> str | list[str]:
        if isinstance(ids, int):
            return self.id_to_token.get(int(ids), "<unk>")
        return [str(self.convert_ids_to_tokens(int(index))) for index in ids]

    def encode(
        self,
        tokens: Sequence[str],
        add_special_tokens: bool = False,
        pad_to_length: int | None = None,
    ) -> list[int]:
        output = [int(self.convert_tokens_to_ids(token)) for token in tokens]
        if add_special_tokens:
            output = [self.bos_token_id, *output, self.eos_token_id]
        if pad_to_length is not None:
            output = output[:pad_to_length]
            output.extend([self.pad_token_id] * max(0, pad_to_length - len(output)))
        return output

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        tokens = []
        specials = set(SPECIAL_TOKENS)
        for index in ids:
            token = str(self.convert_ids_to_tokens(int(index)))
            if skip_special_tokens and token in specials:
                continue
            tokens.append(token)
        return " ".join(tokens)

    def special_token_ids(self) -> set[int]:
        return {int(self.convert_tokens_to_ids(token)) for token in SPECIAL_TOKENS}

    def save_pretrained(self, path: str | Path) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        with open(target / "vocab.json", "w", encoding="utf-8") as handle:
            json.dump(self.token_to_id, handle, indent=2, sort_keys=True)

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "SimpleTokenizer":
        with open(Path(path) / "vocab.json", "r", encoding="utf-8") as handle:
            vocab = json.load(handle)
        tokenizer = cls([])
        tokenizer.token_to_id = {str(token): int(index) for token, index in vocab.items()}
        tokenizer.id_to_token = {index: token for token, index in tokenizer.token_to_id.items()}
        return tokenizer
