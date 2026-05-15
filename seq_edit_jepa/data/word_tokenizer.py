import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from transformers import PreTrainedTokenizer


class SimpleWordTokenizer(PreTrainedTokenizer):
    """Small whitespace tokenizer with HF save/load compatibility."""

    vocab_files_names = {"vocab_file": "vocab.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocab_file: Optional[str] = None, vocab: Optional[Dict[str, int]] = None, **kwargs):
        if vocab_file is not None:
            with open(vocab_file, "r", encoding="utf-8") as handle:
                vocab = json.load(handle)
        if vocab is None:
            raise ValueError("SimpleWordTokenizer requires a vocab or vocab_file.")
        self.vocab = dict(vocab)
        self.ids_to_tokens = {idx: token for token, idx in self.vocab.items()}
        kwargs.setdefault("pad_token", "<pad>")
        kwargs.setdefault("unk_token", "<unk>")
        kwargs.setdefault("bos_token", "<bos>")
        kwargs.setdefault("eos_token", "<eos>")
        kwargs.setdefault("mask_token", "<mask>")
        super().__init__(**kwargs)

    @classmethod
    def from_tokens(cls, tokens: Iterable[str]) -> "SimpleWordTokenizer":
        specials = ["<pad>", "<unk>", "<bos>", "<eos>", "<mask>"]
        ordered = list(dict.fromkeys([*specials, *tokens]))
        return cls(vocab={token: idx for idx, token in enumerate(ordered)})

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.vocab)

    def _tokenize(self, text: str) -> List[str]:
        return text.strip().split()

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab[self.unk_token])

    def _convert_id_to_token(self, index: int) -> str:
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return " ".join(tokens)

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        tokens = [self.bos_token_id, *token_ids_0]
        if token_ids_1 is not None:
            tokens.extend(token_ids_1)
        tokens.append(self.eos_token_id)
        return tokens

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            specials = set(self.all_special_ids)
            return [1 if token_id in specials else 0 for token_id in token_ids_0]
        mask = [1] + [0] * len(token_ids_0)
        if token_ids_1 is not None:
            mask.extend([0] * len(token_ids_1))
        mask.append(1)
        return mask

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        name = "vocab.json" if filename_prefix is None else f"{filename_prefix}-vocab.json"
        vocab_file = path / name
        with open(vocab_file, "w", encoding="utf-8") as handle:
            json.dump(self.vocab, handle, indent=2, sort_keys=True)
        return (str(vocab_file),)
