import os
import random
import sys
import ast
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from seq_edit_jepa.data.sequence import causal_lm_labels, pad_token_ids, tensor_item
from seq_edit_jepa.data.word_tokenizer import SimpleWordTokenizer
from seq_edit_jepa.data.datasets import CleanBatch
from seq_edit_jepa.data.tasks.base import SequenceTask
from seq_edit_jepa.data.tasks.registry import register_task


IGSM_PROBLEM_TOKEN_ID = 222
IGSM_SOLUTION_TOKEN_ID = 223
IGSM_ANSWER_TOKEN_ID = 224
IGSM_EOS_TOKEN_ID = 50256
IGSM_PROBLEM_TOKEN = "<igsm_problem>"
IGSM_SOLUTION_TOKEN = "<igsm_solution>"
IGSM_ANSWER_TOKEN = "<igsm_answer>"
IGSM_GPT2_TOKEN_PREFIX = "<igsm_gpt2_"


def default_igsm_repo_path() -> Path:
    explicit = os.environ.get("IGSM_REPO_PATH")
    if explicit:
        return Path(explicit)
    work = os.environ.get("WORK")
    if work:
        return Path(work) / "codex_research/iGSM"
    return Path("iGSM")


def resolve_igsm_repo(path: Optional[str] = None) -> Path:
    if path and "${" in str(path):
        path = None
    repo = Path(path) if path else default_igsm_repo_path()
    if not (repo / "data_gen/pretrain/id_gen.py").exists():
        raise FileNotFoundError(
            f"Could not find facebookresearch/iGSM at {repo}. "
            "Clone it under $WORK/codex_research/iGSM or set IGSM_REPO_PATH."
        )
    return repo


def import_official_igsm(repo_path: Optional[str] = None):
    repo = resolve_igsm_repo(repo_path)
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    from data_gen.pretrain.id_gen import IdGen  # type: ignore
    from tools.tools import fix_seed, tokenizer  # type: ignore

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return IdGen, fix_seed, tokenizer


def compact_gpt2_token(original_id: int) -> str:
    return f"{IGSM_GPT2_TOKEN_PREFIX}{int(original_id)}>"


class IgsmCompactTokenizer(SimpleWordTokenizer):
    """Compact HF tokenizer for official iGSM GPT-2 token ids."""

    def __init__(self, vocab_file: Optional[str] = None, vocab: Optional[Dict[str, int]] = None, official_tokenizer=None, **kwargs):
        self.official_tokenizer = official_tokenizer
        super().__init__(vocab_file=vocab_file, vocab=vocab, **kwargs)

    @classmethod
    def from_original_ids(cls, original_ids: Iterable[int], official_tokenizer) -> "IgsmCompactTokenizer":
        specials = ["<pad>", "<unk>", "<bos>", "<eos>", "<mask>"]
        marker_tokens = [IGSM_PROBLEM_TOKEN, IGSM_SOLUTION_TOKEN, IGSM_ANSWER_TOKEN]
        content_tokens = [
            compact_gpt2_token(token_id)
            for token_id in sorted(set(int(token_id) for token_id in original_ids))
            if token_id not in {IGSM_PROBLEM_TOKEN_ID, IGSM_SOLUTION_TOKEN_ID, IGSM_ANSWER_TOKEN_ID, IGSM_EOS_TOKEN_ID}
        ]
        ordered = list(dict.fromkeys([*specials, *marker_tokens, *content_tokens]))
        return cls(vocab={token: index for index, token in enumerate(ordered)}, official_tokenizer=official_tokenizer)

    def original_to_token(self, original_id: int) -> str:
        original_id = int(original_id)
        if original_id == IGSM_PROBLEM_TOKEN_ID:
            return IGSM_PROBLEM_TOKEN
        if original_id == IGSM_SOLUTION_TOKEN_ID:
            return IGSM_SOLUTION_TOKEN
        if original_id == IGSM_ANSWER_TOKEN_ID:
            return IGSM_ANSWER_TOKEN
        if original_id == IGSM_EOS_TOKEN_ID:
            return self.eos_token
        return compact_gpt2_token(original_id)

    def original_to_model_id(self, original_id: int) -> int:
        token = self.original_to_token(original_id)
        model_id = int(self.convert_tokens_to_ids(token))
        if model_id == self.unk_token_id:
            raise KeyError(f"Official iGSM token id {original_id} is missing from compact vocabulary.")
        return model_id

    def original_ids_to_model_ids(self, original_ids: Iterable[int]) -> List[int]:
        return [self.original_to_model_id(token_id) for token_id in original_ids]

    def model_id_to_original_id(self, model_id: int) -> Optional[int]:
        token = self.convert_ids_to_tokens(int(model_id))
        if token == IGSM_PROBLEM_TOKEN:
            return IGSM_PROBLEM_TOKEN_ID
        if token == IGSM_SOLUTION_TOKEN:
            return IGSM_SOLUTION_TOKEN_ID
        if token == IGSM_ANSWER_TOKEN:
            return IGSM_ANSWER_TOKEN_ID
        if token == self.eos_token:
            return IGSM_EOS_TOKEN_ID
        if token.startswith(IGSM_GPT2_TOKEN_PREFIX) and token.endswith(">"):
            return int(token[len(IGSM_GPT2_TOKEN_PREFIX) : -1])
        return None

    def decode(self, token_ids, skip_special_tokens: bool = False, **kwargs) -> str:
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        output: List[str] = []
        gpt2_ids: List[int] = []

        def flush_gpt2() -> None:
            if not gpt2_ids:
                return
            if self.official_tokenizer is None:
                output.extend(compact_gpt2_token(token_id) for token_id in gpt2_ids)
            else:
                output.append(self.official_tokenizer.decode(gpt2_ids, skip_special_tokens=skip_special_tokens))
            gpt2_ids.clear()

        for model_id in token_ids:
            token = self.convert_ids_to_tokens(int(model_id))
            original_id = self.model_id_to_original_id(int(model_id))
            if token in {self.pad_token, self.bos_token, self.unk_token}:
                flush_gpt2()
                if not skip_special_tokens:
                    output.append(token)
                continue
            if original_id is not None and original_id not in {
                IGSM_PROBLEM_TOKEN_ID,
                IGSM_SOLUTION_TOKEN_ID,
                IGSM_ANSWER_TOKEN_ID,
                IGSM_EOS_TOKEN_ID,
            }:
                gpt2_ids.append(original_id)
                continue
            flush_gpt2()
            if not skip_special_tokens:
                output.append(token)
        flush_gpt2()
        return "".join(output)


def _collect_official_literal_strings(repo: Path) -> Set[str]:
    strings: Set[str] = set()
    for relative_path in [
        "data_gen/categ.py",
        "data_gen/prototype/id_gen.py",
        "data_gen/pretrain/id_gen.py",
        "math_gen/problem_gen.py",
        "const/params.py",
    ]:
        path = repo / relative_path
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                strings.add(node.value)
            elif isinstance(node, ast.JoinedStr):
                strings.update(
                    value.value
                    for value in node.values
                    if isinstance(value, ast.Constant) and isinstance(value.value, str)
                )
    return {text for text in strings if text}


def _collect_category_strings(repo_path: Optional[str]) -> Set[str]:
    repo = resolve_igsm_repo(repo_path)
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    from data_gen.categ import Data  # type: ignore

    data = Data()
    strings: Set[str] = set()
    for category_sequence in data.categ_list:
        strings.update(str(value) for value in category_sequence)
    for category, group_map in data.categ_dict.items():
        strings.add(str(category))
        for group, items in group_map.items():
            strings.add(str(group))
            strings.update(str(item) for item in items)
    return strings


def _collect_symbol_strings(repo_path: Optional[str]) -> Set[str]:
    repo = resolve_igsm_repo(repo_path)
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    from const.params import feasible_symbols, mod  # type: ignore

    return {str(value) for value in range(int(mod))} | {str(symbol) for symbol in feasible_symbols}


def _encode_context_variants(text: str, official_tokenizer) -> Set[int]:
    variants = {
        text,
        f" {text}",
        f"{text} ",
        f" {text} ",
        f"{text}.",
        f" {text}.",
        f"{text},",
        f" {text},",
        f"{text}?",
        f" {text}?",
        f"{text}'s",
        f" {text}'s",
        f"each {text}",
        f" each {text}",
        f"Define {text}",
        f" Define {text}",
        f"so {text}",
        f" so {text}",
    }
    token_ids: Set[int] = set()
    for variant in variants:
        token_ids.update(int(token_id) for token_id in official_tokenizer.encode(variant))
    return token_ids


def build_compact_igsm_original_vocab(config: "IgsmConfig", official_tokenizer) -> Set[int]:
    """Collect the finite GPT-2 ids that can occur in official iGSM examples."""
    repo = resolve_igsm_repo(config.official_repo_path)
    texts = _collect_official_literal_strings(repo)
    texts.update(_collect_category_strings(config.official_repo_path))
    texts.update(_collect_symbol_strings(config.official_repo_path))
    texts.update(
        {
            "The number of",
            "equals",
            "more than",
            "times as much as",
            "the sum of",
            "the difference of",
            "How many",
            "does",
            "have?",
            "has",
            "Define",
            "as",
            "so",
            "each",
            "and",
            "None",
            "...",
            ".",
            ",",
            ";",
            " = ",
            " + ",
            " - ",
            " * ",
        }
    )
    original_ids = {IGSM_PROBLEM_TOKEN_ID, IGSM_SOLUTION_TOKEN_ID, IGSM_ANSWER_TOKEN_ID, IGSM_EOS_TOKEN_ID}
    for text in texts:
        original_ids.update(_encode_context_variants(text, official_tokenizer))
    if config.compact_vocab_num_prescan_examples > 0:
        IdGen, fix_seed, _tokenizer = import_official_igsm(config.official_repo_path)
        for index in range(config.compact_vocab_num_prescan_examples):
            fix_seed(config.seed + 7000003 + index)
            id_gen = IdGen(
                max_op=config.max_op,
                max_edge=config.max_edge,
                op=config.op,
                perm_level=config.perm_level,
                detail_level=config.detail_level,
            )
            id_gen.gen_prob([i for i in range(23)], p_format=config.p_format)
            original_ids.update(int(token_id) for token_id in id_gen.token_id)
    return original_ids


class IgsmConfig:
    """Config for the official iGSM synthetic grade-school math generator."""

    def __init__(
        self,
        num_examples: int = 4096,
        seed: int = 0,
        max_length: int = 512,
        difficulty: str = "med",
        max_op: Optional[int] = None,
        max_edge: Optional[int] = None,
        op: Optional[int] = None,
        perm_level: int = 5,
        detail_level: int = 0,
        p_format: str = "pq",
        label_mode: str = "all",
        iterable: bool = False,
        official_repo_path: Optional[str] = None,
        compact_vocab: bool = False,
        compact_vocab_num_prescan_examples: int = 0,
        discard_truncated: bool = False,
    ):
        self.num_examples = num_examples
        self.seed = seed
        self.max_length = max_length
        self.difficulty = difficulty
        self.max_op = max_op if max_op is not None else (15 if difficulty == "med" else 21 if difficulty == "hard" else 5)
        self.max_edge = max_edge if max_edge is not None else (20 if difficulty == "med" else 28 if difficulty == "hard" else 8)
        self.op = op
        self.perm_level = perm_level
        self.detail_level = detail_level
        self.p_format = p_format
        self.label_mode = label_mode
        self.iterable = iterable
        self.official_repo_path = official_repo_path
        self.compact_vocab = compact_vocab
        self.compact_vocab_num_prescan_examples = compact_vocab_num_prescan_examples
        self.discard_truncated = discard_truncated
        if self.label_mode not in {"all", "answer", "answer_with_marker", "solution_answer"}:
            raise ValueError("iGSM label_mode must be 'all', 'answer', 'answer_with_marker', or 'solution_answer'.")

    @classmethod
    def from_dict(cls, config: Dict) -> "IgsmConfig":
        return cls(**config)


def build_igsm_tokenizer(config: IgsmConfig):
    _id_gen, _fix_seed, tokenizer = import_official_igsm(config.official_repo_path)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if config.compact_vocab:
        original_ids = build_compact_igsm_original_vocab(config, tokenizer)
        return IgsmCompactTokenizer.from_original_ids(original_ids, tokenizer)
    return tokenizer


class IgsmFixedDataset(Dataset):
    def __init__(self, config: IgsmConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.IdGen, self.fix_seed, self.official_tokenizer = import_official_igsm(config.official_repo_path)
        self.examples = []
        attempt = 0
        max_attempts = max(config.num_examples * 100, config.num_examples + 100)
        while len(self.examples) < config.num_examples and attempt < max_attempts:
            example = self._build_example(config.seed + attempt)
            attempt += 1
            if config.discard_truncated and example["truncated"]:
                continue
            self.examples.append(example)
        if len(self.examples) < config.num_examples:
            raise RuntimeError(
                f"Could only generate {len(self.examples)} non-truncated iGSM examples "
                f"after {attempt} attempts. Increase max_length or relax discard_truncated."
            )

    def _build_example(self, seed: int) -> Dict:
        self.fix_seed(int(seed))
        id_gen = self.IdGen(
            max_op=self.config.max_op,
            max_edge=self.config.max_edge,
            op=self.config.op,
            perm_level=self.config.perm_level,
            detail_level=self.config.detail_level,
        )
        id_gen.gen_prob([i for i in range(23)], p_format=self.config.p_format)
        raw_ids = list(id_gen.token_id)
        model_ids = self._model_ids(raw_ids)
        encoded = pad_token_ids(model_ids, self.config.max_length, self.tokenizer.pad_token_id)
        supervised_positions = self._supervised_positions(raw_ids[: self.config.max_length])
        labels = causal_lm_labels(
            encoded["input_ids"],
            encoded["attention_mask"],
            supervised_positions=None if self.config.label_mode == "all" else supervised_positions,
        )
        answer_start = raw_ids.index(IGSM_ANSWER_TOKEN_ID) + 1 if IGSM_ANSWER_TOKEN_ID in raw_ids else -1
        answer_end = raw_ids.index(IGSM_EOS_TOKEN_ID, answer_start) if answer_start >= 0 and IGSM_EOS_TOKEN_ID in raw_ids[answer_start:] else len(raw_ids)
        solution_start = raw_ids.index(IGSM_SOLUTION_TOKEN_ID) + 1 if IGSM_SOLUTION_TOKEN_ID in raw_ids else -1
        return {
            **encoded,
            "labels": labels,
            "raw_ids": raw_ids,
            "truncated": len(model_ids) > self.config.max_length,
            "problem_prompt_ids": model_ids[:solution_start],
            "answer_prompt_ids": model_ids[:answer_start],
            "answer_ids": model_ids[answer_start:answer_end],
            "solution_answer_ids": model_ids[solution_start:answer_end] if solution_start >= 0 else [],
            "problem_text": self.official_tokenizer.decode(id_gen.prob_token),
            "solution_text": self.official_tokenizer.decode(id_gen.sol_token),
            "answer_text": self.official_tokenizer.decode(id_gen.ans_token),
            "n_op": int(getattr(id_gen.problem, "n_op", -1)),
            "problem": id_gen.problem,
        }

    def _supervised_positions(self, token_ids: list[int]) -> list[int]:
        if IGSM_ANSWER_TOKEN_ID not in token_ids:
            return []
        answer_marker = token_ids.index(IGSM_ANSWER_TOKEN_ID)
        eos = token_ids.index(IGSM_EOS_TOKEN_ID, answer_marker) if IGSM_EOS_TOKEN_ID in token_ids[answer_marker:] else len(token_ids)
        if self.config.label_mode == "answer":
            return list(range(answer_marker + 1, eos))
        if self.config.label_mode == "answer_with_marker":
            return list(range(answer_marker, eos))
        if self.config.label_mode == "solution_answer":
            solution_marker = token_ids.index(IGSM_SOLUTION_TOKEN_ID) if IGSM_SOLUTION_TOKEN_ID in token_ids else answer_marker
            return list(range(solution_marker, eos))
        return list(range(len(token_ids)))

    def _model_ids(self, raw_ids: List[int]) -> List[int]:
        if hasattr(self.tokenizer, "original_ids_to_model_ids"):
            return self.tokenizer.original_ids_to_model_ids(raw_ids)
        return list(raw_ids)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return tensor_item(self.examples[index])



def _tokens_from_ids(tokenizer, ids: Iterable[int]) -> list[str]:
    tokens = []
    special_ids = {int(tokenizer.pad_token_id), int(tokenizer.bos_token_id), int(tokenizer.eos_token_id)}
    for token_id in ids:
        token_id = int(token_id)
        if token_id in special_ids:
            continue
        tokens.append(str(tokenizer.convert_ids_to_tokens(token_id)))
    return tokens


@register_task("official_igsm")
class OfficialIgsmTask(SequenceTask):
    """facebookresearch/iGSM generator wrapped as a sequence-edit task."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.seed = int(config.get("seed", 0))
        self.seq_len = int(config.get("seq_len", config.get("max_length", 1024)))
        self.eval_seed = int(config.get("eval_seed", self.seed + 1000))
        self.ood_seed = int(config.get("ood_seed", self.seed + 2000))
        self.eval_suites = list(config.get("eval_suites", []))
        self.data_config = self._igsm_config(config, seed=self.seed, split="train")
        self.tokenizer = self._load_configured_tokenizer(config) or self.build_tokenizer()
        self._builders: dict[str, IgsmFixedDataset] = {}
        self._counters: dict[str, int] = {}

    def build_tokenizer(self):
        return build_igsm_tokenizer(self.data_config)

    def _load_configured_tokenizer(self, config: dict[str, Any]):
        tokenizer_path = config.get("_tokenizer_path") or config.get("tokenizer_path")
        if not tokenizer_path:
            return None
        vocab_file = Path(str(tokenizer_path)) / "vocab.json"
        if not vocab_file.exists():
            return None
        _id_gen, _fix_seed, tokenizer = import_official_igsm(self.data_config.official_repo_path)
        return IgsmCompactTokenizer(vocab_file=str(vocab_file), official_tokenizer=tokenizer)

    def sample_batch(self, batch_size: int, seq_len: int, split: str, device: torch.device | str) -> CleanBatch:
        builder = self._builder_for_split(split, seq_len)
        rows: list[torch.Tensor] = []
        attention: list[torch.Tensor] = []
        editable: list[torch.Tensor] = []
        segment: list[torch.Tensor] = []
        metadata: list[dict[str, Any]] = []
        for _ in range(batch_size):
            example = self._next_example(builder, split)
            input_ids = example["input_ids"].clone()
            attention_mask = example["attention_mask"].clone()
            edit_mask = self._editable_mask(example, input_ids)
            rows.append(input_ids)
            attention.append(attention_mask)
            editable.append(edit_mask)
            segment.append(edit_mask.long())
            metadata.append(
                {
                    "answer_ids": list(example["answer_ids"]),
                    "solution_start": len(example["problem_prompt_ids"]),
                    "n_op": int(example["n_op"]),
                    "problem_text": example["problem_text"],
                    "solution_text": example["solution_text"],
                    "answer_text": example["answer_text"],
                    "truncated": bool(example["truncated"]),
                }
            )
        return CleanBatch(
            input_ids=torch.stack(rows).to(device),
            attention_mask=torch.stack(attention).to(device),
            editable_mask=torch.stack(editable).to(device),
            segment_ids=torch.stack(segment).to(device),
            metadata=metadata,
        )

    def evaluate_batch(
        self,
        pred_ids: torch.Tensor,
        target_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        metadata: Sequence[dict[str, Any]] | None = None,
    ) -> dict[str, float]:
        base = super().evaluate_batch(pred_ids, target_ids, attention_mask, metadata)
        answer_hits = []
        op_counts = []
        for row in range(pred_ids.shape[0]):
            row_metadata = metadata[row] if metadata is not None and row < len(metadata) else {}
            expected = [int(token_id) for token_id in row_metadata.get("answer_ids", [])]
            predicted = self._extract_answer_ids(pred_ids[row][attention_mask[row].bool()].tolist(), len(expected))
            answer_hits.append(bool(expected) and predicted == expected)
            if "n_op" in row_metadata:
                op_counts.append(int(row_metadata["n_op"]))
        base["igsm/answer_accuracy"] = float(sum(answer_hits) / max(1, len(answer_hits)))
        if op_counts:
            base["igsm/op_count_mean"] = float(sum(op_counts) / len(op_counts))
        return base

    def _igsm_config(self, config: dict[str, Any], seed: int, split: str, seq_len: int | None = None) -> IgsmConfig:
        difficulty = str(config.get("difficulty", "med"))
        max_op = config.get("max_op", 15 if difficulty == "med" else 21)
        max_edge = config.get("max_edge", 20 if difficulty == "med" else 28)
        op = config.get("op")
        if split in {"eval_ood", "ood"}:
            max_op = config.get("ood_max_op", 23 if difficulty == "med" else 32)
            max_edge = config.get("ood_max_edge", 28 if difficulty == "med" else 40)
            op = config.get("ood_op", op)
        exact_op = _exact_op_from_split(split)
        if exact_op is not None:
            max_op = exact_op
            op = exact_op
        return IgsmConfig(
            num_examples=int(config.get("builder_examples", 1024)),
            seed=int(seed),
            max_length=int(seq_len or config.get("max_length", self.seq_len)),
            difficulty=difficulty,
            max_op=None if max_op is None else int(max_op),
            max_edge=None if max_edge is None else int(max_edge),
            op=None if op is None else int(op),
            perm_level=int(config.get("perm_level", 5)),
            detail_level=int(config.get("detail_level", 0)),
            p_format=str(config.get("p_format", "pq")),
            label_mode=str(config.get("label_mode", "all")),
            iterable=False,
            official_repo_path=config.get("official_repo_path"),
            compact_vocab=bool(config.get("compact_vocab", True)),
            compact_vocab_num_prescan_examples=int(config.get("compact_vocab_num_prescan_examples", 0)),
            discard_truncated=bool(config.get("discard_truncated", False)),
        )

    def _builder_for_split(self, split: str, seq_len: int) -> IgsmFixedDataset:
        if split in self._builders:
            return self._builders[split]
        seed = self.seed
        if split.startswith("eval"):
            seed = self.eval_seed
        if "ood" in split or _exact_op_from_split(split) is not None:
            seed = self.ood_seed + (_exact_op_from_split(split) or 0)
        cfg = self._igsm_config(self.config, seed=seed, split=split, seq_len=seq_len)
        builder = IgsmFixedDataset(cfg, self.tokenizer)
        self._builders[split] = builder
        self._counters[split] = 0
        return builder

    def _next_example(self, builder: IgsmFixedDataset, split: str) -> dict[str, Any]:
        index = self._counters.get(split, 0)
        self._counters[split] = index + 1
        if index < len(builder.examples):
            return builder.examples[index]
        return builder._build_example(builder.config.seed + index)

    def _editable_mask(self, example: dict[str, Any], input_ids: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        start = int(len(example["problem_prompt_ids"]))
        valid = input_ids.ne(int(self.tokenizer.pad_token_id))
        if start < int(valid.sum().item()):
            mask[start : int(valid.sum().item())] = True
        return mask & valid

    def _extract_answer_ids(self, ids: Sequence[int], expected_answer_len: int) -> list[int]:
        answer_marker_id = int(self.tokenizer.original_to_model_id(IGSM_ANSWER_TOKEN_ID))
        eos_id = int(self.tokenizer.eos_token_id)
        ids = [int(token_id) for token_id in ids]
        if answer_marker_id not in ids:
            return []
        start = ids.index(answer_marker_id) + 1
        answer = []
        for token_id in ids[start:]:
            if token_id == eos_id:
                break
            answer.append(token_id)
            if len(answer) >= expected_answer_len:
                break
        return answer


def _exact_op_from_split(split: str) -> int | None:
    prefix = "eval_op_"
    if split.startswith(prefix):
        return int(split[len(prefix) :])
    prefix = "op_"
    if split.startswith(prefix):
        return int(split[len(prefix) :])
    return None
