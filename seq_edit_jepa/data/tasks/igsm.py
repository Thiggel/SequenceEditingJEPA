from __future__ import annotations

import random
import re
from typing import Any, Sequence

import torch

from seq_edit_jepa.data.datasets import CleanBatch
from seq_edit_jepa.data.tasks.base import SequenceTask
from seq_edit_jepa.data.tasks.registry import register_task
from seq_edit_jepa.data.tokenize import SimpleTokenizer


@register_task("igsm")
class IgsmTask(SequenceTask):
    """Small arithmetic-trace task with prompt clamping and consistency probes."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.seed = int(config.get("seed", 1))
        self.max_abs_value = int(config.get("max_abs_value", 256))
        self.num_registers = int(config.get("num_registers", 3))
        self.default_ops = int(config.get("operation_count", config.get("num_ops", 2)))
        self.train_min_ops = int(config.get("train_min_ops", self.default_ops))
        self.train_max_ops = int(config.get("train_max_ops", self.default_ops))
        self.eval_min_ops = int(config.get("eval_min_ops", self.train_min_ops))
        self.eval_max_ops = int(config.get("eval_max_ops", self.train_max_ops))
        self.ood_min_ops = int(config.get("ood_min_ops", 20))
        self.ood_max_ops = int(config.get("ood_max_ops", 23))
        self.ood_op_values = [int(value) for value in config.get("ood_op_values", [])]
        self.modulus = _optional_int(config.get("modulus"))
        self.ood_modulus = _optional_int(config.get("ood_modulus", self.modulus))
        self.max_operand = int(config.get("max_operand", 20))
        self.max_multiplier = int(config.get("max_multiplier", 6))
        self.operation_tokens = tuple(config.get("operation_tokens", ["+", "-", "*"]))
        if self.num_registers < 1:
            raise ValueError("iGSM num_registers must be positive.")
        if self.train_min_ops > self.train_max_ops or self.eval_min_ops > self.eval_max_ops or self.ood_min_ops > self.ood_max_ops:
            raise ValueError("iGSM operation ranges must satisfy min <= max.")
        self.rngs = {
            "train": random.Random(self.seed),
            "eval": random.Random(self.seed + 10_000),
            "test": random.Random(self.seed + 20_000),
            "eval_ood": random.Random(self.seed + 30_000),
            "test_ood": random.Random(self.seed + 40_000),
        }
        self.tokenizer = self.build_tokenizer()

    def build_tokenizer(self) -> SimpleTokenizer:
        numbers = [str(value) for value in range(-self.max_abs_value, self.max_abs_value + 1)]
        symbols = [
            "<problem>",
            "<solution>",
            "start",
            "ops",
            "answer",
            *[f"x{index}" for index in range(self.num_registers)],
            "=",
            "+",
            "-",
            "*",
            ";",
        ]
        return SimpleTokenizer([*symbols, *numbers])

    def sample_batch(self, batch_size: int, seq_len: int, split: str, device: torch.device | str) -> CleanBatch:
        rng = self.rngs.setdefault(split, random.Random(self.seed + len(self.rngs) * 10_000))
        rows: list[list[int]] = []
        editable: list[list[int]] = []
        metadata = []
        for _ in range(batch_size):
            op_count = self._sample_op_count(rng, split)
            modulus = self._modulus_for_split(split)
            tokens, solution_start, answer = self._sample_trace(rng, op_count=op_count, modulus=modulus)
            ids = [self.tokenizer.bos_token_id, *self.tokenizer.encode(tokens), self.tokenizer.eos_token_id]
            ids = ids[:seq_len]
            editable_row = [0] * len(ids)
            for pos in range(1 + solution_start, max(1 + solution_start, len(ids) - 1)):
                editable_row[pos] = 1
            pad = max(0, seq_len - len(ids))
            rows.append(ids + [self.tokenizer.pad_token_id] * pad)
            editable.append(editable_row + [0] * pad)
            metadata.append({"tokens": tokens, "answer": answer, "solution_start": solution_start, "op_count": op_count, "modulus": modulus})
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
        correct = []
        consistent = []
        op_counts = []
        for row in range(pred_ids.shape[0]):
            pred_tokens = _tokens(self.tokenizer, pred_ids[row][attention_mask[row].bool()].tolist())
            target_tokens = _tokens(self.tokenizer, target_ids[row][attention_mask[row].bool()].tolist())
            row_metadata = metadata[row] if metadata is not None and row < len(metadata) else {}
            gold_answer = _optional_int(row_metadata.get("answer")) if "answer" in row_metadata else extract_answer(target_tokens)
            correct.append(extract_answer(pred_tokens) == gold_answer)
            modulus = _optional_int(row_metadata.get("modulus", self.modulus))
            consistent.append(trace_is_consistent(pred_tokens, modulus=modulus))
            if "op_count" in row_metadata:
                op_counts.append(int(row_metadata["op_count"]))
        base["igsm/answer_accuracy"] = float(sum(correct) / max(1, len(correct)))
        base["igsm/arithmetic_consistency"] = float(sum(consistent) / max(1, len(consistent)))
        if op_counts:
            base["igsm/op_count_mean"] = float(sum(op_counts) / len(op_counts))
        return base

    def _sample_op_count(self, rng: random.Random, split: str) -> int:
        exact = _exact_op_from_split(split)
        if exact is not None:
            return exact
        if split in {"ood", "eval_ood", "test_ood"}:
            if self.ood_op_values:
                return int(rng.choice(self.ood_op_values))
            return rng.randint(self.ood_min_ops, self.ood_max_ops)
        if split == "train":
            return rng.randint(self.train_min_ops, self.train_max_ops)
        return rng.randint(self.eval_min_ops, self.eval_max_ops)

    def _modulus_for_split(self, split: str) -> int | None:
        if split in {"ood", "eval_ood", "test_ood"} or _exact_op_from_split(split) is not None:
            return self.ood_modulus
        return self.modulus

    def _sample_trace(self, rng: random.Random, op_count: int | None = None, modulus: int | None = None) -> tuple[list[str], int, int]:
        op_count = self.default_ops if op_count is None else int(op_count)
        if op_count == 2 and modulus is None and self.num_registers == 3:
            return self._sample_legacy_trace(rng)
        return self._sample_variable_trace(rng, op_count=op_count, modulus=modulus)

    def _sample_legacy_trace(self, rng: random.Random) -> tuple[list[str], int, int]:
        start = rng.randint(0, 20)
        add = rng.randint(1, 20)
        multiplier = rng.randint(2, 6)
        x1 = start + add
        x2 = x1 * multiplier
        tokens = [
            "<problem>",
            "start",
            str(start),
            "ops",
            "+",
            str(add),
            "*",
            str(multiplier),
            "<solution>",
            "x0",
            "=",
            str(start),
            ";",
            "x1",
            "=",
            "x0",
            "+",
            str(add),
            "=",
            str(x1),
            ";",
            "x2",
            "=",
            "x1",
            "*",
            str(multiplier),
            "=",
            str(x2),
            ";",
            "answer",
            "=",
            str(x2),
        ]
        return tokens, tokens.index("<solution>"), x2

    def _sample_variable_trace(self, rng: random.Random, op_count: int, modulus: int | None) -> tuple[list[str], int, int]:
        start_value = rng.randint(0, min(20, self.max_abs_value))
        if modulus is not None:
            start_value %= modulus
        operations: list[tuple[str, int, int]] = []
        value = start_value
        for _ in range(max(1, op_count)):
            op, operand, value = self._sample_next_operation(rng, value, modulus)
            operations.append((op, operand, value))
        prompt = ["<problem>", "start", str(start_value), "ops"]
        for op, operand, _result in operations:
            prompt.extend([op, str(operand)])
        solution = ["<solution>", "x0", "=", prompt[2], ";"]
        current = start_value
        for index, (op, operand, result) in enumerate(operations, start=1):
            prev_var = f"x{(index - 1) % self.num_registers}"
            var = f"x{index % self.num_registers}"
            current = _apply_operation(current, op, operand, modulus)
            result = current if result != current else result
            solution.extend([var, "=", prev_var, op, str(operand), "=", str(result), ";"])
        tokens = [*prompt, *solution, "answer", "=", str(value)]
        return tokens, tokens.index("<solution>"), int(value)

    def _sample_next_operation(self, rng: random.Random, current: int, modulus: int | None) -> tuple[str, int, int]:
        op = str(rng.choice(self.operation_tokens))
        if op not in {"+", "-", "*"}:
            raise ValueError(f"Unsupported iGSM operation token: {op}")
        if op == "*":
            operand = rng.randint(2, self.max_multiplier)
        else:
            operand = rng.randint(1, self.max_operand)
        result = _apply_operation(current, op, operand, modulus)
        if modulus is None and abs(result) > self.max_abs_value:
            op = str(rng.choice(["+", "-"]))
            operand = rng.randint(1, min(self.max_operand, max(1, self.max_abs_value // 4)))
            result = _apply_operation(current, op, operand, modulus)
            if abs(result) > self.max_abs_value:
                op = "-" if current > 0 else "+"
                operand = min(abs(current), self.max_abs_value)
                result = _apply_operation(current, op, operand, modulus)
        return op, operand, result


def _tokens(tokenizer: SimpleTokenizer, ids: Sequence[int]) -> list[str]:
    specials = {
        tokenizer.pad_token_id,
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.mask_token_id,
        tokenizer.sep_token_id,
    }
    return [str(tokenizer.convert_ids_to_tokens(int(index))) for index in ids if int(index) not in specials]


def extract_answer(tokens: Sequence[str]) -> int | None:
    for index, token in enumerate(tokens):
        if token == "answer" and index + 2 < len(tokens) and tokens[index + 1] == "=":
            try:
                return int(tokens[index + 2])
            except ValueError:
                return None
    return None


def trace_is_consistent(tokens: Sequence[str], modulus: int | None = None) -> bool:
    values: dict[str, int] = {}
    last_value: int | None = None
    index = 0
    while index < len(tokens):
        if _is_register(tokens[index]) and index + 2 < len(tokens) and tokens[index + 1] == "=":
            var = tokens[index]
            try:
                if index + 5 < len(tokens) and tokens[index + 3] in {"+", "-", "*"}:
                    lhs = values.get(tokens[index + 2])
                    if lhs is None:
                        lhs = int(tokens[index + 2])
                    rhs = int(tokens[index + 4])
                    result = int(tokens[index + 6]) if index + 6 < len(tokens) and tokens[index + 5] == "=" else None
                    expected = _apply_operation(lhs, tokens[index + 3], rhs, modulus)
                    if result != expected:
                        return False
                    values[var] = result
                    last_value = result
                else:
                    values[var] = int(tokens[index + 2])
                    last_value = values[var]
            except (ValueError, KeyError):
                return False
        index += 1
    answer = extract_answer(tokens)
    return answer is not None and last_value == answer


def _optional_int(value: Any) -> int | None:
    if value is None or value == "none":
        return None
    return int(value)


def _apply_operation(lhs: int, op: str, rhs: int, modulus: int | None) -> int:
    value = {"+": lhs + rhs, "-": lhs - rhs, "*": lhs * rhs}[op]
    if modulus is not None:
        value %= int(modulus)
    return value


def _is_register(token: str) -> bool:
    return re.fullmatch(r"x\d+", token) is not None


def _exact_op_from_split(split: str) -> int | None:
    match = re.fullmatch(r"(?:eval_|test_)?op_(\d+)", split)
    return int(match.group(1)) if match else None
