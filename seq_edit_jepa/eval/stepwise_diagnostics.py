from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from seq_edit_jepa.actions.action_types import Op
from seq_edit_jepa.data.corruptors import build_corruptor
from seq_edit_jepa.data.datasets import CleanBatch, CorruptionBatch
from seq_edit_jepa.data.tasks import build_task
from seq_edit_jepa.eval.full_denoise import _n_from_remaining
from seq_edit_jepa.eval.posthoc_igsm_ood import (
    _allow_rope_length_extrapolation,
    _checkpoint_dirs,
    _checkpoint_step,
    _eval_splits,
    _load_model,
    _summarize,
)
from seq_edit_jepa.losses import suppress_token_logits
from seq_edit_jepa.models import DenoisingLM, SequenceEditJEPA
from seq_edit_jepa.train.config import load_yaml


DEFAULT_RUNS = [
    "igsm_official_med_step_mask_jepa_T20_200k",
    "igsm_official_med_step_mask_jepa_T50_200k",
]

DEFAULT_FACTOR_MODES = [
    "model_model",
    "oracle_pos_model_token",
    "model_pos_oracle_token",
    "oracle_oracle",
]


@torch.no_grad()
def evaluate_proposal_coverage(
    model: SequenceEditJEPA,
    task,
    corruptor,
    seq_len: int,
    eval_config: dict[str, Any],
    device: torch.device,
    prefix: str,
) -> dict[str, float]:
    was_training = bool(model.training)
    model.eval()
    model.to(device)
    batches = int(eval_config.get("batches", 8))
    batch_size = int(eval_config.get("batch_size", 8))
    split = str(eval_config.get("split", "eval"))
    rollout_steps = max(1, int(eval_config.get("coverage_rollout_steps", 1)))
    m_values = [int(value) for value in eval_config.get("m_values", [1, 2, 4, 8, 16])]
    k_values = [int(value) for value in eval_config.get("k_values", [1, 5, 20, 100])]
    totals: dict[str, float] = {}
    for _ in range(batches):
        clean = task.sample_batch(batch_size, seq_len, split=split, device=device)
        for batch in _coverage_batches(corruptor, clean, rollout_steps):
            metrics = proposal_coverage_for_batch(
                model,
                batch,
                m_values=m_values,
                k_values=k_values,
                mask_token_id=int(model.config.mask_token_id),
            )
            for key, value in metrics.items():
                totals[f"{prefix}/{key}"] = totals.get(f"{prefix}/{key}", 0.0) + float(value)
    denom = max(1, batches * rollout_steps)
    if was_training:
        model.train()
    return {key: value / denom for key, value in totals.items()}


def _coverage_batches(corruptor, clean: CleanBatch, rollout_steps: int) -> list[CorruptionBatch]:
    if rollout_steps <= 1:
        return [corruptor.sample_pair(clean)]
    path = corruptor.sample_path(clean, rollout_steps=rollout_steps)
    batches: list[CorruptionBatch] = []
    for index in range(len(path.action_ops)):
        target_mask = path.action_ops[index] == int(Op.REPLACE)
        batches.append(
            CorruptionBatch(
                clean_ids=clean.input_ids,
                input_ids=path.states[index],
                prev_ids=path.states[index + 1],
                attention_mask=path.attention_masks[index],
                prev_attention_mask=path.attention_masks[index + 1],
                editable_mask=path.editable_mask,
                segment_ids=path.segment_ids,
                n=path.n_values[index],
                action_ops=path.action_ops[index],
                action_tokens=path.action_tokens[index],
                target_mask=target_mask,
                target_n=path.n_values[index + 1],
            )
        )
    return batches


@torch.no_grad()
def proposal_coverage_for_batch(
    model: SequenceEditJEPA,
    batch: CorruptionBatch,
    *,
    m_values: list[int],
    k_values: list[int],
    mask_token_id: int,
) -> dict[str, float]:
    hidden = model.encoder(batch.input_ids, batch.n, batch.attention_mask, batch.segment_ids)
    op_logits, token_logits = model.policy(hidden, attention_mask=batch.attention_mask)
    token_logits = suppress_token_logits(token_logits, [mask_token_id])
    target_mask = (batch.action_ops == int(Op.REPLACE)) & batch.attention_mask.bool()
    remaining = (batch.input_ids != batch.clean_ids) & batch.editable_mask.bool() & batch.attention_mask.bool()
    remaining = remaining | target_mask
    return proposal_coverage_from_logits(
        op_logits,
        token_logits,
        remaining,
        target_mask,
        batch.clean_ids,
        m_values=m_values,
        k_values=k_values,
    )


def proposal_coverage_from_logits(
    op_logits: torch.Tensor,
    token_logits: torch.Tensor,
    remaining: torch.Tensor,
    target_mask: torch.Tensor,
    clean_ids: torch.Tensor,
    *,
    m_values: list[int],
    k_values: list[int],
) -> dict[str, float]:
    replace_prob = op_logits.softmax(dim=-1)[..., int(Op.REPLACE)]
    token_prob = token_logits.softmax(dim=-1).amax(dim=-1)
    rank_score = replace_prob * token_prob
    metrics: dict[str, float] = {
        "target_positions": float(target_mask.float().sum().item()),
        "rows_with_targets": 0.0,
        "corrupted_positions": float(remaining.float().sum().item()),
        "replace_prob_target_mean": 0.0,
        "position_rank_target_mean": 0.0,
        "token_rank_target_mean": 0.0,
    }
    for m in m_values:
        metrics[f"position_recall_at_{m}"] = 0.0
        metrics[f"position_any_hit_at_{m}"] = 0.0
        for k in k_values:
            metrics[f"pair_recall_at_{m}x{k}"] = 0.0
    for k in k_values:
        metrics[f"token_recall_at_{k}"] = 0.0

    target_total = 0
    for row in range(op_logits.shape[0]):
        targets = torch.where(target_mask[row])[0]
        if targets.numel() == 0:
            continue
        metrics["rows_with_targets"] += 1.0
        valid_positions = torch.where(remaining[row])[0]
        if valid_positions.numel() == 0:
            valid_positions = targets
        ordered_positions = valid_positions[torch.argsort(rank_score[row, valid_positions], descending=True)]
        row_targets = set(int(pos.item()) for pos in targets)
        top_position_sets = {
            int(m): set(int(pos.item()) for pos in ordered_positions[: min(int(m), int(ordered_positions.numel()))])
            for m in m_values
        }
        for pos in targets:
            pos_i = int(pos.item())
            target_total += 1
            clean_token = int(clean_ids[row, pos_i].item())
            metrics["replace_prob_target_mean"] += float(replace_prob[row, pos_i].item())
            position_rank = _rank_of_id(ordered_positions, pos_i)
            metrics["position_rank_target_mean"] += float(position_rank)
            token_rank = _rank_of_token(token_logits[row, pos_i], clean_token)
            metrics["token_rank_target_mean"] += float(token_rank)
            position_hits = {}
            for m in m_values:
                pos_hit = pos_i in top_position_sets[int(m)]
                position_hits[int(m)] = pos_hit
                if pos_hit:
                    metrics[f"position_recall_at_{m}"] += 1.0
            for k in k_values:
                token_hit = token_rank <= int(k)
                if token_hit:
                    metrics[f"token_recall_at_{k}"] += 1.0
                for m in m_values:
                    if position_hits[int(m)] and token_hit:
                        metrics[f"pair_recall_at_{m}x{k}"] += 1.0
        for m in m_values:
            if row_targets & top_position_sets[int(m)]:
                metrics[f"position_any_hit_at_{m}"] += 1.0

    if target_total > 0:
        for key in list(metrics):
            if key.endswith("_mean") or key.startswith("position_recall") or key.startswith("token_recall") or key.startswith("pair_recall"):
                metrics[key] /= float(target_total)
    row_total = max(1.0, metrics["rows_with_targets"])
    for m in m_values:
        metrics[f"position_any_hit_at_{m}"] /= row_total
    return metrics


def _rank_of_id(ordered_ids: torch.Tensor, target_id: int) -> int:
    matches = torch.where(ordered_ids == int(target_id))[0]
    if matches.numel() == 0:
        return int(ordered_ids.numel()) + 1
    return int(matches[0].item()) + 1


def _rank_of_token(logits: torch.Tensor, token_id: int) -> int:
    target_score = logits[int(token_id)]
    return int((logits > target_score).sum().item()) + 1


@torch.no_grad()
def evaluate_factorized_sampler(
    model: SequenceEditJEPA,
    task,
    tokenizer,
    corruptor,
    seq_len: int,
    eval_config: dict[str, Any],
    device: torch.device,
    prefix: str,
    proposer: DenoisingLM | None = None,
) -> dict[str, float]:
    was_training = bool(model.training)
    proposer_was_training = bool(proposer.training) if proposer is not None else False
    model.eval()
    model.to(device)
    if proposer is not None:
        proposer.eval()
        proposer.to(device)
    batches = int(eval_config.get("batches", 8))
    batch_size = int(eval_config.get("batch_size", 8))
    split = str(eval_config.get("split", "eval"))
    max_steps = int(eval_config.get("max_steps", 0))
    modes = [str(mode) for mode in eval_config.get("factor_modes", DEFAULT_FACTOR_MODES)]
    totals: dict[str, float] = {}
    for _ in range(batches):
        clean = task.sample_batch(batch_size, seq_len, split=split, device=device)
        for mode in modes:
            pred = factorized_denoise(
                model,
                clean,
                tokenizer,
                corruptor,
                mode=mode,
                max_steps=max_steps,
                proposer=proposer,
            )
            for key, value in task.evaluate_batch(pred, clean.input_ids, clean.attention_mask, clean.metadata).items():
                totals[f"{prefix}/{mode}/{key}"] = totals.get(f"{prefix}/{mode}/{key}", 0.0) + float(value)
            remaining = (pred == int(tokenizer.mask_token_id)) & clean.editable_mask.bool() & clean.attention_mask.bool()
            editable_count = (clean.editable_mask.bool() & clean.attention_mask.bool()).float().sum().clamp_min(1.0)
            totals[f"{prefix}/{mode}/remaining_mask_fraction"] = totals.get(f"{prefix}/{mode}/remaining_mask_fraction", 0.0) + float(
                remaining.float().sum().item() / editable_count.item()
            )
    if was_training:
        model.train()
    if proposer is not None and proposer_was_training:
        proposer.train()
    return {key: value / max(1, batches) for key, value in totals.items()}


@torch.no_grad()
def factorized_denoise(
    model: SequenceEditJEPA,
    clean: CleanBatch,
    tokenizer,
    corruptor,
    *,
    mode: str,
    max_steps: int = 0,
    proposer: DenoisingLM | None = None,
) -> torch.Tensor:
    position_source, token_source = _parse_factorized_mode(mode)
    if token_source == "dlm" and proposer is None:
        raise ValueError("A DenoisingLM proposer checkpoint is required for dlm token modes.")
    input_ids = clean.input_ids.clone()
    attention = clean.attention_mask.bool()
    remaining = clean.editable_mask.bool() & attention
    input_ids[remaining] = int(tokenizer.mask_token_id)
    limit = int(max_steps) if int(max_steps) > 0 else int(remaining.sum(dim=-1).max().item()) + 4
    for _ in range(limit):
        if not bool(remaining.any()):
            break
        n = _n_from_remaining(remaining, clean.editable_mask.bool(), corruptor)
        hidden = model.encoder(input_ids, n, clean.attention_mask, clean.segment_ids)
        op_logits, policy_token_logits = model.policy(hidden, attention_mask=clean.attention_mask)
        policy_token_logits = suppress_token_logits(policy_token_logits, [int(model.config.mask_token_id)])
        if token_source == "dlm":
            assert proposer is not None
            proposer_hidden = proposer.encoder(input_ids, n, clean.attention_mask, clean.segment_ids)
            token_logits = suppress_token_logits(proposer.decoder(proposer_hidden, attention_mask=clean.attention_mask), [int(model.config.mask_token_id)])
        else:
            token_logits = policy_token_logits
        replace_prob = op_logits.softmax(dim=-1)[..., int(Op.REPLACE)]
        token_prob = token_logits.softmax(dim=-1).amax(dim=-1)
        scores = replace_prob * token_prob
        for row in range(input_ids.shape[0]):
            positions = torch.where(remaining[row])[0]
            if positions.numel() == 0:
                continue
            if position_source == "oracle":
                pos = positions[0]
            elif position_source == "model":
                pos = positions[torch.argmax(scores[row, positions])]
            else:
                raise ValueError(f"Unknown position source {position_source!r}.")
            if token_source == "oracle":
                token = clean.input_ids[row, pos]
            elif token_source in {"model", "policy", "dlm"}:
                token = token_logits[row, pos].argmax(dim=-1)
            else:
                raise ValueError(f"Unknown token source {token_source!r}.")
            input_ids[row, pos] = token
            remaining[row, pos] = False
    input_ids = input_ids.masked_fill(~attention, int(tokenizer.pad_token_id))
    return input_ids


def _parse_factorized_mode(mode: str) -> tuple[str, str]:
    normalized = mode.strip().lower()
    aliases = {
        "model_model": ("model", "model"),
        "model_pos_model_token": ("model", "model"),
        "oracle_pos_model_token": ("oracle", "model"),
        "model_pos_oracle_token": ("model", "oracle"),
        "oracle_oracle": ("oracle", "oracle"),
        "oracle_pos_oracle_token": ("oracle", "oracle"),
        "model_pos_dlm_token": ("model", "dlm"),
        "oracle_pos_dlm_token": ("oracle", "dlm"),
    }
    if normalized in aliases:
        return aliases[normalized]
    if "+" in normalized:
        position_source, token_source = normalized.split("+", 1)
        return position_source, token_source
    raise ValueError(f"Unknown factorized mode {mode!r}.")


def main() -> None:
    args = _parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    root = Path(args.runs_root or os.environ.get("SEQ_EDIT_JEPA_WORK_ROOT", "outputs")).expanduser()
    if root.name != "runs" and (root / "runs").exists():
        root = root / "runs"
    run_dirs = [root / name for name in (args.runs or DEFAULT_RUNS)]
    output = Path(args.output or _default_output_path()).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    proposer = _load_proposer(args, root, device) if _needs_proposer(args) else None
    rows: list[dict[str, Any]] = []
    with open(output, "w", encoding="utf-8") as handle:
        for run_dir in run_dirs:
            checkpoints = _checkpoint_dirs(run_dir, args.checkpoint_glob)
            if args.latest_only:
                checkpoints = checkpoints[-1:]
            for checkpoint in checkpoints:
                for split_name, split in _eval_splits(args):
                    metrics = evaluate_checkpoint_on_split(
                        checkpoint,
                        split=split,
                        split_name=split_name,
                        seq_len=args.seq_len,
                        batches=args.batches,
                        batch_size=args.batch_size,
                        device=device,
                        ood_op_values=args.ood_op_values,
                        modulus=args.modulus,
                        diagnostics=args.diagnostics,
                        coverage_rollout_steps=args.coverage_rollout_steps,
                        m_values=args.m_values,
                        k_values=args.k_values,
                        factor_modes=args.factor_modes,
                        max_steps=args.max_steps,
                        proposer=proposer,
                    )
                    row = {
                        "run": run_dir.name,
                        "checkpoint": checkpoint.name,
                        "step": _checkpoint_step(checkpoint),
                        "split_name": split_name,
                        "split": split,
                        "diagnostics": args.diagnostics,
                        **metrics,
                    }
                    rows.append(row)
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    handle.flush()
                    print(json.dumps(row, sort_keys=True), flush=True)

    summary = _summarize(rows)
    summary_path = output.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps({"metrics": str(output), "summary": str(summary_path)}, sort_keys=True))


def evaluate_checkpoint_on_split(
    checkpoint: Path,
    *,
    split: str,
    split_name: str,
    seq_len: int,
    batches: int,
    batch_size: int,
    device: torch.device,
    ood_op_values: list[int],
    modulus: int | None,
    diagnostics: list[str],
    coverage_rollout_steps: int,
    m_values: list[int],
    k_values: list[int],
    factor_modes: list[str],
    max_steps: int,
    proposer: DenoisingLM | None,
) -> dict[str, float]:
    run_dir = checkpoint.parent
    config = load_yaml(run_dir / "config.yaml")
    task_cfg = dict(config.get("task", {}))
    task_cfg["ood_op_values"] = list(ood_op_values)
    task_cfg["ood_modulus"] = modulus
    if split in {"eval_ood", "test_ood"} or split.startswith("op_") or split.startswith("eval_op_"):
        task_cfg["modulus"] = modulus
    tokenizer_path = run_dir / "tokenizer"
    if (tokenizer_path / "vocab.json").exists():
        task_cfg["_tokenizer_path"] = str(tokenizer_path)
    task = build_task(task_cfg)
    tokenizer = task.tokenizer

    model = _load_model(checkpoint, device)
    if not isinstance(model, SequenceEditJEPA):
        raise TypeError(f"Stepwise diagnostics are defined for SequenceEditJEPA, got {type(model).__name__}.")
    _allow_rope_length_extrapolation(model, seq_len)
    if proposer is not None:
        _allow_rope_length_extrapolation(proposer, seq_len)
    corruptor = build_corruptor(dict(config.get("corruptor", {})), tokenizer)
    eval_cfg = {
        "split": split,
        "batches": batches,
        "batch_size": batch_size,
        "coverage_rollout_steps": coverage_rollout_steps,
        "m_values": m_values,
        "k_values": k_values,
        "factor_modes": factor_modes,
        "max_steps": max_steps,
    }
    metrics: dict[str, float] = {}
    if "coverage" in diagnostics:
        coverage = evaluate_proposal_coverage(model, task, corruptor, seq_len, eval_cfg, device, prefix="eval/proposal")
        metrics.update({f"{split_name}/{key.removeprefix('eval/')}": float(value) for key, value in coverage.items()})
    if "factorized" in diagnostics:
        factorized = evaluate_factorized_sampler(model, task, tokenizer, corruptor, seq_len, eval_cfg, device, prefix="eval/factorized", proposer=proposer)
        metrics.update({f"{split_name}/{key.removeprefix('eval/')}": float(value) for key, value in factorized.items()})
    return metrics


def _needs_proposer(args: argparse.Namespace) -> bool:
    return any("dlm" in str(mode).lower() for mode in args.factor_modes)


def _load_proposer(args: argparse.Namespace, root: Path, device: torch.device) -> DenoisingLM:
    if args.proposer_checkpoint:
        checkpoint = Path(args.proposer_checkpoint).expanduser()
    else:
        run_name = args.proposer_run or "igsm_official_med_mask_x0_denoising_lm_200k"
        run_dir = root / run_name
        checkpoints = _checkpoint_dirs(run_dir, args.proposer_checkpoint_glob)
        checkpoint = checkpoints[-1]
    model = _load_model(checkpoint, device)
    if not isinstance(model, DenoisingLM):
        raise TypeError(f"Expected DenoisingLM proposer at {checkpoint}, got {type(model).__name__}.")
    return model


def _default_output_path() -> str:
    root = Path(os.environ.get("SEQ_EDIT_JEPA_WORK_ROOT", "outputs")) / "posthoc" / "igsm_ood"
    root.mkdir(parents=True, exist_ok=True)
    return str(root / "stepwise_diagnostics.jsonl")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stepwise JEPA proposal, factorized-oracle, and proposer diagnostics.")
    parser.add_argument("--runs-root", default=None)
    parser.add_argument("--runs", nargs="*", default=None)
    parser.add_argument("--checkpoint-glob", default="checkpoint-*")
    parser.add_argument("--latest-only", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--ood-op-values", nargs="+", type=int, default=[20, 21, 22, 23])
    parser.add_argument("--modulus", type=int, default=23)
    parser.add_argument("--by-op", action="store_true")
    parser.add_argument("--include-id", action="store_true")
    parser.add_argument("--diagnostics", nargs="+", choices=["coverage", "factorized"], default=["coverage", "factorized"])
    parser.add_argument("--coverage-rollout-steps", type=int, default=1)
    parser.add_argument("--m-values", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 5, 20, 100])
    parser.add_argument("--factor-modes", nargs="+", default=DEFAULT_FACTOR_MODES)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--proposer-run", default=None)
    parser.add_argument("--proposer-checkpoint", default=None)
    parser.add_argument("--proposer-checkpoint-glob", default="checkpoint-*")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    main()
