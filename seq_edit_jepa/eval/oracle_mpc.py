from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from seq_edit_jepa.actions.action_types import Op
from seq_edit_jepa.data.corruptors import build_corruptor
from seq_edit_jepa.data.datasets import CleanBatch
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
from seq_edit_jepa.models import SequenceEditJEPA
from seq_edit_jepa.train.config import load_yaml


DEFAULT_RUNS = [
    "igsm_official_med_step_mask_jepa_T20_200k",
    "igsm_official_med_step_mask_jepa_T50_200k",
]


@torch.no_grad()
def evaluate_oracle_mpc(
    model: SequenceEditJEPA,
    task,
    tokenizer,
    corruptor,
    seq_len: int,
    eval_config: dict[str, Any],
    device: torch.device,
    prefix: str,
) -> dict[str, float]:
    was_training = bool(model.training)
    model.eval()
    model.to(device)
    batches = int(eval_config.get("batches", 4))
    batch_size = int(eval_config.get("batch_size", 1))
    split = str(eval_config.get("split", "eval"))
    totals: dict[str, float] = {}
    for _ in range(batches):
        clean = task.sample_batch(batch_size, seq_len, split=split, device=device)
        pred = oracle_mpc_denoise(
            model,
            clean,
            tokenizer,
            corruptor,
            horizon=int(eval_config.get("horizon", 1)),
            candidates_per_step=int(eval_config.get("candidates_per_step", 4)),
            tokens_per_position=int(eval_config.get("tokens_per_position", 1)),
            max_steps=int(eval_config.get("max_steps", 0)),
            policy_weight=float(eval_config.get("policy_weight", 0.0)),
            score_mask=str(eval_config.get("score_mask", "editable")),
            include_oracle_token=bool(eval_config.get("include_oracle_token", False)),
            rollout_mode=str(eval_config.get("rollout_mode", "latent")),
            score_mode=str(eval_config.get("score_mode", "oracle_goal")),
        )
        for key, value in task.evaluate_batch(pred, clean.input_ids, clean.attention_mask, clean.metadata).items():
            totals[f"{prefix}/{key}"] = totals.get(f"{prefix}/{key}", 0.0) + float(value)
        remaining = (pred == int(tokenizer.mask_token_id)) & clean.editable_mask.bool() & clean.attention_mask.bool()
        editable_count = (clean.editable_mask.bool() & clean.attention_mask.bool()).float().sum().clamp_min(1.0)
        totals[f"{prefix}/mpc_remaining_mask_fraction"] = totals.get(f"{prefix}/mpc_remaining_mask_fraction", 0.0) + float(
            remaining.float().sum().item() / editable_count.item()
        )
    if was_training:
        model.train()
    return {key: value / max(1, batches) for key, value in totals.items()}


@torch.no_grad()
def oracle_mpc_denoise(
    model: SequenceEditJEPA,
    clean: CleanBatch,
    tokenizer,
    corruptor,
    *,
    horizon: int = 1,
    candidates_per_step: int = 4,
    tokens_per_position: int = 1,
    max_steps: int = 0,
    policy_weight: float = 0.0,
    score_mask: str = "editable",
    include_oracle_token: bool = False,
    rollout_mode: str = "latent",
    score_mode: str = "oracle_goal",
) -> torch.Tensor:
    rows = []
    for row in range(clean.input_ids.shape[0]):
        single = replace(
            clean,
            input_ids=clean.input_ids[row : row + 1],
            attention_mask=clean.attention_mask[row : row + 1],
            editable_mask=clean.editable_mask[row : row + 1],
            segment_ids=clean.segment_ids[row : row + 1],
            metadata=clean.metadata[row : row + 1],
        )
        rows.append(
            _oracle_mpc_single(
                model,
                single,
                tokenizer,
                corruptor,
                horizon=max(1, int(horizon)),
                candidates_per_step=max(1, int(candidates_per_step)),
                tokens_per_position=max(1, int(tokens_per_position)),
                max_steps=max(0, int(max_steps)),
                policy_weight=float(policy_weight),
                score_mask=score_mask,
                include_oracle_token=include_oracle_token,
                rollout_mode=rollout_mode,
                score_mode=score_mode,
            )
        )
    return torch.cat(rows, dim=0)


@torch.no_grad()
def _oracle_mpc_single(
    model: SequenceEditJEPA,
    clean: CleanBatch,
    tokenizer,
    corruptor,
    *,
    horizon: int,
    candidates_per_step: int,
    tokens_per_position: int,
    max_steps: int,
    policy_weight: float,
    score_mask: str,
    include_oracle_token: bool,
    rollout_mode: str,
    score_mode: str,
) -> torch.Tensor:
    input_ids = clean.input_ids.clone()
    attention_mask = clean.attention_mask.bool()
    remaining = clean.editable_mask.bool() & attention_mask
    input_ids[remaining] = int(tokenizer.mask_token_id)
    goal_n = torch.zeros((1,), dtype=torch.float32, device=input_ids.device)
    goal_hidden = model.target_encoder(clean.input_ids, goal_n, clean.attention_mask, clean.segment_ids)
    score_positions = _score_positions(clean, remaining, score_mask)

    step = 0
    limit = int(max_steps) if int(max_steps) > 0 else int(remaining.sum().item()) + 4
    while bool(remaining.any()) and step < limit:
        n = _n_from_remaining(remaining, clean.editable_mask.bool(), corruptor)
        hidden = model.encoder(input_ids, n, clean.attention_mask, clean.segment_ids)
        op_logits, token_logits = model.policy(hidden, attention_mask=clean.attention_mask)
        token_logits = suppress_token_logits(token_logits, [int(model.config.mask_token_id)])
        positions, tokens, logprobs = _first_action_candidates(
            op_logits,
            token_logits,
            remaining,
            clean.input_ids,
            candidates_per_step=candidates_per_step,
            tokens_per_position=tokens_per_position,
            include_oracle_token=include_oracle_token,
        )
        if positions.numel() == 0:
            break
        best = _choose_oracle_action(
            model,
            input_ids,
            hidden,
            n,
            clean.attention_mask,
            clean.segment_ids,
            clean.editable_mask.bool(),
            remaining,
            positions,
            tokens,
            logprobs,
            goal_hidden,
            score_positions,
            corruptor,
            horizon=horizon,
            policy_weight=policy_weight,
            rollout_mode=rollout_mode,
            score_mode=score_mode,
        )
        pos = int(positions[best].item())
        token = int(tokens[best].item())
        input_ids[0, pos] = token
        remaining[0, pos] = False
        step += 1
    input_ids = input_ids.masked_fill(~attention_mask, int(tokenizer.pad_token_id))
    return input_ids


def _first_action_candidates(
    op_logits: torch.Tensor,
    token_logits: torch.Tensor,
    remaining: torch.Tensor,
    clean_ids: torch.Tensor,
    *,
    candidates_per_step: int,
    tokens_per_position: int,
    include_oracle_token: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positions = torch.where(remaining[0])[0]
    if positions.numel() == 0:
        empty = torch.empty((0,), dtype=torch.long, device=remaining.device)
        return empty, empty, torch.empty((0,), dtype=op_logits.dtype, device=remaining.device)
    replace_logprob = op_logits.log_softmax(dim=-1)[0, :, int(Op.REPLACE)]
    token_logprob = token_logits.log_softmax(dim=-1)[0]
    token_prob = token_logits.softmax(dim=-1)[0].amax(dim=-1)
    rank_score = op_logits.softmax(dim=-1)[0, :, int(Op.REPLACE)] * token_prob
    top_count = min(int(candidates_per_step), int(positions.numel()))
    top_positions = positions[torch.topk(rank_score[positions], k=top_count).indices]

    candidate_positions: list[torch.Tensor] = []
    candidate_tokens: list[torch.Tensor] = []
    candidate_logprobs: list[torch.Tensor] = []
    seen: set[tuple[int, int]] = set()
    for pos in top_positions:
        token_count = min(int(tokens_per_position), int(token_logits.shape[-1]))
        top_tokens = torch.topk(token_logprob[pos], k=token_count).indices
        if include_oracle_token:
            top_tokens = torch.unique(torch.cat([top_tokens, clean_ids[0, pos].view(1)]), sorted=False)
        for token in top_tokens:
            key = (int(pos.item()), int(token.item()))
            if key in seen:
                continue
            seen.add(key)
            candidate_positions.append(pos)
            candidate_tokens.append(token)
            candidate_logprobs.append(replace_logprob[pos] + token_logprob[pos, token])
    if not candidate_positions:
        empty = torch.empty((0,), dtype=torch.long, device=remaining.device)
        return empty, empty, torch.empty((0,), dtype=op_logits.dtype, device=remaining.device)
    return torch.stack(candidate_positions), torch.stack(candidate_tokens), torch.stack(candidate_logprobs)


@torch.no_grad()
def _choose_oracle_action(
    model: SequenceEditJEPA,
    input_ids: torch.Tensor,
    hidden: torch.Tensor,
    n: torch.Tensor,
    attention_mask: torch.Tensor,
    segment_ids: torch.Tensor,
    editable_mask: torch.Tensor,
    remaining: torch.Tensor,
    positions: torch.Tensor,
    tokens: torch.Tensor,
    logprobs: torch.Tensor,
    goal_hidden: torch.Tensor,
    score_positions: torch.Tensor,
    corruptor,
    *,
    horizon: int,
    policy_weight: float,
    rollout_mode: str,
    score_mode: str,
) -> int:
    normalized_rollout_mode = str(rollout_mode).strip().lower()
    if normalized_rollout_mode not in {"latent", "reencode"}:
        raise ValueError(f"Unknown rollout_mode={rollout_mode!r}; expected latent or reencode.")
    candidate_count = int(positions.numel())
    hidden_roll = hidden.expand(candidate_count, -1, -1)
    attention = attention_mask.expand(candidate_count, -1)
    segments = segment_ids.expand(candidate_count, -1)
    editable = editable_mask.expand(candidate_count, -1)
    rem = remaining.expand(candidate_count, -1).clone()
    current_n = n.expand(candidate_count).clone()
    current_ids = input_ids.expand(candidate_count, -1).clone()
    policy_logprob = logprobs.clone()

    rows = torch.arange(candidate_count, device=rem.device)
    current_ids[rows, positions] = tokens
    rem[rows, positions] = False
    current_n = _n_from_remaining(rem, editable, corruptor)
    if normalized_rollout_mode == "latent":
        hidden_roll = _roll_one_step(model, hidden_roll, n.expand(candidate_count).clone(), attention, positions, tokens)
    else:
        hidden_roll = model.encoder(current_ids, current_n, attention, segments)

    for _ in range(1, int(horizon)):
        active = rem.any(dim=-1)
        if not bool(active.any()):
            break
        op_logits, token_logits = model.policy(hidden_roll, attention_mask=attention)
        token_logits = suppress_token_logits(token_logits, [int(model.config.mask_token_id)])
        replace_logprob = op_logits.log_softmax(dim=-1)[..., int(Op.REPLACE)]
        token_logprob = token_logits.log_softmax(dim=-1)
        best_token_logprob, best_token = token_logprob.max(dim=-1)
        action_score = replace_logprob + best_token_logprob
        action_score = action_score.masked_fill(~rem, torch.finfo(action_score.dtype).min)
        next_positions = action_score.argmax(dim=-1)
        next_tokens = best_token[torch.arange(candidate_count, device=rem.device), next_positions]
        step_n = current_n.clone()
        policy_logprob = policy_logprob + torch.where(
            active,
            action_score[torch.arange(candidate_count, device=rem.device), next_positions],
            torch.zeros_like(policy_logprob),
        )
        active_rows = torch.where(active)[0]
        current_ids[active_rows, next_positions[active_rows]] = next_tokens[active_rows]
        rem[active_rows, next_positions[active_rows]] = False
        current_n = _n_from_remaining(rem, editable, corruptor)
        if normalized_rollout_mode == "latent":
            next_hidden = _roll_one_step(model, hidden_roll, step_n, attention, next_positions, next_tokens)
        else:
            next_hidden = model.encoder(current_ids, current_n, attention, segments)
        hidden_roll = torch.where(active.view(-1, 1, 1), next_hidden, hidden_roll)

    goal = goal_hidden.expand(candidate_count, -1, -1)
    mask = score_positions.expand(candidate_count, -1)
    scores = _score_rollout_terminal(model, hidden_roll, goal, mask, attention, score_mode)
    if policy_weight != 0.0:
        scores = scores + float(policy_weight) * policy_logprob / float(max(1, horizon))
    return int(scores.argmax(dim=0).item())


def _score_rollout_terminal(
    model: SequenceEditJEPA,
    hidden: torch.Tensor,
    goal_hidden: torch.Tensor,
    score_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    score_mode: str,
) -> torch.Tensor:
    normalized = str(score_mode).strip().lower()
    if normalized == "oracle_goal":
        return _latent_goal_score(model, hidden, goal_hidden, score_mask)
    if normalized == "value_head":
        _, pooled = model.value_head(hidden, attention_mask)
        return pooled
    raise ValueError(f"Unknown score_mode={score_mode!r}; expected oracle_goal or value_head.")


def _roll_one_step(
    model: SequenceEditJEPA,
    hidden: torch.Tensor,
    n: torch.Tensor,
    attention_mask: torch.Tensor,
    positions: torch.Tensor,
    tokens: torch.Tensor,
) -> torch.Tensor:
    batch, length, _ = hidden.shape
    ops = torch.full((batch, length), int(Op.KEEP), dtype=torch.long, device=hidden.device)
    action_tokens = torch.full((batch, length), int(model.config.pad_token_id), dtype=torch.long, device=hidden.device)
    rows = torch.arange(batch, device=hidden.device)
    ops[rows, positions] = int(Op.REPLACE)
    action_tokens[rows, positions] = tokens
    return model.predictor(hidden, ops, action_tokens, n, attention_mask)


def _latent_goal_score(model: SequenceEditJEPA, hidden: torch.Tensor, goal_hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred = model.online_projector(hidden)
    goal = model.target_projector(goal_hidden)
    per_token = (pred - goal).pow(2).mean(dim=-1)
    weights = mask.float()
    denom = weights.sum(dim=-1).clamp_min(1.0)
    return -((per_token * weights).sum(dim=-1) / denom)


def _score_positions(clean: CleanBatch, remaining: torch.Tensor, score_mask: str) -> torch.Tensor:
    normalized = str(score_mask).strip().lower()
    attention = clean.attention_mask.bool()
    if normalized == "attention":
        return attention
    if normalized == "remaining":
        return remaining.bool() & attention
    if normalized == "editable":
        return clean.editable_mask.bool() & attention
    raise ValueError(f"Unknown score_mask={score_mask!r}; expected editable, remaining, or attention.")


def main() -> None:
    args = _parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    root = Path(args.runs_root or os.environ.get("SEQ_EDIT_JEPA_WORK_ROOT", "outputs")).expanduser()
    if root.name != "runs" and (root / "runs").exists():
        root = root / "runs"
    run_dirs = [root / name for name in (args.runs or DEFAULT_RUNS)]
    output = Path(args.output or _default_output_path()).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

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
                        horizon=args.horizon,
                        candidates_per_step=args.candidates_per_step,
                        tokens_per_position=args.tokens_per_position,
                        max_steps=args.max_steps,
                        policy_weight=args.policy_weight,
                        score_mask=args.score_mask,
                        include_oracle_token=args.include_oracle_token,
                        rollout_mode=args.rollout_mode,
                        score_mode=args.score_mode,
                        value_head=args.value_head,
                    )
                    row = {
                        "run": run_dir.name,
                        "checkpoint": checkpoint.name,
                        "step": _checkpoint_step(checkpoint),
                        "split_name": split_name,
                        "split": split,
                        "horizon": args.horizon,
                        "candidates_per_step": args.candidates_per_step,
                        "tokens_per_position": args.tokens_per_position,
                        "max_steps": args.max_steps,
                        "policy_weight": args.policy_weight,
                        "score_mask": args.score_mask,
                        "include_oracle_token": args.include_oracle_token,
                        "rollout_mode": args.rollout_mode,
                        "score_mode": args.score_mode,
                        "value_head": args.value_head,
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
    horizon: int,
    candidates_per_step: int,
    tokens_per_position: int,
    max_steps: int,
    policy_weight: float,
    score_mask: str,
    include_oracle_token: bool,
    rollout_mode: str,
    score_mode: str,
    value_head: str | None,
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
        raise TypeError(f"Oracle latent MPC is defined for SequenceEditJEPA, got {type(model).__name__}.")
    if value_head:
        state = torch.load(Path(value_head).expanduser(), map_location=device)
        if isinstance(state, dict) and "value_head" in state:
            state = state["value_head"]
        model.value_head.load_state_dict(state)
    _allow_rope_length_extrapolation(model, seq_len)
    corruptor = build_corruptor(dict(config.get("corruptor", {})), tokenizer)
    eval_cfg = {
        "split": split,
        "batches": batches,
        "batch_size": batch_size,
        "horizon": horizon,
        "candidates_per_step": candidates_per_step,
        "tokens_per_position": tokens_per_position,
        "max_steps": max_steps,
        "policy_weight": policy_weight,
        "score_mask": score_mask,
        "include_oracle_token": include_oracle_token,
        "rollout_mode": rollout_mode,
        "score_mode": score_mode,
    }
    metrics = evaluate_oracle_mpc(model, task, tokenizer, corruptor, seq_len, eval_cfg, device, prefix="eval/oracle_mpc")
    return {f"{split_name}/{key.removeprefix('eval/')}": float(value) for key, value in metrics.items()}


def _default_output_path() -> str:
    root = Path(os.environ.get("SEQ_EDIT_JEPA_WORK_ROOT", "outputs")) / "posthoc" / "igsm_ood"
    root.mkdir(parents=True, exist_ok=True)
    return str(root / "oracle_mpc_metrics.jsonl")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Oracle-goal latent MPC diagnostic for stepwise JEPA checkpoints.")
    parser.add_argument("--runs-root", default=None)
    parser.add_argument("--runs", nargs="*", default=None)
    parser.add_argument("--checkpoint-glob", default="checkpoint-*")
    parser.add_argument("--latest-only", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--ood-op-values", nargs="+", type=int, default=[20, 23])
    parser.add_argument("--modulus", type=int, default=23)
    parser.add_argument("--by-op", action="store_true")
    parser.add_argument("--include-id", action="store_true")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--candidates-per-step", type=int, default=4)
    parser.add_argument("--tokens-per-position", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means one action until no editable masks remain.")
    parser.add_argument("--policy-weight", type=float, default=0.0)
    parser.add_argument("--score-mask", choices=["editable", "remaining", "attention"], default="editable")
    parser.add_argument("--include-oracle-token", action="store_true")
    parser.add_argument("--rollout-mode", choices=["latent", "reencode"], default="latent")
    parser.add_argument("--score-mode", choices=["oracle_goal", "value_head"], default="oracle_goal")
    parser.add_argument("--value-head", default=None, help="Optional path to a posthoc value_head_latest.pt or saved value-head bundle.")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    main()
