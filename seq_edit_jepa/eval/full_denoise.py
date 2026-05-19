from __future__ import annotations

import math
from typing import Any

import torch

from seq_edit_jepa.actions.action_types import Op
from seq_edit_jepa.losses import suppress_token_logits
from seq_edit_jepa.models import DenoisingLM, SequenceEditJEPA


@torch.no_grad()
def evaluate_full_denoise_with_splits(model, task, tokenizer, corruptor, seq_len: int, eval_config: dict[str, Any], device: torch.device) -> dict[str, float]:
    metrics = evaluate_full_denoise(model, task, tokenizer, corruptor, seq_len, eval_config, device, prefix="eval/full_denoise")
    for split_config in eval_config.get("extra_splits", []):
        child_config = dict(eval_config)
        child_config.pop("extra_splits", None)
        child_config.update(dict(split_config))
        name = str(child_config.get("name", child_config.get("split", "extra"))).replace("/", "_")
        metrics.update(
            evaluate_full_denoise(
                model,
                task,
                tokenizer,
                corruptor,
                int(child_config.get("seq_len", seq_len)),
                child_config,
                device,
                prefix=f"eval/{name}/full_denoise",
            )
        )
    return metrics


@torch.no_grad()
def evaluate_full_denoise(model, task, tokenizer, corruptor, seq_len: int, eval_config: dict[str, Any], device: torch.device, prefix: str) -> dict[str, float]:
    was_training = model.training
    model.eval()
    model.to(device)
    batches = int(eval_config.get("batches", 8))
    batch_size = int(eval_config.get("batch_size", eval_config.get("eval_batch_size", 32)))
    split = str(eval_config.get("split", "eval"))
    commit_k = _parse_commit_k(eval_config.get("commit_k", None))
    jepa_inference_mode = str(eval_config.get("jepa_inference_mode", "predictor_decoder"))
    totals: dict[str, float] = {}
    for _ in range(batches):
        clean = task.sample_batch(batch_size, seq_len, split=split, device=device)
        pred = full_denoise(
            model,
            clean,
            tokenizer,
            corruptor,
            commit_k=commit_k,
            jepa_inference_mode=jepa_inference_mode,
        )
        for key, value in task.evaluate_batch(pred, clean.input_ids, clean.attention_mask, clean.metadata).items():
            totals[f"{prefix}/{key}"] = totals.get(f"{prefix}/{key}", 0.0) + float(value)
    if was_training:
        model.train()
    return {key: value / max(1, batches) for key, value in totals.items()}


@torch.no_grad()
def full_denoise(
    model,
    clean,
    tokenizer,
    corruptor,
    *,
    commit_k: int | None = None,
    jepa_inference_mode: str = "predictor_decoder",
) -> torch.Tensor:
    input_ids = clean.input_ids.clone()
    remaining = clean.editable_mask & clean.attention_mask.bool()
    input_ids[remaining] = int(tokenizer.mask_token_id)
    if commit_k is not None:
        return _full_denoise_fixed_k(model, clean, tokenizer, corruptor, input_ids, remaining, int(commit_k), jepa_inference_mode)
    for n_value in range(int(corruptor.num_steps), 0, -1):
        n = torch.full((input_ids.shape[0],), n_value, dtype=torch.long, device=input_ids.device)
        token_logits, scores = _predict_tokens_and_scores(model, input_ids, n, clean.attention_mask, clean.segment_ids, jepa_inference_mode=jepa_inference_mode)
        pred_tokens = token_logits.argmax(dim=-1)
        fill_mask = _scheduled_fill_mask(remaining, scores, clean.editable_mask, corruptor, n_value)
        input_ids[fill_mask] = pred_tokens[fill_mask]
        remaining = remaining & ~fill_mask
        if not bool(remaining.any()):
            break
    input_ids[remaining] = _predict_tokens_and_scores(
        model,
        input_ids,
        torch.ones((input_ids.shape[0],), dtype=torch.long, device=input_ids.device),
        clean.attention_mask,
        clean.segment_ids,
        jepa_inference_mode=jepa_inference_mode,
    )[0].argmax(dim=-1)[remaining]
    input_ids = input_ids.masked_fill(~clean.attention_mask.bool(), int(tokenizer.pad_token_id))
    return input_ids


def _full_denoise_fixed_k(
    model,
    clean,
    tokenizer,
    corruptor,
    input_ids: torch.Tensor,
    remaining: torch.Tensor,
    commit_k: int,
    jepa_inference_mode: str,
) -> torch.Tensor:
    commit_k = max(1, int(commit_k))
    max_iterations = int(input_ids.shape[1]) + int(corruptor.num_steps) + 4
    for _ in range(max_iterations):
        if not bool(remaining.any()):
            break
        n = _n_from_remaining(remaining, clean.editable_mask, corruptor)
        token_logits, scores = _predict_tokens_and_scores(
            model,
            input_ids,
            n,
            clean.attention_mask,
            clean.segment_ids,
            jepa_inference_mode=jepa_inference_mode,
        )
        pred_tokens = token_logits.argmax(dim=-1)
        fill_mask = _fixed_k_fill_mask(remaining, scores, commit_k)
        input_ids[fill_mask] = pred_tokens[fill_mask]
        remaining = remaining & ~fill_mask
    if bool(remaining.any()):
        n = torch.ones((input_ids.shape[0],), dtype=torch.long, device=input_ids.device)
        input_ids[remaining] = _predict_tokens_and_scores(
            model,
            input_ids,
            n,
            clean.attention_mask,
            clean.segment_ids,
            jepa_inference_mode=jepa_inference_mode,
        )[0].argmax(dim=-1)[remaining]
    input_ids = input_ids.masked_fill(~clean.attention_mask.bool(), int(tokenizer.pad_token_id))
    return input_ids


def _predict_tokens_and_scores(
    model,
    input_ids: torch.Tensor,
    n: torch.Tensor,
    attention_mask: torch.Tensor,
    segment_ids: torch.Tensor,
    *,
    jepa_inference_mode: str = "predictor_decoder",
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(model, SequenceEditJEPA):
        hidden = model.encoder(input_ids, n, attention_mask, segment_ids)
        op_logits, action_token_logits = model.policy(hidden, attention_mask=attention_mask)
        action_token_logits = suppress_token_logits(action_token_logits, [model.config.mask_token_id])
        if jepa_inference_mode == "policy_head":
            token_logits = action_token_logits
        elif jepa_inference_mode == "predictor_decoder":
            pred_ops = op_logits.argmax(dim=-1)
            pred_tokens = action_token_logits.argmax(dim=-1)
            pred_tokens = torch.where(
                pred_ops == int(Op.REPLACE),
                pred_tokens,
                torch.full_like(pred_tokens, int(model.config.pad_token_id)),
            )
            hidden_pred = model.predictor(hidden, pred_ops, pred_tokens, n, attention_mask)
            token_logits = suppress_token_logits(model.decoder(hidden_pred, attention_mask=attention_mask), [model.config.mask_token_id])
        else:
            raise ValueError(
                "jepa_inference_mode must be 'predictor_decoder' or 'policy_head', "
                f"got {jepa_inference_mode!r}."
            )
        replace_prob = op_logits.softmax(dim=-1)[..., int(Op.REPLACE)]
        token_prob = token_logits.softmax(dim=-1).amax(dim=-1)
        return token_logits, replace_prob * token_prob
    if isinstance(model, DenoisingLM):
        hidden = model.encoder(input_ids, n, attention_mask, segment_ids)
        token_logits = model.decoder(hidden, attention_mask=attention_mask)
        token_logits = suppress_token_logits(token_logits, [model.config.mask_token_id])
        return token_logits, token_logits.softmax(dim=-1).amax(dim=-1)
    raise TypeError(f"Full denoising is defined for SequenceEditJEPA and DenoisingLM, got {type(model).__name__}.")


def _parse_commit_k(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "none", "schedule", "scheduled", "auto"}:
            return None
        return max(1, int(normalized))
    parsed = int(value)
    return None if parsed <= 0 else parsed


def _n_from_remaining(remaining: torch.Tensor, editable_mask: torch.Tensor, corruptor) -> torch.Tensor:
    n = torch.ones((remaining.shape[0],), dtype=torch.float32, device=remaining.device)
    for row in range(remaining.shape[0]):
        editable_count = int(editable_mask[row].sum().item())
        if editable_count <= 0:
            continue
        ratio = float(remaining[row].sum().item()) / float(editable_count)
        n[row] = max(1.0, min(float(corruptor.num_steps), float(corruptor.inverse_gamma(ratio))))
    return n


def _scheduled_fill_mask(
    remaining: torch.Tensor,
    scores: torch.Tensor,
    editable_mask: torch.Tensor,
    corruptor,
    n_value: int,
) -> torch.Tensor:
    fill_mask = torch.zeros_like(remaining)
    for row in range(remaining.shape[0]):
        positions = torch.where(remaining[row])[0]
        if positions.numel() == 0:
            continue
        editable_count = int(editable_mask[row].sum().item())
        target_remaining = int(math.floor(corruptor.gamma(n_value - 1) * editable_count))
        fill_count = int(positions.numel()) - target_remaining
        if n_value == 1:
            fill_count = int(positions.numel())
        fill_count = max(0, min(fill_count, int(positions.numel())))
        if fill_count == 0:
            continue
        row_scores = scores[row, positions]
        chosen = positions[torch.topk(row_scores, k=fill_count).indices]
        fill_mask[row, chosen] = True
    return fill_mask


def _fixed_k_fill_mask(remaining: torch.Tensor, scores: torch.Tensor, commit_k: int) -> torch.Tensor:
    fill_mask = torch.zeros_like(remaining)
    for row in range(remaining.shape[0]):
        positions = torch.where(remaining[row])[0]
        if positions.numel() == 0:
            continue
        fill_count = min(int(commit_k), int(positions.numel()))
        row_scores = scores[row, positions]
        chosen = positions[torch.topk(row_scores, k=fill_count).indices]
        fill_mask[row, chosen] = True
    return fill_mask
