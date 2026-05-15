from __future__ import annotations

from typing import Any, Sequence

import torch

from seq_edit_jepa.eval.causal_generation import generate_from_prompt
from seq_edit_jepa.eval.full_denoise import (
    _fixed_k_fill_mask,
    _n_from_remaining,
    _parse_commit_k,
    _predict_tokens_and_scores,
    _scheduled_fill_mask,
)
from seq_edit_jepa.models import CausalTransformerLM, DenoisingLM, SequenceEditJEPA


@torch.no_grad()
def format_generation_samples(
    model,
    task,
    tokenizer,
    seq_len: int,
    device: torch.device,
    *,
    corruptor=None,
    splits: Sequence[str] = ("eval",),
    examples_per_split: int = 1,
    trace_steps: Sequence[int] = (16, 8, 4, 1),
    max_chars: int = 1600,
    step: int | None = None,
    commit_k: int | str | None = None,
    jepa_inference_mode: str = "predictor_decoder",
) -> str:
    was_training = bool(model.training)
    model.eval()
    model.to(device)
    lines = ["", f"[generation samples] step={step}" if step is not None else "[generation samples]"]
    for split in splits:
        for sample_index in range(int(examples_per_split)):
            clean = task.sample_batch(1, int(seq_len), split=split, device=device)
            lines.extend(
                _format_one_sample(
                    model,
                    task,
                    tokenizer,
                    clean,
                    split,
                    sample_index,
                    corruptor,
                    trace_steps,
                    max_chars,
                    commit_k,
                    jepa_inference_mode,
                )
            )
    if was_training:
        model.train()
    return "\n".join(lines)


def _format_one_sample(
    model,
    task,
    tokenizer,
    clean,
    split: str,
    sample_index: int,
    corruptor,
    trace_steps: Sequence[int],
    max_chars: int,
    commit_k: int | str | None,
    jepa_inference_mode: str,
) -> list[str]:
    metadata = clean.metadata[0] if clean.metadata else {}
    lines = [
        "",
        f"[sample split={split} index={sample_index} n_op={metadata.get('n_op', '<na>')}]",
        f"problem: {str(metadata.get('problem_text', ''))[:max_chars]}",
        f"gold:    {_decode_focus(tokenizer, clean.input_ids[0], clean.attention_mask[0], max_chars)}",
    ]
    if isinstance(model, CausalTransformerLM):
        pred, pred_mask = generate_from_prompt(model, clean, tokenizer, clean.input_ids.shape[1])
        lines.append(f"causal:  {_decode_focus(tokenizer, pred[0], pred_mask[0], max_chars)}")
        lines.extend(_answer_lines(task, tokenizer, pred[0], pred_mask[0], metadata))
        return lines
    if isinstance(model, (SequenceEditJEPA, DenoisingLM)):
        if corruptor is None:
            lines.append("denoise: <missing corruptor>")
            return lines
        pred, states = _full_denoise_trace(
            model,
            clean,
            tokenizer,
            corruptor,
            set(int(step) for step in trace_steps),
            commit_k=_parse_commit_k(commit_k),
            jepa_inference_mode=jepa_inference_mode,
        )
        for label, state in states:
            lines.append(f"{label}: {_decode_focus(tokenizer, state[0], clean.attention_mask[0], max_chars)}")
        lines.append(f"final:   {_decode_focus(tokenizer, pred[0], clean.attention_mask[0], max_chars)}")
        lines.extend(_answer_lines(task, tokenizer, pred[0], clean.attention_mask[0], metadata))
        return lines
    lines.append(f"generation: <unsupported model {type(model).__name__}>")
    return lines


def _full_denoise_trace(
    model,
    clean,
    tokenizer,
    corruptor,
    trace_steps: set[int],
    *,
    commit_k: int | None = None,
    jepa_inference_mode: str = "predictor_decoder",
):
    input_ids = clean.input_ids.clone()
    remaining = clean.editable_mask & clean.attention_mask.bool()
    input_ids[remaining] = int(tokenizer.mask_token_id)
    states = [(f"n={int(corruptor.num_steps)} start", input_ids.clone())]
    if commit_k is not None:
        commit_k = max(1, int(commit_k))
        step_index = 0
        max_iterations = int(input_ids.shape[1]) + int(corruptor.num_steps) + 4
        while bool(remaining.any()) and step_index < max_iterations:
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
            step_index += 1
            if step_index in trace_steps:
                states.append((f"after k-step={step_index}", input_ids.clone()))
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
        return input_ids, states
    for n_value in range(int(corruptor.num_steps), 0, -1):
        n = torch.full((input_ids.shape[0],), n_value, dtype=torch.long, device=input_ids.device)
        token_logits, scores = _predict_tokens_and_scores(model, input_ids, n, clean.attention_mask, clean.segment_ids, jepa_inference_mode=jepa_inference_mode)
        pred_tokens = token_logits.argmax(dim=-1)
        fill_mask = _scheduled_fill_mask(remaining, scores, clean.editable_mask, corruptor, n_value)
        input_ids[fill_mask] = pred_tokens[fill_mask]
        remaining = remaining & ~fill_mask
        if n_value in trace_steps:
            states.append((f"after n={n_value}", input_ids.clone()))
        if not bool(remaining.any()):
            break
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
    return input_ids, states


def _decode_focus(tokenizer, ids: torch.Tensor, mask: torch.Tensor, max_chars: int) -> str:
    values = ids[mask.bool()].detach().cpu().tolist()
    text = str(tokenizer.decode(values, skip_special_tokens=False)) if hasattr(tokenizer, "decode") else " ".join(str(value) for value in values)
    for marker in ("<igsm_solution>", "<bos>"):
        if marker in text:
            text = text[text.index(marker) :]
            break
    if len(text) > max_chars:
        return text[: max(0, max_chars - 3)] + "..."
    return text


def _answer_lines(task, tokenizer, pred: torch.Tensor, mask: torch.Tensor, metadata: dict[str, Any]) -> list[str]:
    if not hasattr(task, "_extract_answer_ids"):
        return []
    expected = [int(token_id) for token_id in metadata.get("answer_ids", [])]
    predicted = task._extract_answer_ids(pred[mask.bool()].detach().cpu().tolist(), len(expected))
    expected_text = tokenizer.decode(expected, skip_special_tokens=False) if expected else "<missing>"
    predicted_text = tokenizer.decode(predicted, skip_special_tokens=False) if predicted else "<missing>"
    return [
        f"expected_answer:  {expected_text}",
        f"predicted_answer: {predicted_text}",
        f"answer_match: {bool(expected) and predicted == expected}",
    ]
