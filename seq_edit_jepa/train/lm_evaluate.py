from __future__ import annotations

import math
from typing import Any

import torch

from seq_edit_jepa.eval.causal_generation import generate_from_prompt


@torch.no_grad()
def evaluate_causal_lm(model, task, tokenizer, seq_len: int, eval_config: dict[str, Any], device: torch.device) -> dict[str, float]:
    metrics = _evaluate_causal_lm_single(model, task, tokenizer, seq_len, eval_config, device, prefix="eval")
    for split_config in eval_config.get("extra_splits", []):
        child_config = dict(eval_config)
        child_config.pop("extra_splits", None)
        child_config.update(dict(split_config))
        name = str(child_config.get("name", child_config.get("split", "extra"))).replace("/", "_")
        metrics.update(
            _evaluate_causal_lm_single(
                model,
                task,
                tokenizer,
                int(child_config.get("seq_len", seq_len)),
                child_config,
                device,
                prefix=f"eval/{name}",
            )
        )
    return metrics


@torch.no_grad()
def _evaluate_causal_lm_single(model, task, tokenizer, seq_len: int, eval_config: dict[str, Any], device: torch.device, prefix: str) -> dict[str, float]:
    was_training = model.training
    model.eval()
    model.to(device)
    batches = int(eval_config.get("batches", 8))
    batch_size = int(eval_config.get("batch_size", eval_config.get("eval_batch_size", 32)))
    split = str(eval_config.get("split", "eval"))
    total_loss = 0.0
    total_acc = 0.0
    total_gen_exact = 0.0
    task_metric_totals: dict[str, float] = {}
    for _ in range(batches):
        clean = task.sample_batch(batch_size, seq_len, split=split, device=device)
        labels = clean.input_ids.masked_fill(~clean.attention_mask.bool(), -100)
        output = model(input_ids=clean.input_ids, attention_mask=clean.attention_mask, labels=labels)
        total_loss += float(output.loss.detach().cpu())
        shift_logits = output.logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        valid = shift_labels != -100
        total_acc += float((shift_logits.argmax(dim=-1)[valid] == shift_labels[valid]).float().mean().detach().cpu()) if valid.any() else 0.0
        gen_ids, gen_mask = generate_from_prompt(model, clean, tokenizer, seq_len)
        total_gen_exact += _exact(gen_ids, clean.input_ids, clean.attention_mask)
        for key, value in task.evaluate_batch(gen_ids[:, :seq_len], clean.input_ids, clean.attention_mask, clean.metadata).items():
            task_metric_totals[f"{prefix}/gen_{key}"] = task_metric_totals.get(f"{prefix}/gen_{key}", 0.0) + float(value)
    metrics = {
        f"{prefix}/loss": total_loss / max(1, batches),
        f"{prefix}/perplexity": math.exp(min(20.0, total_loss / max(1, batches))),
        f"{prefix}/token_accuracy": total_acc / max(1, batches),
        f"{prefix}/generation_exact_match": total_gen_exact / max(1, batches),
    }
    metrics.update({key: value / max(1, batches) for key, value in task_metric_totals.items()})
    if was_training:
        model.train()
    return metrics


def _exact(pred: torch.Tensor, target: torch.Tensor, attention_mask: torch.Tensor) -> float:
    matches = []
    for row in range(pred.shape[0]):
        mask = attention_mask[row].bool()
        matches.append(bool(torch.equal(pred[row][mask], target[row][mask])))
    return float(sum(matches) / max(1, len(matches)))
