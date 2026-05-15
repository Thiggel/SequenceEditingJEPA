from __future__ import annotations

from typing import Any

import torch

from seq_edit_jepa.actions.apply import apply_fixed_actions
from seq_edit_jepa.actions.metrics import edit_f1, exact_match, token_accuracy
from seq_edit_jepa.losses import suppress_token_logits
from seq_edit_jepa.models import SequenceEditJEPA


@torch.no_grad()
def evaluate(model, task, corruptor, seq_len: int, eval_config: dict[str, Any], device: torch.device) -> dict[str, float]:
    metrics = _evaluate_single(model, task, corruptor, seq_len, eval_config, device, prefix="eval")
    for split_config in eval_config.get("extra_splits", []):
        child_config = dict(eval_config)
        child_config.pop("extra_splits", None)
        child_config.update(dict(split_config))
        name = str(child_config.get("name", child_config.get("split", "extra"))).replace("/", "_")
        metrics.update(_evaluate_single(model, task, corruptor, int(child_config.get("seq_len", seq_len)), child_config, device, prefix=f"eval/{name}"))
    return metrics


@torch.no_grad()
def _evaluate_single(model, task, corruptor, seq_len: int, eval_config: dict[str, Any], device: torch.device, prefix: str) -> dict[str, float]:
    was_training = model.training
    model.eval()
    model.to(device)
    batches = int(eval_config.get("batches", 8))
    batch_size = int(eval_config.get("batch_size", eval_config.get("eval_batch_size", 32)))
    split = str(eval_config.get("split", "eval"))
    totals: dict[str, float] = {}
    count = 0
    for _ in range(batches):
        clean = task.sample_batch(batch_size, seq_len, split=split, device=device)
        batch = corruptor.sample_pair(clean)
        output = model(batch)
        pred_ids = predicted_prev_ids(output, batch, mask_token_id=int(getattr(model.config, "mask_token_id", 2)))
        mask = batch.prev_attention_mask.bool()
        totals[f"{prefix}/loss"] = totals.get(f"{prefix}/loss", 0.0) + float(output.loss.detach().cpu())
        totals[f"{prefix}/token_accuracy"] = totals.get(f"{prefix}/token_accuracy", 0.0) + token_accuracy(pred_ids, batch.prev_ids, mask)
        totals[f"{prefix}/exact_match"] = totals.get(f"{prefix}/exact_match", 0.0) + exact_match(pred_ids, batch.prev_ids, mask)
        if output.op_logits is not None:
            op_pred = output.op_logits.argmax(dim=-1)
            for key, value in edit_f1(op_pred, batch.action_ops, batch.editable_mask & batch.attention_mask.bool()).items():
                totals[f"{prefix}/{key}"] = totals.get(f"{prefix}/{key}", 0.0) + float(value)
        for key, value in task.evaluate_batch(pred_ids, batch.prev_ids, batch.prev_attention_mask, clean.metadata).items():
            totals[f"{prefix}/{key}"] = totals.get(f"{prefix}/{key}", 0.0) + float(value)
        count += 1
    metrics = {key: value / max(1, count) for key, value in totals.items()}
    rollout_steps = [int(step) for step in eval_config.get("rollout_steps", [])]
    if isinstance(model, SequenceEditJEPA):
        for steps in rollout_steps:
            values = []
            for _ in range(max(1, int(eval_config.get("rollout_batches", 2)))):
                clean = task.sample_batch(batch_size, seq_len, split=split, device=device)
                path = corruptor.sample_path(clean, rollout_steps=steps)
                values.append(float(model.rollout_loss(path).detach().cpu()))
            metrics[f"{prefix}/rollout_mse_k{steps}"] = float(sum(values) / max(1, len(values)))
    if was_training:
        model.train()
    return metrics


def predicted_prev_ids(output, batch, mask_token_id: int = 2) -> torch.Tensor:
    if output.op_logits is not None and output.token_logits is not None:
        token_logits = suppress_token_logits(output.token_logits, [mask_token_id])
        return apply_fixed_actions(batch.input_ids, output.op_logits.argmax(dim=-1), token_logits.argmax(dim=-1))
    token_logits = suppress_token_logits(output.logits, [mask_token_id])
    pred = batch.input_ids.clone()
    target_mask = batch.target_mask if batch.target_mask is not None else batch.prev_attention_mask.bool()
    pred[target_mask.bool()] = token_logits.argmax(dim=-1)[target_mask.bool()]
    return pred
