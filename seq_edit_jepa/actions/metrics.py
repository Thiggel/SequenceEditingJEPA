from __future__ import annotations

import torch

from seq_edit_jepa.actions.action_types import Op


def token_accuracy(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    valid = mask.bool()
    denom = int(valid.sum().item())
    if denom == 0:
        return 0.0
    return float((pred[valid] == target[valid]).float().mean().item())


def exact_match(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    valid = mask.bool()
    per_row = []
    for row in range(pred.shape[0]):
        row_mask = valid[row]
        per_row.append(bool(torch.equal(pred[row][row_mask], target[row][row_mask])))
    return float(sum(per_row) / max(1, len(per_row)))


def edit_f1(pred_ops: torch.Tensor, target_ops: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    valid = mask.bool()
    pred = (pred_ops == int(Op.REPLACE)) & valid
    target = (target_ops == int(Op.REPLACE)) & valid
    tp = float((pred & target).sum().item())
    fp = float((pred & ~target).sum().item())
    fn = float((~pred & target).sum().item())
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {"edit_precision": precision, "edit_recall": recall, "edit_f1": f1}
