from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from seq_edit_jepa.data.corruptors import build_corruptor
from seq_edit_jepa.data.tasks import build_task
from seq_edit_jepa.eval.posthoc_igsm_ood import _allow_rope_length_extrapolation, _checkpoint_dirs, _load_model
from seq_edit_jepa.models import SequenceEditJEPA
from seq_edit_jepa.train.config import load_yaml


def main() -> None:
    args = _parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    run_dir = Path(args.run_dir or _default_run_dir()).expanduser()
    checkpoint = Path(args.checkpoint).expanduser() if args.checkpoint else _checkpoint_dirs(run_dir, args.checkpoint_glob)[-1]
    output_dir = Path(args.output_dir or _default_output_dir(run_dir.name, checkpoint.name)).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_yaml(run_dir / "config.yaml")
    task_cfg = dict(config.get("task", {}))
    tokenizer_path = run_dir / "tokenizer"
    if (tokenizer_path / "vocab.json").exists():
        task_cfg["_tokenizer_path"] = str(tokenizer_path)
    task = build_task(task_cfg)
    tokenizer = task.tokenizer
    corruptor = build_corruptor(dict(config.get("corruptor", {})), tokenizer)
    seq_len = int(args.seq_len or task_cfg.get("seq_len", task_cfg.get("max_length", 1024)))

    model = _load_model(checkpoint, device)
    if not isinstance(model, SequenceEditJEPA):
        raise TypeError(f"Value-head training expects SequenceEditJEPA, got {type(model).__name__}.")
    _allow_rope_length_extrapolation(model, seq_len)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.value_head.parameters():
        parameter.requires_grad_(True)

    optimizer = torch.optim.AdamW(model.value_head.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    metrics_path = output_dir / "metrics.jsonl"
    for step in range(1, int(args.steps) + 1):
        clean = task.sample_batch(int(args.batch_size), seq_len, split="train", device=device)
        batch = corruptor.sample_pair(clean)
        with torch.no_grad():
            hidden = model.encoder(batch.input_ids, batch.n, batch.attention_mask, batch.segment_ids)
            goal_n = torch.zeros((clean.input_ids.shape[0],), dtype=torch.float32, device=device)
            goal_hidden = model.target_encoder(clean.input_ids, goal_n, clean.attention_mask, clean.segment_ids)
            token_target, pooled_target = _oracle_goal_targets(model, hidden, goal_hidden, clean.editable_mask & clean.attention_mask.bool())
        token_value, pooled_value = model.value_head(hidden.detach(), batch.attention_mask)
        score_mask = (clean.editable_mask & clean.attention_mask.bool()).float()
        token_loss = (((token_value - token_target) ** 2) * score_mask).sum() / score_mask.sum().clamp_min(1.0)
        pooled_loss = F.mse_loss(pooled_value.float(), pooled_target.float())
        loss = pooled_loss + float(args.token_loss_weight) * token_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % int(args.log_every) == 0 or step == 1:
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "loss_pooled": float(pooled_loss.detach().cpu()),
                "loss_token": float(token_loss.detach().cpu()),
                "target_mean": float(pooled_target.mean().detach().cpu()),
                "prediction_mean": float(pooled_value.mean().detach().cpu()),
                "run": run_dir.name,
                "checkpoint": checkpoint.name,
            }
            with open(metrics_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps(row, sort_keys=True), flush=True)
        if step % int(args.save_every) == 0:
            _save_value_head(model, output_dir, step, run_dir, checkpoint, args)
    _save_value_head(model, output_dir, int(args.steps), run_dir, checkpoint, args)


@torch.no_grad()
def _oracle_goal_targets(
    model: SequenceEditJEPA,
    hidden: torch.Tensor,
    goal_hidden: torch.Tensor,
    score_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred = model.online_projector(hidden)
    goal = model.target_projector(goal_hidden)
    token_target = -((pred - goal).pow(2).mean(dim=-1))
    weights = score_mask.float()
    pooled_target = (token_target * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)
    return token_target, pooled_target


def _save_value_head(
    model: SequenceEditJEPA,
    output_dir: Path,
    step: int,
    run_dir: Path,
    checkpoint: Path,
    args: argparse.Namespace,
) -> None:
    path = output_dir / f"value_head_step_{step}.pt"
    torch.save(
        {
            "value_head": model.value_head.state_dict(),
            "step": int(step),
            "source_run": run_dir.name,
            "source_checkpoint": str(checkpoint),
            "args": vars(args),
        },
        path,
    )
    torch.save(model.value_head.state_dict(), output_dir / "value_head_latest.pt")


def _default_run_dir() -> str:
    root = Path(os.environ.get("SEQ_EDIT_JEPA_WORK_ROOT", "outputs")) / "runs"
    return str(root / "igsm_official_med_step_mask_jepa_T20_200k")


def _default_output_dir(run_name: str, checkpoint_name: str) -> str:
    root = Path(os.environ.get("SEQ_EDIT_JEPA_WORK_ROOT", "outputs")) / "posthoc" / "value_heads"
    return str(root / f"{run_name}_{checkpoint_name}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight JEPA value head to predict oracle-goal latent energy.")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-glob", default="checkpoint-*")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--token-loss-weight", type=float, default=0.25)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    main()
