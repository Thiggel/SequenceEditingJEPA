from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from seq_edit_jepa.data.corruptors import build_corruptor
from seq_edit_jepa.data.tasks import build_task
from seq_edit_jepa.eval.sample_printing import format_generation_samples
from seq_edit_jepa.models import CausalTransformerLM, DenoisingLM, SequenceEditJEPA
from seq_edit_jepa.train.config import load_yaml


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir)
    checkpoint = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint(run_dir)
    config = load_yaml(run_dir / "config.yaml")
    task_cfg = dict(config.get("task", {}))
    task_cfg["_tokenizer_path"] = str(run_dir / "tokenizer")
    task_cfg["builder_examples"] = max(1, int(args.examples))
    task_cfg["max_length"] = int(args.seq_len)
    task_cfg["seq_len"] = int(args.seq_len)
    task = build_task(task_cfg)
    tokenizer = task.tokenizer
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    model = _load_model(checkpoint, config).to(device).eval()
    corruptor = build_corruptor(dict(config.get("corruptor", {"name": "mask", "num_steps": 16})), tokenizer)
    print(
        format_generation_samples(
            model,
            task,
            tokenizer,
            int(args.seq_len),
            device,
            corruptor=corruptor,
            splits=[args.split],
            examples_per_split=int(args.examples),
            trace_steps=args.trace_steps,
            max_chars=int(args.max_chars),
        ),
        flush=True,
    )


def _load_model(checkpoint: Path, config: dict[str, Any]):
    model_type = str(config.get("model", {}).get("type", ""))
    if model_type == "causal_lm":
        return CausalTransformerLM.from_pretrained(checkpoint)
    if model_type == "denoising_lm":
        return DenoisingLM.from_pretrained(checkpoint)
    return SequenceEditJEPA.from_pretrained(checkpoint)


def _latest_checkpoint(run_dir: Path) -> Path:
    checkpoints = [path for path in run_dir.glob("checkpoint-*") if path.is_dir()]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint-* under {run_dir}")
    return max(checkpoints, key=lambda path: int(path.name.rsplit("-", 1)[1]))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print iGSM generation/debug samples for a saved run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="eval")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--examples", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trace-steps", type=int, nargs="*", default=[16, 8, 4, 1])
    parser.add_argument("--max-chars", type=int, default=2000)
    return parser.parse_args()


if __name__ == "__main__":
    main()
