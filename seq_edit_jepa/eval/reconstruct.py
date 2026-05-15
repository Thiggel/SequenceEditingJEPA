from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from seq_edit_jepa.data.corruptors import build_corruptor
from seq_edit_jepa.data.tasks import build_task
from seq_edit_jepa.models import DenoisingLM, SequenceEditJEPA, build_model
from seq_edit_jepa.train.evaluate import evaluate


def evaluate_checkpoint(checkpoint_path: str | Path, batches: int = 8, device: str = "auto") -> dict[str, float]:
    target_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_dir():
        config_path = checkpoint_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Could not find config.json under {checkpoint_path}.")
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
        cls = DenoisingLM if "DenoisingLM" in raw_config.get("architectures", []) else SequenceEditJEPA
        model = cls.from_pretrained(checkpoint_path).to(target_device)
        source_config_path = checkpoint_path.parent / "config.yaml"
        if not source_config_path.exists():
            raise FileNotFoundError("HF model directory evaluation also needs the saved experiment config.yaml beside model/.")
        from seq_edit_jepa.train.config import load_yaml

        config = load_yaml(source_config_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=target_device)
        config = checkpoint["config"]
        model = None
    task = build_task(dict(config.get("task", {})))
    tokenizer = task.build_tokenizer()
    seq_len = int(config.get("task", {}).get("seq_len", config.get("model", {}).get("max_length", 128)))
    corruptor = build_corruptor(dict(config.get("corruptor", {})), tokenizer)
    if model is None:
        model = build_model(dict(config.get("model", {})), tokenizer, corruptor.num_steps, seq_len).to(target_device)
        model.load_state_dict(checkpoint["model"])
    eval_cfg = dict(config.get("eval", {}))
    eval_cfg["batches"] = batches
    return evaluate(model, task, corruptor, seq_len, eval_cfg, target_device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    print(json.dumps(evaluate_checkpoint(args.checkpoint, args.batches, args.device), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
