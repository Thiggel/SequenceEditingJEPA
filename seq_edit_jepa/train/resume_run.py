from __future__ import annotations

import argparse
from pathlib import Path

from seq_edit_jepa.train.config import load_yaml
from seq_edit_jepa.train.hf_experiment import run_experiment_from_config
from seq_edit_jepa.train.hf_lm_experiment import run_lm_experiment_from_config


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir)
    config = load_yaml(run_dir / "config.yaml")
    checkpoint = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint(run_dir)
    config.setdefault("experiment", {})["resume_from_checkpoint"] = str(checkpoint)
    model_type = str(config.get("model", {}).get("type", ""))
    print(f"Resuming {run_dir.name} from {checkpoint}", flush=True)
    if model_type == "causal_lm":
        run_lm_experiment_from_config(config)
    else:
        run_experiment_from_config(config)


def _latest_checkpoint(run_dir: Path) -> Path:
    checkpoints = []
    for checkpoint in run_dir.glob("checkpoint-*"):
        if not checkpoint.is_dir():
            continue
        if not (checkpoint / "trainer_state.json").is_file():
            continue
        if not (checkpoint / "model.safetensors").is_file():
            continue
        checkpoints.append(checkpoint)
    if not checkpoints:
        raise FileNotFoundError(f"No complete checkpoint-* directory found under {run_dir}")
    return max(checkpoints, key=_checkpoint_step)


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return -1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume a saved sequence-edit run from its own config.yaml.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
