from __future__ import annotations

import argparse
import json
from pathlib import Path

from seq_edit_jepa.train.hf_experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Compatibility wrapper around the HF Trainer experiment runner.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    metrics = run_experiment(Path(args.config))
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
