from __future__ import annotations

import json
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from seq_edit_jepa.train.hf_experiment import run_experiment_from_config
from seq_edit_jepa.train.hf_lm_experiment import run_lm_experiment_from_config


@hydra.main(version_base=None, config_path="../../configs", config_name="smoke_lano_mask")
def main(cfg: DictConfig) -> None:
    config = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(config, dict):
        raise TypeError("Hydra config must resolve to a mapping.")
    if "model" not in config and len(config) == 1:
        only_value = next(iter(config.values()))
        if isinstance(only_value, dict) and "model" in only_value:
            config = only_value
    if str(config.get("model", {}).get("type", "")) == "causal_lm":
        metrics = run_lm_experiment_from_config(config)
    else:
        metrics = run_experiment_from_config(config)
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
