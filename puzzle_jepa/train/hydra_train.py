from __future__ import annotations

import json

import hydra
from omegaconf import DictConfig, OmegaConf

from puzzle_jepa.train.smoke import run_smoke_experiment


@hydra.main(version_base=None, config_path="../../configs/puzzle", config_name="jepa_sudoku_smoke")
def main(cfg: DictConfig) -> None:
    config = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(config, dict):
        raise TypeError("Hydra config must resolve to a mapping.")
    print(json.dumps(run_smoke_experiment(config), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
