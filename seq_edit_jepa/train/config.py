from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping.")
    return data


def save_yaml(config: dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def default_work_root() -> Path:
    explicit = os.environ.get("SEQ_EDIT_JEPA_WORK_ROOT")
    if explicit:
        return Path(explicit)
    work = os.environ.get("WORK")
    if work:
        return Path(work) / "sequence-editing"
    return Path("outputs")


def default_output_dir(config: dict[str, Any]) -> Path:
    experiment = dict(config.get("experiment", {}))
    if experiment.get("output_dir"):
        return Path(str(experiment["output_dir"]))
    name = str(experiment.get("name", "debug"))
    return default_work_root() / "runs" / name
