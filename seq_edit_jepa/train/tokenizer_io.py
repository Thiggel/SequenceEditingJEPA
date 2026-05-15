from __future__ import annotations

from pathlib import Path
import json
from typing import Any

from seq_edit_jepa.data.tokenize import SimpleTokenizer
from seq_edit_jepa.data.word_tokenizer import SimpleWordTokenizer


def load_or_build_tokenizer(task, task_config: dict[str, Any], output_dir: Path, prefer_saved: bool = False):
    """Return the run tokenizer, if present, otherwise build the task tokenizer.

    Checkpoint resume must use the exact tokenizer saved by the original run,
    because compact vocabularies can depend on task setup and source-code state.
    """

    tokenizer_dir = output_dir / "tokenizer"
    tokenizer = None
    if prefer_saved and (tokenizer_dir / "vocab.json").exists():
        tokenizer = _load_saved_tokenizer(task_config, tokenizer_dir)
    if tokenizer is None:
        tokenizer = task.build_tokenizer()
    if hasattr(task, "tokenizer"):
        task.tokenizer = tokenizer
    return tokenizer


def _load_saved_tokenizer(task_config: dict[str, Any], tokenizer_dir: Path):
    task_name = str(task_config.get("name", ""))
    if task_name == "official_igsm":
        from seq_edit_jepa.data.tasks.official_igsm import IgsmCompactTokenizer, import_official_igsm

        _id_gen, _fix_seed, official_tokenizer = import_official_igsm(task_config.get("official_repo_path"))
        return IgsmCompactTokenizer(vocab_file=str(tokenizer_dir / "vocab.json"), official_tokenizer=official_tokenizer)
    if (tokenizer_dir / "tokenizer_config.json").exists():
        if _tokenizer_class(tokenizer_dir) == "SimpleWordTokenizer":
            return SimpleWordTokenizer(vocab_file=str(tokenizer_dir / "vocab.json"))
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(str(tokenizer_dir))
    return SimpleTokenizer.from_pretrained(tokenizer_dir)


def _tokenizer_class(tokenizer_dir: Path) -> str:
    config_path = tokenizer_dir / "tokenizer_config.json"
    if not config_path.exists():
        return ""
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    return str(config.get("tokenizer_class", ""))
