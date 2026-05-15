from __future__ import annotations

from typing import Any

from seq_edit_jepa.data.corruptors.base import Corruptor, tokenizer_special_ids
from seq_edit_jepa.data.corruptors.edit_script_corruptor import EditScript, EditScriptCorruptor
from seq_edit_jepa.data.corruptors.mask_corruptor import MaskCorruptor
from seq_edit_jepa.data.corruptors.replacement_corruptor import ReplacementCorruptor


def build_corruptor(config: dict[str, Any], tokenizer) -> Corruptor:
    name = str(config.get("name", "mask"))
    vocab_size = int(len(tokenizer))
    mask_token_id = int(getattr(tokenizer, "mask_token_id"))
    pad_token_id = int(getattr(tokenizer, "pad_token_id"))
    special_ids = tokenizer_special_ids(tokenizer)
    kwargs = {
        "config": config,
        "vocab_size": vocab_size,
        "mask_token_id": mask_token_id,
        "pad_token_id": pad_token_id,
        "special_token_ids": special_ids,
    }
    if name == "mask":
        return MaskCorruptor(**kwargs)
    if name == "replacement":
        return ReplacementCorruptor(**kwargs)
    raise ValueError(f"Unknown corruptor '{name}'.")


__all__ = ["Corruptor", "EditScript", "EditScriptCorruptor", "MaskCorruptor", "ReplacementCorruptor", "build_corruptor"]
