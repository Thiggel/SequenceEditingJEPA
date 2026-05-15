from __future__ import annotations

from typing import Any

from seq_edit_jepa.models.causal_lm import CausalTransformerLM
from seq_edit_jepa.models.config import SequenceEditJEPAConfig
from seq_edit_jepa.models.denoising_lm import DenoisingLM
from seq_edit_jepa.models.seq_edit_jepa import SeqEditJEPAOutput, SequenceEditJEPA

try:
    from transformers import AutoConfig, AutoModel

    AutoConfig.register(SequenceEditJEPAConfig.model_type, SequenceEditJEPAConfig)
    AutoModel.register(SequenceEditJEPAConfig, SequenceEditJEPA)
except Exception:
    pass


def build_model(config: dict[str, Any], tokenizer, num_steps: int, max_length: int):
    model_type = str(config.get("type", "seq_edit_jepa"))
    vocab_size = int(len(tokenizer))
    cfg = dict(config)
    cfg.pop("type", None)
    cfg.setdefault("vocab_size", vocab_size)
    cfg.setdefault("pad_token_id", int(getattr(tokenizer, "pad_token_id")))
    cfg.setdefault("mask_token_id", int(getattr(tokenizer, "mask_token_id")))
    cfg.setdefault("num_steps", int(num_steps))
    cfg.setdefault("max_position_embeddings", int(max_length))
    model_config = SequenceEditJEPAConfig.from_dict(cfg)
    if model_type == "seq_edit_jepa":
        return SequenceEditJEPA(model_config)
    if model_type == "denoising_lm":
        return DenoisingLM(model_config)
    if model_type == "causal_lm":
        return CausalTransformerLM(model_config)
    raise ValueError(f"Unknown model type: {model_type}")


__all__ = ["CausalTransformerLM", "DenoisingLM", "SeqEditJEPAOutput", "SequenceEditJEPA", "SequenceEditJEPAConfig", "build_model"]
