from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig


class SequenceEditJEPAConfig(PretrainedConfig):
    """Hugging Face config for sequence-edit JEPA models."""

    model_type = "seq_edit_jepa"

    def __init__(
        self,
        vocab_size: int = 32000,
        pad_token_id: int = 0,
        mask_token_id: int = 2,
        max_length: int = 128,
        max_position_embeddings: int | None = None,
        num_steps: int = 16,
        num_ops: int = 3,
        hidden_size: int = 256,
        intermediate_size: int = 1024,
        encoder_layers: int = 4,
        policy_layers: int = 0,
        predictor_layers: int = 8,
        decoder_layers: int = 0,
        num_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        qk_norm: bool = True,
        timestep_embedding_size: int | None = None,
        predictor_type: str = "transformer",
        action_conditioned: bool = True,
        action_conditioning: str = "concat",
        ema_tau: float = 0.995,
        lambda_act: float = 1.0,
        lambda_action_op: float = 1.0,
        lambda_action_token: float = 1.0,
        lambda_dyn: float = 1.0,
        lambda_tok: float = 0.5,
        lambda_sig: float = 0.05,
        lambda_roll: float = 0.0,
        lambda_val: float = 0.0,
        lambda_value_token: float = 0.25,
        detach_token_head: bool = False,
        detach_value_head: bool = True,
        predictor_action_source: str = "gold",
        predictor_action_temperature: float = 1.0,
        initializer_range: float = 0.02,
        **kwargs: Any,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = int(vocab_size)
        self.mask_token_id = int(mask_token_id)
        self.max_position_embeddings = int(max_position_embeddings or max_length)
        self.num_steps = int(num_steps)
        self.num_ops = int(num_ops)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.encoder_layers = int(encoder_layers)
        self.policy_layers = int(policy_layers)
        self.predictor_layers = int(predictor_layers)
        self.decoder_layers = int(decoder_layers)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.attention_dropout = float(attention_dropout)
        self.norm_eps = float(norm_eps)
        self.rope_theta = float(rope_theta)
        self.qk_norm = bool(qk_norm)
        self.timestep_embedding_size = int(timestep_embedding_size or hidden_size)
        self.predictor_type = str(predictor_type)
        self.action_conditioned = bool(action_conditioned)
        self.action_conditioning = str(action_conditioning)
        self.ema_tau = float(ema_tau)
        self.lambda_act = float(lambda_act)
        self.lambda_action_op = float(lambda_action_op)
        self.lambda_action_token = float(lambda_action_token)
        self.lambda_dyn = float(lambda_dyn)
        self.lambda_tok = float(lambda_tok)
        self.lambda_sig = float(lambda_sig)
        self.lambda_roll = float(lambda_roll)
        self.lambda_val = float(lambda_val)
        self.lambda_value_token = float(lambda_value_token)
        self.detach_token_head = bool(detach_token_head)
        self.detach_value_head = bool(detach_value_head)
        self.predictor_action_source = str(predictor_action_source)
        self.predictor_action_temperature = float(predictor_action_temperature)
        self.initializer_range = float(initializer_range)
