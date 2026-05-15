import torch

from seq_edit_jepa.data.corruptors import build_corruptor
from seq_edit_jepa.data.tasks import build_task
from seq_edit_jepa.eval.full_denoise import full_denoise
from seq_edit_jepa.models import build_model


def test_full_denoise_supports_jepa_predictor_decoder_and_fixed_k():
    task = build_task({"name": "lano", "seed": 6, "seq_len": 32, "max_depth": 3})
    tokenizer = task.build_tokenizer()
    clean = task.sample_batch(2, 32, "eval", "cpu")
    corruptor = build_corruptor({"name": "mask", "num_steps": 4}, tokenizer)
    model = build_model(
        {
            "type": "seq_edit_jepa",
            "hidden_size": 32,
            "intermediate_size": 64,
            "encoder_layers": 1,
            "predictor_layers": 1,
            "num_heads": 4,
            "dropout": 0.0,
        },
        tokenizer,
        num_steps=4,
        max_length=32,
    )
    pred = full_denoise(model, clean, tokenizer, corruptor, commit_k=2, jepa_inference_mode="predictor_decoder")
    assert pred.shape == clean.input_ids.shape
    assert not torch.any(pred[clean.attention_mask.bool()] == tokenizer.mask_token_id)


def test_full_denoise_supports_jepa_policy_head_ablation():
    task = build_task({"name": "lano", "seed": 7, "seq_len": 32, "max_depth": 3})
    tokenizer = task.build_tokenizer()
    clean = task.sample_batch(2, 32, "eval", "cpu")
    corruptor = build_corruptor({"name": "mask", "num_steps": 4}, tokenizer)
    model = build_model(
        {
            "type": "seq_edit_jepa",
            "hidden_size": 32,
            "intermediate_size": 64,
            "encoder_layers": 1,
            "predictor_layers": 1,
            "num_heads": 4,
            "dropout": 0.0,
        },
        tokenizer,
        num_steps=4,
        max_length=32,
    )
    pred = full_denoise(model, clean, tokenizer, corruptor, jepa_inference_mode="policy_head")
    assert pred.shape == clean.input_ids.shape
    assert not torch.any(pred[clean.attention_mask.bool()] == tokenizer.mask_token_id)
