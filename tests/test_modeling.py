import torch

from seq_edit_jepa.data.corruptors import build_corruptor
from seq_edit_jepa.data.tasks import build_task
from seq_edit_jepa.models import build_model
from seq_edit_jepa.models.seq_edit_jepa import SequenceEditJEPA


def _batch_and_tokenizer(corruptor_name="mask"):
    task = build_task({"name": "lano", "seed": 5, "seq_len": 32, "max_depth": 3})
    tokenizer = task.build_tokenizer()
    clean = task.sample_batch(4, 32, "train", "cpu")
    corruptor = build_corruptor({"name": corruptor_name, "num_steps": 4}, tokenizer)
    return tokenizer, corruptor.sample_pair(clean), corruptor.sample_path(clean, 2)


def test_seq_edit_jepa_forward_backward_transformer():
    tokenizer, batch, path = _batch_and_tokenizer("mask")
    model = build_model(
        {
            "type": "seq_edit_jepa",
            "hidden_size": 32,
            "intermediate_size": 64,
            "encoder_layers": 1,
            "predictor_layers": 1,
            "num_heads": 4,
            "dropout": 0.0,
            "lambda_sig": 0.01,
        },
        tokenizer,
        num_steps=4,
        max_length=32,
    )
    assert isinstance(model, SequenceEditJEPA)
    output = model(batch)
    assert torch.isfinite(output.loss)
    assert output.logits.shape == (4, 32, tokenizer.vocab_size)
    rollout = model.rollout_loss(path)
    assert torch.isfinite(rollout)
    (output.loss + 0.1 * rollout).backward()


def test_seq_edit_jepa_action_free_mlp_replacement():
    tokenizer, batch, _ = _batch_and_tokenizer("replacement")
    model = build_model(
        {
            "type": "seq_edit_jepa",
            "hidden_size": 32,
            "intermediate_size": 64,
            "encoder_layers": 1,
            "predictor_layers": 0,
            "num_heads": 4,
            "predictor_type": "mlp",
            "action_conditioned": False,
            "dropout": 0.0,
        },
        tokenizer,
        num_steps=4,
        max_length=32,
    )
    output = model(batch)
    assert torch.isfinite(output.loss)
    output.loss.backward()


def test_seq_edit_jepa_soft_action_deep_decoder_value():
    tokenizer, batch, _ = _batch_and_tokenizer("mask")
    model = build_model(
        {
            "type": "seq_edit_jepa",
            "hidden_size": 32,
            "intermediate_size": 64,
            "encoder_layers": 1,
            "policy_layers": 1,
            "predictor_layers": 1,
            "decoder_layers": 1,
            "num_heads": 4,
            "dropout": 0.0,
            "predictor_action_source": "predicted_soft",
            "lambda_action_op": 1.5,
            "lambda_action_token": 2.0,
            "lambda_val": 0.1,
            "lambda_value_token": 0.1,
            "detach_token_head": True,
        },
        tokenizer,
        num_steps=4,
        max_length=32,
    )
    output = model(batch)
    assert torch.isfinite(output.loss)
    assert output.logits.shape == (4, 32, tokenizer.vocab_size)
    assert "loss/value" in output.components
    output.loss.backward()


def test_denoising_lm_baseline_forward_backward():
    tokenizer, batch, _ = _batch_and_tokenizer("mask")
    model = build_model(
        {
            "type": "denoising_lm",
            "hidden_size": 32,
            "intermediate_size": 64,
            "encoder_layers": 1,
            "num_heads": 4,
            "dropout": 0.0,
        },
        tokenizer,
        num_steps=4,
        max_length=32,
    )
    output = model(batch)
    assert output.op_logits is None
    assert torch.isfinite(output.loss)
    output.loss.backward()


def test_denoising_lm_deep_decoder_forward_backward():
    tokenizer, batch, _ = _batch_and_tokenizer("mask")
    model = build_model(
        {
            "type": "denoising_lm",
            "hidden_size": 32,
            "intermediate_size": 64,
            "encoder_layers": 1,
            "decoder_layers": 1,
            "num_heads": 4,
            "dropout": 0.0,
        },
        tokenizer,
        num_steps=4,
        max_length=32,
    )
    output = model(batch)
    assert torch.isfinite(output.loss)
    output.loss.backward()
