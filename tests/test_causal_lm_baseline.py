from seq_edit_jepa.data.tasks import build_task
from seq_edit_jepa.models import CausalTransformerLM, build_model
from seq_edit_jepa.train.lm_data import CleanLMDataCollator, CleanLMIterableDataset
from seq_edit_jepa.train.lm_evaluate import evaluate_causal_lm


def test_causal_lm_forward_and_eval():
    task = build_task({"name": "lano", "seed": 8, "seq_len": 32, "max_depth": 3})
    tokenizer = task.build_tokenizer()
    model = build_model(
        {
            "type": "causal_lm",
            "hidden_size": 32,
            "intermediate_size": 64,
            "encoder_layers": 2,
            "num_heads": 4,
            "dropout": 0.0,
        },
        tokenizer,
        num_steps=1,
        max_length=32,
    )
    assert isinstance(model, CausalTransformerLM)
    dataset = CleanLMIterableDataset(task, 32, "train", length=2)
    batch = CleanLMDataCollator()(list(dataset))
    output = model(**batch)
    assert output.loss is not None
    output.loss.backward()
    metrics = evaluate_causal_lm(model, task, tokenizer, 32, {"batches": 1, "batch_size": 2}, next(model.parameters()).device)
    assert "eval/loss" in metrics
    assert "eval/gen_task/token_accuracy" in metrics
