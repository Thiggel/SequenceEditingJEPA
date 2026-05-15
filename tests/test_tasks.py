from seq_edit_jepa.data.tasks import build_task
import torch

from seq_edit_jepa.data.tasks.igsm import trace_is_consistent
from seq_edit_jepa.data.tasks.lano import is_valid_lano, parse_depths


def test_lano_generation_is_valid():
    task = build_task({"name": "lano", "seed": 1, "seq_len": 32, "max_depth": 4})
    tokenizer = task.build_tokenizer()
    batch = task.sample_batch(16, 32, "train", "cpu")
    for row in batch.input_ids:
        tokens = tokenizer.decode(row.tolist()).split()
        assert is_valid_lano(tokens)
        assert len(parse_depths(tokens)) == len(tokens)


def test_igsm_generation_is_consistent():
    task = build_task({"name": "igsm", "seed": 2, "seq_len": 96})
    tokenizer = task.build_tokenizer()
    batch = task.sample_batch(8, 96, "train", "cpu")
    for row in batch.input_ids:
        tokens = tokenizer.decode(row.tolist()).split()
        assert trace_is_consistent(tokens)


def test_igsm_ood_generation_uses_requested_operations():
    task = build_task(
        {
            "name": "igsm",
            "seed": 3,
            "seq_len": 256,
            "modulus": 23,
            "ood_op_values": [20, 21, 22, 23],
        }
    )
    tokenizer = task.build_tokenizer()
    batch = task.sample_batch(8, 256, "eval_ood", "cpu")
    op_counts = [row["op_count"] for row in batch.metadata]
    assert min(op_counts) >= 20
    assert max(op_counts) <= 23
    for row in batch.input_ids:
        tokens = tokenizer.decode(row.tolist()).split()
        assert trace_is_consistent(tokens, modulus=23)


def test_igsm_answer_accuracy_uses_clean_metadata_answer():
    task = build_task({"name": "igsm", "seed": 4, "seq_len": 96})
    batch = task.sample_batch(2, 96, "train", "cpu")
    masked_target = batch.input_ids.clone()
    answer_id = task.tokenizer.convert_tokens_to_ids("answer")
    for row in range(masked_target.shape[0]):
        answer_positions = torch.where(masked_target[row].eq(answer_id))[0]
        if answer_positions.numel():
            masked_target[row, int(answer_positions[0].item()) :] = task.tokenizer.mask_token_id
    metrics = task.evaluate_batch(masked_target, masked_target, batch.attention_mask, batch.metadata)
    assert metrics["igsm/answer_accuracy"] == 0.0
