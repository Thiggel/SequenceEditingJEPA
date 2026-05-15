import torch

from seq_edit_jepa.actions.apply import apply_fixed_actions
from seq_edit_jepa.data.corruptors import build_corruptor
from seq_edit_jepa.data.tasks import build_task


def _clean_batch():
    task = build_task({"name": "lano", "seed": 4, "seq_len": 32, "max_depth": 3})
    tokenizer = task.build_tokenizer()
    clean = task.sample_batch(batch_size=6, seq_len=32, split="train", device="cpu")
    return task, tokenizer, clean


def test_mask_corruptor_oracle_actions_reconstruct_prev():
    _, tokenizer, clean = _clean_batch()
    corruptor = build_corruptor({"name": "mask", "num_steps": 8}, tokenizer)
    batch = corruptor.sample_pair(clean)
    reconstructed = apply_fixed_actions(batch.input_ids, batch.action_ops, batch.action_tokens)
    assert torch.equal(reconstructed, batch.prev_ids)
    assert torch.equal(batch.prev_ids, clean.input_ids)
    assert batch.target_mask is not None
    assert torch.equal(batch.target_mask, batch.input_ids == tokenizer.mask_token_id)
    assert not torch.any(batch.prev_ids[batch.target_mask] == tokenizer.mask_token_id)
    assert torch.equal(batch.input_ids[~batch.editable_mask], clean.input_ids[~batch.editable_mask])


def test_replacement_corruptor_oracle_actions_reconstruct_prev():
    _, tokenizer, clean = _clean_batch()
    corruptor = build_corruptor({"name": "replacement", "num_steps": 8}, tokenizer)
    batch = corruptor.sample_pair(clean)
    reconstructed = apply_fixed_actions(batch.input_ids, batch.action_ops, batch.action_tokens)
    assert torch.equal(reconstructed, batch.prev_ids)
    assert torch.equal(batch.prev_ids, clean.input_ids)
    assert batch.target_mask is not None
    assert torch.equal(batch.target_mask, batch.input_ids != clean.input_ids)
    assert torch.equal(batch.input_ids[~batch.editable_mask], clean.input_ids[~batch.editable_mask])


def test_sample_path_has_consistent_lengths_and_actions():
    _, tokenizer, clean = _clean_batch()
    corruptor = build_corruptor({"name": "mask", "num_steps": 8}, tokenizer)
    path = corruptor.sample_path(clean, rollout_steps=3)
    assert len(path.states) == 4
    assert len(path.action_ops) == 3
    for index in range(3):
        reconstructed = apply_fixed_actions(path.states[index], path.action_ops[index], path.action_tokens[index])
        assert torch.equal(reconstructed, path.states[index + 1])


def test_stepwise_mask_corruptor_targets_next_state():
    _, tokenizer, clean = _clean_batch()
    corruptor = build_corruptor({"name": "mask", "num_steps": 8, "target_mode": "step"}, tokenizer)
    batch = corruptor.sample_pair(clean)
    reconstructed = apply_fixed_actions(batch.input_ids, batch.action_ops, batch.action_tokens)
    assert torch.equal(reconstructed, batch.prev_ids)
    assert batch.target_mask is not None
    assert torch.all(batch.action_ops[batch.target_mask] == 1)
    assert torch.any(batch.input_ids == tokenizer.mask_token_id)
    assert not torch.any(batch.prev_ids[batch.target_mask] == tokenizer.mask_token_id)
    unrevealed = (batch.input_ids == tokenizer.mask_token_id) & ~batch.target_mask
    assert torch.equal(batch.prev_ids[unrevealed], batch.input_ids[unrevealed])


def test_stepwise_replacement_corruptor_targets_next_state():
    _, tokenizer, clean = _clean_batch()
    corruptor = build_corruptor({"name": "replacement", "num_steps": 8, "target_mode": "step"}, tokenizer)
    batch = corruptor.sample_pair(clean)
    reconstructed = apply_fixed_actions(batch.input_ids, batch.action_ops, batch.action_tokens)
    assert torch.equal(reconstructed, batch.prev_ids)
    assert batch.target_mask is not None
    assert torch.equal(batch.prev_ids[batch.target_mask], clean.input_ids[batch.target_mask])
    unrevealed = (batch.input_ids != clean.input_ids) & ~batch.target_mask
    assert torch.equal(batch.prev_ids[unrevealed], batch.input_ids[unrevealed])
