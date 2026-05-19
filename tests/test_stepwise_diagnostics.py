from __future__ import annotations

import torch

from seq_edit_jepa.actions.action_types import Op
from seq_edit_jepa.eval.stepwise_diagnostics import _parse_factorized_mode, proposal_coverage_from_logits


def test_proposal_coverage_counts_position_token_and_pair_hits() -> None:
    op_logits = torch.zeros(1, 4, 6)
    token_logits = torch.zeros(1, 4, 8)
    op_logits[0, 2, int(Op.REPLACE)] = 5.0
    op_logits[0, 1, int(Op.REPLACE)] = 3.0
    token_logits[0, 2, 6] = 7.0
    token_logits[0, 1, 5] = 4.0
    remaining = torch.tensor([[False, True, True, False]])
    target_mask = torch.tensor([[False, False, True, False]])
    clean_ids = torch.tensor([[0, 5, 6, 0]])

    metrics = proposal_coverage_from_logits(
        op_logits,
        token_logits,
        remaining,
        target_mask,
        clean_ids,
        m_values=[1, 2],
        k_values=[1, 5],
    )

    assert metrics["target_positions"] == 1.0
    assert metrics["rows_with_targets"] == 1.0
    assert metrics["position_recall_at_1"] == 1.0
    assert metrics["token_recall_at_1"] == 1.0
    assert metrics["pair_recall_at_1x1"] == 1.0
    assert metrics["position_rank_target_mean"] == 1.0
    assert metrics["token_rank_target_mean"] == 1.0


def test_parse_factorized_mode_aliases() -> None:
    assert _parse_factorized_mode("model_model") == ("model", "model")
    assert _parse_factorized_mode("oracle_pos_dlm_token") == ("oracle", "dlm")
    assert _parse_factorized_mode("model+oracle") == ("model", "oracle")
