import numpy as np
import torch
from hydra import compose, initialize_config_dir

from pathlib import Path

from puzzle_jepa.data import SudokuWorld, collate_rollouts, sample_oracle_rollout_transition
from puzzle_jepa.models import SigRegActionJEPA, sigreg_loss


SUDOKU_PUZZLE = (
    "530070000"
    "600195000"
    "098000060"
    "800060003"
    "400803001"
    "700020006"
    "060000280"
    "000419005"
    "000080079"
)
SUDOKU_SOLUTION = (
    "534678912"
    "672195348"
    "198342567"
    "859761423"
    "426853791"
    "713924856"
    "961537284"
    "287419635"
    "345286179"
)


def _rollout_batch(steps=3, batch_size=2):
    world = SudokuWorld()
    example = world.example_from_strings(SUDOKU_PUZZLE, SUDOKU_SOLUTION)
    rollouts = [
        sample_oracle_rollout_transition(world, example, np.random.default_rng(seed), steps=steps)
        for seed in range(batch_size)
    ]
    return world, collate_rollouts(rollouts)


def test_sigreg_penalizes_degenerate_embeddings_more_than_gaussian():
    torch.manual_seed(0)
    degenerate = torch.zeros(256, 16)
    gaussian = torch.randn(256, 16)
    degenerate_loss = sigreg_loss(degenerate, projections=64, knots=16)
    gaussian_loss = sigreg_loss(gaussian, projections=64, knots=16)
    assert gaussian_loss < degenerate_loss


def test_grid5_variants_produce_single_latent_and_backpropagate():
    world, batch = _rollout_batch(steps=3, batch_size=2)
    for encoder_type in ("mlp", "cls_transformer"):
        for predictor_type in ("mlp", "ar_transformer"):
            for predict_delta in (False, True):
                model = SigRegActionJEPA(
                    vocab_size=world.vocab_size,
                    latent_size=32,
                    encoder_type=encoder_type,
                    predictor_type=predictor_type,
                    predict_delta=predict_delta,
                    encoder_hidden_size=64,
                    predictor_hidden_size=64,
                    transformer_layers=1,
                    predictor_layers=1,
                    num_heads=4,
                    max_rollout_steps=3,
                    sigreg_projections=8,
                    sigreg_knots=4,
                )
                output = model.rollout_loss(batch.states, batch.actions, batch.target_states, batch.goals)
                assert torch.isfinite(output.loss)
                assert output.pred_latents.shape == output.target_latents.shape == (2, 3, 32)
                output.loss.backward()
                assert any(param.grad is not None for param in model.encoder.parameters())
                assert any(param.grad is not None for param in model.goal_energy_head.parameters())


def test_ar_transformer_predictor_is_causal_over_training_sequence():
    torch.manual_seed(0)
    model = SigRegActionJEPA(
        vocab_size=10,
        latent_size=16,
        encoder_type="mlp",
        predictor_type="ar_transformer",
        encoder_hidden_size=32,
        predictor_hidden_size=32,
        predictor_layers=1,
        num_heads=4,
        max_rollout_steps=4,
        sigreg_projections=8,
        sigreg_knots=4,
    )
    model.eval()
    latents = torch.randn(2, 4, 16)
    actions = torch.tensor(
        [
            [[0, 0, 0, 1], [0, 0, 1, 2], [0, 0, 2, 3], [0, 0, 3, 4]],
            [[0, 1, 0, 1], [0, 1, 1, 2], [0, 1, 2, 3], [0, 1, 3, 4]],
        ],
        dtype=torch.long,
    )
    baseline = model.predict_sequence(latents, actions)
    changed_latents = latents.clone()
    changed_actions = actions.clone()
    changed_latents[:, 2:] = torch.randn_like(changed_latents[:, 2:]) * 100.0
    changed_actions[:, 2:, 1:] = torch.tensor([8, 8, 9])
    changed = model.predict_sequence(changed_latents, changed_actions)
    assert torch.allclose(baseline[:, :2], changed[:, :2], atol=1.0e-5)
    assert not torch.allclose(baseline[:, 3], changed[:, 3])


def test_grid5_hydra_config_composes():
    repo_root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(version_base=None, config_dir=str(repo_root / "configs" / "puzzle")):
        cfg = compose(config_name="grid5_sudoku_sigreg")
    assert cfg.model.sigreg_weight == 1.0
    assert cfg.model.action_size == 16
    assert cfg.training.goal_energy_weight == 1.0
