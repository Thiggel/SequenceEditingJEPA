import math
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir

from puzzle_jepa.data import SudokuWorld, WorldAction, collate_rollouts, sample_oracle_rollout_transition
from puzzle_jepa.eval.grid6_planner_matrix import run_closed_loop, score_sequence
from puzzle_jepa.models import CausalTrajectoryJEPA


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


def _small_model(horizons=(1, 2, 4), max_sequence_steps=6):
    return CausalTrajectoryJEPA(
        vocab_size=10,
        d_model=24,
        board_hidden_size=48,
        action_dim=8,
        encoder_layers=1,
        predictor_layers=1,
        action_chunk_layers=1,
        num_heads=4,
        intermediate_size=48,
        max_sequence_steps=max_sequence_steps,
        max_horizon=4,
        horizons=horizons,
        sigreg_projections=8,
        sigreg_knots=4,
        target_encoder_momentum=0.5,
    )


def _rollout_batch(steps=4, batch_size=2):
    world = SudokuWorld()
    example = world.example_from_strings(SUDOKU_PUZZLE, SUDOKU_SOLUTION)
    rollouts = [
        sample_oracle_rollout_transition(world, example, np.random.default_rng(seed), steps=steps)
        for seed in range(batch_size)
    ]
    return world, example, collate_rollouts(rollouts)


def test_grid6_rollout_loss_backpropagates_for_multiple_horizons():
    _world, _example, batch = _rollout_batch(steps=4, batch_size=2)
    model = _small_model(horizons=(1, 2, 4), max_sequence_steps=5)
    output = model.rollout_loss(
        batch.states,
        batch.actions,
        batch.target_states,
        batch.goals,
        clue_masks=batch.clue_masks,
    )
    assert torch.isfinite(output.loss)
    assert torch.isfinite(output.prediction_loss)
    assert torch.isfinite(output.sigreg_loss)
    assert torch.isfinite(output.goal_energy_loss)
    assert set(output.horizon_losses) == {1, 2, 4}
    assert output.pred_latents.shape == output.target_latents.shape == (2, 4, 24)

    output.loss.backward()
    assert any(param.grad is not None for param in model.encoder.parameters())
    assert any(param.grad is not None for param in model.chunk_encoder.parameters())
    assert any(param.grad is not None for param in model.predictor.parameters())
    assert any(param.grad is not None for param in model.goal_energy_head.parameters())
    assert not any(param.grad is not None for param in model.target_encoder.parameters())


def test_grid6_horizons_longer_than_rollout_are_omitted():
    _world, _example, batch = _rollout_batch(steps=3, batch_size=2)
    model = _small_model(horizons=(1, 2, 4), max_sequence_steps=4)
    output = model.rollout_loss(
        batch.states,
        batch.actions,
        batch.target_states,
        batch.goals,
        clue_masks=batch.clue_masks,
    )
    assert set(output.horizon_losses) == {1, 2}


def test_grid6_encoder_is_causal_over_board_action_history():
    _world, _example, batch = _rollout_batch(steps=4, batch_size=2)
    model = _small_model(horizons=(1,), max_sequence_steps=5)
    model.eval()
    states = torch.cat([batch.states[:, None], batch.target_states], dim=1)
    actions = batch.actions.clone()
    with torch.no_grad():
        baseline = model.encode_context(states, actions, clue_masks=batch.clue_masks)
        changed_states = states.clone()
        changed_actions = actions.clone()
        changed_states[:, 2:] = torch.flip(changed_states[:, 2:], dims=[-1])
        changed_actions[:, 2:, 1:] = torch.tensor([8, 8, 9])
        changed = model.encode_context(changed_states, changed_actions, clue_masks=batch.clue_masks)
    assert torch.allclose(baseline[:, :2], changed[:, :2], atol=1.0e-5)
    assert not torch.allclose(baseline[:, -1], changed[:, -1])


def test_grid6_target_encoder_is_frozen_and_ema_updates():
    _world, _example, batch = _rollout_batch(steps=2, batch_size=2)
    model = _small_model(horizons=(1,), max_sequence_steps=3)
    assert not any(param.requires_grad for param in model.target_encoder.parameters())
    output = model.rollout_loss(
        batch.states,
        batch.actions,
        batch.target_states,
        batch.goals,
        clue_masks=batch.clue_masks,
    )
    output.loss.backward()
    assert not any(param.grad is not None for param in model.target_encoder.parameters())
    with torch.no_grad():
        next(model.encoder.parameters()).add_(1.0)
    before = next(model.target_encoder.parameters()).detach().clone()
    model.update_target_encoder()
    after = next(model.target_encoder.parameters()).detach()
    assert not torch.allclose(before, after)


def test_grid6_direct_multi_horizon_score_sequence_runs():
    world, example, _batch = _rollout_batch(steps=4, batch_size=1)
    model = _small_model(horizons=(1, 2, 4), max_sequence_steps=8)
    clue_mask = world.clue_mask_from_puzzle(example.state)
    actions = []
    board = example.state.copy()
    for row, col in np.argwhere((example.state == 0) & (board != example.goal))[:4]:
        action = WorldAction(int(row), int(col), int(example.goal[row, col]))
        actions.append(action)
        board = world.apply(board, action, clue_mask=clue_mask, allow_overwrite=True)
    score, leaf, terminal = score_sequence(
        model,
        world,
        [example.state.copy()],
        [],
        actions,
        example.goal,
        example.state,
        clue_mask,
        transition_mode="latent_rollout",
        score_mode="latent_goal",
        action_mode="mutable_overwrite",
        aggregate="single",
        prefix_horizons=[4],
        device=torch.device("cpu"),
    )
    assert math.isfinite(score)
    assert leaf.shape == example.state.shape
    assert terminal is False


def test_grid6_closed_loop_planners_smoke():
    world, example, _batch = _rollout_batch(steps=2, batch_size=1)
    model = _small_model(horizons=(1, 2), max_sequence_steps=6)
    for planner in ("beam", "cem", "mcts"):
        result = run_closed_loop(
            model,
            world,
            example,
            planner=planner,
            transition_mode="symbolic_reencode",
            score_mode="latent_goal",
            action_mode="mutable_overwrite",
            aggregate="single",
            horizon=2,
            prefix_horizons=[1, 2],
            max_steps=1,
            beam_width=1,
            branch_size=2,
            cem_candidates=4,
            cem_elites=2,
            cem_iterations=1,
            cem_smoothing=0.7,
            mcts_simulations=4,
            rng=np.random.default_rng(0),
            device=torch.device("cpu"),
        )
        assert result["steps"] in {0, 1}
        assert result["remaining_hamming"] >= 0
        assert result["start_hamming"] >= result["remaining_hamming"] or planner in {"cem", "mcts"}


def test_grid6_hydra_config_composes():
    repo_root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(version_base=None, config_dir=str(repo_root / "configs" / "puzzle")):
        cfg = compose(config_name="grid6_sudoku_trajectory")
    assert cfg.model.d_model == 320
    assert cfg.model.action_dim == 32
    assert cfg.model.horizons == [1]
    assert cfg.model.stabilizer_type == "sigreg"
    assert cfg.training.rollout_steps == 32
