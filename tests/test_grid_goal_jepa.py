import numpy as np
import pytest
import torch

from puzzle_jepa.data.grid_goal_sudoku import (
    collate_grid_goal_sudoku_trajectories,
    sample_grid_goal_sudoku_trajectory,
)
from puzzle_jepa.data.worlds import PuzzleExample, SudokuWorld, WorldAction
from puzzle_jepa.models.grid_goal_jepa import GridTokenGoalJEPA
from puzzle_jepa.planning.grid_goal_planner import run_beam_mpc


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


def _example() -> PuzzleExample:
    world = SudokuWorld()
    return world.example_from_strings(SUDOKU_PUZZLE, SUDOKU_SOLUTION)


def _small_model(**kwargs) -> GridTokenGoalJEPA:
    defaults = dict(
        d_model=32,
        distance_dim=16,
        context_layers=1,
        state_layers=1,
        predictor_layers=1,
        goal_layers=1,
        num_heads=4,
        dropout=0.0,
        multi_step_horizons=(1, 4),
    )
    defaults.update(kwargs)
    return GridTokenGoalJEPA(**defaults)


def _small_batch(batch_size=2):
    rng = np.random.default_rng(0)
    example = _example()
    trajectories = [
        sample_grid_goal_sudoku_trajectory(example, rng, oracle_probability=1.0)
        for _ in range(batch_size)
    ]
    return collate_grid_goal_sudoku_trajectories(trajectories)


def test_grid_goal_trajectory_is_fill_only_and_context_conditioned():
    rng = np.random.default_rng(1)
    example = _example()
    trajectory = sample_grid_goal_sudoku_trajectory(example, rng, oracle_probability=1.0)
    assert trajectory.boards.shape[0] == int(np.count_nonzero(example.state == 0)) + 1
    assert np.array_equal(trajectory.context, example.state)
    assert np.array_equal(trajectory.clue_mask, example.state != 0)
    for before, action_values, after in zip(trajectory.boards[:-1], trajectory.actions[:-1], trajectory.boards[1:], strict=True):
        action = WorldAction(*[int(x) for x in action_values])
        assert before[action.row, action.col] == 0
        assert after[action.row, action.col] == int(example.goal[action.row, action.col])
        assert np.count_nonzero(before != after) == 1


def test_model_uses_full_grid_latent_without_cls_vector():
    batch = _small_batch()
    model = _small_model()
    context = model.encode_context(batch.context, batch.clue_mask, batch.editable_mask, batch.active_mask)
    state = model.encode_state(batch.boards[:, 0], context, batch.clue_mask, batch.editable_mask, batch.active_mask)
    assert context.shape == (2, 81, 32)
    assert state.shape == (2, 81, 32)


def test_goal_predictor_depends_on_context_and_outputs_board_tokens():
    batch = _small_batch()
    model = _small_model()
    context = model.encode_context(batch.context, batch.clue_mask, batch.editable_mask, batch.active_mask)
    predicted_goal = model.predict_goal(context, batch.active_mask)
    changed_context = model.encode_context(batch.goals, batch.clue_mask, batch.editable_mask, batch.active_mask)
    changed_goal = model.predict_goal(changed_context, batch.active_mask)
    assert predicted_goal.shape == (2, 81, 32)
    assert not torch.allclose(predicted_goal, changed_goal)


def test_markov_predictor_accepts_current_board_latent_and_one_action_token():
    batch = _small_batch()
    model = _small_model()
    context = model.encode_context(batch.context, batch.clue_mask, batch.editable_mask, batch.active_mask)
    state = model.encode_state(batch.boards[:, 0], context, batch.clue_mask, batch.editable_mask, batch.active_mask)
    pred = model.predict_next(state, batch.actions[:, 0], context)
    assert pred.shape == state.shape
    changed_action = batch.actions[:, 0].clone()
    changed_action[:, 2] = (changed_action[:, 2] % 9) + 1
    changed_pred = model.predict_next(state, changed_action, context)
    assert not torch.allclose(pred, changed_pred)


def test_forward_computes_all_requested_losses():
    batch = _small_batch()
    model = _small_model()
    negative_actions = batch.actions[:, 0].clone()
    negative_actions[:, 2] = (negative_actions[:, 2] % 9) + 1
    corrupt_goals = batch.goals.clone()
    corrupt_goals[:, 0, 0] = (corrupt_goals[:, 0, 0] % 9) + 1
    output = model(
        batch.boards,
        batch.actions,
        batch.context,
        batch.clue_mask,
        batch.editable_mask,
        batch.active_mask,
        batch.goals,
        masks=batch.masks,
        negative_actions=negative_actions,
        corrupt_goals=corrupt_goals,
    )
    assert torch.isfinite(output.loss)
    assert output.predicted_goal_latents.shape == (2, 81, 32)
    assert output.predicted_next_latents.shape[2:] == (81, 32)
    assert output.progress_rank_loss.ndim == 0
    assert output.action_rank_loss.ndim == 0
    assert output.temporal_straightening_loss.ndim == 0
    assert output.terminal_corrupt_loss.ndim == 0


def test_mean_pooled_distance_mode_runs():
    batch = _small_batch()
    model = _small_model(distance_mode="mean_pooled")
    context = model.encode_context(batch.context, batch.clue_mask, batch.editable_mask, batch.active_mask)
    state = model.encode_state(batch.boards[:, 0], context, batch.clue_mask, batch.editable_mask, batch.active_mask)
    goal = model.predict_goal(context, batch.active_mask)
    distance = model.distance(state, goal, batch.active_mask)
    assert distance.shape == (2,)
    assert torch.isfinite(distance).all()


def test_beam_mpc_runs_with_oracle_goal_distance():
    example = _example()
    # Keep the planner smoke test tiny by using an almost solved board.
    state = example.goal.copy()
    state[0, 2] = 0
    state[0, 3] = 0
    tiny = PuzzleExample(state, example.goal)
    model = _small_model()
    result = run_beam_mpc(
        model,
        tiny.state,
        tiny.goal,
        score_mode="oracle_goal_distance",
        transition_mode="symbolic_reencode",
        beam_width=2,
        beam_depth=2,
        max_steps=2,
        device=torch.device("cpu"),
    )
    assert result.steps <= 2
    assert result.beam_width == 2
    assert result.beam_depth == 2
    assert result.action_evals > 0


def test_invalid_distance_mode_is_rejected():
    with pytest.raises(ValueError, match="distance_mode"):
        _small_model(distance_mode="bad")
