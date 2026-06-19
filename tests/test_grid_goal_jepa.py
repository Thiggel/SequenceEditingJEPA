import numpy as np
import pytest
import torch

from puzzle_jepa.data.grid_goal_sudoku import (
    collate_grid_goal_sudoku_trajectories,
    sample_grid_goal_sudoku_trajectory,
)
from puzzle_jepa.data.worlds import PuzzleExample, SudokuWorld, WorldAction
from puzzle_jepa.models.grid_goal_jepa import GridTokenGoalJEPA, _temporal_straightening_loss
from puzzle_jepa.planning.grid_goal_planner import (
    changed_cell_raw_euclidean_distance,
    projected_tokenwise_euclidean_distance,
    raw_tokenwise_cosine_distance,
    raw_tokenwise_euclidean_distance,
    raw_tokenwise_squared_euclidean_distance,
    run_beam_mpc,
)


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


def test_beam_mpc_runs_with_raw_oracle_euclidean_goal_distance():
    example = _example()
    state = example.goal.copy()
    state[0, 2] = 0
    state[0, 3] = 0
    tiny = PuzzleExample(state, example.goal)
    model = _small_model()
    result = run_beam_mpc(
        model,
        tiny.state,
        tiny.goal,
        score_mode="oracle_goal_raw_euclidean_distance",
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


@pytest.mark.parametrize(
    "score_mode",
    [
        "oracle_goal_raw_squared_euclidean_distance",
        "predicted_goal_raw_squared_euclidean_distance",
        "oracle_goal_raw_cosine_distance",
        "predicted_goal_raw_cosine_distance",
        "oracle_goal_raw_hybrid_distance",
        "predicted_goal_raw_hybrid_distance",
        "oracle_goal_raw_euclidean_progress",
        "predicted_goal_raw_euclidean_progress",
        "oracle_goal_changed_cell_raw_euclidean_distance",
        "predicted_goal_changed_cell_raw_euclidean_distance",
        "oracle_goal_projected_euclidean_distance",
        "predicted_goal_projected_euclidean_distance",
    ],
)
def test_beam_mpc_runs_with_oracle_metric_probe_scores(score_mode):
    example = _example()
    state = example.goal.copy()
    state[0, 2] = 0
    state[0, 3] = 0
    tiny = PuzzleExample(state, example.goal)
    model = _small_model()
    result = run_beam_mpc(
        model,
        tiny.state,
        tiny.goal,
        score_mode=score_mode,
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


def test_progress_metric_does_not_trigger_zero_distance_early_stop():
    example = _example()
    state = example.goal.copy()
    state[0, 2] = 0
    state[0, 3] = 0
    tiny = PuzzleExample(state, example.goal)
    model = _small_model()
    result = run_beam_mpc(
        model,
        tiny.state,
        tiny.goal,
        score_mode="oracle_goal_raw_euclidean_progress",
        transition_mode="symbolic_reencode",
        beam_width=2,
        beam_depth=2,
        max_steps=1,
        device=torch.device("cpu"),
    )
    assert result.action_evals > 18


def test_raw_tokenwise_euclidean_distance_uses_unprojected_latents():
    a = torch.tensor([[[0.0, 0.0], [3.0, 4.0]]])
    b = torch.zeros_like(a)
    mask = torch.tensor([[True, False]])

    distance = raw_tokenwise_euclidean_distance(a, b, mask)

    assert distance.item() == pytest.approx(0.0)


def test_metric_probe_distance_variants_are_task_agnostic_token_metrics():
    a = torch.tensor([[[0.0, 0.0], [3.0, 4.0]]])
    b = torch.zeros_like(a)
    mask = torch.tensor([[False, True]])
    projector = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        projector.weight.copy_(torch.eye(2))
    action = WorldAction(0, 1, 7)

    assert raw_tokenwise_euclidean_distance(a, b, mask).item() == pytest.approx(5.0)
    assert raw_tokenwise_squared_euclidean_distance(a, b, mask).item() == pytest.approx(25.0)
    assert raw_tokenwise_cosine_distance(a, b, mask).item() == pytest.approx(1.0)
    assert projected_tokenwise_euclidean_distance(a, b, mask, projector).item() == pytest.approx(5.0)
    assert changed_cell_raw_euclidean_distance(a, b, action).item() == pytest.approx(5.0)


def test_invalid_distance_mode_is_rejected():
    with pytest.raises(ValueError, match="distance_mode"):
        _small_model(distance_mode="bad")


def test_temporal_straightening_needs_three_valid_frames():
    states = torch.tensor([[[[0.0, 0.0]], [[1.0, 0.0]]]])
    goal = torch.tensor([[[0.0, 1.0]]])
    masks = torch.ones((1, 2), dtype=torch.bool)
    active_mask = torch.ones((1, 1), dtype=torch.bool)

    loss = _temporal_straightening_loss(states, goal, masks=masks, active_mask=active_mask)

    assert loss.item() == pytest.approx(0.0, abs=1.0e-8)


def test_temporal_straightening_uses_only_fully_valid_triplets():
    states = torch.tensor([[[[0.0, 0.0]], [[1.0, 0.0]], [[2.0, 0.0]]]])
    goal = torch.tensor([[[0.0, 1.0]]])
    masks = torch.tensor([[True, True, False]])
    active_mask = torch.ones((1, 1), dtype=torch.bool)

    loss = _temporal_straightening_loss(states, goal, masks=masks, active_mask=active_mask)

    assert loss.item() == pytest.approx(0.0, abs=1.0e-8)


def test_temporal_straightening_is_goal_independent_curvature():
    states = torch.tensor([[[[0.0, 0.0]], [[1.0, 0.0]], [[2.0, 0.0]]]])
    aligned_goal = torch.tensor([[[3.0, 0.0]]])
    off_axis_goal = torch.tensor([[[0.0, 3.0]]])
    masks = torch.ones((1, 3), dtype=torch.bool)
    active_mask = torch.ones((1, 1), dtype=torch.bool)

    aligned_loss = _temporal_straightening_loss(states, aligned_goal, masks=masks, active_mask=active_mask)
    off_axis_loss = _temporal_straightening_loss(states, off_axis_goal, masks=masks, active_mask=active_mask)

    torch.testing.assert_close(aligned_loss, off_axis_loss)
    assert aligned_loss.item() == pytest.approx(0.0, abs=1.0e-8)


def test_temporal_straightening_penalizes_adjacent_velocity_turns():
    straight = torch.tensor([[[[0.0, 0.0]], [[1.0, 0.0]], [[2.0, 0.0]]]])
    curved = torch.tensor([[[[0.0, 0.0]], [[1.0, 0.0]], [[1.0, 1.0]]]])
    goal = torch.tensor([[[3.0, 0.0]]])
    masks = torch.ones((1, 3), dtype=torch.bool)
    active_mask = torch.ones((1, 1), dtype=torch.bool)

    straight_loss = _temporal_straightening_loss(straight, goal, masks=masks, active_mask=active_mask)
    curved_loss = _temporal_straightening_loss(curved, goal, masks=masks, active_mask=active_mask)

    assert straight_loss.item() == pytest.approx(0.0, abs=1.0e-8)
    assert curved_loss > straight_loss


def test_temporal_straightening_uses_full_grid_latent_not_mean_summary():
    states = torch.tensor(
        [
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 1.0], [1.0, 1.0]],
            ]
        ]
    )
    goal = states[:, -1]
    masks = torch.ones((1, 3), dtype=torch.bool)
    active_mask = torch.ones((1, 2), dtype=torch.bool)

    loss = _temporal_straightening_loss(states, goal, masks=masks, active_mask=active_mask)

    assert loss.item() == pytest.approx(1.0, abs=1.0e-8)
