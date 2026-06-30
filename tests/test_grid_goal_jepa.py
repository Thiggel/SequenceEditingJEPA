import numpy as np
import pytest
import torch
import torch.nn.functional as F

from puzzle_jepa.data.grid_goal_sudoku import (
    collate_grid_goal_sudoku_trajectories,
    sample_grid_goal_sudoku_trajectory,
)
from puzzle_jepa.data.worlds import PuzzleExample, SudokuWorld, WorldAction
from puzzle_jepa.models.grid_goal_jepa import GridTokenGoalJEPA, _affected_token_weights, _temporal_straightening_loss
from puzzle_jepa.planning.grid_goal_planner import (
    affected_context_raw_euclidean_distances,
    changed_cell_raw_euclidean_distance,
    changed_cell_raw_euclidean_distances,
    delta_topk_raw_euclidean_distances,
    projected_tokenwise_euclidean_distance,
    raw_full_board_mse_distance,
    raw_tokenwise_cosine_distance,
    raw_tokenwise_euclidean_distance,
    raw_tokenwise_squared_euclidean_distance,
    run_beam_mpc,
    run_categorical_cem_mpc,
    run_hierarchical_beam_mpc,
    run_hierarchical_cem_mpc,
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
    model = _small_model(goal_conditioning="context")
    context = model.encode_context(batch.context, batch.clue_mask, batch.editable_mask, batch.active_mask)
    predicted_goal = model.predict_goal(context, batch.active_mask)
    changed_context = model.encode_context(batch.goals, batch.clue_mask, batch.editable_mask, batch.active_mask)
    changed_goal = model.predict_goal(changed_context, batch.active_mask)
    assert predicted_goal.shape == (2, 81, 32)
    assert not torch.allclose(predicted_goal, changed_goal)


def test_conditional_goal_predictor_depends_on_current_state_latent():
    batch = _small_batch()
    model = _small_model(goal_conditioning="initial_current")
    context = model.encode_context(batch.context, batch.clue_mask, batch.editable_mask, batch.active_mask)
    initial = model.encode_state(batch.boards[:, 0], context, batch.clue_mask, batch.editable_mask, batch.active_mask)
    current = model.encode_state(batch.boards[:, 1], context, batch.clue_mask, batch.editable_mask, batch.active_mask)

    initial_goal = model.predict_goal(
        context,
        batch.active_mask,
        initial_latents=initial,
        current_latents=initial,
    )
    current_goal = model.predict_goal(
        context,
        batch.active_mask,
        initial_latents=initial,
        current_latents=current,
    )

    assert initial_goal.shape == (2, 81, 32)
    assert not torch.allclose(initial_goal, current_goal)


def test_goal_conditioning_defaults_to_initial_and_current_state():
    model = _small_model()

    assert model.goal_conditioning == "initial_current"


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


@pytest.mark.parametrize(
    "action_conditioning",
    [
        "action_token",
        "affected_marker",
        "local_action_feature",
        "old_local_value",
        "old_local_concat",
        "action_cross_attention",
        "adaln_action",
    ],
)
def test_action_conditioning_variants_predict_board_latents(action_conditioning):
    batch = _small_batch()
    model = _small_model(action_conditioning=action_conditioning)
    context = model.encode_context(batch.context, batch.clue_mask, batch.editable_mask, batch.active_mask)
    state = model.encode_state(batch.boards[:, 0], context, batch.clue_mask, batch.editable_mask, batch.active_mask)

    pred = model.predict_next(state, batch.actions[:, 0], context)

    assert pred.shape == state.shape
    assert torch.isfinite(pred).all()


def test_old_local_value_conditioning_does_not_require_action_token():
    class ExplodingActionToken(torch.nn.Module):
        def forward(self, actions):
            raise AssertionError("old_local_value should not prepend or compute an action token")

    batch = _small_batch()
    model = _small_model(action_conditioning="old_local_value")
    model.action_token = ExplodingActionToken()
    context = model.encode_context(batch.context, batch.clue_mask, batch.editable_mask, batch.active_mask)
    state = model.encode_state(batch.boards[:, 0], context, batch.clue_mask, batch.editable_mask, batch.active_mask)

    pred = model.predict_next(state, batch.actions[:, 0], context)

    assert pred.shape == state.shape


def test_old_local_value_injects_normalized_digit_only_into_edited_cell():
    model = _small_model(action_conditioning="old_local_value")
    state = torch.zeros((2, 81, model.d_model))
    actions = torch.tensor([[0, 1, 5], [2, 3, 7]])

    conditioned = model._condition_state_latents(state, actions, None)
    expected0 = F.layer_norm(model.local_action_digit(torch.tensor([5])), (model.d_model,))[0]
    expected1 = F.layer_norm(model.local_action_digit(torch.tensor([7])), (model.d_model,))[0]

    torch.testing.assert_close(conditioned[0, 1], expected0)
    torch.testing.assert_close(conditioned[1, 21], expected1)
    assert conditioned[0].abs().sum().item() == pytest.approx(conditioned[0, 1].abs().sum().item())
    assert conditioned[1].abs().sum().item() == pytest.approx(conditioned[1, 21].abs().sum().item())


def test_old_local_value_preserves_bfloat16_latent_dtype():
    model = _small_model(action_conditioning="old_local_value")
    state = torch.zeros((1, 81, model.d_model), dtype=torch.bfloat16)
    action = torch.tensor([[0, 1, 5]])

    conditioned = model._condition_state_latents(state, action, None)

    assert conditioned.dtype == torch.bfloat16


def test_old_local_concat_replaces_the_edited_cell_with_concat_projection():
    model = _small_model(action_conditioning="old_local_concat")
    state = torch.randn((1, 81, model.d_model))
    action = torch.tensor([[4, 5, 6]])
    value = model._old_local_action_values(action, state.dtype)
    expected = model.old_local_concat(torch.cat([state[:, 41], value], dim=-1))

    conditioned = model._condition_state_latents(state, action, None)

    torch.testing.assert_close(conditioned[:, 41], expected)
    torch.testing.assert_close(conditioned[:, 40], state[:, 40])


def test_delta_predictor_returns_residual_over_current_latent():
    batch = _small_batch()
    base = _small_model(predict_delta=False)
    delta = _small_model(predict_delta=True)
    delta.load_state_dict(base.state_dict(), strict=False)
    context = base.encode_context(batch.context, batch.clue_mask, batch.editable_mask, batch.active_mask)
    state = base.encode_state(batch.boards[:, 0], context, batch.clue_mask, batch.editable_mask, batch.active_mask)

    base_pred = base.predict_next(state, batch.actions[:, 0], context)
    delta_pred = delta.predict_next(state, batch.actions[:, 0], context)

    torch.testing.assert_close(delta_pred, state + base_pred)


def test_affected_dynamics_weights_emphasize_action_cells():
    actions = torch.tensor([[[0, 1, 5], [2, 2, 6], [0, 1, 7]]])

    weights = _affected_token_weights(actions, token_count=9, rows=3, cols=3, affected_weight=11.0, horizon=2)

    assert weights.shape == (1, 2, 9)
    assert weights[0, 0, 1].item() == pytest.approx(11.0)
    assert weights[0, 0, 8].item() == pytest.approx(11.0)
    assert weights[0, 1, 8].item() == pytest.approx(11.0)
    assert weights[0, 1, 1].item() == pytest.approx(11.0)
    assert weights[0, 0, 0].item() == pytest.approx(1.0)


def test_affected_context_dynamics_weights_mark_sudoku_row_col_and_block():
    actions = torch.tensor([[4, 4, 5]])

    weights = _affected_token_weights(
        actions,
        token_count=81,
        rows=9,
        cols=9,
        affected_weight=8.0,
        context_weight=2.0,
        horizon=1,
    )

    assert weights.shape == (1, 81)
    assert weights[0, 40].item() == pytest.approx(8.0)
    assert weights[0, 36].item() == pytest.approx(2.0)
    assert weights[0, 4].item() == pytest.approx(2.0)
    assert weights[0, 30].item() == pytest.approx(2.0)
    assert weights[0, 0].item() == pytest.approx(1.0)


def test_forward_runs_with_affected_weighting_vicreg_and_ema_target_encoder():
    batch = _small_batch()
    model = _small_model(
        action_conditioning="affected_marker",
        dynamics_weighting="affected",
        regularizer="vicreg",
        use_ema_target_encoder=True,
    )
    before = next(model.target_state_encoder.parameters()).detach().clone()

    output = model(
        batch.boards,
        batch.actions,
        batch.context,
        batch.clue_mask,
        batch.editable_mask,
        batch.active_mask,
        batch.goals,
        masks=batch.masks,
    )
    output.loss.backward()
    with torch.no_grad():
        next(model.state_encoder.parameters()).add_(0.01)
    model.update_ema_target_encoder(decay=0.5)
    after = next(model.target_state_encoder.parameters()).detach()

    assert torch.isfinite(output.loss)
    assert not torch.allclose(before, after)


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
    assert output.dense_future_loss.ndim == 0
    assert output.hierarchy_loss.ndim == 0
    assert output.predicted_goal_latents.shape == (2, 81, 32)
    assert output.predicted_next_latents.shape[2:] == (81, 32)
    assert output.progress_rank_loss.ndim == 0
    assert output.action_rank_loss.ndim == 0
    assert output.policy_prior_loss.ndim == 0
    assert output.temporal_straightening_loss.ndim == 0
    assert output.terminal_corrupt_loss.ndim == 0


def test_dense_future_prediction_and_truncated_rollout_run():
    batch = _small_batch()
    model = _small_model(
        dense_future_weight=0.5,
        rollout_detach_interval=2,
        multi_step_horizons=(1, 4, 8),
    )

    output = model(
        batch.boards,
        batch.actions,
        batch.context,
        batch.clue_mask,
        batch.editable_mask,
        batch.active_mask,
        batch.goals,
        masks=batch.masks,
    )
    output.loss.backward()

    assert torch.isfinite(output.loss)
    assert output.dense_future_loss.item() > 0.0


def test_dense_rollout_all_steps_supervises_every_intermediate_horizon_once():
    batch = _small_batch()
    model = _small_model(
        dense_future_weight=1.0,
        dense_rollout_all_steps=True,
        multi_step_horizons=(4,),
    )
    seen_horizons = []
    predict_calls = 0
    original = model._dynamics_error
    original_predict_next = model.predict_next

    def wrapped_dynamics_error(*args, **kwargs):
        seen_horizons.append(int(kwargs.get("horizon", 1)))
        return original(*args, **kwargs)

    def wrapped_predict_next(*args, **kwargs):
        nonlocal predict_calls
        predict_calls += 1
        return original_predict_next(*args, **kwargs)

    model._dynamics_error = wrapped_dynamics_error
    model.predict_next = wrapped_predict_next

    output = model(
        batch.boards,
        batch.actions,
        batch.context,
        batch.clue_mask,
        batch.editable_mask,
        batch.active_mask,
        batch.goals,
        masks=batch.masks,
    )

    assert torch.isfinite(output.loss)
    assert output.dense_future_loss.item() > 0.0
    assert seen_horizons.count(1) == 1
    assert seen_horizons.count(2) == 1
    assert seen_horizons.count(3) == 1
    assert seen_horizons.count(4) == 1
    assert predict_calls == 5


def test_hierarchy_uses_one_encoder_and_multiple_predictors():
    batch = _small_batch()
    model = _small_model(hierarchy_levels=(2, 4), hierarchy_loss_weight=0.5)
    context = model.encode_context(batch.context, batch.clue_mask, batch.editable_mask, batch.active_mask)
    state = model.encode_state(batch.boards[:, 0], context, batch.clue_mask, batch.editable_mask, batch.active_mask)
    pred_level2 = model.predict_high_level(state, batch.actions[:, :2], context, level=2)
    pred_level4 = model.predict_high_level(state, batch.actions[:, :4], context, level=4)

    assert model.embedder is not None
    assert not hasattr(model, "high_level_state_encoder")
    assert sorted(model.high_level_predictors.keys()) == ["2", "4"]
    assert pred_level2.shape == state.shape
    assert pred_level4.shape == state.shape


def test_old_local_value_hierarchy_grounds_macro_actions_on_affected_cells():
    model = _small_model(action_conditioning="old_local_value", hierarchy_levels=(2,), hierarchy_loss_weight=1.0)
    state = torch.zeros((1, 81, model.d_model))
    actions = torch.tensor([[[0, 1, 5], [2, 3, 7]]])

    conditioned = model._condition_state_latents_with_macro_actions(state, actions)

    assert conditioned[0, 1].abs().sum().item() > 0.0
    assert conditioned[0, 21].abs().sum().item() > 0.0
    inactive = conditioned.clone()
    inactive[:, [1, 21]] = 0.0
    assert inactive.abs().sum().item() == pytest.approx(0.0)


def test_old_local_value_hierarchy_keeps_macro_token_and_adds_grounding_when_actions_are_known():
    batch = _small_batch()
    model = _small_model(action_conditioning="old_local_value", hierarchy_levels=(2,), hierarchy_loss_weight=1.0)
    context = model.encode_context(batch.context, batch.clue_mask, batch.editable_mask, batch.active_mask)
    state = model.encode_state(batch.boards[:, 0], context, batch.clue_mask, batch.editable_mask, batch.active_mask)
    actions = batch.actions[:, :2]
    macro = model.encode_macro_action(actions)

    without_grounding = model.predict_high_level_from_macro(state, macro, context, level=2)
    with_grounding = model.predict_high_level(state, actions, context, level=2)

    assert without_grounding.shape == with_grounding.shape == state.shape
    assert not torch.allclose(without_grounding, with_grounding)


def test_shared_hierarchy_predictor_uses_one_predictor_with_level_conditioning():
    batch = _small_batch()
    model = _small_model(hierarchy_levels=(2, 4), hierarchy_loss_weight=1.0, shared_hierarchy_predictor=True)
    context = model.encode_context(batch.context, batch.clue_mask, batch.editable_mask, batch.active_mask)
    state = model.encode_state(batch.boards[:, 0], context, batch.clue_mask, batch.editable_mask, batch.active_mask)

    pred_level2 = model.predict_high_level(state, batch.actions[:, :2], context, level=2)
    pred_level4 = model.predict_high_level(state, batch.actions[:, :4], context, level=4)

    assert sorted(model.high_level_predictors.keys()) == []
    assert model.shared_high_level_predictor is not None
    assert pred_level2.shape == state.shape
    assert pred_level4.shape == state.shape


def test_hierarchy_loss_runs_with_multiple_predictors():
    batch = _small_batch()
    model = _small_model(hierarchy_levels=(2, 4), hierarchy_loss_weight=1.0)

    output = model(
        batch.boards,
        batch.actions,
        batch.context,
        batch.clue_mask,
        batch.editable_mask,
        batch.active_mask,
        batch.goals,
        masks=batch.masks,
    )

    assert torch.isfinite(output.loss)
    assert output.hierarchy_loss.item() > 0.0


def test_forward_runs_with_conditional_goal_and_listwise_oracle_ranking():
    batch = _small_batch(batch_size=1)
    model = _small_model(
        goal_conditioning="initial_current",
        progress_rank_target="both",
        action_rank_mode="listwise",
        action_rank_target="both",
        listwise_action_rank_max_actions=32,
    )
    positive = batch.actions[:, 0]
    negative = positive.clone()
    negative[:, 2] = (negative[:, 2] % 9) + 1

    output = model(
        batch.boards,
        batch.actions,
        batch.context,
        batch.clue_mask,
        batch.editable_mask,
        batch.active_mask,
        batch.goals,
        masks=batch.masks,
        oracle_mask=batch.oracle_mask,
        action_rank_states=batch.boards[:, 0],
        positive_actions=positive,
        negative_actions=negative,
    )

    assert torch.isfinite(output.loss)
    assert output.predicted_goal_latents.shape == (1, 81, 32)
    assert output.action_rank_loss.item() >= 0.0


def test_policy_prior_scores_primitive_and_macro_actions():
    batch = _small_batch(batch_size=1)
    model = _small_model(hierarchy_levels=(2,), hierarchy_loss_weight=1.0, policy_prior_weight=1.0)
    context = model.encode_context(batch.context, batch.clue_mask, batch.editable_mask, batch.active_mask)
    state = model.encode_state(batch.boards[:, 0], context, batch.clue_mask, batch.editable_mask, batch.active_mask)
    goal = model.predict_goal(context, batch.active_mask, initial_latents=state, current_latents=state)
    primitive_actions = torch.stack([batch.actions[:, 0], batch.actions[:, 1]], dim=1)
    macro_actions = batch.actions[:, :2].unsqueeze(1)

    primitive_logits = model.score_action_prior(state, goal, context, batch.active_mask, primitive_actions)
    macro_logits = model.score_macro_action_prior(state, goal, context, batch.active_mask, macro_actions, level=2)

    assert primitive_logits.shape == (1, 2)
    assert macro_logits.shape == (1, 1)
    assert torch.isfinite(primitive_logits).all()
    assert torch.isfinite(macro_logits).all()


def test_forward_runs_with_policy_prior_loss_on_hierarchy_macro_actions():
    batch = _small_batch(batch_size=1)
    model = _small_model(
        hierarchy_levels=(2,),
        hierarchy_loss_weight=1.0,
        policy_prior_weight=1.0,
        policy_prior_mode="listwise",
        listwise_action_rank_max_actions=32,
    )
    positive = batch.actions[:, 0]
    negative = positive.clone()
    negative[:, 2] = (negative[:, 2] % 9) + 1

    output = model(
        batch.boards,
        batch.actions,
        batch.context,
        batch.clue_mask,
        batch.editable_mask,
        batch.active_mask,
        batch.goals,
        masks=batch.masks,
        oracle_mask=batch.oracle_mask,
        action_rank_states=batch.boards[:, 0],
        positive_actions=positive,
        negative_actions=negative,
    )

    assert torch.isfinite(output.loss)
    assert output.policy_prior_loss.item() > 0.0


def test_mean_pooled_distance_mode_runs():
    batch = _small_batch()
    model = _small_model(distance_mode="mean_pooled")
    context = model.encode_context(batch.context, batch.clue_mask, batch.editable_mask, batch.active_mask)
    state = model.encode_state(batch.boards[:, 0], context, batch.clue_mask, batch.editable_mask, batch.active_mask)
    goal = model.predict_goal(context, batch.active_mask, initial_latents=state, current_latents=state)
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
        "oracle_goal_raw_mse_distance",
        "predicted_goal_raw_mse_distance",
        "oracle_goal_raw_cosine_distance",
        "predicted_goal_raw_cosine_distance",
        "oracle_goal_raw_hybrid_distance",
        "predicted_goal_raw_hybrid_distance",
        "oracle_goal_raw_euclidean_progress",
        "predicted_goal_raw_euclidean_progress",
        "oracle_goal_changed_cell_raw_euclidean_distance",
        "predicted_goal_changed_cell_raw_euclidean_distance",
        "oracle_goal_affected_context_raw_euclidean_distance",
        "predicted_goal_affected_context_raw_euclidean_distance",
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


@pytest.mark.parametrize(
    "score_mode",
    [
        "oracle_goal_raw_squared_euclidean_distance",
        "predicted_goal_raw_hybrid_distance",
        "oracle_goal_raw_euclidean_progress",
        "predicted_goal_changed_cell_raw_euclidean_distance",
        "oracle_goal_affected_context_raw_euclidean_distance",
    ],
)
def test_latent_rollout_beam_mpc_runs_with_metric_probe_scores(score_mode):
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
        transition_mode="latent_rollout",
        beam_width=2,
        beam_depth=2,
        max_steps=2,
        device=torch.device("cpu"),
    )
    assert result.steps <= 2
    assert result.beam_width == 2
    assert result.beam_depth == 2
    assert result.action_evals > 0


def test_categorical_cem_mpc_runs_with_latent_rollout():
    example = _example()
    state = example.goal.copy()
    state[0, 2] = 0
    state[0, 3] = 0
    tiny = PuzzleExample(state, example.goal)
    model = _small_model()
    result = run_categorical_cem_mpc(
        model,
        tiny.state,
        tiny.goal,
        score_mode="oracle_goal_distance",
        transition_mode="latent_rollout",
        beam_width=2,
        beam_depth=2,
        max_steps=1,
        cem_samples=4,
        cem_iters=2,
        cem_elites=2,
        device=torch.device("cpu"),
    )
    assert result.steps <= 1
    assert result.beam_depth == 2
    assert result.action_evals > 0


def test_categorical_cem_mpc_handles_horizon_longer_than_remaining_blanks():
    example = _example()
    state = example.goal.copy()
    state[0, 2] = 0
    tiny = PuzzleExample(state, example.goal)
    model = _small_model()

    result = run_categorical_cem_mpc(
        model,
        tiny.state,
        tiny.goal,
        score_mode="oracle_goal_distance",
        transition_mode="symbolic_reencode",
        beam_width=2,
        beam_depth=4,
        max_steps=1,
        cem_samples=4,
        cem_iters=1,
        cem_elites=1,
        device=torch.device("cpu"),
    )

    assert result.steps <= 1
    assert result.action_evals > 0


def test_hierarchical_cem_mpc_plans_high_level_subgoal_then_low_level_action():
    example = _example()
    state = example.goal.copy()
    state[0, 2] = 0
    state[0, 3] = 0
    tiny = PuzzleExample(state, example.goal)
    model = _small_model(hierarchy_levels=(2, 4), hierarchy_loss_weight=1.0)
    result = run_hierarchical_cem_mpc(
        model,
        tiny.state,
        tiny.goal,
        score_mode="oracle_goal_distance",
        transition_mode="latent_rollout",
        beam_width=2,
        beam_depth=4,
        max_steps=1,
        cem_samples=4,
        cem_iters=1,
        cem_elites=2,
        high_cem_samples=4,
        high_cem_iters=1,
        high_cem_elites=2,
        device=torch.device("cpu"),
    )
    assert result.steps <= 1
    assert result.beam_depth == 4
    assert result.action_evals > 0


def test_hierarchical_cem_mpc_handles_subgoal_horizon_longer_than_remaining_blanks():
    example = _example()
    state = example.goal.copy()
    state[0, 2] = 0
    tiny = PuzzleExample(state, example.goal)
    model = _small_model(hierarchy_levels=(2, 4), hierarchy_loss_weight=1.0)

    result = run_hierarchical_cem_mpc(
        model,
        tiny.state,
        tiny.goal,
        score_mode="oracle_goal_distance",
        transition_mode="symbolic_reencode",
        beam_width=2,
        beam_depth=4,
        max_steps=1,
        cem_samples=4,
        cem_iters=1,
        cem_elites=1,
        high_cem_samples=4,
        high_cem_iters=1,
        high_cem_elites=1,
        device=torch.device("cpu"),
    )

    assert result.steps <= 1
    assert result.action_evals > 0


def test_hierarchical_beam_mpc_runs_high_level_subgoal_then_low_level_beam():
    example = _example()
    state = example.goal.copy()
    state[0, 2] = 0
    state[0, 3] = 0
    tiny = PuzzleExample(state, example.goal)
    model = _small_model(hierarchy_levels=(2, 4), hierarchy_loss_weight=1.0)

    result = run_hierarchical_beam_mpc(
        model,
        tiny.state,
        tiny.goal,
        score_mode="oracle_goal_distance",
        transition_mode="latent_rollout",
        beam_width=2,
        beam_depth=4,
        max_steps=1,
        device=torch.device("cpu"),
    )

    assert result.steps <= 1
    assert result.beam_depth == 4
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
    assert raw_full_board_mse_distance(a, b, mask).item() == pytest.approx(12.5)
    assert raw_tokenwise_cosine_distance(a, b, mask).item() == pytest.approx(1.0)
    assert projected_tokenwise_euclidean_distance(a, b, mask, projector).item() == pytest.approx(5.0)
    assert changed_cell_raw_euclidean_distance(a, b, action).item() == pytest.approx(5.0)


def test_batched_changed_cell_distance_matches_single_item_distance():
    a = torch.tensor([[[0.0, 0.0], [3.0, 4.0]], [[5.0, 12.0], [0.0, 0.0]]])
    b = torch.zeros_like(a)
    actions = [WorldAction(0, 1, 7), WorldAction(0, 0, 3)]

    batched = changed_cell_raw_euclidean_distances(a, b, actions)

    assert batched.tolist() == pytest.approx([5.0, 13.0])
    assert batched[0].item() == pytest.approx(changed_cell_raw_euclidean_distance(a[:1], b[:1], actions[0]).item())


def test_affected_context_distance_uses_local_context_weights():
    a = torch.zeros((1, 81, 1))
    b = torch.zeros_like(a)
    a[0, 40, 0] = 1.0
    a[0, 36, 0] = 1.0
    a[0, 0, 0] = 1.0
    actions = [WorldAction(4, 4, 5)]

    score = affected_context_raw_euclidean_distances(a, b, actions, affected_weight=8.0, context_weight=2.0)

    assert score.item() == pytest.approx((8.0 + 2.0 + 1.0) / 108.0)


def test_delta_topk_distance_scores_largest_predicted_changes():
    previous = torch.zeros((1, 3, 2))
    next_latents = torch.tensor([[[0.1, 0.0], [5.0, 0.0], [0.2, 0.0]]])
    goal = torch.tensor([[[10.0, 0.0], [6.0, 0.0], [10.0, 0.0]]])
    mask = torch.ones((1, 3), dtype=torch.bool)

    top1 = delta_topk_raw_euclidean_distances(next_latents, previous, goal, mask, top_k=1)
    top2 = delta_topk_raw_euclidean_distances(next_latents, previous, goal, mask, top_k=2)

    assert top1.item() == pytest.approx(1.0)
    assert top2.item() == pytest.approx((1.0 + 9.8) / 2.0)


def test_delta_topk_distance_does_not_average_inactive_tokens_when_k_is_large():
    previous = torch.zeros((1, 3, 1))
    next_latents = torch.tensor([[[100.0], [1.0], [50.0]]])
    goal = torch.zeros_like(next_latents)
    mask = torch.tensor([[False, True, False]])

    score = delta_topk_raw_euclidean_distances(next_latents, previous, goal, mask, top_k=3)

    assert score.item() == pytest.approx(1.0)


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
