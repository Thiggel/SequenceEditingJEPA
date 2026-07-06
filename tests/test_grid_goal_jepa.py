import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import puzzle_jepa.models.grid_goal_jepa as grid_goal_jepa_module
import puzzle_jepa.planning.grid_goal_planner as planner_module
from puzzle_jepa.data.grid_goal_sudoku import (
    apply_sudoku_action,
    array_to_action,
    collate_grid_goal_sudoku_trajectories,
    legal_sudoku_actions,
    sample_grid_goal_sudoku_trajectory,
)
from puzzle_jepa.data.worlds import PuzzleExample, SudokuWorld, WorldAction
from puzzle_jepa.models.grid_goal_jepa import (
    GridTokenGoalJEPA,
    _affected_token_weights,
    _sudoku_bad_state_labels,
    _sudoku_one_wrong_corruptions,
    _sudoku_remaining_edit_targets,
    _sudoku_wrong_commitment_targets,
    _temporal_straightening_loss,
)
from puzzle_jepa.planning.grid_goal_planner import (
    ACTION_VOCAB,
    _valid_action_vocab_mask,
    _predict_goal_for_board,
    _sample_macro_action_sequences,
    affected_context_raw_euclidean_distances,
    changed_cell_raw_euclidean_distance,
    changed_cell_raw_euclidean_distances,
    delta_topk_raw_euclidean_distances,
    hierarchical_subgoal_cem,
    latent_distance,
    projected_tokenwise_euclidean_distance,
    planning_horizon,
    raw_full_board_mse_distance,
    raw_tokenwise_cosine_distance,
    raw_tokenwise_euclidean_distance,
    raw_tokenwise_squared_euclidean_distance,
    run_beam_mpc,
    run_categorical_cem_mpc,
    run_hierarchical_beam_mpc,
    run_hierarchical_cem_mpc,
    run_waypoint_hierarchical_cem_mpc,
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


def test_editable_counterfactual_sampling_overwrites_only_non_clue_cells_and_collates():
    rng = np.random.default_rng(11)
    example = _example()
    trajectory = sample_grid_goal_sudoku_trajectory(
        example,
        rng,
        oracle_probability=1.0,
        allow_overwrite=True,
        editable_noise_probability=1.0,
        counterfactual_branches=3,
        counterfactual_depth=2,
        counterfactual_max_pairs=24,
    )

    assert trajectory.counterfactual_states is not None
    assert trajectory.counterfactual_actions is not None
    assert trajectory.counterfactual_next_boards is not None
    assert trajectory.counterfactual_action_sequences is not None
    assert trajectory.counterfactual_future_boards is not None
    assert trajectory.counterfactual_step_mask is not None
    assert trajectory.counterfactual_states.shape[0] <= 24
    assert np.all(trajectory.boards[:, trajectory.clue_mask] == example.state[trajectory.clue_mask][None])
    assert any(
        before[action.row, action.col] != 0 and after[action.row, action.col] != before[action.row, action.col]
        for before, action_values, after in zip(trajectory.boards[:-1], trajectory.actions[:-1], trajectory.boards[1:], strict=True)
        for action in [WorldAction(*[int(x) for x in action_values])]
    )
    for state, action_values, next_board in zip(
        trajectory.counterfactual_states[:5],
        trajectory.counterfactual_action_sequences[:5, 0],
        trajectory.counterfactual_future_boards[:5, 0],
        strict=True,
    ):
        expected = apply_sudoku_action(
            state,
            array_to_action(action_values),
            clue_mask=trajectory.clue_mask,
            allow_conflicts=True,
            allow_overwrite=True,
        )
        assert np.array_equal(next_board, expected)

    batch = collate_grid_goal_sudoku_trajectories([trajectory])
    assert batch.counterfactual_states is not None
    assert batch.counterfactual_actions is not None
    assert batch.counterfactual_next_boards is not None
    assert batch.counterfactual_action_sequences is not None
    assert batch.counterfactual_future_boards is not None
    assert batch.counterfactual_step_mask is not None
    assert bool(batch.counterfactual_mask.any())


def test_counterfactual_depth_targets_include_multistep_future_boards():
    rng = np.random.default_rng(13)
    trajectory = sample_grid_goal_sudoku_trajectory(
        _example(),
        rng,
        oracle_probability=1.0,
        allow_overwrite=False,
        counterfactual_branches=1,
        counterfactual_depth=3,
        counterfactual_max_pairs=512,
    )

    assert trajectory.counterfactual_states is not None
    assert trajectory.counterfactual_next_boards is not None
    changed_cells = np.count_nonzero(
        trajectory.counterfactual_states != trajectory.counterfactual_next_boards,
        axis=(1, 2),
    )

    assert int(changed_cells.max()) >= 2


def test_editable_legal_actions_include_overwrites_but_never_clue_cells():
    example = _example()
    board = example.state.copy()
    clue_mask = board != 0
    row, col = np.argwhere(~clue_mask)[0]
    row = int(row)
    col = int(col)
    board[row, col] = int(example.goal[row, col])

    fill_only = legal_sudoku_actions(board, clue_mask=clue_mask, allow_conflicts=True, allow_overwrite=False)
    editable = legal_sudoku_actions(board, clue_mask=clue_mask, allow_conflicts=True, allow_overwrite=True)

    assert all(board[action.row, action.col] == 0 for action in fill_only)
    assert any(action.row == row and action.col == col and action.value != board[row, col] for action in editable)
    assert not any(clue_mask[action.row, action.col] for action in editable)


def test_editable_planning_keeps_depth_on_full_wrong_board_and_masks_clues():
    example = _example()
    full_wrong = example.goal.copy()
    clue_mask = example.state != 0
    row, col = np.argwhere(~clue_mask)[0]
    row = int(row)
    col = int(col)
    full_wrong[row, col] = (int(example.goal[row, col]) % 9) + 1

    assert planning_horizon(16, full_wrong, allow_overwrite=False) == 0
    assert planning_horizon(16, full_wrong, allow_overwrite=True) == 16

    fill_mask = _valid_action_vocab_mask(full_wrong, clue_mask=clue_mask, allow_overwrite=False)
    edit_mask = _valid_action_vocab_mask(full_wrong, clue_mask=clue_mask, allow_overwrite=True)
    clue_action_ids = [
        index
        for index, action in enumerate(ACTION_VOCAB)
        if bool(clue_mask[action.row, action.col])
    ]
    changed_cell_ids = [
        index
        for index, action in enumerate(ACTION_VOCAB)
        if action.row == row and action.col == col
    ]

    assert not bool(fill_mask.any())
    assert not bool(edit_mask[clue_action_ids].any())
    assert bool(edit_mask[changed_cell_ids].any())


def test_hierarchical_beam_passes_allow_overwrite_to_primitive_tracker(monkeypatch):
    import puzzle_jepa.planning.grid_goal_planner as planner_module

    example = _example()
    board = example.goal.copy()
    clue_mask = example.state != 0
    row, col = np.argwhere(~clue_mask)[0]
    board[int(row), int(col)] = (int(example.goal[int(row), int(col)]) % 9) + 1
    model = _small_model(hierarchy_levels=(4,))
    captured = {}

    def fake_hierarchical_subgoal_beam(
        model,
        board,
        start_latent,
        goal_latent,
        context_latents,
        clue_mask,
        editable_mask,
        active_mask,
        *,
        score_mode,
        level,
        beam_width,
        device,
        allow_overwrite=False,
    ):
        captured["subgoal_allow_overwrite"] = allow_overwrite
        return goal_latent, 0

    def fake_beam_plan_once(
        model,
        board,
        goal,
        context_latents,
        predicted_goal,
        oracle_goal,
        clue_mask,
        editable_mask,
        active_mask,
        *,
        score_mode,
        transition_mode,
        beam_width,
        beam_depth,
        device,
        allow_overwrite=False,
    ):
        captured["primitive_allow_overwrite"] = allow_overwrite
        return None, 0

    monkeypatch.setattr(planner_module, "hierarchical_subgoal_beam", fake_hierarchical_subgoal_beam)
    monkeypatch.setattr(planner_module, "beam_plan_once", fake_beam_plan_once)

    planner_module.run_hierarchical_beam_mpc(
        model,
        board,
        example.goal,
        score_mode="oracle_goal_raw_euclidean_distance",
        transition_mode="latent_rollout",
        beam_width=1,
        beam_depth=4,
        max_steps=1,
        device=torch.device("cpu"),
        allow_overwrite=True,
    )

    assert captured["subgoal_allow_overwrite"] is True
    assert captured["primitive_allow_overwrite"] is True


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


def test_waypoint_loss_uses_only_successful_trajectory_rows():
    batch = _small_batch(batch_size=2)
    model = _small_model(
        waypoint_horizons=(2,),
        waypoint_loss_weight=1.0,
        waypoint_final_weight=0.25,
        goal_mse_weight=0.0,
        goal_nce_weight=0.0,
        progress_rank_weight=0.0,
        action_rank_weight=0.0,
        terminal_corrupt_weight=0.0,
        sigreg_weight=0.0,
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
        oracle_mask=batch.oracle_mask,
    )
    no_oracle = model(
        batch.boards,
        batch.actions,
        batch.context,
        batch.clue_mask,
        batch.editable_mask,
        batch.active_mask,
        batch.goals,
        masks=batch.masks,
        oracle_mask=torch.zeros_like(batch.oracle_mask),
    )

    assert torch.isfinite(output.loss)
    assert output.waypoint_loss.item() > 0.0
    assert output.waypoint_final_loss.item() > 0.0
    assert no_oracle.waypoint_loss.item() == pytest.approx(0.0)
    assert no_oracle.waypoint_final_loss.item() == pytest.approx(0.0)


def test_multi_horizon_waypoint_predictor_outputs_one_latent_per_horizon():
    batch = _small_batch(batch_size=1)
    model = _small_model(
        waypoint_horizons=(4, 8),
        waypoint_loss_weight=1.0,
        goal_mse_weight=0.0,
        goal_nce_weight=0.0,
        progress_rank_weight=0.0,
        action_rank_weight=0.0,
        terminal_corrupt_weight=0.0,
        sigreg_weight=0.0,
    )
    context = model.encode_context(batch.context, batch.clue_mask, batch.editable_mask, batch.active_mask)
    current = model.encode_state(batch.boards[:, 0], context, batch.clue_mask, batch.editable_mask, batch.active_mask)

    waypoints = model.predict_waypoint(context, batch.active_mask, current)

    assert waypoints.shape == (1, len(model.waypoint_horizons), 81, model.d_model)


def test_counterfactual_dynamics_loss_is_reported_and_backprops():
    rng = np.random.default_rng(12)
    example = _example()
    trajectories = [
        sample_grid_goal_sudoku_trajectory(
            example,
            rng,
            oracle_probability=1.0,
            allow_overwrite=True,
            counterfactual_branches=2,
            counterfactual_depth=1,
            counterfactual_max_pairs=12,
        )
        for _ in range(2)
    ]
    batch = collate_grid_goal_sudoku_trajectories(trajectories)
    model = _small_model(
        counterfactual_dynamics_weight=1.0,
        goal_mse_weight=0.0,
        goal_nce_weight=0.0,
        progress_rank_weight=0.0,
        action_rank_weight=0.0,
        terminal_corrupt_weight=0.0,
        sigreg_weight=0.0,
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
        oracle_mask=batch.oracle_mask,
        counterfactual_states=batch.counterfactual_states,
        counterfactual_actions=batch.counterfactual_actions,
        counterfactual_next_boards=batch.counterfactual_next_boards,
        counterfactual_mask=batch.counterfactual_mask,
        counterfactual_action_sequences=batch.counterfactual_action_sequences,
        counterfactual_future_boards=batch.counterfactual_future_boards,
        counterfactual_step_mask=batch.counterfactual_step_mask,
    )

    assert torch.isfinite(output.loss)
    assert output.counterfactual_dynamics_loss.item() > 0.0
    output.loss.backward()
    assert any(param.grad is not None and torch.isfinite(param.grad).all() for param in model.parameters())


def test_counterfactual_sequences_supervise_hierarchy_predictor_without_main_trajectory_chunks():
    rng = np.random.default_rng(14)
    example = _example()
    trajectory = sample_grid_goal_sudoku_trajectory(
        example,
        rng,
        oracle_probability=1.0,
        allow_overwrite=False,
        counterfactual_branches=2,
        counterfactual_depth=3,
        counterfactual_max_pairs=16,
    )
    batch = collate_grid_goal_sudoku_trajectories([trajectory])
    model = _small_model(
        hierarchy_levels=(2,),
        hierarchy_loss_weight=1.0,
        counterfactual_dynamics_weight=1.0,
        dynamics_weighting="affected_context",
        goal_mse_weight=0.0,
        goal_nce_weight=0.0,
        progress_rank_weight=0.0,
        action_rank_weight=0.0,
        terminal_corrupt_weight=0.0,
        sigreg_weight=0.0,
    )
    seen_levels = []
    original = model.predict_high_level

    def record_predict_high_level(state_latents, macro_actions, context_latents, *, level):
        seen_levels.append(int(level))
        return original(state_latents, macro_actions, context_latents, level=level)

    model.predict_high_level = record_predict_high_level
    output = model(
        batch.boards[:, :1],
        batch.actions[:, :1],
        batch.context,
        batch.clue_mask,
        batch.editable_mask,
        batch.active_mask,
        batch.goals,
        masks=batch.masks[:, :1],
        oracle_mask=batch.oracle_mask,
        counterfactual_states=batch.counterfactual_states,
        counterfactual_actions=batch.counterfactual_actions,
        counterfactual_next_boards=batch.counterfactual_next_boards,
        counterfactual_mask=batch.counterfactual_mask,
        counterfactual_action_sequences=batch.counterfactual_action_sequences,
        counterfactual_future_boards=batch.counterfactual_future_boards,
        counterfactual_step_mask=batch.counterfactual_step_mask,
    )

    assert torch.isfinite(output.loss)
    assert 2 in seen_levels
    assert output.counterfactual_dynamics_loss.item() > 0.0


def test_affected_marker_adaln_action_conditioning_changes_prediction_for_same_state():
    batch = _small_batch(batch_size=1)
    model = _small_model(action_conditioning="affected_marker_adaln")
    context = model.encode_context(batch.context, batch.clue_mask, batch.editable_mask, batch.active_mask)
    state = model.encode_state(batch.boards[:, 0], context, batch.clue_mask, batch.editable_mask, batch.active_mask)
    action_a = torch.tensor([[0, 2, 1]], dtype=torch.long)
    action_b = torch.tensor([[8, 8, 9]], dtype=torch.long)

    pred_a = model.predict_next(state, action_a, context)
    pred_b = model.predict_next(state, action_b, context)

    assert pred_a.shape == pred_b.shape == state.shape
    assert not torch.allclose(pred_a, pred_b)


def test_delta_jepa_set_ldad_loss_is_invariant_to_action_order():
    row_logits = torch.full((1, 2, 9), -8.0)
    col_logits = torch.full((1, 2, 9), -8.0)
    digit_logits = torch.full((1, 2, 10), -8.0)
    row_logits[0, 0, 0] = 8.0
    col_logits[0, 0, 1] = 8.0
    digit_logits[0, 0, 2] = 8.0
    row_logits[0, 1, 3] = 8.0
    col_logits[0, 1, 4] = 8.0
    digit_logits[0, 1, 5] = 8.0
    rows_ab = torch.tensor([[0, 3]])
    cols_ab = torch.tensor([[1, 4]])
    digits_ab = torch.tensor([[2, 5]])
    rows_ba = torch.tensor([[3, 0]])
    cols_ba = torch.tensor([[4, 1]])
    digits_ba = torch.tensor([[5, 2]])

    loss_ab = grid_goal_jepa_module._order_invariant_action_sequence_loss(
        row_logits,
        col_logits,
        digit_logits,
        rows_ab,
        cols_ab,
        digits_ab,
    )
    loss_ba = grid_goal_jepa_module._order_invariant_action_sequence_loss(
        row_logits,
        col_logits,
        digit_logits,
        rows_ba,
        cols_ba,
        digits_ba,
    )

    assert loss_ab.item() == pytest.approx(loss_ba.item(), abs=1.0e-7)


def test_goal_conditioning_defaults_to_initial_and_current_state():
    model = _small_model()

    assert model.goal_conditioning == "initial_current"


def test_sudoku_bad_state_labels_detect_wrong_digit_and_duplicates():
    batch = _small_batch(batch_size=1)
    boards = batch.boards[:, :3].clone()

    assert not bool(_sudoku_bad_state_labels(boards, batch.goals).any())

    wrong = boards.clone()
    row, col = torch.nonzero(wrong[0, 1] == 0, as_tuple=False)[0]
    wrong[0, 1, row, col] = (batch.goals[0, row, col] % 9) + 1
    assert bool(_sudoku_bad_state_labels(wrong, batch.goals)[0, 1])

    duplicate = boards.clone()
    duplicate[0, 2, 0, 2] = duplicate[0, 2, 0, 0]
    assert bool(_sudoku_bad_state_labels(duplicate, batch.goals)[0, 2])


def test_verifier_targets_separate_compatibility_from_remaining_work():
    batch = _small_batch(batch_size=1)
    initial = batch.boards[:, 0]
    solved = batch.goals
    wrong = solved.clone()
    wrong[:, 0, 2] = (wrong[:, 0, 2] % 9) + 1
    boards = torch.stack([initial, solved, wrong], dim=1)

    wrong_targets = _sudoku_wrong_commitment_targets(boards, batch.goals, batch.editable_mask)
    remaining_targets = _sudoku_remaining_edit_targets(boards, batch.goals, batch.editable_mask)

    assert wrong_targets.sum(dim=-1).tolist() == [[0.0, 0.0, 1.0]]
    assert remaining_targets.sum(dim=-1)[0, 0].item() == int((initial != solved).sum().item())
    assert remaining_targets.sum(dim=-1).tolist()[0][1:] == [0.0, 1.0]


def test_verifier_corruption_preserves_clues_and_creates_wrong_commitment():
    batch = _small_batch(batch_size=1)
    corrupt = _sudoku_one_wrong_corruptions(batch.boards[:, :2], batch.goals, batch.editable_mask)

    assert torch.equal(corrupt[:, :, batch.clue_mask[0]], batch.boards[:, :2, batch.clue_mask[0]])
    wrong_counts = _sudoku_wrong_commitment_targets(corrupt, batch.goals, batch.editable_mask).sum(dim=-1)
    assert torch.all(wrong_counts >= 1.0)


def test_verifier_losses_are_finite_and_reported_on_encoded_and_predicted_latents():
    from puzzle_jepa.train.grid_goal_sudoku import _sample_rank_actions

    batch = _small_batch(batch_size=2)
    rank_states, positives, negatives = _sample_rank_actions(
        batch.boards,
        batch.goals,
        np.random.default_rng(10),
        masks=batch.masks,
        device=torch.device("cpu"),
        allow_overwrite=True,
    )
    model = _small_model(
        goal_mse_weight=0.0,
        goal_nce_weight=0.0,
        progress_rank_weight=0.0,
        action_rank_weight=0.0,
        terminal_corrupt_weight=0.0,
        compatibility_weight=0.2,
        remaining_weight=0.2,
        verifier_predicted_weight=0.2,
        verifier_corruption_weight=0.5,
        verifier_rank_weight=0.2,
        verifier_rank_mode="pairwise",
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
        oracle_mask=batch.oracle_mask,
        action_rank_states=rank_states,
        positive_actions=positives,
        negative_actions=negatives,
    )

    assert torch.isfinite(output.loss)
    assert output.compatibility_loss.item() > 0.0
    assert output.remaining_loss.item() > 0.0
    assert output.verifier_predicted_loss.item() > 0.0
    assert output.verifier_rank_loss.item() > 0.0


def test_verifier_listwise_rank_loss_runs_over_editable_actions():
    batch = _small_batch(batch_size=1)
    model = _small_model(
        goal_mse_weight=0.0,
        goal_nce_weight=0.0,
        progress_rank_weight=0.0,
        action_rank_weight=0.0,
        terminal_corrupt_weight=0.0,
        compatibility_weight=0.1,
        remaining_weight=0.1,
        verifier_rank_weight=0.1,
        verifier_rank_mode="listwise",
        listwise_action_rank_max_actions=64,
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
        oracle_mask=batch.oracle_mask,
        action_rank_states=batch.boards[:, 0],
        positive_actions=batch.actions[:, 0],
        negative_actions=batch.actions[:, 0],
    )

    assert torch.isfinite(output.verifier_rank_loss)
    assert output.verifier_rank_loss.item() > 0.0


def test_verifier_planner_score_modes_do_not_depend_on_goal_latent():
    batch = _small_batch(batch_size=1)
    model = _small_model(verifier_score_alpha=2.0, verifier_score_beta=0.5)
    context = model.encode_context(batch.context, batch.clue_mask, batch.editable_mask, batch.active_mask)
    latents = model.encode_state(batch.boards[:, 0], context, batch.clue_mask, batch.editable_mask, batch.active_mask)
    fake_goal_a = torch.zeros_like(latents)
    fake_goal_b = torch.randn_like(latents)

    score_a = latent_distance(model, latents, fake_goal_a, fake_goal_b, batch.active_mask, "verifier_energy")
    score_b = latent_distance(model, latents, fake_goal_b, fake_goal_a, batch.active_mask, "verifier_energy")

    assert torch.allclose(score_a, score_b)
    assert torch.allclose(score_a, model.verifier_score(latents, batch.active_mask))


def test_rank_sampler_with_overwrite_repairs_wrong_filled_cells():
    from puzzle_jepa.train.grid_goal_sudoku import _sample_rank_actions

    batch = _small_batch(batch_size=1)
    board = batch.goals.clone()
    board[:, 0, 2] = (board[:, 0, 2] % 9) + 1
    rank_states, positives, negatives = _sample_rank_actions(
        board,
        batch.goals,
        np.random.default_rng(0),
        device=torch.device("cpu"),
        allow_overwrite=True,
    )

    assert torch.equal(rank_states, board)
    assert positives.tolist() == [[0, 2, int(batch.goals[0, 0, 2].item())]]
    assert negatives[0, 0].item() == 0
    assert negatives[0, 1].item() == 2
    assert negatives[0, 2].item() != positives[0, 2].item()


def test_verifier_diagnostics_report_calibration_and_successor_metrics():
    from puzzle_jepa.eval.grid_goal_verifier_diagnostics import run_verifier_diagnostics

    model = _small_model(
        goal_mse_weight=0.0,
        goal_nce_weight=0.0,
        compatibility_weight=1.0,
        remaining_weight=1.0,
    ).eval()

    metrics = run_verifier_diagnostics(
        model,
        [_example()],
        device=torch.device("cpu"),
        seed=0,
        max_examples=1,
        max_actions=32,
    )

    for key in (
        "compatibility_auc",
        "remaining_mae",
        "remaining_spearman",
        "successor_top1",
        "successor_top5",
        "successor_true_score_gap",
        "successor_rollout_score_gap",
    ):
        assert key in metrics
        assert math.isfinite(metrics[key])


@pytest.mark.parametrize("metric_geometry_mode", ["terminal_progress", "hindsight", "contrastive", "iql", "success", "success_iql", "terminal_value"])
def test_metric_geometry_losses_are_reported_and_finite(metric_geometry_mode):
    batch = _small_batch(batch_size=2)
    model = _small_model(
        metric_geometry_mode=metric_geometry_mode,
        metric_geometry_weight=0.5,
        metric_goal_mse_weight=0.25,
        bad_state_weight=0.1,
        metric_bad_margin_weight=0.1,
        goal_mse_weight=0.0,
        goal_nce_weight=0.0,
        progress_rank_weight=0.0,
        action_rank_weight=0.0,
        terminal_corrupt_weight=0.0,
        sigreg_weight=0.0,
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
        oracle_mask=batch.oracle_mask,
    )

    assert torch.isfinite(output.loss)
    assert torch.isfinite(output.metric_geometry_loss)
    assert torch.isfinite(output.metric_goal_mse_loss)
    assert torch.isfinite(output.bad_state_loss)
    assert torch.isfinite(output.bad_margin_loss)
    assert output.metric_geometry_loss.item() >= 0.0
    output.loss.backward()
    grads = [param.grad.detach() for param in model.parameters() if param.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_quasimetric_distance_is_directional_and_self_zero():
    model = _small_model(metric_distance_type="quasimetric")
    source = torch.tensor([[[0.0, 2.0] + [0.0] * 14]], dtype=torch.float32)
    goal = torch.tensor([[[2.0, 3.0] + [0.0] * 14]], dtype=torch.float32)
    mask = torch.ones(1, 1, dtype=torch.bool)

    forward = model._metric_distance_from_projected(source, goal, mask)
    backward = model._metric_distance_from_projected(goal, source, mask)
    same = model._metric_distance_from_projected(source, source, mask)

    assert same.item() == pytest.approx(0.0)
    assert forward.item() == pytest.approx(5.0)
    assert backward.item() == pytest.approx(0.0)


@pytest.mark.parametrize("score_mode", ["success_metric_distance", "terminal_value"])
def test_non_oracle_value_planner_scores_do_not_require_goal_latents(score_mode):
    batch = _small_batch(batch_size=1)
    model = _small_model(metric_geometry_mode="success_iql", metric_geometry_weight=1.0)
    context = model.encode_context(batch.context, batch.clue_mask, batch.editable_mask, batch.active_mask)
    state = model.encode_state(batch.boards[:, 0], context, batch.clue_mask, batch.editable_mask, batch.active_mask)
    bogus_goal = torch.zeros_like(state)

    scores = latent_distance(
        model,
        state,
        predicted_goal=bogus_goal,
        oracle_goal=bogus_goal,
        mask=batch.active_mask,
        score_mode=score_mode,
    )

    assert scores.shape == (1,)
    assert torch.isfinite(scores).all()


def test_metric_projected_planner_distance_uses_metric_heads_not_legacy_projector():
    batch = _small_batch(batch_size=1)
    model = _small_model()
    context = model.encode_context(batch.context, batch.clue_mask, batch.editable_mask, batch.active_mask)
    state = model.encode_state(batch.boards[:, 0], context, batch.clue_mask, batch.editable_mask, batch.active_mask)
    goal = model.encode_state(batch.goals, context, batch.clue_mask, batch.editable_mask, batch.active_mask)
    mask = batch.active_mask

    with torch.no_grad():
        model.distance_projector.weight.zero_()
        model.distance_projector.bias.zero_()
        model.metric_src_projector.weight.zero_()
        for index in range(model.metric_src_projector.weight.shape[0]):
            model.metric_src_projector.weight[index, index] = 1.0
        model.metric_src_projector.bias.fill_(0.0)

    legacy_zero = projected_tokenwise_euclidean_distance(state, goal, mask, model.distance_projector)
    metric_score = latent_distance(
        model,
        state,
        predicted_goal=goal,
        oracle_goal=goal,
        mask=mask,
        score_mode="oracle_goal_projected_euclidean_distance",
    )

    assert legacy_zero.item() == pytest.approx(0.0)
    assert metric_score.item() > 0.0


def test_asymmetric_metric_projection_has_distinct_goal_head():
    model = _small_model(metric_asymmetric_projection=True)

    assert model.metric_goal_projector is not model.metric_src_projector


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


def test_old_local_concat_preserves_bfloat16_latent_dtype():
    model = _small_model(action_conditioning="old_local_concat")
    state = torch.zeros((1, 81, model.d_model), dtype=torch.bfloat16)
    action = torch.tensor([[4, 5, 6]])

    conditioned = model._condition_state_latents(state, action, None)

    assert conditioned.dtype == torch.bfloat16


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
    assert output.goal_distance_field_loss.ndim == 0
    assert output.policy_prior_loss.ndim == 0
    assert output.temporal_straightening_loss.ndim == 0
    assert output.terminal_corrupt_loss.ndim == 0


def test_goal_online_no_stopgrad_uses_online_encoder_for_goal_target():
    batch = _small_batch(batch_size=1)
    model = _small_model(
        goal_conditioning="context",
        goal_target_mode="online_no_stopgrad",
        use_ema_target_encoder=True,
        regularizer="none",
        goal_nce_weight=0.0,
        dense_future_weight=0.0,
        progress_rank_weight=0.0,
        action_rank_mode="none",
        action_rank_weight=0.0,
        temporal_straightening_weight=0.0,
        terminal_corrupt_weight=0.0,
    )

    output = model(
        batch.boards[:, :1],
        batch.actions[:, :1],
        batch.context,
        batch.clue_mask,
        batch.editable_mask,
        batch.active_mask,
        batch.goals,
        masks=batch.masks[:, :1],
    )
    output.loss.backward()

    online_grad = sum(
        0.0 if parameter.grad is None else float(parameter.grad.detach().abs().sum().item())
        for parameter in model.state_encoder.parameters()
    )
    target_grad = sum(
        0.0 if parameter.grad is None else float(parameter.grad.detach().abs().sum().item())
        for parameter in model.target_state_encoder.parameters()
    )
    assert online_grad > 0.0
    assert target_grad == pytest.approx(0.0)


def test_goal_target_stopgrad_does_not_backprop_goal_loss_to_online_state_encoder():
    batch = _small_batch(batch_size=1)
    model = _small_model(
        goal_conditioning="context",
        goal_target_mode="target_stopgrad",
        use_ema_target_encoder=True,
        regularizer="none",
        goal_nce_weight=0.0,
        dense_future_weight=0.0,
        progress_rank_weight=0.0,
        action_rank_mode="none",
        action_rank_weight=0.0,
        temporal_straightening_weight=0.0,
        terminal_corrupt_weight=0.0,
    )

    output = model(
        batch.boards[:, :1],
        batch.actions[:, :1],
        batch.context,
        batch.clue_mask,
        batch.editable_mask,
        batch.active_mask,
        batch.goals,
        masks=batch.masks[:, :1],
    )
    output.loss.backward()

    online_grad = sum(
        0.0 if parameter.grad is None else float(parameter.grad.detach().abs().sum().item())
        for parameter in model.state_encoder.parameters()
    )
    assert online_grad == pytest.approx(0.0)


def test_initial_current_goal_conditioning_backprops_goal_loss_to_state_encoder():
    batch = _small_batch(batch_size=1)
    model = _small_model(
        goal_conditioning="initial_current",
        goal_target_mode="target_stopgrad",
        use_ema_target_encoder=True,
        regularizer="none",
        goal_nce_weight=0.0,
        dense_future_weight=0.0,
        progress_rank_weight=0.0,
        action_rank_mode="none",
        action_rank_weight=0.0,
        temporal_straightening_weight=0.0,
        terminal_corrupt_weight=0.0,
    )

    output = model(
        batch.boards[:, :1],
        batch.actions[:, :1],
        batch.context,
        batch.clue_mask,
        batch.editable_mask,
        batch.active_mask,
        batch.goals,
        masks=batch.masks[:, :1],
    )
    output.loss.backward()

    online_grad = sum(
        0.0 if parameter.grad is None else float(parameter.grad.detach().abs().sum().item())
        for parameter in model.state_encoder.parameters()
    )
    assert online_grad > 0.0


def test_detached_initial_current_goal_conditioning_keeps_goal_loss_out_of_state_encoder():
    batch = _small_batch(batch_size=1)
    model = _small_model(
        goal_conditioning="initial_current",
        goal_conditioning_detach_state=True,
        goal_target_mode="target_stopgrad",
        use_ema_target_encoder=True,
        regularizer="none",
        goal_nce_weight=0.0,
        dense_future_weight=0.0,
        progress_rank_weight=0.0,
        action_rank_mode="none",
        action_rank_weight=0.0,
        temporal_straightening_weight=0.0,
        terminal_corrupt_weight=0.0,
    )

    output = model(
        batch.boards[:, :1],
        batch.actions[:, :1],
        batch.context,
        batch.clue_mask,
        batch.editable_mask,
        batch.active_mask,
        batch.goals,
        masks=batch.masks[:, :1],
    )
    output.loss.backward()

    online_grad = sum(
        0.0 if parameter.grad is None else float(parameter.grad.detach().abs().sum().item())
        for parameter in model.state_encoder.parameters()
    )
    assert online_grad == pytest.approx(0.0)


def test_goal_mse_weight_can_disable_token_goal_mse_objective():
    batch = _small_batch(batch_size=1)
    model = _small_model(
        goal_conditioning="context",
        goal_mse_weight=0.0,
        goal_nce_weight=0.0,
        goal_distance_field_weight=0.0,
        regularizer="none",
        dense_future_weight=0.0,
        progress_rank_weight=0.0,
        action_rank_mode="none",
        action_rank_weight=0.0,
        temporal_straightening_weight=0.0,
        terminal_corrupt_weight=0.0,
    )

    output = model(
        batch.boards[:, :1],
        batch.actions[:, :1],
        batch.context,
        batch.clue_mask,
        batch.editable_mask,
        batch.active_mask,
        batch.goals,
        masks=batch.masks[:, :1],
    )

    assert output.goal_mse_loss.item() == pytest.approx(0.0)
    assert output.loss.item() == pytest.approx(0.0, abs=1.0e-6)


def test_planner_goal_recompute_uses_current_board_for_initial_current_goal_conditioning():
    example = _example()
    model = _small_model(goal_conditioning="initial_current")
    device = torch.device("cpu")
    clue_mask = example.state != 0
    editable_mask = ~clue_mask
    active_mask = np.ones((9, 9), dtype=bool)
    context_t = torch.as_tensor(example.state[None], dtype=torch.long, device=device)
    clue_t = torch.as_tensor(clue_mask[None], dtype=torch.bool, device=device)
    edit_t = torch.as_tensor(editable_mask[None], dtype=torch.bool, device=device)
    active_t = torch.as_tensor(active_mask[None], dtype=torch.bool, device=device)
    context = model.encode_context(context_t, clue_t, edit_t, active_t)
    initial = model.encode_state(context_t, context, clue_t, edit_t, active_t)
    current = example.state.copy()
    first_blank = tuple(np.argwhere(current == 0)[0])
    current[first_blank] = example.goal[first_blank]

    direct_current = model.encode_state(
        torch.as_tensor(current[None], dtype=torch.long, device=device),
        context,
        clue_t,
        edit_t,
        active_t,
    )
    expected = model.predict_goal(context, active_t, initial_latents=initial, current_latents=direct_current)
    recomputed = _predict_goal_for_board(
        model,
        current,
        context,
        initial,
        clue_mask,
        editable_mask,
        active_mask,
        device=device,
    )

    torch.testing.assert_close(recomputed, expected)


def test_goal_distance_field_distillation_loss_is_active_when_requested():
    batch = _small_batch(batch_size=1)
    model = _small_model(
        goal_distance_field_weight=1.0,
        regularizer="none",
        goal_nce_weight=0.0,
        progress_rank_weight=0.0,
        action_rank_mode="none",
        action_rank_weight=0.0,
        temporal_straightening_weight=0.0,
        terminal_corrupt_weight=0.0,
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

    assert torch.isfinite(output.loss)
    assert torch.isfinite(output.goal_distance_field_loss)
    assert output.goal_distance_field_loss.item() >= 0.0


def test_regularizer_can_combine_sigreg_and_vicreg():
    batch = _small_batch()
    model = _small_model(regularizer="both")

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
    assert output.sigreg_loss.item() > 0.0


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


def test_dense_rollout_all_steps_smooth_count_weights_short_horizons_more():
    batch = _small_batch(batch_size=1)
    model = _small_model(
        dense_future_weight=1.0,
        dense_rollout_all_steps=True,
        dense_rollout_weighting="smooth_count",
        multi_step_horizons=(4,),
        sigreg_weight=0.0,
        goal_mse_weight=0.0,
        goal_nce_weight=0.0,
        progress_rank_weight=0.0,
        action_rank_weight=0.0,
        temporal_straightening_weight=0.0,
        terminal_corrupt_weight=0.0,
    )

    def fake_dynamics_error(predicted, *args, **kwargs):
        horizon = int(kwargs.get("horizon", 1))
        return predicted.new_full(predicted.shape[:2], float(horizon))

    model._dynamics_error = fake_dynamics_error

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

    expected_terms = [
        2.0 * (3.0 / math.sqrt(2.0)),
        3.0 * (2.0 / math.sqrt(3.0)),
        4.0 * (1.0 / math.sqrt(4.0)),
    ]
    assert output.dynamics_loss.item() == pytest.approx(1.0)
    assert output.dense_future_loss.item() == pytest.approx(sum(expected_terms) / len(expected_terms))


def test_variable_start_dense_rollout_uses_all_available_starts_once():
    batch = _small_batch(batch_size=1)
    frames = batch.boards.shape[1]
    model = _small_model(
        dense_future_weight=1.0,
        dense_rollout_variable_starts=True,
        multi_step_horizons=(4,),
    )
    seen = []
    predict_calls = 0
    original = model._dynamics_error
    original_predict_next = model.predict_next

    def wrapped_dynamics_error(predicted, *args, **kwargs):
        seen.append((int(kwargs.get("horizon", 1)), predicted.shape[1]))
        return original(predicted, *args, **kwargs)

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
    assert output.dynamics_loss.item() > 0.0
    assert output.dense_future_loss.item() == pytest.approx(0.0)
    assert seen == [(1, frames - 1), (2, frames - 2), (3, frames - 3), (4, frames - 4)]
    assert predict_calls == 4


@pytest.mark.parametrize(
    ("weighting", "gamma"),
    [
        ("uniform", 0.5),
        ("inverse_sqrt", 0.5),
        ("geometric", 0.5),
    ],
)
def test_variable_start_dense_rollout_applies_configured_horizon_weights(weighting, gamma):
    batch = _small_batch(batch_size=1)
    model = _small_model(
        dense_future_weight=1.0,
        dense_rollout_variable_starts=True,
        dense_rollout_weighting=weighting,
        dense_rollout_gamma=gamma,
        multi_step_horizons=(3,),
    )

    def fake_dynamics_error(predicted, *args, **kwargs):
        horizon = int(kwargs.get("horizon", 1))
        return predicted.new_full(predicted.shape[:2], float(horizon))

    model._dynamics_error = fake_dynamics_error
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

    expected_num = 0.0
    expected_den = 0.0
    for horizon in (1, 2, 3):
        start_count = batch.boards.shape[1] - horizon
        valid_count = (batch.masks[:, :start_count] & batch.masks[:, horizon : horizon + start_count]).sum().item()
        if weighting == "uniform":
            weight = 1.0
        elif weighting == "inverse_sqrt":
            weight = horizon**-0.5
        else:
            weight = gamma ** (horizon - 1)
        expected_num += weight * horizon * valid_count
        expected_den += weight * valid_count

    assert output.dynamics_loss.item() == pytest.approx(expected_num / expected_den)


def test_legacy_equivalent_refactor_matches_old_multi_horizon_dense_objective_without_dropout():
    batch = _small_batch(batch_size=1)
    common_kwargs = dict(
        dropout=0.0,
        dense_future_weight=1.0,
        multi_step_horizons=(1, 4, 8),
        sigreg_weight=0.0,
        goal_mse_weight=0.0,
        goal_nce_weight=0.0,
        goal_distance_field_weight=0.0,
        progress_rank_weight=0.0,
        action_rank_weight=0.0,
        temporal_straightening_weight=0.0,
        terminal_corrupt_weight=0.0,
    )
    old_model = _small_model(**common_kwargs)
    refactored_model = _small_model(**common_kwargs, dense_rollout_refactor_mode="legacy_equivalent")
    refactored_model.load_state_dict(old_model.state_dict())
    old_model.eval()
    refactored_model.eval()

    old_output = old_model(
        batch.boards,
        batch.actions,
        batch.context,
        batch.clue_mask,
        batch.editable_mask,
        batch.active_mask,
        batch.goals,
        masks=batch.masks,
    )
    refactored_output = refactored_model(
        batch.boards,
        batch.actions,
        batch.context,
        batch.clue_mask,
        batch.editable_mask,
        batch.active_mask,
        batch.goals,
        masks=batch.masks,
    )

    assert refactored_output.dynamics_loss.item() == pytest.approx(old_output.dynamics_loss.item(), rel=1e-5)
    assert refactored_output.dense_future_loss.item() == pytest.approx(old_output.dense_future_loss.item(), rel=1e-5)
    assert refactored_output.loss.item() == pytest.approx(old_output.loss.item(), rel=1e-5)

    old_output.loss.backward()
    refactored_output.loss.backward()
    for (old_name, old_param), (ref_name, ref_param) in zip(
        old_model.named_parameters(), refactored_model.named_parameters(), strict=True
    ):
        assert old_name == ref_name
        if old_param.grad is None and ref_param.grad is None:
            continue
        assert old_param.grad is not None
        assert ref_param.grad is not None
        assert torch.allclose(old_param.grad, ref_param.grad, atol=1.0e-5, rtol=1.0e-5), old_name


def test_zero_weight_auxiliary_losses_are_not_computed_or_backpropagated(monkeypatch):
    batch = _small_batch(batch_size=1)
    model = _small_model(
        goal_mse_weight=0.0,
        goal_nce_weight=0.0,
        goal_distance_field_weight=0.0,
        progress_rank_weight=0.0,
        action_rank_mode="pairwise",
        action_rank_weight=0.0,
        temporal_straightening_weight=0.0,
        terminal_corrupt_weight=0.0,
        sigreg_weight=0.0,
    )

    def disabled_loss_called(*args, **kwargs):
        raise AssertionError("disabled auxiliary loss should not be computed")

    monkeypatch.setattr(grid_goal_jepa_module, "_distance_field_distillation_loss", disabled_loss_called)
    monkeypatch.setattr(grid_goal_jepa_module, "_temporal_straightening_loss", disabled_loss_called)
    monkeypatch.setattr(model, "_progress_rank_objective", disabled_loss_called)
    monkeypatch.setattr(model, "_regularizer_loss", disabled_loss_called)

    corrupt_goals = torch.full_like(batch.goals, 99)
    output = model(
        batch.boards,
        batch.actions,
        batch.context,
        batch.clue_mask,
        batch.editable_mask,
        batch.active_mask,
        batch.goals,
        masks=batch.masks,
        corrupt_goals=corrupt_goals,
        negative_actions=batch.actions[:, 0],
    )
    assert torch.isfinite(output.loss)
    assert output.goal_mse_loss.item() == pytest.approx(0.0)
    assert output.goal_nce_loss.item() == pytest.approx(0.0)
    assert output.goal_distance_field_loss.item() == pytest.approx(0.0)
    assert output.progress_rank_loss.item() == pytest.approx(0.0)
    assert output.action_rank_loss.item() == pytest.approx(0.0)
    assert output.temporal_straightening_loss.item() == pytest.approx(0.0)
    assert output.terminal_corrupt_loss.item() == pytest.approx(0.0)

    output.loss.backward()
    assert all(param.grad is None or torch.isfinite(param.grad).all() for param in model.parameters())


def test_legacy_count_refactor_uses_horizon_counts_without_endpoint_terms():
    batch = _small_batch(batch_size=1)
    model = _small_model(
        dense_future_weight=1.0,
        dense_rollout_refactor_mode="legacy_count",
        multi_step_horizons=(1, 4, 8),
        sigreg_weight=0.0,
        goal_mse_weight=0.0,
        goal_nce_weight=0.0,
        progress_rank_weight=0.0,
        action_rank_weight=0.0,
        temporal_straightening_weight=0.0,
        terminal_corrupt_weight=0.0,
    )

    def fake_dynamics_error(predicted, *args, **kwargs):
        horizon = int(kwargs.get("horizon", 1))
        return predicted.new_full(predicted.shape[:2], float(horizon))

    model._dynamics_error = fake_dynamics_error
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

    expected_dense_terms = []
    for horizon in range(1, 9):
        count = 2 if horizon <= 4 else 1
        expected_dense_terms.append(float(horizon) * model._dense_horizon_weight(horizon) * count)

    assert output.dynamics_loss.item() == pytest.approx(1.0)
    assert output.dense_future_loss.item() == pytest.approx(sum(expected_dense_terms) / len(expected_dense_terms))


def test_dense_rollout_refactor_mode_rejects_other_dense_rollout_modes():
    with pytest.raises(ValueError, match="dense_rollout_refactor_mode cannot be combined"):
        _small_model(dense_rollout_refactor_mode="legacy_equivalent", dense_rollout_all_steps=True)
    with pytest.raises(ValueError, match="dense_rollout_refactor_mode cannot be combined"):
        _small_model(dense_rollout_refactor_mode="legacy_count", dense_rollout_variable_starts=True)


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


def test_macro_action_bottleneck_projects_to_high_level_predictor_width():
    batch = _small_batch()
    model = _small_model(hierarchy_levels=(2,), hierarchy_loss_weight=1.0, macro_action_dim=6)
    context = model.encode_context(batch.context, batch.clue_mask, batch.editable_mask, batch.active_mask)
    state = model.encode_state(batch.boards[:, 0], context, batch.clue_mask, batch.editable_mask, batch.active_mask)

    macro = model.encode_macro_action(batch.actions[:, :2])
    projected = model.project_macro_action(macro)
    predicted = model.predict_high_level_from_macro(state, macro, context, level=2)

    assert model.macro_action_dim == 6
    assert macro.shape == (2, 6)
    assert projected.shape == (2, model.d_model)
    assert predicted.shape == state.shape
    with pytest.raises(ValueError, match="Macro action must have shape"):
        model.predict_high_level_from_macro(state, torch.zeros((2, model.d_model)), context, level=2)


def test_macro_action_codebook_sequences_are_valid_fill_chunks():
    example = _example()
    board = example.goal.copy()
    board[0, 2] = 0
    board[0, 3] = 0
    rng = np.random.default_rng(3)

    sequences = _sample_macro_action_sequences(board, level=2, samples=8, rng=rng)

    assert sequences.shape == (8, 2, 3)
    for sequence in sequences:
        current = board.copy()
        for row, col, value in sequence:
            assert current[row, col] == 0
            assert 1 <= value <= 9
            current[row, col] = value


def test_hierarchical_subgoal_optimizer_uses_macro_bottleneck_with_and_without_codebook():
    batch = _small_batch(batch_size=1)
    model = _small_model(hierarchy_levels=(2,), hierarchy_loss_weight=1.0, macro_action_dim=5)
    context = model.encode_context(batch.context, batch.clue_mask, batch.editable_mask, batch.active_mask)
    start = model.encode_state(batch.boards[:, 0], context, batch.clue_mask, batch.editable_mask, batch.active_mask)
    goal = model.encode_state(batch.goals, context, batch.clue_mask, batch.editable_mask, batch.active_mask)

    for optimizer, codebook in (("cem", "none"), ("cem", "init"), ("mppi", "none"), ("mppi", "init")):
        subgoal, evals = hierarchical_subgoal_cem(
            model,
            start,
            goal,
            context,
            batch.active_mask[0].numpy(),
            board=batch.boards[0, 0].numpy(),
            score_mode="oracle_goal_raw_euclidean_distance",
            level=2,
            macro_horizon=1,
            samples=3,
            iterations=1,
            elites=1,
            momentum=0.0,
            init_std=1.0,
            optimizer=optimizer,
            codebook=codebook,
            rng=np.random.default_rng(4),
            device=torch.device("cpu"),
        )

        assert subgoal.shape == start.shape
        assert evals == 3


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


def test_waypoint_hierarchical_cem_tracks_waypoint_with_macro_optimizer(monkeypatch):
    example = _example()
    state = example.goal.copy()
    state[0, 2] = 0
    state[0, 3] = 0
    tiny = PuzzleExample(state, example.goal)
    model = _small_model(
        hierarchy_levels=(2, 4),
        hierarchy_loss_weight=1.0,
        macro_action_dim=6,
        waypoint_horizons=(2,),
        waypoint_loss_weight=1.0,
    )
    calls = []

    def record_macro_optimizer(*args, **kwargs):
        calls.append(kwargs)
        return args[2].detach(), 1

    primitive_calls = []

    def record_primitive_tracker(*args, **kwargs):
        primitive_calls.append(kwargs)
        return WorldAction(row=0, col=2, value=int(tiny.goal[0, 2])), 1

    monkeypatch.setattr(planner_module, "hierarchical_subgoal_cem", record_macro_optimizer)
    monkeypatch.setattr(planner_module, "categorical_cem_plan_once", record_primitive_tracker)

    result = run_waypoint_hierarchical_cem_mpc(
        model,
        tiny.state,
        tiny.goal,
        score_mode="oracle_waypoint_raw_euclidean_distance",
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
        high_cem_optimizer="cem",
        device=torch.device("cpu"),
        allow_overwrite=True,
        waypoint_horizon=2,
    )

    assert result.steps <= 1
    assert result.action_evals > 0
    assert calls
    assert {call["optimizer"] for call in calls} == {"cem"}
    assert {call["score_mode"] for call in calls} == {"oracle_goal_raw_euclidean_distance"}
    assert all(call["allow_overwrite"] is True for call in calls)
    assert primitive_calls
    assert primitive_calls[0]["score_mode"] == "oracle_goal_raw_euclidean_distance"
    assert primitive_calls[0]["allow_overwrite"] is True


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


def test_delta_jepa_online_target_mode_does_not_call_target_encoder():
    batch = _small_batch(batch_size=1)
    model = _small_model(
        sigreg_weight=0.0,
        goal_mse_weight=0.0,
        goal_nce_weight=0.0,
        progress_rank_weight=0.0,
        action_rank_weight=0.0,
        temporal_straightening_weight=0.0,
        terminal_corrupt_weight=0.0,
        regularizer="none",
        dynamics_target_mode="online_no_stopgrad",
        goal_target_mode="online_no_stopgrad",
        delta_action_weight=10.0,
        delta_action_horizons=(1, 2),
        use_ema_target_encoder=False,
    )

    def fail_encode_state_target(*args, **kwargs):
        raise AssertionError("Delta-JEPA online target mode should not use the target encoder path")

    model.encode_state_target = fail_encode_state_target
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

    assert output.delta_action_loss.item() > 0.0
    torch.testing.assert_close(
        output.loss.detach(),
        output.dynamics_loss + 10.0 * output.delta_action_loss,
    )


def test_delta_jepa_ldad_decodes_each_configured_horizon():
    batch = _small_batch(batch_size=1)
    model = _small_model(
        sigreg_weight=0.0,
        goal_mse_weight=0.0,
        goal_nce_weight=0.0,
        progress_rank_weight=0.0,
        action_rank_weight=0.0,
        temporal_straightening_weight=0.0,
        terminal_corrupt_weight=0.0,
        regularizer="none",
        dynamics_target_mode="online_no_stopgrad",
        goal_target_mode="online_no_stopgrad",
        delta_action_weight=1.0,
        delta_action_horizons=(1, 3),
    )
    calls = []
    original_forward = model.delta_action_decoder.forward

    def record_forward(delta_tokens, active_mask, steps):
        calls.append((int(steps), tuple(delta_tokens.shape), tuple(active_mask.shape)))
        return original_forward(delta_tokens, active_mask, steps)

    model.delta_action_decoder.forward = record_forward
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

    assert output.delta_action_loss.item() > 0.0
    assert [call[0] for call in calls] == [1, 3]
    assert all(shape[-2] == 81 for _, shape, _ in calls)


def test_goal_online_no_stopgrad_uses_online_goal_encoder_without_ema_target():
    batch = _small_batch(batch_size=1)
    model = _small_model(
        sigreg_weight=0.0,
        goal_mse_weight=1.0,
        goal_nce_weight=0.0,
        progress_rank_weight=0.0,
        action_rank_weight=0.0,
        temporal_straightening_weight=0.0,
        terminal_corrupt_weight=0.0,
        regularizer="none",
        dynamics_target_mode="online_no_stopgrad",
        goal_target_mode="online_no_stopgrad",
        use_ema_target_encoder=False,
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

    assert output.goal_target_latents.requires_grad
    output.loss.backward()
    assert any(param.grad is not None for param in model.state_encoder.parameters())


def test_online_dynamics_target_rejects_ema_target_encoder_noop():
    with pytest.raises(ValueError, match="use_ema_target_encoder"):
        _small_model(dynamics_target_mode="online_no_stopgrad", use_ema_target_encoder=True)


def test_single_hidden_state_represents_board_with_one_token_and_causal_history_predictor():
    batch = _small_batch(batch_size=2)
    model = _small_model(
        latent_representation="single",
        goal_conditioning="context_current",
        sigreg_weight=0.0,
        goal_mse_weight=1.0,
        goal_nce_weight=0.0,
        progress_rank_weight=0.0,
        action_rank_weight=0.0,
        temporal_straightening_weight=0.0,
        terminal_corrupt_weight=0.0,
        regularizer="none",
        dynamics_target_mode="online_no_stopgrad",
        delta_action_weight=1.0,
        delta_action_horizons=(1,),
    )
    context = model.encode_context(batch.context, batch.clue_mask, batch.editable_mask, batch.active_mask)
    first_state = model.encode_state(batch.boards[:, 0], context, batch.clue_mask, batch.editable_mask, batch.active_mask)

    assert model.single_state_cls is not None
    assert model.single_state_cls.shape == (1, 1, 32)
    assert first_state.shape == (2, 1, 32)
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

    assert output.state_latents.shape[:3] == (2, batch.boards.shape[1], 1)
    assert output.predicted_next_latents.shape[:3] == (2, batch.boards.shape[1] - 1, 1)
    assert output.predicted_goal_latents.shape == (2, 1, 32)
    assert output.delta_action_loss.item() > 0.0
    output.loss.backward()
    assert model.single_state_cls.grad is not None
    assert model.single_state_cls.grad.abs().sum() > 0.0


def test_single_hidden_state_planner_distances_accept_board_masks():
    action = WorldAction(3, 4, 5)
    a = torch.tensor([[[1.0, 2.0]], [[2.0, 2.0]]])
    b = torch.zeros_like(a)
    mask = torch.ones((2, 9, 9), dtype=torch.bool)

    raw = raw_tokenwise_euclidean_distance(a, b, mask)
    changed = changed_cell_raw_euclidean_distances(a, b, [action, action])
    affected = affected_context_raw_euclidean_distances(a, b, [action, action])

    expected = a.square().sum(dim=-1).sqrt().squeeze(-1)
    torch.testing.assert_close(raw, expected)
    torch.testing.assert_close(changed, expected)
    torch.testing.assert_close(affected, expected)
