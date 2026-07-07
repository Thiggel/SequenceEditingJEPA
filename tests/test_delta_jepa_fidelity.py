import numpy as np
import pytest
import torch

from puzzle_jepa.data.grid_goal_sudoku import collate_grid_goal_sudoku_trajectories, sample_grid_goal_sudoku_trajectory
from puzzle_jepa.data.worlds import PuzzleExample, SudokuWorld
from puzzle_jepa.models.grid_goal_jepa import GridTokenGoalJEPA
from puzzle_jepa.planning.grid_goal_planner import ACTION_VOCAB, _prepare_goal_latents, _score_cem_sequences


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
    return SudokuWorld().example_from_strings(SUDOKU_PUZZLE, SUDOKU_SOLUTION)


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
        sigreg_weight=0.0,
        goal_mse_weight=0.0,
        goal_nce_weight=0.0,
        goal_distance_field_weight=0.0,
        progress_rank_weight=0.0,
        action_rank_weight=0.0,
        action_rank_mode="none",
        temporal_straightening_weight=0.0,
        terminal_corrupt_weight=0.0,
        policy_prior_weight=0.0,
        regularizer="none",
        multi_step_horizons=(1,),
    )
    defaults.update(kwargs)
    return GridTokenGoalJEPA(**defaults)


def _small_batch(batch_size: int = 1):
    rng = np.random.default_rng(123)
    example = _example()
    trajectories = [
        sample_grid_goal_sudoku_trajectory(example, rng, oracle_probability=1.0)
        for _ in range(batch_size)
    ]
    return collate_grid_goal_sudoku_trajectories(trajectories)


def _grad_sum(tensor: torch.Tensor | None) -> float:
    if tensor is None:
        return 0.0
    return float(tensor.detach().abs().sum().item())


def test_delta_jepa_online_dynamics_mse_backprops_through_next_state_target():
    batch = _small_batch(batch_size=1)
    model = _small_model(
        dynamics_target_mode="online_no_stopgrad",
        goal_target_mode="online_no_stopgrad",
        delta_action_weight=0.0,
    )

    output = model(
        batch.boards[:, :2],
        batch.actions[:, :2],
        batch.context,
        batch.clue_mask,
        batch.editable_mask,
        batch.active_mask,
        batch.goals,
        masks=batch.masks[:, :2],
    )
    output.state_latents.retain_grad()
    output.loss.backward()

    assert _grad_sum(output.state_latents.grad[:, 1]) > 0.0


def test_stopgrad_dynamics_mse_blocks_gradients_to_next_state_target_latent():
    batch = _small_batch(batch_size=1)
    model = _small_model(
        dynamics_target_mode="target_stopgrad",
        goal_target_mode="target_stopgrad",
        delta_action_weight=0.0,
        use_ema_target_encoder=True,
    )

    output = model(
        batch.boards[:, :2],
        batch.actions[:, :2],
        batch.context,
        batch.clue_mask,
        batch.editable_mask,
        batch.active_mask,
        batch.goals,
        masks=batch.masks[:, :2],
    )
    output.state_latents.retain_grad()
    output.loss.backward()

    assert _grad_sum(output.state_latents.grad[:, 1]) == pytest.approx(0.0)
    assert all(parameter.grad is None for parameter in model.target_state_encoder.parameters())


def test_ldad_loss_backprops_to_both_displacement_endpoints():
    model = _small_model(delta_action_weight=1.0, delta_action_horizons=(1,))
    state_latents = torch.randn(1, 3, 81, model.d_model, requires_grad=True)
    actions = torch.tensor([[[0, 0, 1], [0, 1, 2], [0, 2, 3]]])
    masks = torch.ones((1, 3), dtype=torch.bool)
    active_mask = torch.ones((1, 9, 9), dtype=torch.bool)

    loss = model._delta_action_objective(state_latents, actions, masks, active_mask)
    loss.backward()

    assert _grad_sum(state_latents.grad[:, 0]) > 0.0
    assert _grad_sum(state_latents.grad[:, 1]) > 0.0


def test_ldad_predicted_delta_backprops_through_transition_predictor():
    batch = _small_batch(batch_size=1)
    model = _small_model(delta_action_weight=1.0, delta_action_horizons=(1,))
    model.zero_grad(set_to_none=True)
    batch_size, frames = batch.boards.shape[:2]
    context = model.encode_context(batch.context, batch.clue_mask, batch.editable_mask, batch.active_mask)
    flat_context = context[:, None].expand(batch_size, frames, *context.shape[1:]).reshape(
        batch_size * frames,
        context.shape[1],
        model.d_model,
    )
    flat_clue = batch.clue_mask[:, None].expand(batch_size, frames, 9, 9).reshape(batch_size * frames, 9, 9)
    flat_edit = batch.editable_mask[:, None].expand(batch_size, frames, 9, 9).reshape(batch_size * frames, 9, 9)
    flat_active = batch.active_mask[:, None].expand(batch_size, frames, 9, 9).reshape(batch_size * frames, 9, 9)
    state_latents = model.encode_state(
        batch.boards.reshape(batch_size * frames, 9, 9),
        flat_context,
        flat_clue,
        flat_edit,
        flat_active,
    ).reshape(batch_size, frames, 81, model.d_model)

    loss = model._delta_action_objective(state_latents, batch.actions, batch.masks, batch.active_mask, context)
    loss.backward()

    predictor_grad = sum(_grad_sum(parameter.grad) for parameter in model.predictor.parameters())
    assert predictor_grad > 0.0


def test_delta_action_weight_with_no_horizons_is_rejected_instead_of_silently_disabling_ldad():
    with pytest.raises(ValueError, match="delta_action_horizons"):
        _small_model(delta_action_weight=1.0, delta_action_horizons=())


def test_ema_target_encoder_with_online_dynamics_target_is_not_silently_ignored():
    batch = _small_batch(batch_size=1)
    noema = _small_model(dynamics_target_mode="online_no_stopgrad", use_ema_target_encoder=False)
    try:
        ema = _small_model(dynamics_target_mode="online_no_stopgrad", use_ema_target_encoder=True)
    except ValueError:
        return
    ema.load_state_dict(noema.state_dict(), strict=False)
    with torch.no_grad():
        for parameter in ema.target_state_encoder.parameters():
            parameter.add_(0.5)

    common_args = (
        batch.boards[:, :2],
        batch.actions[:, :2],
        batch.context,
        batch.clue_mask,
        batch.editable_mask,
        batch.active_mask,
        batch.goals,
    )
    noema_loss = noema(*common_args, masks=batch.masks[:, :2]).dynamics_loss
    ema_loss = ema(*common_args, masks=batch.masks[:, :2]).dynamics_loss

    assert not torch.allclose(noema_loss, ema_loss)


def test_single_state_training_predictor_receives_the_full_past_history_once():
    batch = _small_batch(batch_size=1)
    model = _small_model(
        latent_representation="single",
        goal_conditioning="context_current",
        dynamics_target_mode="online_no_stopgrad",
        goal_target_mode="online_no_stopgrad",
        delta_action_weight=0.0,
    )
    seen_history_lengths = []
    original = model.predict_next_sequence

    def record_predict_next_sequence(state_history, action_history, context_latents):
        seen_history_lengths.append(int(action_history.shape[1]))
        return original(state_history, action_history, context_latents)

    model.predict_next_sequence = record_predict_next_sequence

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
    assert seen_history_lengths == [batch.boards.shape[1] - 1]


def test_single_state_latent_rollout_planner_passes_growing_history_to_autoregressive_predictor():
    example = _example()
    board = example.goal.copy()
    blanks = [(0, 2), (0, 3), (0, 5)]
    for row, col in blanks:
        board[row, col] = 0
    goal = example.goal
    clue_mask = board != 0
    editable_mask = ~clue_mask
    active_mask = np.ones((9, 9), dtype=bool)
    model = _small_model(
        latent_representation="single",
        goal_conditioning="context_current",
        dynamics_target_mode="online_no_stopgrad",
        goal_target_mode="online_no_stopgrad",
    )
    context, predicted_goal, oracle_goal, _ = _prepare_goal_latents(
        model,
        board,
        goal,
        clue_mask,
        editable_mask,
        active_mask,
        device=torch.device("cpu"),
    )
    action_ids = []
    for row, col in blanks:
        target = int(goal[row, col])
        action = next(
            candidate
            for candidate in ACTION_VOCAB
            if (candidate.row, candidate.col, candidate.value) == (row, col, target)
        )
        action_ids.append(ACTION_VOCAB.index(action))
    seq_ids = np.asarray([action_ids], dtype=np.int64)
    seen_history_lengths = []

    def record_predict_next_sequence(state_history, action_history, context_latents):
        seen_history_lengths.append(int(action_history.shape[1]))
        batch = state_history.shape[0]
        return torch.zeros((batch, action_history.shape[1], 1, model.d_model), dtype=state_history.dtype)

    model.predict_next_sequence = record_predict_next_sequence

    _score_cem_sequences(
        model,
        board,
        [goal],
        seq_ids,
        context,
        predicted_goal,
        oracle_goal,
        clue_mask,
        editable_mask,
        active_mask,
        score_mode="oracle_goal_raw_euclidean_distance",
        transition_mode="latent_rollout",
        device=torch.device("cpu"),
    )

    assert seen_history_lengths == [1, 2, 3]
