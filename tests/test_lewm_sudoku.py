import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from pathlib import Path

from puzzle_jepa.data.lewm_sudoku import (
    action_to_array,
    apply_fill_action,
    collate_sudoku_trajectories,
    legal_fill_actions,
    sample_sudoku_trajectory,
)
from puzzle_jepa.data.worlds import SudokuWorld, WorldAction
from puzzle_jepa.models.lewm import LeWMSIGReg, LeWMSudokuModel
from puzzle_jepa.planning.lewm_planner import (
    beam_plan_once,
    best_first_plan_once,
    categorical_cem_plan_once,
    greedy_plan_once,
    local_search_plan_once,
    mcts_plan_once,
    run_mpc,
    solve_sudoku_exact,
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


def _example():
    world = SudokuWorld()
    return world, world.example_from_strings(SUDOKU_PUZZLE, SUDOKU_SOLUTION)


def _small_model(sigreg_projections=8):
    return LeWMSudokuModel(
        d_model=32,
        latent_dim=32,
        encoder_layers=1,
        predictor_layers=1,
        num_heads=4,
        dropout=0.0,
        action_component_dim=4,
        action_dim=8,
        max_history=8,
        projector_hidden_dim=64,
        sigreg_projections=sigreg_projections,
        sigreg_knots=5,
    )


def test_lewm_trajectory_samples_fill_only_sequences():
    _world, example = _example()
    rng = np.random.default_rng(0)
    trajectory = sample_sudoku_trajectory(example, rng, num_frames=6, oracle_probability=1.0)
    assert trajectory.boards.shape == (6, 9, 9)
    assert trajectory.actions.shape == (6, 3)
    for before, action_values, after in zip(
        trajectory.boards[:-1],
        trajectory.actions[:-1],
        trajectory.boards[1:],
        strict=True,
    ):
        action = WorldAction(*[int(x) for x in action_values])
        assert before[action.row, action.col] == 0
        assert after[action.row, action.col] == action.value
        assert action.value == example.goal[action.row, action.col]
        changed = np.argwhere(before != after)
        assert changed.shape[0] == 1
    assert trajectory.actions[-1].tolist() == [0, 0, 0]


def test_lewm_full_trajectory_reaches_terminal_length():
    _world, example = _example()
    rng = np.random.default_rng(0)
    trajectory = sample_sudoku_trajectory(example, rng, num_frames=None, oracle_probability=1.0)
    blanks = int(np.count_nonzero(example.state == 0))
    assert trajectory.boards.shape == (blanks + 1, 9, 9)
    assert np.array_equal(trajectory.boards[-1], example.goal)


def test_lewm_trajectory_batch_collates_variable_lengths_with_masks():
    _world, example = _example()
    rng = np.random.default_rng(1)
    trajectories = [
        sample_sudoku_trajectory(example, rng, num_frames=4),
        sample_sudoku_trajectory(example, rng, num_frames=6),
    ]
    batch = collate_sudoku_trajectories(trajectories)
    assert batch.boards.shape == (2, 6, 9, 9)
    assert batch.actions.shape == (2, 6, 3)
    assert batch.goals.shape == (2, 9, 9)
    assert batch.masks.tolist() == [[True, True, True, True, False, False], [True] * 6]
    assert torch.equal(batch.boards[0, 4], batch.boards[0, 3])
    assert batch.oracle_mask.dtype == torch.bool


def test_sigreg_is_stepwise_and_penalizes_degenerate_embeddings():
    torch.manual_seed(0)
    sigreg = LeWMSIGReg(knots=5, num_proj=32)
    degenerate = torch.zeros(4, 64, 16)
    gaussian = torch.randn(4, 64, 16)
    assert sigreg(gaussian) < sigreg(degenerate)
    masked = sigreg(gaussian, torch.ones(4, 64, dtype=torch.bool))
    assert torch.isfinite(masked)
    with pytest.raises(ValueError, match="time, batch, dim"):
        sigreg(torch.randn(64, 16))


def test_encoder_accepts_current_board_only_and_uses_cls_representation():
    world, example = _example()
    model = _small_model()
    model.eval()
    boards = torch.as_tensor(np.stack([example.state, example.goal]), dtype=torch.long)
    emb = model.encode_board(boards)
    assert emb.shape == (2, 32)
    with pytest.raises(TypeError):
        model.encode_board(boards, torch.ones_like(boards))


def test_adaln_zero_initialized_and_predictor_is_causal():
    torch.manual_seed(0)
    model = _small_model()
    block = model.predictor.layers[0]
    assert torch.count_nonzero(block.adaLN_modulation[-1].weight).item() == 0
    assert torch.count_nonzero(block.adaLN_modulation[-1].bias).item() == 0
    model.eval()
    embeddings = torch.randn(2, 4, 32)
    actions = torch.tensor(
        [
            [[0, 0, 1], [0, 1, 2], [0, 2, 3], [0, 3, 4]],
            [[1, 0, 1], [1, 1, 2], [1, 2, 3], [1, 3, 4]],
        ],
        dtype=torch.long,
    )
    baseline = model.predict_sequence(embeddings, actions)
    changed_embeddings = embeddings.clone()
    changed_actions = actions.clone()
    changed_embeddings[:, 2:] = torch.randn_like(changed_embeddings[:, 2:]) * 100.0
    changed_actions[:, 2:] = torch.tensor([8, 8, 9])
    changed = model.predict_sequence(changed_embeddings, changed_actions)
    assert torch.allclose(baseline[:, :2], changed[:, :2], atol=1.0e-5)
    assert not torch.allclose(baseline[:, 3], changed[:, 3])


def test_lewm_loss_backpropagates_and_goal_board_has_zero_target_distance():
    _world, example = _example()
    rng = np.random.default_rng(2)
    trajectories = [sample_sudoku_trajectory(example, rng, num_frames=4) for _ in range(2)]
    batch = collate_sudoku_trajectories(trajectories)
    model = _small_model()
    output = model(batch.boards, batch.actions, batch.goals, masks=batch.masks)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert any(param.grad is not None for param in model.encoder.parameters())
    assert any(param.grad is not None for param in model.predictor.parameters())
    assert any(param.grad is not None for param in model.value_head.parameters())

    model.eval()
    solved = torch.as_tensor(np.stack([[example.goal, example.goal], [example.goal, example.goal]]), dtype=torch.long)
    pad_actions = torch.zeros((2, 2, 3), dtype=torch.long)
    goals = torch.as_tensor(np.stack([example.goal, example.goal]), dtype=torch.long)
    solved_output = model(solved, pad_actions, goals)
    assert torch.allclose(solved_output.goal_distances, torch.zeros_like(solved_output.goal_distances), atol=1.0e-5)


def test_fill_action_helpers_never_overwrite():
    _world, example = _example()
    action = WorldAction(0, 2, 4)
    filled = apply_fill_action(example.state, action)
    assert filled[0, 2] == 4
    with pytest.raises(ValueError, match="empty"):
        apply_fill_action(filled, WorldAction(0, 2, 9))
    assert all(example.state[action.row, action.col] == 0 for action in legal_fill_actions(example.state))
    assert action_to_array(action).tolist() == [0, 2, 4]


def test_hamming_planners_pick_goal_consistent_actions():
    _world, example = _example()
    device = torch.device("cpu")
    rng = np.random.default_rng(0)
    planners = [
        greedy_plan_once(
            None,
            example.state,
            example.goal,
            transition_mode="symbolic_reencode",
            score_mode="true_hamming_oracle",
            device=device,
        ),
        beam_plan_once(
            None,
            example.state,
            example.goal,
            horizon=2,
            beam_width=2,
            branch_size=4,
            transition_mode="symbolic_reencode",
            score_mode="true_hamming_oracle",
            device=device,
        ),
        best_first_plan_once(
            None,
            example.state,
            example.goal,
            horizon=2,
            max_expansions=16,
            branch_size=4,
            heuristic_weight=1.0,
            transition_mode="symbolic_reencode",
            score_mode="true_hamming_oracle",
            device=device,
        ),
        categorical_cem_plan_once(
            None,
            example.state,
            example.goal,
            horizon=1,
            candidates=256,
            elites=16,
            iterations=2,
            smoothing=0.2,
            transition_mode="symbolic_reencode",
            score_mode="true_hamming_oracle",
            rng=rng,
            device=device,
        ),
        local_search_plan_once(
            None,
            example.state,
            example.goal,
            horizon=1,
            candidates=256,
            iterations=32,
            temperature=0.0,
            transition_mode="symbolic_reencode",
            score_mode="true_hamming_oracle",
            rng=rng,
            device=device,
        ),
    ]
    for action in planners:
        assert action is not None
        assert action.value == example.goal[action.row, action.col]


def test_mcts_and_mpc_work_on_one_empty_cell():
    world = SudokuWorld()
    goal = world.from_string(SUDOKU_SOLUTION)
    board = goal.copy()
    board[0, 0] = 0
    rng = np.random.default_rng(0)
    action = mcts_plan_once(
        None,
        board,
        goal,
        horizon=1,
        simulations=128,
        exploration=1.4,
        transition_mode="symbolic_reencode",
        score_mode="true_hamming_oracle",
        rng=rng,
        device=torch.device("cpu"),
    )
    assert action == WorldAction(0, 0, int(goal[0, 0]))
    result = run_mpc(
        None,
        board,
        goal,
        planner="mcts",
        horizon=1,
        score_mode="true_hamming_oracle",
        max_steps=1,
        mcts_simulations=128,
        rng=np.random.default_rng(0),
    )
    assert result.solved
    assert result.remaining_hamming == 0


def test_exact_symbolic_solver_solves_known_puzzle():
    _world, example = _example()
    solved = solve_sudoku_exact(example.state)
    assert solved is not None
    assert np.array_equal(solved, example.goal)


def test_lewm_hydra_config_composes():
    repo_root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(version_base=None, config_dir=str(repo_root / "configs" / "puzzle")):
        cfg = compose(config_name="lewm_sudoku")
    assert cfg.model.encoder_layers == 6
    assert cfg.model.predictor_layers == 6
    assert cfg.model.max_history == 82
    assert cfg.model.projector_hidden_dim == 2048
    assert cfg.model.sigreg_projections == 1024
    assert cfg.model.sigreg_weight == 0.1
    assert cfg.model.stop_gradient_target is False
    assert cfg.training.num_frames is None
