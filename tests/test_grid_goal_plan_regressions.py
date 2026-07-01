import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from puzzle_jepa.data.grid_goal_sudoku import (
    collate_grid_goal_sudoku_trajectories,
    sample_grid_goal_sudoku_trajectory,
    sample_random_grid_goal_sudoku_trajectory,
)
from puzzle_jepa.data.worlds import SudokuWorld
from puzzle_jepa.eval.grid_goal_diagnostics import run_grid_goal_diagnostics
from puzzle_jepa.eval.grid_goal_planner_matrix import load_checkpoint, run_planner_matrix
from puzzle_jepa.models.grid_goal_jepa import GridTokenGoalJEPA
from puzzle_jepa.train.grid_goal_sudoku import _zero_context_masks


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
        multi_step_horizons=(1, 4),
    )
    defaults.update(kwargs)
    return GridTokenGoalJEPA(**defaults)


def _small_model_config(**kwargs):
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
    return defaults


def test_action_rank_positives_are_target_consistent():
    """Action-rank positives should be solution actions, not random trajectory fills."""
    rng = np.random.default_rng(0)
    trajectory = sample_grid_goal_sudoku_trajectory(_example(), rng, oracle_probability=0.0)
    first_action = trajectory.actions[0]

    row, col, value = [int(x) for x in first_action]

    assert trajectory.is_oracle
    assert value == int(trajectory.goal[row, col])


def test_no_context_ablation_removes_context_value_conditioning():
    """R1 is specified as removing context conditioning, not only clue/editable mask embeddings."""
    rng = np.random.default_rng(1)
    batch = collate_grid_goal_sudoku_trajectories(
        [sample_grid_goal_sudoku_trajectory(_example(), rng, oracle_probability=1.0)]
    )
    batch = _zero_context_masks(batch)
    model = _small_model().eval()

    state = batch.boards[:, 0]
    with torch.no_grad():
        context_a = model.encode_context(batch.context, batch.clue_mask, batch.editable_mask, batch.active_mask)
        context_b = model.encode_context(batch.goals, batch.clue_mask, batch.editable_mask, batch.active_mask)
        latent_a = model.encode_state(state, context_a, batch.clue_mask, batch.editable_mask, batch.active_mask)
        latent_b = model.encode_state(state, context_b, batch.clue_mask, batch.editable_mask, batch.active_mask)

    assert torch.allclose(latent_a, latent_b, atol=1.0e-6)


def test_forward_accepts_non_9x9_active_grid_tokens():
    """The architecture is grid-token based and should not hard-code Sudoku's 81 tokens internally."""
    model = _small_model()
    boards = torch.zeros((1, 3, 4, 4), dtype=torch.long)
    actions = torch.zeros((1, 3, 3), dtype=torch.long)
    context = torch.zeros((1, 4, 4), dtype=torch.long)
    clue_mask = torch.zeros((1, 4, 4), dtype=torch.bool)
    editable_mask = torch.ones((1, 4, 4), dtype=torch.bool)
    active_mask = torch.ones((1, 4, 4), dtype=torch.bool)
    goals = torch.zeros((1, 4, 4), dtype=torch.long)
    masks = torch.ones((1, 3), dtype=torch.bool)

    output = model(
        boards,
        actions,
        context,
        clue_mask,
        editable_mask,
        active_mask,
        goals,
        masks=masks,
    )

    assert output.state_latents.shape == (1, 3, 16, 32)


def test_progress_rank_loss_ignores_non_oracle_trajectories():
    batch = collate_grid_goal_sudoku_trajectories(
        [sample_grid_goal_sudoku_trajectory(_example(), np.random.default_rng(2), oracle_probability=1.0)]
    )
    model = _small_model()
    output = model(
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

    assert output.progress_rank_loss.item() == pytest.approx(0.0)


def test_action_rank_shapes_encoder_geometry_not_predictor_rollout():
    batch = collate_grid_goal_sudoku_trajectories(
        [sample_grid_goal_sudoku_trajectory(_example(), np.random.default_rng(3), oracle_probability=1.0)]
    )
    positive = batch.actions[:, 0].clone()
    negative = positive.clone()
    negative[:, 2] = (negative[:, 2] % 9) + 1

    torch.manual_seed(0)
    model_a = _small_model().eval()
    model_b = _small_model().eval()
    model_b.load_state_dict(model_a.state_dict())
    with torch.no_grad():
        for param in model_b.predictor.parameters():
            param.add_(torch.randn_like(param))
        for param in model_b.predictor_out.parameters():
            param.add_(torch.randn_like(param))

    kwargs = dict(
        boards=batch.boards,
        actions=batch.actions,
        context=batch.context,
        clue_mask=batch.clue_mask,
        editable_mask=batch.editable_mask,
        active_mask=batch.active_mask,
        goals=batch.goals,
        masks=batch.masks,
        oracle_mask=batch.oracle_mask,
        positive_actions=positive,
        negative_actions=negative,
    )
    with torch.no_grad():
        loss_a = model_a(**kwargs).action_rank_loss
        loss_b = model_b(**kwargs).action_rank_loss

    torch.testing.assert_close(loss_a, loss_b)


@pytest.mark.parametrize(
    "legacy_path",
    [
        "puzzle_jepa/models/action_jepa.py",
        "puzzle_jepa/models/sigreg_jepa.py",
        "puzzle_jepa/models/trajectory_jepa.py",
        "puzzle_jepa/train/grid5.py",
        "puzzle_jepa/train/grid6.py",
        "puzzle_jepa/eval/grid5_planner_matrix.py",
        "puzzle_jepa/eval/grid6_planner_matrix.py",
    ],
)
def test_legacy_cls_value_and_causal_paths_are_removed_from_active_tree(legacy_path):
    assert not Path(legacy_path).exists()


def test_progress_rank_ignores_non_oracle_trajectories():
    """Progress ranking is defined only along successful trajectories."""
    rng = np.random.default_rng(2)
    batch = collate_grid_goal_sudoku_trajectories(
        [sample_random_grid_goal_sudoku_trajectory(_example(), rng)]
    )
    assert not bool(batch.oracle_mask.item())

    model = _small_model().eval()
    with torch.no_grad():
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

    assert output.progress_rank_loss.item() == pytest.approx(0.0, abs=1.0e-8)


def test_action_rank_does_not_depend_on_predictor_rollout(monkeypatch):
    """Action ranking should shape encoded successor geometry, not P_phi outputs."""
    example = _example()
    model = _small_model().eval()
    board = torch.as_tensor(example.state[None, None], dtype=torch.long)
    actions = torch.zeros((1, 1, 3), dtype=torch.long)
    context = torch.as_tensor(example.state[None], dtype=torch.long)
    clue_mask = torch.as_tensor((example.state != 0)[None], dtype=torch.bool)
    editable_mask = ~clue_mask
    active_mask = torch.ones_like(clue_mask)
    goals = torch.as_tensor(example.goal[None], dtype=torch.long)
    masks = torch.ones((1, 1), dtype=torch.bool)
    empty_row, empty_col = [int(x) for x in np.argwhere(example.state == 0)[0]]
    correct = int(example.goal[empty_row, empty_col])
    wrong = 1 + (correct % 9)
    positive_actions = torch.as_tensor([[empty_row, empty_col, correct]], dtype=torch.long)
    negative_actions = torch.as_tensor([[empty_row, empty_col, wrong]], dtype=torch.long)

    original_predict_next = model.predict_next

    def forbid_nonempty_predictor_actions(state_latent, action, context_latents):
        if action.numel() > 0:
            raise AssertionError("action ranking should not call predict_next")
        return original_predict_next(state_latent, action, context_latents)

    monkeypatch.setattr(model, "predict_next", forbid_nonempty_predictor_actions)

    with torch.no_grad():
        model(
            board,
            actions,
            context,
            clue_mask,
            editable_mask,
            active_mask,
            goals,
            masks=masks,
            positive_actions=positive_actions,
            negative_actions=negative_actions,
        )


def test_training_action_rank_state_is_not_hard_wired_to_initial_board(monkeypatch, tmp_path):
    """Action ranking should train branch discrimination at trajectory states, not only s_0."""
    import puzzle_jepa.train.grid_goal_sudoku as train_mod

    rng = np.random.default_rng(4)
    example = _example()
    batch = collate_grid_goal_sudoku_trajectories(
        [sample_grid_goal_sudoku_trajectory(example, rng, oracle_probability=1.0)]
    )
    captured_rank_boards = []

    class DummyModel(torch.nn.Module):
        def __init__(self, **_kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

        def forward(self, *args, **kwargs):
            loss = self.weight.square()
            zero = loss.detach()
            return SimpleNamespace(
                loss=loss,
                dynamics_loss=zero,
                sigreg_loss=zero,
                goal_mse_loss=zero,
                goal_nce_loss=zero,
                progress_rank_loss=zero,
                action_rank_loss=zero,
                temporal_straightening_loss=zero,
                terminal_corrupt_loss=zero,
            )

    def fake_sample_batch(*_args, **_kwargs):
        return batch

    def fake_sample_rank_actions(boards, goals, rng, *, device):
        del goals, rng
        captured_rank_boards.append(boards.detach().cpu().clone())
        action = batch.actions[:, 0].to(device)
        negative = action.clone()
        negative[:, 2] = (negative[:, 2] % 9) + 1
        return action, negative

    monkeypatch.setattr(train_mod, "GridTokenGoalJEPA", DummyModel)
    monkeypatch.setattr(train_mod, "_load_examples", lambda *_args, **_kwargs: [example])
    monkeypatch.setattr(train_mod, "_sample_batch", fake_sample_batch)
    monkeypatch.setattr(train_mod, "_sample_rank_actions", fake_sample_rank_actions)
    monkeypatch.setattr(train_mod, "run_grid_goal_diagnostics", lambda *_args, **_kwargs: {})

    train_mod.run_grid_goal_sudoku(
        {
            "seed": 0,
            "ablation": "M0_full",
            "output_dir": str(tmp_path),
            "task": {"repo_id": "unused", "train_split": "train", "eval_split": "test"},
            "model": {},
            "training": {
                "max_steps": 1,
                "batch_size": 1,
                "oracle_probability": 1.0,
                "learning_rate": 1.0e-4,
                "eval_every_steps": 1,
                "save_every_steps": 999,
                "bf16": False,
            },
            "eval": {"diagnostic_examples": 1},
        }
    )

    assert captured_rank_boards
    initial_boards = batch.boards[:, 0].cpu()
    rank_boards = captured_rank_boards[0]
    assert rank_boards.ndim == 4 or not torch.equal(rank_boards, initial_boards)


@pytest.mark.parametrize(
    "legacy_path",
    [
        "puzzle_jepa/models/recursive.py",
        "puzzle_jepa/models/layers.py",
    ],
)
def test_recursive_baseline_scaffolding_is_kept(legacy_path):
    assert Path(legacy_path).exists()


def test_planner_matrix_records_beam_and_cem_rows(tmp_path):
    example = _example()
    state = example.goal.copy()
    state[0, 2] = 0
    tiny = type(example)(state, example.goal)
    model = _small_model()
    records = run_planner_matrix(
        model,
        [tiny],
        output_path=tmp_path / "planner_matrix.jsonl",
        device=torch.device("cpu"),
        beam_widths=(2,),
        beam_depths=(1,),
        scores=("oracle_goal_distance",),
        transitions=("latent_rollout",),
        planners=("mpc_beam", "categorical_cem"),
        max_examples=1,
        max_steps=1,
        cem_samples=3,
        cem_iters=1,
        cem_elites=1,
    )

    assert [record["planner"] for record in records] == ["mpc_beam", "categorical_cem"]
    assert (tmp_path / "planner_matrix.jsonl").read_text().count("\n") == 2
    rerun = run_planner_matrix(
        model,
        [tiny],
        output_path=tmp_path / "planner_matrix.jsonl",
        device=torch.device("cpu"),
        beam_widths=(2,),
        beam_depths=(1,),
        scores=("oracle_goal_distance",),
        transitions=("latent_rollout",),
        planners=("mpc_beam", "categorical_cem"),
        max_examples=1,
        max_steps=1,
        cem_samples=3,
        cem_iters=1,
        cem_elites=1,
    )
    assert rerun == []
    assert (tmp_path / "planner_matrix.jsonl").read_text().count("\n") == 2


def test_planner_matrix_resume_appends_only_missing_cells(tmp_path):
    example = _example()
    state = example.goal.copy()
    state[0, 2] = 0
    tiny = type(example)(state, example.goal)
    output_path = tmp_path / "planner_matrix.jsonl"
    output_path.write_text(
        json.dumps(
            {
                "planner": "mpc_beam",
                "transition_mode": "latent_rollout",
                "score_mode": "oracle_goal_distance",
                "beam_width": 2,
                "beam_depth": 1,
                "examples": 1,
                "solved": 0,
                "solve_rate": 0.0,
                "remaining_hamming_mean": 1.0,
                "steps_mean": 1.0,
                "action_evals_mean": 1.0,
                "elapsed_seconds_mean": 0.0,
            },
            sort_keys=True,
        )
        + "\n"
        + '{"planner": '
    )

    records = run_planner_matrix(
        _small_model(),
        [tiny],
        output_path=output_path,
        device=torch.device("cpu"),
        beam_widths=(2,),
        beam_depths=(1,),
        scores=("oracle_goal_distance",),
        transitions=("latent_rollout",),
        planners=("mpc_beam", "categorical_cem"),
        max_examples=1,
        max_steps=1,
        cem_samples=3,
        cem_iters=1,
        cem_elites=1,
    )

    assert [record["planner"] for record in records] == ["categorical_cem"]
    parseable = []
    malformed = 0
    for line in output_path.read_text().splitlines():
        try:
            parseable.append(json.loads(line))
        except json.JSONDecodeError:
            malformed += 1
    assert [record["planner"] for record in parseable] == ["mpc_beam", "categorical_cem"]
    assert malformed == 1


def test_diagnostics_include_rollout_and_goal_alignment_metrics(tmp_path):
    model = _small_model().eval()
    metrics = run_grid_goal_diagnostics(
        model,
        [_example()],
        tmp_path,
        device=torch.device("cpu"),
        panel_examples=1,
        panel_steps=1,
        panel_actions=2,
    )

    required = {
        "predictor_rollout_mse_h1",
        "predictor_rollout_mse_h4",
        "latent_rollout_action_top1",
        "goal_prediction_token_mse",
        "predicted_vs_oracle_goal_distance",
        "distance_hamming_spearman",
    }
    assert required <= set(metrics)


def test_rollout_diagnostics_include_configured_long_horizons(tmp_path):
    model = _small_model(multi_step_horizons=(1, 4, 8, 16, 32)).eval()
    metrics = run_grid_goal_diagnostics(
        model,
        [_example()],
        tmp_path,
        device=torch.device("cpu"),
        panel_examples=1,
        panel_steps=1,
        panel_actions=2,
    )

    assert "predictor_rollout_mse_h32" in metrics
    assert "latent_rollout_drift_mse_h32" in metrics


def test_planner_checkpoint_loader_accepts_training_metadata_numpy_scalars(tmp_path):
    model_config = _small_model_config()
    model = GridTokenGoalJEPA(**model_config)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": {"model": model_config, "task": {}, "seed": 0},
            "metrics": {"loss": np.float64(1.0)},
        },
        checkpoint_path,
    )

    loaded_model, loaded_config = load_checkpoint(checkpoint_path, torch.device("cpu"))

    assert isinstance(loaded_model, GridTokenGoalJEPA)
    assert loaded_config["model"] == model_config
