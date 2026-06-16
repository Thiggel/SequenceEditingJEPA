from pathlib import Path

import numpy as np
import pytest
import torch

from puzzle_jepa.data.grid_goal_sudoku import collate_grid_goal_sudoku_trajectories, sample_grid_goal_sudoku_trajectory
from puzzle_jepa.data.worlds import SudokuWorld
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
