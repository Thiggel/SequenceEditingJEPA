from __future__ import annotations

import inspect

import torch

from puzzle_jepa.models.grid_goal_jepa import GridTokenGoalJEPA, sigreg_regularizer


def test_grid_goal_sigreg_detects_non_gaussian_samples_with_matched_covariance() -> None:
    torch.manual_seed(19)
    gaussian = torch.randn(2048, 1)
    rademacher = (2 * torch.randint(0, 2, (2048, 1)) - 1).float()
    mask = torch.ones((2048,), dtype=torch.bool)

    gaussian_loss = sigreg_regularizer(gaussian, mask)
    rademacher_loss = sigreg_regularizer(rademacher, mask)
    assert float(gaussian_loss) < 0.25 * float(rademacher_loss)


def test_grid_goal_ldad_never_substitutes_predictor_displacement_for_encoded_endpoints() -> None:
    source = inspect.getsource(GridTokenGoalJEPA._delta_action_objective)
    assert "_delta_action_predicted_future" not in source
