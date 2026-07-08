from __future__ import annotations

import numpy as np
import torch

from puzzle_jepa.object_dynamics.batching import sample_object_dynamics_batch
from puzzle_jepa.object_dynamics.domain import apply_low_level_action
from puzzle_jepa.object_dynamics.generator import ObjectDynamicsGenerator, ObjectDynamicsSpec, TRAJECTORY_KINDS
from puzzle_jepa.object_dynamics.model import ObjectDynamicsJEPA
from puzzle_jepa.train.object_dynamics import run_object_dynamics_training


def test_generator_trajectories_apply_actions() -> None:
    rng = np.random.default_rng(123)
    for kind in TRAJECTORY_KINDS:
        spec = ObjectDynamicsSpec(
            grid_size=12,
            max_objects=3,
            max_shape_extent=4,
            trajectory_kind=kind,
            counterfactual_ratio=0.0,
            wrong_ratio=0.0,
        )
        trajectory = ObjectDynamicsGenerator(spec).sample_trajectory(rng, min_actions=1)
        assert trajectory.states.shape[1:] == (12, 12)
        assert len(trajectory.actions) == len(trajectory.action_object_ids)
        current = trajectory.states[0]
        for index, action in enumerate(trajectory.actions):
            current = apply_low_level_action(current, action)
            np.testing.assert_array_equal(current, trajectory.states[index + 1])


def test_batch_sampler_shapes() -> None:
    rng = np.random.default_rng(321)
    generator = ObjectDynamicsGenerator(
        ObjectDynamicsSpec(grid_size=12, max_objects=3, max_shape_extent=4, trajectory_kind="semantic_mix")
    )
    batch = sample_object_dynamics_batch(generator, rng, batch_size=5, horizon=4)
    assert batch.states.shape == (5, 12, 12)
    assert batch.actions.shape == (5, 4, 4)
    assert batch.futures.shape == (5, 4, 12, 12)
    assert batch.completion.shape == (5, 3)


def test_model_forward_base_ldad_and_hierarchy() -> None:
    rng = np.random.default_rng(456)
    generator = ObjectDynamicsGenerator(
        ObjectDynamicsSpec(grid_size=12, max_objects=3, max_shape_extent=4, trajectory_kind="frontier_build")
    )
    batch = sample_object_dynamics_batch(generator, rng, batch_size=3, horizon=4)
    base = ObjectDynamicsJEPA(grid_size=12, d_model=32, encoder_layers=1, encoder_heads=4, rollout_horizon=2)
    base_output = base(batch)
    assert torch.isfinite(base_output.loss)
    assert base_output.predicted.shape == (3, 2, 32)

    ldad = ObjectDynamicsJEPA(grid_size=12, d_model=32, encoder_layers=1, encoder_heads=4, rollout_horizon=2, ldad_weight=0.1)
    ldad_output = ldad(batch)
    assert torch.isfinite(ldad_output.ldad_loss)

    hierarchy = ObjectDynamicsJEPA(
        grid_size=12,
        d_model=32,
        encoder_layers=1,
        encoder_heads=4,
        rollout_horizon=1,
        hierarchy_horizon=4,
    )
    hierarchy_output = hierarchy(batch)
    assert torch.isfinite(hierarchy_output.hierarchy_loss)
    assert float(hierarchy_output.hierarchy_loss.detach()) > 0.0


def test_trainer_smoke_run(tmp_path) -> None:
    config = {
        "seed": 7,
        "device": "cpu",
        "output_dir": str(tmp_path / "run"),
        "data": {
            "name": "test_frontier",
            "grid_size": 8,
            "num_colors": 6,
            "min_objects": 1,
            "max_objects": 2,
            "max_shape_extent": 4,
            "trajectory_kind": "frontier_build",
            "counterfactual_ratio": 0.0,
            "wrong_ratio": 0.0,
            "max_scene_retries": 64,
        },
        "model": {
            "name": "test_cls32_r2",
            "d_model": 32,
            "encoder_layers": 1,
            "encoder_heads": 4,
            "rollout_horizon": 2,
            "hierarchy_horizon": 0,
        },
        "objective": {
            "name": "base",
            "target_ema": False,
            "ema_decay": 0.99,
            "ldad_weight": 0.0,
            "regularizer": "none",
            "regularizer_weight": 0.0,
        },
        "training": {
            "max_steps": 2,
            "batch_size": 2,
            "learning_rate": 1.0e-3,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "bf16": False,
            "eval_every_steps": 2,
            "save_every_steps": 2,
        },
        "eval": {
            "probe_train_samples": 8,
            "probe_eval_samples": 6,
            "probe_batch_size": 3,
            "probe_steps": 2,
            "probe_learning_rate": 1.0e-2,
        },
    }
    metrics = run_object_dynamics_training(config)
    assert metrics["step"] == 2
    assert (tmp_path / "run" / "checkpoint.pt").exists()
    assert "probe_object_count_acc" in metrics
