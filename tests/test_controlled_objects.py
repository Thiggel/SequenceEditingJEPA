from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from puzzle_jepa.controlled_objects.batching import (
    ControlledObjectBatch,
    build_controlled_dataset,
)
from puzzle_jepa.controlled_objects.domain import RigidAction
from puzzle_jepa.controlled_objects.evaluation import _exact_receding_plan
from puzzle_jepa.controlled_objects.generator import (
    ControlledObjectGenerator,
    ControlledObjectSpec,
)
from puzzle_jepa.controlled_objects.model import ControlledObjectJEPA


ROOT = Path(__file__).resolve().parents[1]


def _generator(*, horizon: int = 64) -> ControlledObjectGenerator:
    return ControlledObjectGenerator(
        ControlledObjectSpec(
            grid_size=8,
            num_colors=6,
            object_count=2,
            trajectory_length=horizon,
            invalid_action_ratio=0.1,
        )
    )


def _model(**kwargs) -> ControlledObjectJEPA:
    return ControlledObjectJEPA(
        grid_size=8,
        num_colors=6,
        token_dim=16,
        latent_dim=8,
        encoder_layers=1,
        encoder_heads=4,
        action_token_dim=16,
        action_heads=4,
        **kwargs,
    )


def test_rigid_action_moves_the_whole_selected_component_without_object_id() -> None:
    generator = _generator()
    state = np.zeros((8, 8), dtype=np.int64)
    state[2:4, 2:4] = 3

    moved, valid = generator.apply_action(state, RigidAction(3, 3, 4))

    assert valid
    assert np.count_nonzero(moved == 3) == 4
    assert np.all(moved[2:4, 3:5] == 3)
    assert not np.any(moved[:, :2] == 3)


def test_invalid_background_boundary_and_collision_actions_are_noops() -> None:
    generator = _generator()
    state = np.zeros((8, 8), dtype=np.int64)
    state[0:2, 0:2] = 1
    state[0:2, 2:4] = 2
    for action in (
        RigidAction(7, 7, 4),
        RigidAction(0, 0, 1),
        RigidAction(0, 0, 4),
    ):
        next_state, valid = generator.apply_action(state, action)
        assert not valid
        np.testing.assert_array_equal(next_state, state)


def test_controlled_trajectories_replay_exactly_and_keep_action_labels() -> None:
    generator = _generator(horizon=16)
    trajectory = generator.sample_trajectory(np.random.default_rng(7))
    replayed = trajectory.states[0]
    replay_validity = []
    for action_values in trajectory.actions:
        replayed, valid = generator.apply_action(
            replayed, RigidAction(*(int(value) for value in action_values))
        )
        replay_validity.append(valid)
    np.testing.assert_array_equal(replayed, trajectory.states[-1])
    np.testing.assert_array_equal(replay_validity, trajectory.action_validity)
    assert trajectory.actions.shape == (16, 3)


def test_exact_dynamics_receding_planner_solves_known_reachable_goals() -> None:
    generator = _generator(horizon=8)
    rng = np.random.default_rng(9)
    for _ in range(8):
        trajectory = generator.sample_trajectory(rng)
        planned = _exact_receding_plan(
            generator,
            trajectory.states[0],
            trajectory.states[-1],
            trajectory.actions,
            rng,
            candidates=8,
        )
        np.testing.assert_array_equal(planned, trajectory.states[-1])


def test_dataset_samples_matched_contiguous_state_action_windows() -> None:
    generator = _generator(horizon=16)
    dataset = build_controlled_dataset(generator, trajectory_count=8, seed=11)
    batch = dataset.sample_batch(
        np.random.default_rng(13), batch_size=4, horizon=8
    )
    assert batch.states.shape == (4, 9, 8, 8)
    assert batch.actions.shape == (4, 8, 3)
    for sample in range(4):
        state = batch.states[sample, 0].numpy()
        for step in range(8):
            action = RigidAction(*(int(value) for value in batch.actions[sample, step]))
            state, valid = generator.apply_action(state, action)
            assert valid == bool(batch.action_validity[sample, step])
            np.testing.assert_array_equal(state, batch.states[sample, step + 1].numpy())


def test_hierarchy_depth_and_stride_define_multiplicative_temporal_levels() -> None:
    assert _model(hierarchy_depth=1, hierarchy_stride=4).level_spans == (1,)
    assert _model(hierarchy_depth=4, hierarchy_stride=4).level_spans == (1, 4, 16, 64)
    assert _model(hierarchy_depth=3, hierarchy_stride=2).level_spans == (1, 2, 4)
    assert _model(hierarchy_depth=3, hierarchy_stride=8).level_spans == (1, 8, 64)


def test_rollout_horizon_and_all_level_supervision_are_independent_axes() -> None:
    flat = _model(hierarchy_depth=1, rollout_steps=8)
    low_only = _model(
        hierarchy_depth=3,
        hierarchy_stride=4,
        rollout_steps=4,
        rollout_all_levels=False,
    )
    all_levels = _model(
        hierarchy_depth=3,
        hierarchy_stride=4,
        rollout_steps=4,
        rollout_all_levels=True,
    )
    assert flat.required_horizon == 8
    assert low_only.required_horizon == 16
    assert all_levels.required_horizon == 64
    assert [low_only.level_rollout_steps(level) for level in range(3)] == [4, 1, 1]
    assert [all_levels.level_rollout_steps(level) for level in range(3)] == [4, 4, 4]


def test_dense_rollout_feeds_predictions_back_and_uses_geometric_weights() -> None:
    generator = _generator(horizon=4)
    dataset = build_controlled_dataset(generator, trajectory_count=4, seed=17)
    batch = dataset.sample_batch(np.random.default_rng(19), batch_size=2, horizon=4)
    model = _model(
        hierarchy_depth=1,
        rollout_steps=4,
        rollout_lambda=0.75,
        target_mode="shared",
        stop_gradient_targets=False,
        vicreg_weight=0.0,
    )
    inputs = []

    def capture(level: int, latent: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        inputs.append(latent)
        return latent + float(len(inputs))

    model.predict_chunk = capture  # type: ignore[method-assign]
    output = model(batch)

    assert len(inputs) == 4
    torch.testing.assert_close(inputs[1], inputs[0] + 1.0)
    torch.testing.assert_close(inputs[2], inputs[1] + 2.0)
    expected = torch.tensor([1.0, 0.75, 0.75**2, 0.75**3])
    expected /= expected.sum()
    torch.testing.assert_close(output.rollout_weights[0], expected)


def test_target_gradient_modes_match_requested_ldad_ablation() -> None:
    generator = _generator(horizon=4)
    batch = build_controlled_dataset(generator, trajectory_count=4, seed=23).sample_batch(
        np.random.default_rng(29), batch_size=2, horizon=4
    )
    online = _model(
        rollout_steps=4,
        target_mode="shared",
        stop_gradient_targets=False,
        vicreg_weight=0.0,
        ldad_weight=1.0,
    )
    stopgrad = _model(
        rollout_steps=4,
        target_mode="shared",
        stop_gradient_targets=True,
        vicreg_weight=0.05,
        ldad_weight=1.0,
    )
    ema = _model(
        rollout_steps=4,
        target_mode="ema",
        stop_gradient_targets=True,
        vicreg_weight=0.0,
        ldad_weight=1.0,
    )

    assert online(batch).targets[0].requires_grad
    assert not stopgrad(batch).targets[0].requires_grad
    assert not ema(batch).targets[0].requires_grad
    assert online.target_encoder is None
    assert stopgrad.target_encoder is None
    assert ema.target_encoder is not None
    assert all(not parameter.requires_grad for parameter in ema.target_encoder.parameters())


def test_ldad_decodes_full_action_sequence_from_long_horizon_displacement() -> None:
    generator = _generator(horizon=4)
    batch = build_controlled_dataset(generator, trajectory_count=4, seed=31).sample_batch(
        np.random.default_rng(37), batch_size=2, horizon=4
    )
    model = _model(
        rollout_steps=4,
        target_mode="shared",
        stop_gradient_targets=False,
        vicreg_weight=0.0,
        ldad_weight=1.0,
    )
    captured = []
    original = model.ldad_decoder.forward

    def capture(delta: torch.Tensor, *, horizon: int) -> torch.Tensor:
        captured.append((delta, horizon))
        return original(delta, horizon=horizon)

    model.ldad_decoder.forward = capture  # type: ignore[method-assign,union-attr]
    output = model(batch)
    current = model.encode(batch.states[:, 0])
    endpoint = model.encode(batch.states[:, 4], target=True)

    assert captured[0][1] == 4
    torch.testing.assert_close(captured[0][0], endpoint - current)
    assert float(output.ldad_loss.detach()) >= 0.0


def test_full_grid_ldad_is_paired_and_trains_its_displacement_pooler() -> None:
    generator = _generator(horizon=4)
    batch = build_controlled_dataset(generator, trajectory_count=4, seed=41).sample_batch(
        np.random.default_rng(43), batch_size=2, horizon=4
    )
    model = _model(
        latent_representation="grid",
        rollout_steps=4,
        target_mode="shared",
        stop_gradient_targets=False,
        vicreg_weight=0.0,
        ldad_weight=1.0,
    )
    output = model(batch)
    output.loss.backward()

    assert output.predictions[0].shape == (2, 4, 64, 8)
    assert model.delta_pool is not None
    assert model.delta_pool.weight.grad is not None
    assert bool(torch.isfinite(model.delta_pool.weight.grad).all())


def test_hierarchy_stage_freezes_encoder_and_lower_temporal_models() -> None:
    model = _model(hierarchy_depth=3)
    model.freeze_below_level(1)

    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(not parameter.requires_grad for parameter in model.dynamics[0].parameters())
    assert all(parameter.requires_grad for parameter in model.dynamics[1].parameters())
    assert all(parameter.requires_grad for parameter in model.dynamics[2].parameters())


def test_controlled_model_forward_backward_reports_each_hierarchy_level() -> None:
    generator = _generator(horizon=16)
    batch = build_controlled_dataset(generator, trajectory_count=4, seed=47).sample_batch(
        np.random.default_rng(53), batch_size=2, horizon=16
    )
    model = _model(
        hierarchy_depth=3,
        hierarchy_stride=4,
        rollout_steps=2,
        rollout_all_levels=False,
    )
    output = model(batch)
    output.loss.backward()

    assert len(output.level_losses) == 3
    assert [prediction.shape[1] for prediction in output.predictions] == [2, 1, 1]
    assert model.dynamics[2].predictor[-1].weight.grad is not None


def test_config_tree_contains_all_five_unique_ldad_variants() -> None:
    objective_dir = ROOT / "configs/controlled_objects/objective"
    configs = {
        path.stem: path.read_text(encoding="utf-8")
        for path in objective_dir.glob("ldad*.yaml")
    }
    assert set(configs) == {
        "ldad_online",
        "ldad_ema",
        "ldad_vicreg_stopgrad",
        "ldad_vicreg_ema",
        "ldad_vicreg_online",
    }
    assert "target_mode: shared" in configs["ldad_online"]
    assert "stop_gradient_targets: false" in configs["ldad_online"]
    assert "target_mode: ema" in configs["ldad_ema"]
    assert "vicreg_weight: 0.05" in configs["ldad_vicreg_stopgrad"]
    assert "target_mode: ema" in configs["ldad_vicreg_ema"]
    assert "stop_gradient_targets: false" in configs["ldad_vicreg_online"]


def test_launcher_has_separate_axes_and_paired_delta_jepa_rows() -> None:
    launcher = (
        ROOT / "scripts/experiments/submit_controlled_objects_hwm.sh"
    ).read_text(encoding="utf-8")
    slurm = (ROOT / "scripts/slurm/run_controlled_objects_train.slurm").read_text(
        encoding="utf-8"
    )
    assert "DEPTHS=(1 2 3 4)" in launcher
    assert "STRIDES=(2 4 8)" in launcher
    assert "ROLLOUTS=(1 2 4 8)" in launcher
    assert "LAMBDAS=(0.75 0.9 0.95 1.0)" in launcher
    assert "for lambda in 0.75 0.9 0.95" in launcher
    assert "for all_levels in false true" in launcher
    assert "REPRESENTATIONS=(cls grid)" in launcher
    assert "LDAD_OBJECTIVES=(" in launcher
    assert 'if [[ "${JOB_COUNT}" -ne 72 ]]' in launcher
    assert "training.init_checkpoint" in slurm
    assert "training.train_from_level" in slurm
