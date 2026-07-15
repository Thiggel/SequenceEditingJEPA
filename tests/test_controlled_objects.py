from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from puzzle_jepa.controlled_objects import evaluation as controlled_evaluation
from puzzle_jepa.controlled_objects import probes as controlled_probes
from puzzle_jepa.controlled_objects.batching import build_controlled_dataset
from puzzle_jepa.controlled_objects.domain import RigidTransform
from puzzle_jepa.controlled_objects.generator import (
    ControlledObjectGenerator,
    ControlledObjectSpec,
)
from puzzle_jepa.controlled_objects.model import (
    ActionChunkEncoder,
    CausalLatentPredictor,
    ControlledObjectJEPA,
    ControlledStateEncoder,
)
from puzzle_jepa.controlled_objects.probes import run_controlled_object_probes
from puzzle_jepa.train.controlled_objects import _initialize_low_level


def _generator(*, objects: int = 4, horizon: int = 128) -> ControlledObjectGenerator:
    return ControlledObjectGenerator(
        ControlledObjectSpec(
            object_count=objects,
            trajectory_length=horizon,
            invalid_action_ratio=0.0,
            noop_ratio=0.0,
            require_state_change=True,
        )
    )


def _model(
    *,
    architecture: str = "transformer",
    spans: list[int] | None = None,
    rollout: int = 2,
    target_mode: str = "shared",
) -> ControlledObjectJEPA:
    return ControlledObjectJEPA(
        hidden_dim=16,
        level_spans=spans or [1],
        macro_dim=8,
        predictor_architecture=architecture,
        predictor_layers=1,
        predictor_heads=4,
        rollout_steps=rollout,
        rollout_all_levels=True,
        target_mode=target_mode,
        stop_gradient_targets=True,
        vicreg_weight=0.0,
    )


@pytest.mark.parametrize("object_count", [1, 2, 4, 8])
def test_deterministic_object_trajectory_uses_valid_rigid_motion_frames(
    object_count: int,
) -> None:
    generator = _generator(objects=object_count, horizon=32)
    trajectory = generator.sample_trajectory(np.random.default_rng(7))

    assert trajectory.scene.object_count == object_count
    assert len(set(trajectory.scene.colors.tolist())) == object_count
    changed = (trajectory.states[1:] != trajectory.states[:-1]).reshape(32, -1)
    assert np.all(changed.sum(axis=1) >= 2)
    initial_areas = {
        int(color): int(np.count_nonzero(trajectory.states[0] == color))
        for color in trajectory.scene.colors
    }
    for state in trajectory.states:
        assert {
            color: int(np.count_nonzero(state == color)) for color in initial_areas
        } == initial_areas
    for state, action_values, successor in zip(
        trajectory.states[:-1],
        trajectory.actions,
        trajectory.states[1:],
        strict=True,
    ):
        action = RigidTransform(*(int(value) for value in action_values))
        replayed, valid = generator.apply_action(state, action)
        assert valid
        np.testing.assert_array_equal(replayed, successor)


def test_rigid_action_rejects_background_and_updates_multiple_pixels() -> None:
    generator = _generator(objects=1, horizon=2)
    state = generator.sample_scene(np.random.default_rng(1)).grid
    row, col = (int(value) for value in np.argwhere(state == 0)[0])
    _, valid = generator.apply_action(state, RigidTransform(row, col, 1))
    assert not valid
    action = generator.candidate_actions(state)[0]
    edited, valid = generator.apply_action(state, action)
    assert valid
    assert np.count_nonzero(edited != state) >= 2


def test_hidden_motion_policy_is_not_a_visible_color_alias() -> None:
    generator = _generator(objects=4, horizon=2)
    scenes = [generator.sample_scene(np.random.default_rng(seed)) for seed in range(16)]
    assert any(
        not np.array_equal(scene.motion_ids, scene.colors - 1) for scene in scenes
    )


def test_dataset_samples_contiguous_windows() -> None:
    dataset = build_controlled_dataset(_generator(horizon=16), trajectory_count=4, seed=3)
    batch = dataset.sample_batch(
        np.random.default_rng(4), batch_size=3, horizon=8, device=torch.device("cpu")
    )
    assert batch.states.shape == (3, 9, 16, 16)
    assert batch.actions.shape == (3, 8, 3)
    assert torch.all(
        (batch.states[:, 1:] != batch.states[:, :-1]).flatten(2).sum(dim=2) >= 2
    )


def test_state_encoder_is_only_a_768_to_hidden_mlp() -> None:
    encoder = ControlledStateEncoder(grid_size=16, num_colors=10, hidden_dim=32)
    assert isinstance(encoder.mlp[0], nn.Linear)
    assert encoder.mlp[0].in_features == 16 * 16 * 3 == 768
    assert encoder.mlp[0].out_features == 32
    assert not any(isinstance(module, nn.TransformerEncoder) for module in encoder.modules())
    output = encoder(torch.zeros(2, 16, 16, dtype=torch.long))
    assert output.shape == (2, 32)


def test_action_chunk_is_ordered_nonlinear_eight_dimensional_bottleneck() -> None:
    torch.manual_seed(2)
    encoder = ActionChunkEncoder(
        grid_size=16, num_action_types=7, chunk_length=4, macro_dim=8
    )
    assert isinstance(encoder.project, nn.Sequential)
    assert encoder.project[0].in_features == 4 * (16 + 16 + 7)
    assert encoder.project[-1].out_features == 8
    actions = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 2], [9, 3, 1]]])
    assert not torch.allclose(encoder(actions), encoder(actions.flip(1)))


@pytest.mark.parametrize("architecture", ["transformer", "gated_deltanet", "lstm"])
def test_predictors_are_causal_and_backpropagate(architecture: str) -> None:
    torch.manual_seed(5)
    predictor = CausalLatentPredictor(
        latent_dim=16,
        macro_dim=8,
        architecture=architecture,
        num_layers=1,
        num_heads=4,
        max_context=8,
    )
    states = torch.randn(2, 4, 16, requires_grad=True)
    macros = torch.randn(2, 4, 8, requires_grad=True)
    changed_states = states.detach().clone()
    changed_macros = macros.detach().clone()
    changed_states[:, 2:] += 100.0
    changed_macros[:, 2:] -= 100.0
    output = predictor(states, macros)
    changed = predictor(changed_states, changed_macros)
    torch.testing.assert_close(output[:, :2], changed[:, :2])
    output.square().mean().backward()
    assert states.grad is not None and torch.isfinite(states.grad).all()
    assert macros.grad is not None and torch.isfinite(macros.grad).all()


@pytest.mark.parametrize(
    ("spans", "rollout", "required"),
    [([1], 4, 4), ([1, 10], 4, 40), ([1, 10, 100], 4, 400)],
)
def test_hierarchy_schedules_have_expected_horizon(
    spans: list[int], rollout: int, required: int
) -> None:
    model = _model(spans=spans, rollout=rollout)
    assert model.level_spans == tuple(spans)
    assert model.required_horizon == required


@pytest.mark.parametrize("architecture", ["transformer", "gated_deltanet", "lstm"])
def test_every_hierarchy_level_gets_teacher_forcing_and_dense_rollout_loss(
    architecture: str,
) -> None:
    model = _model(architecture=architecture, spans=[1, 4], rollout=2)
    dataset = build_controlled_dataset(_generator(horizon=8), trajectory_count=4, seed=8)
    batch = dataset.sample_batch(
        np.random.default_rng(9), batch_size=2, horizon=8
    )
    output = model(batch)
    assert len(output.level_losses) == 2
    assert all(prediction.shape == (2, 2, 16) for prediction in output.predictions)
    assert all(
        prediction.shape == (2, 2, 16)
        for prediction in output.teacher_forced_predictions
    )
    assert output.teacher_forcing_loss.requires_grad
    assert output.rollout_loss.requires_grad
    output.loss.backward()
    assert all(
        parameter.grad is not None
        for parameter in model.dynamics[1].parameters()
        if parameter.requires_grad
    )


def test_joint_hierarchy_every_level_backpropagates_into_shared_encoder() -> None:
    model = _model(spans=[1, 2, 4], rollout=1, target_mode="ema")
    batch = build_controlled_dataset(
        _generator(horizon=4), trajectory_count=4, seed=18
    ).sample_batch(np.random.default_rng(19), batch_size=2, horizon=4)
    output = model(batch)
    for level, loss in enumerate(output.level_losses):
        model.zero_grad(set_to_none=True)
        loss.backward(retain_graph=level + 1 < len(output.level_losses))
        encoder_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.encoder.parameters()
            if parameter.grad is not None
        )
        assert encoder_gradient > 0.0, f"level {level} did not update the encoder"


def test_rollout_weighting_is_lambda_power_normalized() -> None:
    model = ControlledObjectJEPA(
        hidden_dim=16,
        level_spans=[1],
        rollout_steps=4,
        rollout_lambda=0.9,
        predictor_layers=1,
        predictor_heads=4,
        target_mode="shared",
        stop_gradient_targets=True,
        vicreg_weight=0.0,
    )
    batch = build_controlled_dataset(
        _generator(horizon=4), trajectory_count=2, seed=10
    ).sample_batch(np.random.default_rng(11), batch_size=2, horizon=4)
    weights = model(batch).rollout_weights[0]
    expected = torch.tensor([1.0, 0.9, 0.9**2, 0.9**3])
    torch.testing.assert_close(weights, expected / expected.sum())


def test_staged_hierarchy_initialization_copies_and_freezes_lower_levels(
    tmp_path: Path,
) -> None:
    source = _model(spans=[1], rollout=1)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model": source.state_dict()}, checkpoint)
    target = _model(spans=[1, 4], rollout=1)
    _initialize_low_level(target, checkpoint)
    target.freeze_below_level(1)
    for source_value, target_value in zip(
        source.encoder.parameters(), target.encoder.parameters(), strict=True
    ):
        torch.testing.assert_close(source_value, target_value)
        assert not target_value.requires_grad
    assert all(not parameter.requires_grad for parameter in target.dynamics[0].parameters())
    assert all(parameter.requires_grad for parameter in target.dynamics[1].parameters())


def test_three_level_stage_rejects_checkpoint_missing_middle_level(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model": _model(spans=[1], rollout=1).state_dict()}, checkpoint)
    with pytest.raises(ValueError, match="action_encoders.1"):
        _initialize_low_level(_model(spans=[1, 4, 16], rollout=1), checkpoint)


def test_ema_target_encoder_is_frozen_and_updated() -> None:
    model = _model(target_mode="ema")
    assert model.target_encoder is not None
    before = [parameter.clone() for parameter in model.target_encoder.parameters()]
    with torch.no_grad():
        next(model.encoder.parameters()).add_(1.0)
    model.update_target_encoder()
    assert all(not parameter.requires_grad for parameter in model.target_encoder.parameters())
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, model.target_encoder.parameters(), strict=True)
    )


def test_vicreg_uses_separate_adjusted_variance_and_covariance_coefficients() -> None:
    model = ControlledObjectJEPA(
        hidden_dim=16,
        level_spans=[1],
        rollout_steps=1,
        predictor_layers=1,
        predictor_heads=4,
        target_mode="ema",
        stop_gradient_targets=True,
        vicreg_weight=1.0,
        vicreg_variance_weight=3.0,
        vicreg_covariance_weight=5.0,
        vicreg_adjust_cov=True,
    )
    batch = build_controlled_dataset(
        _generator(horizon=2), trajectory_count=8, seed=10
    ).sample_batch(np.random.default_rng(11), batch_size=8, horizon=1)
    output = model(batch)
    torch.testing.assert_close(
        output.vicreg_loss,
        3.0 * output.vicreg_variance_loss + 5.0 * output.vicreg_covariance_loss,
    )


def test_sigreg_is_finite_and_updates_the_encoder() -> None:
    model = ControlledObjectJEPA(
        hidden_dim=16,
        level_spans=[1, 2],
        rollout_steps=1,
        predictor_layers=1,
        predictor_heads=4,
        target_mode="shared",
        stop_gradient_targets=False,
        vicreg_weight=0.0,
        sigreg_weight=0.05,
        sigreg_num_slices=8,
        sigreg_num_points=5,
    )
    batch = build_controlled_dataset(
        _generator(horizon=2), trajectory_count=8, seed=20
    ).sample_batch(np.random.default_rng(21), batch_size=8, horizon=2)
    output = model(batch)
    assert torch.isfinite(output.sigreg_loss)
    assert float(output.sigreg_loss.detach()) > 0.0
    output.loss.backward()
    assert all(parameter.grad is not None for parameter in model.encoder.parameters())


def test_ldad_uses_rigid_transform_action_classes() -> None:
    model = ControlledObjectJEPA(
        hidden_dim=16,
        level_spans=[1],
        rollout_steps=1,
        predictor_layers=1,
        predictor_heads=4,
        target_mode="ema",
        stop_gradient_targets=True,
        vicreg_weight=0.0,
        ldad_weight=1.0,
    )
    batch = build_controlled_dataset(
        _generator(horizon=2), trajectory_count=2, seed=10
    ).sample_batch(np.random.default_rng(11), batch_size=2, horizon=1)
    output = model(batch)
    assert output.ldad_logits is not None
    assert output.ldad_logits[2].shape == (2, 7)


def test_probe_suite_covers_all_properties_and_pixel_reconstruction() -> None:
    metrics = run_controlled_object_probes(
        _model(architecture="lstm", spans=[1, 4], rollout=2),
        _generator(objects=2, horizon=8),
        seed=12,
        train_samples=16,
        eval_samples=8,
        batch_size=8,
        device=torch.device("cpu"),
        steps=1,
        learning_rate=1.0e-3,
    )
    assert metrics["probe_schema"] == "controlled_objects_v5"
    assert metrics["probe_motion_policy_interpretation"] == (
        "unobservable_single_frame_negative_control"
    )
    required = {
        "probe_object_count_balanced_acc",
        "probe_object_presence_balanced_acc",
        "probe_shape_balanced_acc",
        "probe_motion_policy_balanced_acc",
        "probe_position_r2",
        "probe_area_r2",
        "probe_relation_r2",
        "probe_delta_action_row_balanced_acc",
        "probe_delta_action_col_balanced_acc",
        "probe_delta_action_transform_balanced_acc",
        "probe_pixel_decoder_acc",
        "probe_pixel_decoder_foreground_iou",
        "probe_level1_rollout2_pixel_decoder_acc",
    }
    assert required <= metrics.keys()
    assert all(np.isfinite(float(metrics[name])) for name in required)


def test_masked_regression_standardizes_tiny_targets_before_optimization() -> None:
    torch.manual_seed(22)
    train_x = torch.randn(128, 4)
    train_y = torch.stack(
        (
            0.02 + 0.001 * train_x[:, 0],
            0.03 - 0.002 * train_x[:, 1],
        ),
        dim=1,
    ).unsqueeze(-1)
    mask = torch.ones(128, 2, dtype=torch.bool)
    standardized, _, _ = controlled_probes._standardize_masked_targets(train_y, mask)
    torch.testing.assert_close(
        standardized.mean(dim=0), torch.zeros(2, 1), atol=1.0e-5, rtol=0.0
    )
    torch.testing.assert_close(
        standardized.std(dim=0, unbiased=False),
        torch.ones(2, 1),
        atol=1.0e-4,
        rtol=0.0,
    )
    value, _ = controlled_probes._fit_masked_regressor(
        train_x,
        train_y,
        mask,
        train_x,
        train_y,
        mask,
        transfer_x=None,
        transfer_y=None,
        transfer_mask=None,
        steps=200,
        learning_rate=0.03,
    )
    assert value > 0.98


def test_symbolic_pixel_planner_exactly_reaches_short_goal() -> None:
    generator = _generator(objects=1, horizon=4)
    trajectory = generator.sample_trajectory(np.random.default_rng(13))
    planned = controlled_evaluation._symbolic_receding_plan(
        generator,
        trajectory.states[0],
        trajectory.states[-1],
        max_depth=4,
        beam_width=4,
    )
    np.testing.assert_array_equal(planned, trajectory.states[-1])


def test_hierarchical_planners_return_atomic_actions() -> None:
    generator = _generator(objects=1, horizon=8)
    dataset = build_controlled_dataset(generator, trajectory_count=8, seed=14)
    model = _model(architecture="lstm", spans=[1, 4], rollout=1)
    trajectory = generator.sample_trajectory(np.random.default_rng(15), horizon=4)
    target = model.encode(torch.as_tensor(trajectory.states[-1:]), target=True)
    supports = {
        level: controlled_evaluation._estimate_macro_support(
            model,
            dataset,
            level=level,
            seed=16 + level,
            sample_count=8,
            device=torch.device("cpu"),
        )
        for level in range(2)
    }
    action = controlled_evaluation._recursive_on_support_action(
        model,
        generator,
        trajectory.states[0],
        target,
        np.random.default_rng(17),
        candidates=2,
        device=torch.device("cpu"),
        supports=supports,
    )
    assert isinstance(action, RigidTransform)


def test_mppi_performs_every_requested_update() -> None:
    class CountingDynamics:
        def __init__(self) -> None:
            self.calls = 0

        def rollout(self, initial: torch.Tensor, macros: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            padded = torch.nn.functional.pad(macros, (0, initial.shape[-1] - macros.shape[-1]))
            return initial[:, None] + padded.cumsum(dim=1)

    dynamics = CountingDynamics()

    class FakeModel:
        def __init__(self) -> None:
            self.dynamics = [None, dynamics]

    latent_dim = 16
    macro_dim = 8
    support = controlled_evaluation.MacroSupport(
        state_bank=torch.zeros(4, latent_dim),
        bank=torch.zeros(4, macro_dim),
        action_bank=torch.zeros(4, 1, 3, dtype=torch.long),
        lower=torch.full((macro_dim,), -2.0),
        upper=torch.full((macro_dim,), 2.0),
        state_mean=torch.zeros(latent_dim),
        state_std=torch.ones(latent_dim),
        mean=torch.zeros(macro_dim),
        std=torch.ones(macro_dim),
    )
    controlled_evaluation._mppi_macro_sequence(
        FakeModel(),
        torch.zeros(1, latent_dim),
        torch.ones(1, latent_dim),
        level=1,
        transition_count=2,
        support=support,
        candidates=8,
        iterations=3,
        support_weight=0.0,
        torch_rng=torch.Generator().manual_seed(3),
    )
    assert dynamics.calls == 4


def test_conditional_support_energy_scores_macro_not_state_dimension() -> None:
    torch.manual_seed(23)
    state_bank = torch.randn(32, 256)
    macro_bank = torch.randn(32, 8)
    support = controlled_evaluation.MacroSupport(
        state_bank=state_bank,
        bank=macro_bank,
        action_bank=torch.zeros(32, 1, 3, dtype=torch.long),
        lower=macro_bank.quantile(0.02, dim=0),
        upper=macro_bank.quantile(0.98, dim=0),
        state_mean=state_bank.mean(dim=0),
        state_std=state_bank.std(dim=0, unbiased=False).clamp_min(1.0e-3),
        mean=macro_bank.mean(dim=0),
        std=macro_bank.std(dim=0, unbiased=False).clamp_min(1.0e-3),
    )
    on = controlled_evaluation._conditional_macro_energy(
        macro_bank, state_bank, support
    )
    off = controlled_evaluation._conditional_macro_energy(
        macro_bank + 5.0 * support.std, state_bank, support
    )
    torch.testing.assert_close(on, torch.zeros_like(on))
    assert float(off.mean()) > 4.0


def test_vicreg_grid_has_only_two_regularizer_axes_and_three_seeds(
    tmp_path: Path,
) -> None:
    repo = Path(__file__).resolve().parents[1]
    env = {
        "SUBMIT": "0",
        "MAX_STEPS": "1",
        "SWEEP_NAME": "pytest_controlled_vicreg_hwm",
        "PUZZLE_JEPA_WORK_ROOT": str(tmp_path),
    }
    completed = subprocess.run(
        ["bash", "scripts/experiments/submit_controlled_objects_vicreg_hwm.sh"],
        cwd=repo,
        env={**os.environ, **env},
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Final comparison cells: 48" in completed.stdout
    manifest_line = next(
        line for line in completed.stdout.splitlines() if line.startswith("Final comparison")
    )
    manifest = Path(manifest_line.split("Task manifest: ", maxsplit=1)[1])
    rows = manifest.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 49
    values = [row.split("\t") for row in rows[1:]]
    assert {row[1] for row in values} == {"0.05", "1", "10", "29.409"}
    assert {row[2] for row in values} == {"0.1", "1", "10", "17.866"}
    assert {row[3] for row in values} == {"1707", "2707", "3707"}


def test_joint_objective_gate_has_twelve_objectives_and_three_seeds(
    tmp_path: Path,
) -> None:
    repo = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        ["bash", "scripts/experiments/submit_controlled_objects_joint_objectives.sh"],
        cwd=repo,
        env={
            **os.environ,
            "SUBMIT": "0",
            "MAX_STEPS": "1",
            "SWEEP_NAME": "pytest_controlled_joint_hwm",
            "PUZZLE_JEPA_WORK_ROOT": str(tmp_path),
        },
        check=True,
        capture_output=True,
        text=True,
    )
    assert "36 joint [1,10,100] trainers" in completed.stdout
    manifest_line = next(
        line for line in completed.stdout.splitlines() if line.startswith("Task manifest")
    )
    rows = Path(manifest_line.split(": ", maxsplit=1)[1]).read_text(
        encoding="utf-8"
    ).splitlines()
    values = [row.split("\t") for row in rows[1:]]
    assert len(values) == 36
    assert len({row[1] for row in values}) == 12
    assert {row[2] for row in values} == {"1707", "2707", "3707"}


def test_slurm_training_keeps_hydra_metadata_off_home_filesystem() -> None:
    repo = Path(__file__).resolve().parents[1]
    script = (repo / "scripts/slurm/run_controlled_objects_train.slurm").read_text(
        encoding="utf-8"
    )
    assert '"hydra.run.dir=${OUTPUT_ROOT}/${RUN_NAME}/hydra"' in script


def test_checkpoint_config_is_json_serializable() -> None:
    model = _model(spans=[1, 2, 4], rollout=8)
    payload = {
        "level_spans": list(model.level_spans),
        "architecture": model.predictor_architecture,
        "required_horizon": model.required_horizon,
    }
    assert json.loads(json.dumps(payload))["level_spans"] == [1, 2, 4]
