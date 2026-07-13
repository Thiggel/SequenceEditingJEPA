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
from puzzle_jepa.controlled_objects.batching import build_controlled_dataset
from puzzle_jepa.controlled_objects.domain import PixelEdit
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
def test_deterministic_object_trajectory_uses_atomic_pixel_edits(
    object_count: int,
) -> None:
    generator = _generator(objects=object_count, horizon=32)
    trajectory = generator.sample_trajectory(np.random.default_rng(7))

    assert trajectory.scene.object_count == object_count
    assert len(set(trajectory.scene.colors.tolist())) == object_count
    assert len(set(trajectory.scene.motion_ids.tolist())) == object_count
    changed = (trajectory.states[1:] != trajectory.states[:-1]).reshape(32, -1)
    assert np.all(changed.sum(axis=1) == 1)
    for state, action_values, successor in zip(
        trajectory.states[:-1],
        trajectory.actions,
        trajectory.states[1:],
        strict=True,
    ):
        action = PixelEdit(*(int(value) for value in action_values))
        replayed, valid = generator.apply_action(state, action)
        assert valid
        np.testing.assert_array_equal(replayed, successor)


def test_pixel_edit_rejects_noop_and_changes_exactly_one_cell() -> None:
    generator = _generator(objects=1, horizon=2)
    state = generator.sample_scene(np.random.default_rng(1)).grid
    row, col = (int(value) for value in np.argwhere(state == 0)[0])
    edited, valid = generator.apply_action(state, PixelEdit(row, col, 1))
    assert valid
    assert np.count_nonzero(edited != state) == 1
    _, valid = generator.apply_action(edited, PixelEdit(row, col, 1))
    assert not valid


def test_dataset_samples_contiguous_windows() -> None:
    dataset = build_controlled_dataset(_generator(horizon=16), trajectory_count=4, seed=3)
    batch = dataset.sample_batch(
        np.random.default_rng(4), batch_size=3, horizon=8, device=torch.device("cpu")
    )
    assert batch.states.shape == (3, 9, 16, 16)
    assert batch.actions.shape == (3, 8, 3)
    assert torch.all(
        (batch.states[:, 1:] != batch.states[:, :-1]).flatten(2).sum(dim=2) == 1
    )


def test_state_encoder_is_only_a_768_to_hidden_mlp() -> None:
    encoder = ControlledStateEncoder(grid_size=16, num_colors=10, hidden_dim=32)
    assert isinstance(encoder.mlp[0], nn.Linear)
    assert encoder.mlp[0].in_features == 16 * 16 * 3 == 768
    assert encoder.mlp[0].out_features == 32
    assert not any(isinstance(module, nn.TransformerEncoder) for module in encoder.modules())
    output = encoder(torch.zeros(2, 16, 16, dtype=torch.long))
    assert output.shape == (2, 32)


def test_action_chunk_is_ordered_concat_plus_one_linear_projection() -> None:
    torch.manual_seed(2)
    encoder = ActionChunkEncoder(
        grid_size=16, num_colors=10, chunk_length=4, macro_dim=8
    )
    assert isinstance(encoder.project, nn.Linear)
    assert encoder.project.in_features == 4 * (16 + 16 + 10)
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
    [([1], 8, 8), ([1, 4], 8, 32), ([1, 4, 16], 8, 128), ([1, 2, 4], 8, 32)],
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
    assert metrics["probe_schema"] == "controlled_objects_v3"
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
        "probe_delta_action_color_balanced_acc",
        "probe_pixel_decoder_acc",
        "probe_pixel_decoder_foreground_iou",
        "probe_level1_rollout2_pixel_decoder_acc",
    }
    assert required <= metrics.keys()
    assert all(np.isfinite(float(metrics[name])) for name in required)


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
    assert isinstance(action, PixelEdit)


def test_new_grid_has_exact_axis_product_and_no_transformer_encoder() -> None:
    repo = Path(__file__).resolve().parents[1]
    env = {
        "SUBMIT": "0",
        "MAX_STEPS": "1",
        "SWEEP_NAME": "pytest_controlled_mlp_grid",
    }
    completed = subprocess.run(
        ["bash", "scripts/experiments/submit_controlled_objects_mlp_grid.sh"],
        cwd=repo,
        env={**os.environ, **env},
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Final comparison cells: 1152" in completed.stdout
    manifest_line = next(
        line for line in completed.stdout.splitlines() if line.startswith("Final comparison")
    )
    manifest = Path(manifest_line.split("Task manifest: ", maxsplit=1)[1])
    rows = manifest.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 289
    values = [row.split("\t") for row in rows[1:]]
    assert {row[1] for row in values} == {"transformer", "gated_deltanet", "lstm"}
    assert {row[2] for row in values} == {"1", "2", "4", "8"}
    assert {row[3] for row in values} == {"weighted", "unweighted"}
    assert {row[5] for row in values} == {"1", "2", "4", "8"}


def test_checkpoint_config_is_json_serializable() -> None:
    model = _model(spans=[1, 2, 4], rollout=8)
    payload = {
        "level_spans": list(model.level_spans),
        "architecture": model.predictor_architecture,
        "required_horizon": model.required_horizon,
    }
    assert json.loads(json.dumps(payload))["level_spans"] == [1, 2, 4]
