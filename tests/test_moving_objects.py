from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from puzzle_jepa.moving_objects.batching import _pair_distance, sample_moving_object_batch
from puzzle_jepa.moving_objects.generator import MovingObjectGenerator, MovingObjectSpec
from puzzle_jepa.moving_objects.model import MovingObjectJEPA
from puzzle_jepa.moving_objects.probes import run_moving_object_probes
from puzzle_jepa.moving_objects.probes import run_moving_object_dynamics_diagnostics
from scripts.analysis.analyze_moving_objects import KEYS, analyze


ROOT = Path(__file__).resolve().parents[1]


def test_reflected_motion_is_collision_free_and_keeps_attributes() -> None:
    generator = MovingObjectGenerator(
        MovingObjectSpec(grid_size=12, min_objects=3, max_objects=3, sequence_length=8)
    )
    trajectory = generator.sample_trajectory(np.random.default_rng(7))
    assert trajectory.states.shape == (8, 12, 12)
    assert trajectory.object_count == 3
    assert np.all((trajectory.object_maps >= -1) & (trajectory.object_maps < 3))
    assert np.all(np.count_nonzero(trajectory.object_maps == 0, axis=(1, 2)) > 0)
    assert trajectory.shape_ids.shape == trajectory.colors.shape == (3,)
    assert trajectory.angular_velocities.shape == (8, 3)
    assert len(set(trajectory.colors.tolist())) == 3
    assert np.any(trajectory.positions[1:] != trajectory.positions[:-1])


def test_motion_batch_uses_two_frames_and_probeable_object_load() -> None:
    generator = MovingObjectGenerator(
        MovingObjectSpec(grid_size=10, min_objects=1, max_objects=3, sequence_length=8)
    )
    batch = sample_moving_object_batch(
        generator, np.random.default_rng(11), batch_size=12, horizon=3
    )
    assert batch.contexts.shape == (12, 2, 10, 10)
    assert batch.future_contexts.shape == (12, 3, 2, 10, 10)
    assert set(batch.object_count.tolist()).issubset({1, 2, 3})
    assert batch.shape_counts.shape == (12, 5)
    assert batch.velocity_counts.shape == (12, 24)
    assert batch.relations.shape == (12, 5)
    assert batch.future_velocity_counts.shape == (12, 24)
    assert batch.angular_velocity_counts.shape == (12, 3)
    assert batch.future_angular_velocity_counts.shape == (12, 3)
    assert batch.future_relations.shape == (12, 5)


def test_collision_retries_do_not_bias_away_from_requested_high_load() -> None:
    generator = MovingObjectGenerator(
        MovingObjectSpec(grid_size=16, min_objects=8, max_objects=8, sequence_length=8)
    )
    trajectory = generator.sample_trajectory(np.random.default_rng(12))
    assert trajectory.object_count == 8


def test_rotating_motion_preserves_identity_and_exposes_angular_velocity() -> None:
    generator = MovingObjectGenerator(
        MovingObjectSpec(
            grid_size=12,
            min_objects=4,
            max_objects=4,
            sequence_length=8,
            rotate_objects=True,
        )
    )
    trajectory = generator.sample_trajectory(np.random.default_rng(29))
    assert set(np.unique(trajectory.angular_velocities)).issubset({-1, 1})
    assert np.all(trajectory.angular_velocities == trajectory.angular_velocities[0])
    assert trajectory.states.shape == trajectory.object_maps.shape
    assert all(np.count_nonzero(frame) > 0 for frame in trajectory.states)


def test_wrapped_relations_use_shortest_toroidal_distance() -> None:
    left = np.asarray([0, 0])
    right = np.asarray([0, 15])
    assert _pair_distance(left, right, 16, "reflect") == 15.0
    assert _pair_distance(left, right, 16, "wrap") == 1.0


def test_latent_dim_is_a_projection_not_the_visual_token_width() -> None:
    small = MovingObjectJEPA(grid_size=8, token_dim=32, latent_dim=2, encoder_layers=1, encoder_heads=4)
    wide = MovingObjectJEPA(grid_size=8, token_dim=32, latent_dim=16, encoder_layers=1, encoder_heads=4)
    contexts = torch.zeros(3, 2, 8, 8, dtype=torch.long)
    assert small.encode(contexts).shape == (3, 2)
    assert wide.encode(contexts).shape == (3, 16)
    assert small.encoder.token_dim == wide.encoder.token_dim == 32
    assert not hasattr(small, "latent_representation")


def test_motion_jepa_forward_backward_and_frozen_probes() -> None:
    generator = MovingObjectGenerator(
        MovingObjectSpec(grid_size=8, min_objects=1, max_objects=2, sequence_length=7)
    )
    batch = sample_moving_object_batch(generator, np.random.default_rng(13), batch_size=4, horizon=2)
    model = MovingObjectJEPA(
        grid_size=8, token_dim=16, latent_dim=4, encoder_layers=1, encoder_heads=4, rollout_horizon=2
    )
    output = model(batch)
    output.loss.backward()
    assert output.predictions.shape == (4, 2, 4)
    assert model.encoder.project[1].weight.grad is not None
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())

    metrics = run_moving_object_probes(
        model,
        generator,
        np.random.default_rng(17),
        train_samples=16,
        eval_samples=8,
        batch_size=8,
        device=torch.device("cpu"),
        steps=1,
        learning_rate=1.0e-2,
    )
    for key in (
        "probe_object_count_balanced_acc",
        "probe_shape_count_mae",
        "probe_shape_count_r2",
        "probe_velocity_count_mae",
        "probe_velocity_count_r2",
        "probe_relations_mae",
        "probe_grid_foreground_iou",
    ):
        assert np.isfinite(metrics[key])

    diagnostics = run_moving_object_dynamics_diagnostics(
        model,
        generator,
        np.random.default_rng(19),
        samples=8,
        batch_size=4,
        device=torch.device("cpu"),
    )
    assert diagnostics["dynamics_pixel_change_rate"] > 0.0
    assert np.isfinite(diagnostics["dynamics_prediction_gain_fraction"])


def test_temporal_delta_objective_forces_nonconstant_online_differences() -> None:
    generator = MovingObjectGenerator(
        MovingObjectSpec(grid_size=8, min_objects=2, max_objects=2, sequence_length=7)
    )
    batch = sample_moving_object_batch(generator, np.random.default_rng(23), batch_size=8, horizon=1)
    model = MovingObjectJEPA(
        grid_size=8,
        token_dim=16,
        latent_dim=4,
        encoder_layers=1,
        encoder_heads=4,
        rollout_horizon=1,
        temporal_delta_weight=0.1,
    )
    output = model(batch)
    output.loss.backward()
    assert float(output.temporal_delta_loss.detach()) > 0.0
    assert model.encoder.project[1].weight.grad is not None


def test_new_sweep_is_single_cls_only_and_crosses_requested_axes() -> None:
    submit = (ROOT / "scripts/experiments/submit_moving_objects_bottleneck.sh").read_text()
    slurm = (ROOT / "scripts/slurm/run_moving_objects_train.slurm").read_text()
    model = (ROOT / "configs/moving_objects/model/cls_bottleneck.yaml").read_text()
    assert "LATENT_DIMS=(2 4 8 16 32 64)" in submit
    assert "MAX_OBJECT_COUNTS=(1 2 4 6 8)" in submit
    assert "SEEDS=(1707 2707 3707)" in submit
    assert "grid128" not in submit + slurm + model
    assert "latent_representation" not in submit + slurm + model
    assert "Retired:" in (ROOT / "scripts/experiments/submit_object_dynamics_phase1.sh").read_text()
    assert "Retired:" in (ROOT / "scripts/experiments/submit_object_dynamics_trajectory_gate.sh").read_text()


def test_six_hour_watcher_is_configured() -> None:
    watcher = (ROOT / "scripts/experiments/submit_moving_objects_oversight.sh").read_text()
    assert 'CADENCE_HOURS="${CADENCE_HOURS:-6}"' in watcher
    assert "--begin=" in watcher


def test_temporal_gate_keeps_single_cls_and_selected_axes() -> None:
    script = (ROOT / "scripts/experiments/submit_moving_objects_temporal.sh").read_text()
    assert "LATENT_DIMS=(4 8 16 32)" in script
    assert "MAX_OBJECT_COUNTS=(4 8)" in script
    assert "SEEDS=(1707 2707 3707)" in script
    assert "ema_vicreg_temporal" in script
    assert "grid" not in script.lower()


def test_transfer_gate_pairs_base_and_temporal_single_cls_rows() -> None:
    script = (ROOT / "scripts/experiments/submit_moving_objects_transfer.sh").read_text()
    assert "DATASETS=(wrapped_motion rotating_motion)" in script
    assert "OBJECTIVES=(ema_vicreg ema_vicreg_temporal)" in script
    assert "SEEDS=(1707 2707 3707)" in script
    assert "LATENT_DIM=4" in script
    assert "MAX_OBJECTS=8" in script
    assert "grid" not in script.lower()


def test_analyzer_keeps_trajectory_objective_and_bottleneck_axes_separate(tmp_path: Path) -> None:
    run = tmp_path / "motion_n4_z8_test_seed1707"
    run.mkdir()
    initial = {
        "step": 0, "data": "reflected_motion", "objective": "ema_vicreg",
        "latent_dim": 8, "max_objects": 4, "seed": 1707,
    }
    final = {**initial, "step": 5000}
    for index, key in enumerate(KEYS):
        initial[key] = float(index)
        final[key] = float(index) + 0.25
    (run / "metrics.jsonl").write_text("\n".join((json.dumps(initial), json.dumps(final))))

    transfer = tmp_path / "motion_n4_z8_transfer_seed1707"
    transfer.mkdir()
    transfer_initial = {**initial, "data": "wrapped_motion", "objective": "ema_vicreg_temporal"}
    transfer_final = {**final, "data": "wrapped_motion", "objective": "ema_vicreg_temporal"}
    (transfer / "metrics.jsonl").write_text(
        "\n".join((json.dumps(transfer_initial), json.dumps(transfer_final)))
    )

    summary = analyze(tmp_path, {run.name, transfer.name})

    assert len(summary["runs"]) == 2
    assert len(summary["aggregates"]) == 2
    assert {(row["data"], row["objective"]) for row in summary["aggregates"]} == {
        ("reflected_motion", "ema_vicreg"),
        ("wrapped_motion", "ema_vicreg_temporal"),
    }
    assert summary["aggregates"][0]["latent_dim"] == 8
    assert summary["aggregates"][0]["max_objects"] == 4
    assert summary["aggregates"][0]["delta"][KEYS[0]]["mean"] == 0.25
    assert analyze(tmp_path, {"another_run"})["runs"] == []
