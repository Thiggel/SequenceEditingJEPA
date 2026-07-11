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
from puzzle_jepa.moving_objects.probes import _fit_slot_regressor
from scripts.analysis.analyze_moving_objects import KEYS, _manifest_run_names, analyze


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
    assert batch.object_slot_features.shape == (12, 9, 11)
    assert batch.future_object_slot_features.shape == (12, 9, 11)
    assert torch.equal(batch.object_slot_present.sum(dim=1), batch.object_count)
    assert torch.allclose(
        batch.object_slot_features[..., :5].sum(dim=-1)[batch.object_slot_present],
        torch.ones_like(batch.object_slot_features[..., 0][batch.object_slot_present]),
    )


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


def test_construction_trajectory_families_preserve_objects_and_progress() -> None:
    kinds = (
        "object_blocked",
        "frontier_build",
        "random_within_object",
        "interleaved_build",
        "global_random",
        "completion",
        "noisy_repair",
    )
    for index, kind in enumerate(kinds):
        generator = MovingObjectGenerator(
            MovingObjectSpec(
                grid_size=16,
                min_objects=4,
                max_objects=4,
                trajectory_kind=kind,
            )
        )
        trajectory = generator.sample_trajectory(np.random.default_rng(101 + index))
        assert trajectory.kind == kind
        assert trajectory.object_count == 4
        assert len(trajectory.states) >= 6
        assert np.all(trajectory.velocities == 0)
        assert np.all(trajectory.angular_velocities == 0)
        assert np.all(np.diff(trajectory.completion, axis=0) >= 0.0)
        assert np.allclose(trajectory.completion[-1], 1.0)
        changed = np.count_nonzero(trajectory.states[1:] != trajectory.states[:-1], axis=(1, 2))
        assert np.all(changed == 1)


def test_construction_batch_has_visible_count_completion_and_no_fake_motion() -> None:
    generator = MovingObjectGenerator(
        MovingObjectSpec(
            grid_size=16,
            min_objects=8,
            max_objects=8,
            trajectory_kind="interleaved_build",
        )
    )
    batch = sample_moving_object_batch(
        generator, np.random.default_rng(131), batch_size=16, horizon=4
    )
    assert torch.all(batch.object_count == 8)
    assert torch.all((batch.visible_object_count >= 1) & (batch.visible_object_count <= 8))
    assert torch.all((batch.future_visible_object_count >= 1) & (batch.future_visible_object_count <= 8))
    assert batch.completion_features.shape == (16, 5)
    assert torch.all((batch.completion_features >= 0.0) & (batch.completion_features <= 1.0))
    assert torch.all(
        (batch.future_completion_features >= 0.0) & (batch.future_completion_features <= 1.0)
    )
    assert torch.count_nonzero(batch.velocity_counts) == 0
    assert torch.count_nonzero(batch.angular_velocity_counts) == 0
    assert torch.equal(batch.object_slot_present.sum(dim=1), batch.visible_object_count)
    assert torch.equal(
        batch.future_object_slot_present.sum(dim=1), batch.future_visible_object_count
    )


def test_construction_ordering_regimes_are_behaviorally_distinct() -> None:
    trajectories = {}
    for index, kind in enumerate(
        ("object_blocked", "random_within_object", "interleaved_build", "frontier_build")
    ):
        generator = MovingObjectGenerator(
            MovingObjectSpec(
                grid_size=16,
                min_objects=4,
                max_objects=4,
                trajectory_kind=kind,
            )
        )
        trajectories[kind] = generator.sample_trajectory(np.random.default_rng(211 + index))

    for kind in ("object_blocked", "random_within_object"):
        ids = _changed_object_ids(trajectories[kind])
        segments = [ids[0]] + [current for previous, current in zip(ids, ids[1:]) if current != previous]
        assert len(segments) == len(set(segments)) == 4

    interleaved_ids = _changed_object_ids(trajectories["interleaved_build"])
    assert len(set(interleaved_ids[:4])) == 4
    assert interleaved_ids[0] == interleaved_ids[4]

    frontier = trajectories["frontier_build"]
    for step in range(1, len(frontier.states)):
        changed = np.argwhere(frontier.states[step] != frontier.states[step - 1])
        row, col = (int(value) for value in changed[0])
        object_id = int(frontier.object_maps[step, row, col])
        previous_cells = np.argwhere(frontier.object_maps[step - 1] == object_id)
        if len(previous_cells):
            assert np.any(np.max(np.abs(previous_cells - np.asarray([row, col])), axis=1) == 1)


def test_completion_and_repair_start_with_all_objects_visible_but_incomplete() -> None:
    for index, kind in enumerate(("completion", "noisy_repair")):
        generator = MovingObjectGenerator(
            MovingObjectSpec(
                grid_size=16,
                min_objects=5,
                max_objects=5,
                trajectory_kind=kind,
            )
        )
        trajectory = generator.sample_trajectory(np.random.default_rng(251 + index))
        assert len(np.unique(trajectory.object_maps[0][trajectory.object_maps[0] >= 0])) == 5
        assert np.all((trajectory.completion[0] > 0.0) & (trajectory.completion[0] < 1.0))
        assert np.allclose(trajectory.completion[-1], 1.0)


def _changed_object_ids(trajectory) -> list[int]:
    output = []
    for step in range(1, len(trajectory.states)):
        changed = np.argwhere(trajectory.states[step] != trajectory.states[step - 1])
        row, col = (int(value) for value in changed[0])
        output.append(int(trajectory.object_maps[step, row, col]))
    return output


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
        "probe_visible_object_count_balanced_acc",
        "probe_rollout_object_count_balanced_acc",
        "probe_shape_count_mae",
        "probe_shape_count_r2",
        "probe_velocity_count_mae",
        "probe_velocity_count_r2",
        "probe_relations_mae",
        "probe_completion_r2",
        "probe_bound_shape_acc",
        "probe_bound_velocity_r2",
        "probe_bound_position_r2",
        "raw_probe_bound_shape_acc",
        "probe_rollout_bound_shape_acc",
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


def test_bound_rollout_probe_clamps_predictions_to_observed_factor_ranges() -> None:
    torch.manual_seed(41)
    train_x = torch.randn(6, 3)
    train_y = torch.zeros(6, 2, 11)
    for sample in range(6):
        train_y[sample, :, sample % 5] = 1.0
        train_y[sample, :, 5:7] = -1.0 if sample % 2 == 0 else 1.0
        train_y[sample, :, 7] = -1.0 if sample % 2 == 0 else 1.0
        train_y[sample, :, 8:11] = float(sample % 2)
    mask = torch.ones(6, 2, dtype=torch.bool)
    _, transfer = _fit_slot_regressor(
        train_x,
        train_y,
        mask,
        train_x,
        train_y,
        mask,
        steps=1,
        learning_rate=1.0e-2,
        standardize=False,
        transfer_x=torch.full_like(train_x, 1.0e6),
        transfer_y=train_y,
        transfer_mask=mask,
    )
    assert transfer["shape_mae"] <= 1.0
    assert transfer["velocity_mae"] <= 2.0
    assert transfer["angular_velocity_mae"] <= 2.0
    assert transfer["position_mae"] <= 1.0
    assert transfer["completion_mae"] <= 1.0


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


def test_reconstruction_control_trains_same_single_cls_without_jepa_gradient() -> None:
    generator = MovingObjectGenerator(
        MovingObjectSpec(grid_size=8, min_objects=2, max_objects=2, sequence_length=7)
    )
    batch = sample_moving_object_batch(
        generator, np.random.default_rng(47), batch_size=4, horizon=1
    )
    model = MovingObjectJEPA(
        grid_size=8,
        token_dim=16,
        latent_dim=4,
        encoder_layers=1,
        encoder_heads=4,
        rollout_horizon=1,
        regularizer_weight=0.0,
        prediction_weight=0.0,
        reconstruction_weight=1.0,
    )
    output = model(batch)
    output.loss.backward()
    assert float(output.reconstruction_loss.detach()) > 0.0
    assert model.reconstruction_decoder is not None
    assert model.reconstruction_decoder.weight.grad is not None
    assert model.encoder.project[1].weight.grad is not None
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        for parameter in model.predictor.parameters()
    )
    metrics = run_moving_object_probes(
        model,
        generator,
        np.random.default_rng(53),
        train_samples=8,
        eval_samples=8,
        batch_size=4,
        device=torch.device("cpu"),
        steps=1,
        learning_rate=1.0e-2,
    )
    assert np.isfinite(metrics["model_reconstruction_grid_acc"])
    assert np.isfinite(metrics["model_reconstruction_foreground_iou"])


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


def test_moving_training_defaults_to_deterministic_gpu_kernels() -> None:
    config = (ROOT / "configs/moving_objects/train.yaml").read_text()
    trainer = (ROOT / "puzzle_jepa/train/moving_objects.py").read_text()
    assert "deterministic: true" in config
    assert 'CUBLAS_WORKSPACE_CONFIG", ":4096:8"' in trainer
    reproducibility = (ROOT / "puzzle_jepa/moving_objects/reproducibility.py").read_text()
    assert "configure_reproducibility" in trainer
    assert "torch.use_deterministic_algorithms(deterministic)" in reproducibility
    assert "enable_flash_sdp(not deterministic)" in reproducibility
    assert "enable_mem_efficient_sdp(not deterministic)" in reproducibility


def test_bound_object_reprobe_is_manifest_driven_and_dependency_safe() -> None:
    submit = (ROOT / "scripts/experiments/submit_moving_objects_probe_eval.sh").read_text()
    slurm = (ROOT / "scripts/slurm/run_moving_objects_probe_eval.slurm").read_text()
    evaluator = (ROOT / "puzzle_jepa/eval/moving_object_probes.py").read_text()
    assert 'dependency_args=(--dependency="afterany:${job_id}")' in submit
    assert "probe_eval_v4.json" in slurm
    assert '"schema": "moving_objects_probe_eval_v4"' in evaluator
    assert "run_moving_object_probes" in evaluator


def test_six_hour_watcher_is_configured() -> None:
    watcher = (ROOT / "scripts/experiments/submit_moving_objects_oversight.sh").read_text()
    assert 'CADENCE_HOURS="${CADENCE_HOURS:-6}"' in watcher
    assert "--begin=" in watcher


def test_dynamics_evaluation_can_wait_for_its_own_training_row() -> None:
    script = (ROOT / "scripts/experiments/submit_moving_objects_dynamics_eval.sh").read_text()
    assert 'DEPEND_ON_TRAIN:-0' in script
    assert 'dependency_args=(--dependency="afterany:${job_id}")' in script


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


def test_capacity_transfer_restores_size_and_load_axes_without_cell_latents() -> None:
    script = (ROOT / "scripts/experiments/submit_moving_objects_capacity_transfer.sh").read_text()
    assert "DATASETS=(wrapped_motion rotating_motion)" in script
    assert "BASE_LATENT_DIMS=(2 4 8 16 32 64)" in script
    assert "BASE_MAX_OBJECT_COUNTS=(1 2 4 6 8)" in script
    assert "TEMPORAL_LATENT_DIMS=(4 8 16 32)" in script
    assert "TEMPORAL_MAX_OBJECT_COUNTS=(4 8)" in script
    assert "SEEDS=(1707 2707 3707)" in script
    assert "228 single-CLS jobs" in script
    assert "grid" not in script.lower()


def test_sequence_transfer_covers_all_ordering_completion_and_repair_families() -> None:
    script = (ROOT / "scripts/experiments/submit_moving_objects_sequence_transfer.sh").read_text()
    for data in (
        "object_blocked_build",
        "frontier_build",
        "random_within_object_build",
        "interleaved_build",
        "global_random_build",
        "completion",
        "noisy_repair",
    ):
        assert data in script
    assert "OBJECTIVES=(ema_vicreg ema_vicreg_temporal)" in script
    assert "LATENT_DIMS=(2 4 8 16 32)" in script
    assert "MAX_OBJECT_COUNTS=(4 8)" in script
    assert "SEEDS=(1707 2707 3707)" in script
    assert "420 single-CLS jobs" in script
    assert 'dependency_args=(--dependency="afterany:${previous_stage_ids}")' in script
    assert "grid" not in script.lower()


def test_deterministic_confirmation_crosses_selected_capacity_and_load_rows() -> None:
    script = (
        ROOT / "scripts/experiments/submit_moving_objects_deterministic_confirmation.sh"
    ).read_text()
    assert "DATASETS=(reflected_motion wrapped_motion rotating_motion)" in script
    assert '"4 4 ema_vicreg"' in script
    assert '"4 4 ema_vicreg_temporal"' in script
    assert '"4 8 ema_vicreg"' in script
    assert '"4 8 ema_vicreg_temporal"' in script
    assert '"32 4 ema_vicreg"' in script
    assert '"32 8 ema_vicreg"' in script
    assert "SEEDS=(1707 2707 3707)" in script
    assert "54 deterministic single-CLS jobs" in script
    assert "grid" not in script.lower()


def test_reconstruction_confirmation_matches_selected_capacity_and_load_rows() -> None:
    script = (
        ROOT / "scripts/experiments/submit_moving_objects_reconstruction_confirmation.sh"
    ).read_text()
    assert "DATASETS=(reflected_motion wrapped_motion rotating_motion)" in script
    assert "LATENT_DIMS=(4 32)" in script
    assert "MAX_OBJECT_COUNTS=(4 8)" in script
    assert "SEEDS=(1707 2707 3707)" in script
    assert "OBJECTIVE_CONFIG=reconstruction" in script
    assert "36 deterministic single-CLS controls" in script
    assert "grid" not in script.lower()


def test_selected_sequence_matrix_keeps_binding_load_and_control_axes() -> None:
    script = (ROOT / "scripts/experiments/submit_moving_objects_sequence_selected.sh").read_text()
    for data in (
        "object_blocked_build",
        "frontier_build",
        "random_within_object_build",
        "interleaved_build",
        "global_random_build",
        "completion",
        "noisy_repair",
    ):
        assert data in script
    for row in (
        '"4 4 ema_vicreg"',
        '"4 4 ema_vicreg_temporal"',
        '"4 4 reconstruction"',
        '"4 8 ema_vicreg"',
        '"4 8 ema_vicreg_temporal"',
        '"4 8 reconstruction"',
        '"32 4 ema_vicreg"',
        '"32 4 reconstruction"',
        '"32 8 ema_vicreg"',
        '"32 8 reconstruction"',
    ):
        assert row in script
    assert 'dependency_args=(--dependency="afterany:${previous_stage_ids}")' in script
    assert "210 deterministic single-CLS jobs" in script
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


def test_analyzer_prefers_matched_v4_reprobe_metrics(tmp_path: Path) -> None:
    run = tmp_path / "motion_n8_z4_reprobe_seed1707"
    run.mkdir()
    initial = {
        "step": 0,
        "data": "wrapped_motion",
        "objective": "ema_vicreg_temporal",
        "latent_dim": 4,
        "max_objects": 8,
        "seed": 1707,
        "probe_bound_shape_acc": 0.1,
    }
    final = {**initial, "step": 5000, "probe_bound_shape_acc": 0.2}
    initial["train_reconstruction_loss"] = 1.2
    final["train_reconstruction_loss"] = 0.4
    (run / "metrics.jsonl").write_text("\n".join((json.dumps(initial), json.dumps(final))))
    (run / "probe_eval_v4.json").write_text(
        json.dumps(
            {
                "schema": "moving_objects_probe_eval_v4",
                "initial": {"probe_bound_shape_acc": 0.3},
                "final": {"probe_bound_shape_acc": 0.8},
            }
        )
    )

    summary = analyze(tmp_path, {run.name})

    assert summary["runs"][0]["probe_source"] == "probe_eval_v4.json"
    assert summary["aggregates"][0]["probe_v4_n"] == 1
    assert summary["runs"][0]["absolute"]["probe_bound_shape_acc"] == 0.8
    assert summary["runs"][0]["delta"]["probe_bound_shape_acc"] == 0.5
    assert summary["runs"][0]["absolute"]["train_reconstruction_loss"] == 0.4


def test_analyzer_unions_explicit_manifests_without_root_scanning(tmp_path: Path) -> None:
    first = tmp_path / "first.tsv"
    second = tmp_path / "second.tsv"
    first.write_text("run_name\tjob_id\nrun_a\t1\n")
    second.write_text("run_name\tjob_id\nrun_b\t2\nrun_c\t3\n")
    assert _manifest_run_names([first, second]) == {"run_a", "run_b", "run_c"}
