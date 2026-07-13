from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from puzzle_jepa.object_dynamics.batching import RELATION_NAMES, _object_relations
from puzzle_jepa.object_dynamics.domain import ObjectSpec
from puzzle_jepa.object_dynamics.generator import ObjectDynamicsGenerator, ObjectDynamicsSpec, TRAJECTORY_KINDS
from puzzle_jepa.object_dynamics.model import ObjectDynamicsJEPA
from puzzle_jepa.object_dynamics.probes import run_object_dynamics_probes


ROOT = Path(__file__).resolve().parents[1]


def test_prestage_has_three_well_separated_train_lengths() -> None:
    config = OmegaConf.load(ROOT / "configs/object_dynamics/sweep/prestage.yaml")
    train_lengths = list(config.max_steps)
    assert len(train_lengths) >= 3
    assert max(train_lengths) >= 10 * min(train_lengths)


def test_object_experiment_configs_have_no_full_grid_rows() -> None:
    assert not (ROOT / "configs/object_dynamics/model/grid128_r8.yaml").exists()
    assert not (ROOT / "configs/object_dynamics/model/h_grid128_h8.yaml").exists()
    assert not (ROOT / "configs/controlled_objects/model/grid_ldad.yaml").exists()
    sweep = (ROOT / "configs/object_dynamics/sweep/phase1.yaml").read_text()
    assert "grid128_r8" not in sweep
    assert "h_grid128_h8" not in sweep


def test_generator_has_pure_random_edit_control() -> None:
    assert "random_off_manifold" in TRAJECTORY_KINDS
    assert (ROOT / "configs/object_dynamics/data/random_off_manifold.yaml").exists()


def test_hierarchy_exposes_high_and_low_level_planning_rollouts() -> None:
    model = ObjectDynamicsJEPA(
        grid_size=8,
        d_model=32,
        encoder_layers=1,
        encoder_heads=4,
        rollout_horizon=1,
        hierarchy_horizon=4,
        hierarchy_planning=True,
        hierarchy_rollout_steps=2,
    )
    states = torch.zeros(2, 8, 8, dtype=torch.long)
    goals = states.clone()
    candidates = torch.tensor(
        [
            [
                [[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 1]],
                [[0, 6, 6, 2], [0, 6, 7, 2], [0, 7, 6, 2], [0, 7, 7, 2]],
            ],
            [
                [[0, 2, 2, 3], [0, 2, 3, 3], [0, 3, 2, 3], [0, 3, 3, 3]],
                [[0, 4, 4, 4], [0, 4, 5, 4], [0, 5, 4, 4], [0, 5, 5, 4]],
            ],
        ],
        dtype=torch.long,
    )
    rollout = model.rollout_high_level(states, candidates[:, :1])
    plan = model.plan_macro_actions(states, candidates, goals)
    continuous = model.optimize_macro_actions(
        states,
        goals,
        num_samples=8,
        num_elites=2,
        num_iterations=2,
    )
    tracked_indices, tracked_scores, _ = model.track_subgoal(states, candidates, continuous.predicted_states[:, 0])
    primitive = model.optimize_primitive_actions(
        states,
        continuous.predicted_states[:, 0],
        num_samples=8,
        num_elites=2,
        num_iterations=2,
    )

    assert rollout.shape == (2, 1, 32)
    assert plan.high_level_indices.shape == (2,)
    assert plan.low_level_indices.shape == (2,)
    assert plan.subgoals.shape == (2, 32)
    assert torch.isfinite(plan.high_level_scores).all()
    assert torch.isfinite(plan.low_level_scores).all()
    assert model.macro_action_dim < model.d_model
    assert model.chunk_encoder.cls.shape == (1, 1, model.d_model)
    assert continuous.macro_actions.shape == (2, 1, model.macro_action_dim)
    assert continuous.predicted_states.shape == (2, 1, 32)
    assert tracked_indices.shape == (2,)
    assert torch.isfinite(tracked_scores).all()
    assert primitive.actions.shape == (2, 4, 4)
    assert primitive.predicted_endpoints.shape == (2, 32)
    assert torch.isfinite(primitive.subgoal_scores).all()
    erase = primitive.actions[..., 0] == 1
    assert bool(torch.all(primitive.actions[..., 3][erase] == 0))
    assert bool(torch.all(primitive.actions[..., 3][~erase] > 0))
    simulated = states.clone()
    for step in range(primitive.actions.shape[1]):
        action = primitive.actions[:, step]
        values = simulated[torch.arange(len(states)), action[:, 1], action[:, 2]]
        assert bool(torch.all((action[:, 0] == 0) == (values == 0)))
        simulated[torch.arange(len(states)), action[:, 1], action[:, 2]] = action[:, 3]


def test_scene_metadata_covers_parts_and_inside_relations() -> None:
    rng = np.random.default_rng(5)
    generator = ObjectDynamicsGenerator(
        ObjectDynamicsSpec(
            grid_size=12,
            min_objects=2,
            max_objects=2,
            inside_ratio=1.0,
            trajectory_kind="object_blocked",
            counterfactual_ratio=0.0,
            wrong_ratio=0.0,
        )
    )
    trajectory = generator.sample_trajectory(rng)
    scene = trajectory.scene

    assert all(obj.parts for obj in scene.objects)
    assert "inside" in RELATION_NAMES
    present, relations = _object_relations(trajectory, len(trajectory.states) - 1, max_objects=2)
    assert present.tolist() == [True]
    assert bool(relations[0, RELATION_NAMES.index("inside")])

    mask = np.zeros((6, 6), dtype=bool)
    mask[:2, :2] = True
    mask[-2:, -2:] = True
    multipart = ObjectSpec(0, "multipart", 1, mask)
    assert len(multipart.parts) == 2


def test_all_phase_trajectory_regimes_support_two_h16_macro_steps() -> None:
    for path in sorted((ROOT / "configs/object_dynamics/data").glob("*.yaml")):
        config = {
            key: value
            for key, value in dict(OmegaConf.load(path)).items()
            if key != "name"
        }
        generator = ObjectDynamicsGenerator(ObjectDynamicsSpec(**config))
        trajectory = generator.sample_trajectory(np.random.default_rng(1), min_actions=32)
        assert trajectory.sample_start_indices(32).size, path.name


def test_probe_suite_includes_attention_evidence() -> None:
    model = ObjectDynamicsJEPA(grid_size=8, d_model=16, encoder_layers=1, encoder_heads=4)
    attention = model.attention_maps(torch.zeros(2, 8, 8, dtype=torch.long))
    assert attention.shape == (2, 4, 8, 8)
    torch.testing.assert_close(attention.flatten(2).sum(dim=-1), torch.ones(2, 4))


def test_reconstruction_trained_encoder_baseline_config_is_preserved() -> None:
    assert (ROOT / "configs/object_dynamics/objective/reconstruction.yaml").exists()

    generator = ObjectDynamicsGenerator(
        ObjectDynamicsSpec(grid_size=8, max_objects=2, max_shape_extent=4, counterfactual_ratio=0.0, wrong_ratio=0.0)
    )
    from puzzle_jepa.object_dynamics.batching import sample_object_dynamics_batch

    batch = sample_object_dynamics_batch(generator, np.random.default_rng(6), batch_size=2, horizon=1)
    model = ObjectDynamicsJEPA(
        grid_size=8,
        d_model=16,
        encoder_layers=1,
        encoder_heads=4,
        dynamics_weight=0.0,
        reconstruction_weight=1.0,
        target_mode="shared",
    )
    output = model(batch)
    output.loss.backward()
    assert float(output.reconstruction_loss.detach()) > 0.0
    assert model.encoder.color.weight.grad is not None
    assert model.predictor[0].weight.grad is None


def test_probe_suite_includes_nonlinear_upper_bound_and_correction_chunks() -> None:
    metrics = _small_hierarchy_probe_run()
    assert np.isfinite(metrics["probe_mlp_object_count_acc"])
    assert np.isfinite(metrics["probe_delta_action_process_acc"])
    assert np.isfinite(metrics["probe_delta_action_process_balanced_acc"])
    assert np.isfinite(metrics["raw_probe_action_process_provenance_acc"])
    assert np.isfinite(metrics["raw_probe_action_process_provenance_balanced_acc"])
    assert np.isfinite(metrics["probe_action_process_provenance_majority_acc"])
    assert np.isfinite(metrics["probe_action_process_provenance_majority_balanced_acc"])
    assert np.isfinite(metrics["probe_chunk_correction_acc"])


def test_nearest_neighbor_probe_reports_semantic_object_factors() -> None:
    metrics = _small_hierarchy_probe_run()
    assert "latent_nn_current_shape_acc" in metrics
    assert "pixel_nn_current_shape_acc" in metrics
    assert "latent_nn_current_color_acc" in metrics
    assert "pixel_nn_current_color_acc" in metrics
    assert "latent_nn_current_completion_mae" in metrics
    assert "pixel_nn_current_completion_mae" in metrics


def test_probe_suite_decodes_object_count_after_rollout() -> None:
    metrics = _small_hierarchy_probe_run()
    assert np.isfinite(metrics["probe_rollout_object_count_acc"])
    assert np.isfinite(metrics["probe_rollout_object_count_balanced_acc"])


def test_probe_suite_evaluates_high_level_predictions() -> None:
    metrics = _small_hierarchy_probe_run()
    assert np.isfinite(metrics["probe_hierarchy_endpoint_mse"])
    assert 0.0 <= metrics["probe_hierarchy_macro_retrieval_acc"] <= 1.0
    assert 0.0 <= metrics["probe_hierarchy_low_level_retrieval_acc"] <= 1.0
    assert np.isfinite(metrics["probe_hierarchy_optimized_goal_l1"])
    assert np.isfinite(metrics["probe_hierarchy_subgoal_reachability_l1"])
    assert np.isfinite(metrics["probe_hierarchy_cem_subgoal_l1"])
    assert np.isfinite(metrics["probe_hierarchy_cem_goal_l1"])
    assert 0.0 <= metrics["probe_hierarchy_retrieval_goal_hamming"] <= 1.0
    assert 0.0 <= metrics["probe_hierarchy_retrieval_goal_success"] <= 1.0
    assert 0.0 <= metrics["probe_hierarchy_cem_executed_goal_hamming"] <= 1.0
    assert 0.0 <= metrics["probe_hierarchy_cem_executed_goal_success"] <= 1.0
    assert np.isfinite(metrics["probe_hierarchy_cem_model_bias_l1"])


def test_retired_object_launchers_contain_no_experiment_rows() -> None:
    for name in (
        "submit_object_dynamics_phase1.sh",
        "submit_object_dynamics_trajectory_gate.sh",
    ):
        script = (ROOT / "scripts/experiments" / name).read_text()
        assert script.index("Retired:") < script.index("exit 2")
        assert "sbatch" not in script
        assert "grid128_r8" not in script


def _small_hierarchy_probe_run() -> dict[str, float]:
    generator = ObjectDynamicsGenerator(
        ObjectDynamicsSpec(
            grid_size=8,
            max_objects=2,
            max_shape_extent=4,
            trajectory_kind="semantic_mix",
            counterfactual_ratio=0.15,
            wrong_ratio=0.05,
        )
    )
    model = ObjectDynamicsJEPA(
        grid_size=8,
        d_model=16,
        encoder_layers=1,
        encoder_heads=4,
        rollout_horizon=1,
        hierarchy_horizon=4,
        hierarchy_planning=True,
        hierarchy_rollout_steps=2,
    )
    return run_object_dynamics_probes(
        model,
        generator,
        np.random.default_rng(13),
        train_samples=16,
        eval_samples=8,
        batch_size=8,
        horizon=model.training_horizon,
        device=torch.device("cpu"),
        steps=1,
    )
