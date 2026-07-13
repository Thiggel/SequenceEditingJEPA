from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from puzzle_jepa.object_dynamics.batching import _visible_object_map, _visible_slot_id, sample_object_dynamics_batch
from puzzle_jepa.object_dynamics.domain import (
    ActionOp,
    LowLevelAction,
    ObjectSpec,
    ObjectTrajectory,
    SceneSpec,
    TrajectoryCategory,
)
from puzzle_jepa.object_dynamics.generator import ObjectDynamicsGenerator, ObjectDynamicsSpec
from puzzle_jepa.object_dynamics.losses import covariance_loss, sigreg_regularizer, vicreg_regularizer
from puzzle_jepa.object_dynamics.model import ObjectDynamicsJEPA
from puzzle_jepa.object_dynamics.probes import _class_balanced_weights, run_object_dynamics_probes
from puzzle_jepa.object_dynamics.shapes import SHAPE_TYPES
from puzzle_jepa.train.object_dynamics import (
    _fork_rng_devices,
    _initialize_from_checkpoint,
    _set_trainable_components,
)


def test_object_dynamics_ldad_uses_encoded_future_displacement() -> None:
    rng = np.random.default_rng(11)
    generator = ObjectDynamicsGenerator(
        ObjectDynamicsSpec(
            grid_size=8,
            max_objects=2,
            max_shape_extent=4,
            trajectory_kind="frontier_build",
            counterfactual_ratio=0.0,
            wrong_ratio=0.0,
        )
    )
    batch = sample_object_dynamics_batch(generator, rng, batch_size=2, horizon=2)
    model = ObjectDynamicsJEPA(
        grid_size=8,
        d_model=32,
        encoder_layers=1,
        encoder_heads=4,
        rollout_horizon=1,
        ldad_weight=1.0,
    )
    captured: list[torch.Tensor] = []
    original = model._ldad_loss

    def capture(delta: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        captured.append(delta.detach())
        return original(delta, actions)

    model._ldad_loss = capture  # type: ignore[method-assign]
    output = model(batch)
    current = model.encode(batch.states).detach()
    encoded_delta = output.targets[:, 0].detach() - current
    assert captured
    torch.testing.assert_close(captured[0], encoded_delta)


def test_end_to_end_objectives_keep_future_targets_in_gradient_graph() -> None:
    rng = np.random.default_rng(13)
    generator = ObjectDynamicsGenerator(
        ObjectDynamicsSpec(
            grid_size=8,
            max_objects=2,
            max_shape_extent=4,
            counterfactual_ratio=0.0,
            wrong_ratio=0.0,
        )
    )
    batch = sample_object_dynamics_batch(generator, rng, batch_size=2, horizon=1)
    model = ObjectDynamicsJEPA(
        grid_size=8,
        d_model=32,
        encoder_layers=1,
        encoder_heads=4,
        rollout_horizon=1,
        target_mode="shared",
        ldad_weight=0.1,
    )

    output = model(batch)
    assert output.targets.requires_grad


def test_sigreg_distinguishes_gaussian_from_rademacher_samples() -> None:
    torch.manual_seed(17)
    gaussian = torch.randn(2048, 1)
    rademacher = (2 * torch.randint(0, 2, (2048, 1)) - 1).float()

    gaussian_loss = sigreg_regularizer(gaussian, num_slices=64)
    rademacher_loss = sigreg_regularizer(rademacher, num_slices=64)
    assert float(gaussian_loss) < 0.25 * float(rademacher_loss)


def test_vicreg_penalizes_collapse_and_correlated_features() -> None:
    torch.manual_seed(19)
    independent = torch.randn(4096, 8)
    collapsed = torch.zeros_like(independent)
    shared = torch.randn(4096, 1).expand_as(independent)

    assert float(vicreg_regularizer(independent)) < 0.1 * float(vicreg_regularizer(collapsed))
    assert float(covariance_loss(shared)) > 100.0 * float(covariance_loss(independent))


def test_ema_target_encoder_tracks_online_encoder_without_gradients() -> None:
    model = ObjectDynamicsJEPA(
        grid_size=8,
        d_model=16,
        encoder_layers=1,
        encoder_heads=4,
        target_mode="ema",
        ema_decay=0.5,
    )
    before = model.target_encoder.color.weight.detach().clone()
    with torch.no_grad():
        model.encoder.color.weight.add_(2.0)

    model.update_target_encoder()

    torch.testing.assert_close(model.target_encoder.color.weight, before + 1.0)
    assert all(not parameter.requires_grad for parameter in model.target_encoder.parameters())


def test_scene_generator_can_emit_touching_or_diagonal_contact_objects() -> None:
    rng = np.random.default_rng(23)
    generator = ObjectDynamicsGenerator(
        ObjectDynamicsSpec(
            grid_size=16,
            min_objects=3,
            max_objects=4,
            max_shape_extent=5,
            counterfactual_ratio=0.0,
            wrong_ratio=0.0,
        )
    )
    scenes = [generator.sample_scene(rng) for _ in range(200)]
    assert any(_has_touching_pair(scene) for scene in scenes)


def test_scene_generator_keeps_dense_object_ids_and_min_object_count() -> None:
    rng = np.random.default_rng(31)
    generator = ObjectDynamicsGenerator(
        ObjectDynamicsSpec(
            grid_size=8,
            min_objects=4,
            max_objects=4,
            max_shape_extent=6,
            max_scene_retries=1,
            counterfactual_ratio=0.0,
            wrong_ratio=0.0,
        )
    )
    scenes = []
    for _ in range(100):
        try:
            scenes.append(generator.sample_scene(rng))
        except RuntimeError:
            pass

    assert scenes
    for scene in scenes:
        assert scene.object_count >= generator.spec.min_objects
        assert [obj.object_id for obj in scene.objects] == list(range(scene.object_count))


def test_wrong_trajectory_validity_is_labeled_per_state() -> None:
    rng = np.random.default_rng(37)
    generator = ObjectDynamicsGenerator(
        ObjectDynamicsSpec(
            grid_size=10,
            max_objects=3,
            max_shape_extent=4,
            trajectory_kind="frontier_build",
            counterfactual_ratio=0.0,
            wrong_ratio=1.0,
        )
    )
    trajectory = generator.sample_trajectory(rng)
    expected = np.asarray(
        [
            bool(np.all((state == 0) | (state == trajectory.scene.grid)))
            for state in trajectory.states
        ],
        dtype=bool,
    )

    np.testing.assert_array_equal(trajectory.state_validity, expected)


def test_counterfactuals_include_structured_local_off_path_edits() -> None:
    rng = np.random.default_rng(39)
    generator = ObjectDynamicsGenerator(
        ObjectDynamicsSpec(
            grid_size=10,
            min_objects=2,
            max_objects=3,
            max_shape_extent=4,
            trajectory_kind="frontier_build",
            counterfactual_ratio=1.0,
            wrong_ratio=0.0,
        )
    )
    trajectories = [generator.sample_trajectory(rng, min_actions=8) for _ in range(20)]

    assert all(trajectory.category.name == "COUNTERFACTUAL" for trajectory in trajectories)
    assert all(trajectory.sample_start_indices(8).size for trajectory in trajectories)
    assert all(bool(np.any(~trajectory.state_validity)) for trajectory in trajectories)
    assert any(
        any(not np.all((state == 0) | (state == trajectory.scene.grid)) for state in trajectory.states)
        for trajectory in trajectories
    )


def test_object_blocked_is_distinct_from_random_within_object() -> None:
    mask = np.zeros((6, 6), dtype=bool)
    mask[1:4, 2:5] = True
    grid = np.zeros((6, 6), dtype=np.int64)
    grid[mask] = 3
    scene = SceneSpec(grid=grid, objects=(ObjectSpec(object_id=0, shape_type="solid_rectangle", color=3, mask=mask),))
    generator = ObjectDynamicsGenerator(ObjectDynamicsSpec(grid_size=6, max_shape_extent=4, counterfactual_ratio=0.0))

    blocked = generator._object_blocked(scene, np.random.default_rng(0))
    blocked_order = tuple((action.row, action.col) for action in blocked.actions)
    assert blocked_order == tuple(sorted(blocked_order))

    random_orders = set()
    for seed in range(8):
        trajectory = generator._random_within_object(scene, np.random.default_rng(seed))
        random_orders.add(tuple((action.row, action.col) for action in trajectory.actions))
    assert len(random_orders) > 1
    assert any(order != blocked_order for order in random_orders)


def test_scene_object_ids_are_spatially_canonical_and_touching_objects_are_distinguishable() -> None:
    rng = np.random.default_rng(41)
    generator = ObjectDynamicsGenerator(
        ObjectDynamicsSpec(
            grid_size=12,
            min_objects=3,
            max_objects=4,
            max_shape_extent=5,
            duplicate_shape_ratio=0.5,
            same_color_ratio=0.8,
            counterfactual_ratio=0.0,
            wrong_ratio=0.0,
        )
    )

    for _ in range(100):
        scene = generator.sample_scene(rng)
        assert [obj.bbox for obj in scene.objects] == sorted(obj.bbox for obj in scene.objects)
        for left_index, left in enumerate(scene.objects):
            for right in scene.objects[left_index + 1 :]:
                if _cells_touch(np.argwhere(left.mask), np.argwhere(right.mask)):
                    assert left.color != right.color


def test_hidden_object_maps_track_only_objects_present_in_each_state() -> None:
    rng = np.random.default_rng(43)
    generator = ObjectDynamicsGenerator(
        ObjectDynamicsSpec(
            grid_size=10,
            min_objects=3,
            max_objects=3,
            max_shape_extent=4,
            trajectory_kind="object_blocked",
            counterfactual_ratio=0.0,
            wrong_ratio=0.0,
        )
    )
    trajectory = generator.sample_trajectory(rng, min_actions=4)
    assert not bool(np.any(trajectory.object_maps[0] >= 0))
    for state, object_map in zip(trajectory.states, trajectory.object_maps, strict=True):
        assert bool(np.all(object_map[state == 0] == -1))
        assert bool(np.all(object_map[state != 0] >= 0))

    batch = sample_object_dynamics_batch(generator, rng, batch_size=32, horizon=4)
    visible_counts = torch.tensor(
        [torch.unique(item[item > 0]).numel() for item in batch.object_map], dtype=torch.long
    )
    future_counts = torch.tensor(
        [torch.unique(item[item > 0]).numel() for item in batch.future_object_map[:, -1]], dtype=torch.long
    )
    torch.testing.assert_close(batch.object_count, visible_counts)
    torch.testing.assert_close(batch.future_object_count[:, -1], future_counts)
    torch.testing.assert_close(batch.object_present.sum(dim=1), visible_counts)
    torch.testing.assert_close(batch.future_object_present[:, -1].sum(dim=1), future_counts)


def test_transform_identity_preserves_hidden_shape_class() -> None:
    generator = ObjectDynamicsGenerator(
        ObjectDynamicsSpec(
            grid_size=10,
            max_objects=3,
            max_shape_extent=4,
            trajectory_kind="transform_identity",
            counterfactual_ratio=0.0,
            wrong_ratio=0.0,
        )
    )
    trajectory = generator.sample_trajectory(np.random.default_rng(42))

    assert len(trajectory.scene.objects) == 1
    assert trajectory.scene.objects[0].shape_type in SHAPE_TYPES


def test_probe_object_slots_remain_stable_as_partial_bboxes_grow() -> None:
    shape = (6, 6)
    first_mask = np.zeros(shape, dtype=bool)
    first_mask[:5, 4] = True
    second_mask = np.zeros(shape, dtype=bool)
    second_mask[2, 0] = True
    target = np.zeros(shape, dtype=np.int64)
    target[first_mask] = 2
    target[second_mask] = 3

    state0 = np.zeros(shape, dtype=np.int64)
    state0[4, 4] = 2
    state0[2, 0] = 3
    state1 = state0.copy()
    state1[0, 4] = 2
    map0 = np.full(shape, -1, dtype=np.int64)
    map0[4, 4] = 0
    map0[2, 0] = 1
    map1 = map0.copy()
    map1[0, 4] = 0
    trajectory = ObjectTrajectory(
        states=np.stack([state0, state1]),
        object_maps=np.stack([map0, map1]),
        actions=(LowLevelAction(ActionOp.PAINT, 0, 4, 2),),
        action_object_ids=(0,),
        scene=SceneSpec(
            grid=target,
            objects=(
                ObjectSpec(0, "line", 2, first_mask),
                ObjectSpec(1, "line", 3, second_mask),
            ),
        ),
        kind="slot_stability",
        semantic=True,
        category=TrajectoryCategory.SEMANTIC,
        transition_categories=(TrajectoryCategory.SEMANTIC,),
        state_validity=np.ones(2, dtype=bool),
    )

    for state_index in (0, 1):
        assert _visible_slot_id(trajectory, state_index, 0, max_objects=2) == 1
        assert _visible_slot_id(trajectory, state_index, 1, max_objects=2) == 2
        object_map = _visible_object_map(trajectory, state_index, max_objects=2)
        assert object_map[4, 4] == 1
        assert object_map[2, 0] == 2


def test_probe_evaluation_preserves_model_mode_and_torch_rng() -> None:
    rng = np.random.default_rng(47)
    generator = ObjectDynamicsGenerator(
        ObjectDynamicsSpec(
            grid_size=8,
            min_objects=2,
            max_objects=2,
            max_shape_extent=4,
            trajectory_kind="object_blocked",
            counterfactual_ratio=0.0,
            wrong_ratio=0.0,
        )
    )
    model = ObjectDynamicsJEPA(
        grid_size=8,
        d_model=16,
        encoder_layers=1,
        encoder_heads=4,
        rollout_horizon=1,
    )
    model.train()
    torch.manual_seed(123)
    rng_state = torch.random.get_rng_state().clone()

    run_object_dynamics_probes(
        model,
        generator,
        rng,
        train_samples=24,
        eval_samples=12,
        batch_size=12,
        horizon=1,
        device=torch.device("cpu"),
        steps=1,
    )

    assert model.training
    torch.testing.assert_close(torch.random.get_rng_state(), rng_state)


def test_phase_sweep_includes_non_empty_start_trajectory_regimes() -> None:
    sweep = Path(__file__).resolve().parents[1] / "configs" / "object_dynamics" / "sweep" / "phase1.yaml"
    text = sweep.read_text()
    assert "completion" in text
    assert "transform_identity" in text
    assert "random_off_manifold" in text


def test_phase_sweep_uses_a_common_probe_distribution() -> None:
    config = (Path(__file__).resolve().parents[1] / "configs" / "object_dynamics" / "train.yaml").read_text()
    trainer = (Path(__file__).resolve().parents[1] / "puzzle_jepa" / "train" / "object_dynamics.py").read_text()
    assert "probe_trajectory_kind: semantic_mix" in config
    assert "probe_generator" in trainer


def test_probe_class_weights_equalize_observed_class_mass() -> None:
    target = torch.tensor([0, 0, 0, 1, 2, 2])
    weights = _class_balanced_weights(target, num_classes=4)
    weighted_mass = torch.bincount(target, minlength=4).float() * weights
    torch.testing.assert_close(weighted_mass[:3], torch.full((3,), 2.0))
    assert float(weights[3]) == 0.0


def test_phase_sweep_is_retired_before_submission() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "experiments" / "submit_object_dynamics_phase1.sh"
    text = script.read_text()
    assert text.index("Retired:") < text.index("exit 2")
    assert "sbatch" not in text


def test_stability_prestage_excludes_unpaired_delta_objective() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "experiments"
        / "submit_object_dynamics_stability_prestage.sh"
    )
    text = script.read_text()
    assert "OBJECTIVES=(ema vicreg sigreg)" in text
    assert "ldad" not in text
    assert "SEEDS_OVERRIDE" in text


def test_balanced_reprobe_is_paired_to_every_stability_training_job() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "experiments"
        / "submit_object_dynamics_balanced_reprobe.sh"
    ).read_text()
    for train_job in range(3831210, 3831228):
        assert f":{train_job}\"" in script
    for train_job in range(3831379, 3831394, 2):
        assert f":{train_job}\"" in script
    assert 'dependency_args=(--dependency="afterok:${train_job}")' in script
    assert 'sbatch --parsable "${dependency_args[@]}"' in script
    assert "probe_eval_balanced_v4.json" in script
    assert "run_object_dynamics_probe_eval.slurm" in script


def test_stability_replication_is_multiseed_and_excludes_delta() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "experiments"
        / "submit_object_dynamics_stability_replication.sh"
    ).read_text()
    assert "OBJECTIVES=(ema sigreg)" in script
    assert "SEEDS=(2707 3707)" in script
    assert "ldad" not in script
    assert 'sbatch --parsable --dependency="afterok:${train_job}"' in script


def test_length_calibration_covers_dimensions_seeds_and_saturation_scale() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "experiments"
        / "submit_object_dynamics_length_calibration.sh"
    ).read_text()
    assert "MODELS=(cls64_r8 cls128_r8)" in script
    assert "SEEDS=(1707 2707 3707)" in script
    assert "STEP_COUNTS=(5000 15000 50000)" in script
    assert "probe_eval_balanced_v4.json" in script
    assert 'SAVE_EVERY_STEPS="${steps}"' in script


def test_staged_hwm_loads_and_freezes_only_low_level_components(tmp_path: Path) -> None:
    low = ObjectDynamicsJEPA(grid_size=8, d_model=16, encoder_layers=1, encoder_heads=4)
    with torch.no_grad():
        low.encoder.color.weight.fill_(0.75)
    checkpoint = tmp_path / "low.pt"
    torch.save({"model": low.state_dict()}, checkpoint)

    high = ObjectDynamicsJEPA(
        grid_size=8,
        d_model=16,
        encoder_layers=1,
        encoder_heads=4,
        rollout_horizon=1,
        hierarchy_horizon=4,
        hierarchy_planning=True,
        hierarchy_rollout_steps=2,
        macro_action_dim=4,
    )
    loaded = _initialize_from_checkpoint(high, checkpoint, device=torch.device("cpu"))
    _set_trainable_components(high, "hierarchy_only")

    assert "encoder.color.weight" in loaded
    torch.testing.assert_close(high.encoder.color.weight, torch.full_like(high.encoder.color.weight, 0.75))
    assert all(not parameter.requires_grad for parameter in high.encoder.parameters())
    assert all(parameter.requires_grad for parameter in high.chunk_encoder.parameters())
    assert all(parameter.requires_grad for parameter in high.hierarchy_predictor.parameters())
    assert high.rollout_weight == 0.0


def test_probe_rng_forks_the_active_cuda_device() -> None:
    assert _fork_rng_devices(torch.device("cpu")) == []
    assert _fork_rng_devices(torch.device("cuda:3")) == [3]


def test_hwm_calibration_crosses_macro_dimension_and_training_schedule() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "experiments"
        / "submit_object_dynamics_hwm_calibration.sh"
    ).read_text()
    assert "MACRO_DIMS=(4 8 16)" in script
    assert "for schedule in joint staged" in script
    assert "TRAINABLE_COMPONENTS=\"${trainable_components}\"" in script
    assert "model.macro_action_dim=${macro_dim}" in script
    assert "eval.run_probes_during_training=false" in script
    assert "probe_eval_balanced_v4.json" in script


def _has_touching_pair(scene: SceneSpec) -> bool:
    masks = [obj.mask for obj in scene.objects]
    for left_index, left in enumerate(masks):
        left_cells = np.argwhere(left)
        for right in masks[left_index + 1 :]:
            right_cells = np.argwhere(right)
            if _cells_touch(left_cells, right_cells):
                return True
    return False


def _cells_touch(left_cells: np.ndarray, right_cells: np.ndarray) -> bool:
    for row, col in left_cells:
        delta = np.abs(right_cells - np.asarray([row, col]))
        if bool(np.any(np.max(delta, axis=1) == 1)):
            return True
    return False
