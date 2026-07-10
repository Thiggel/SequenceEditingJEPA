from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from puzzle_jepa.object_dynamics.batching import sample_object_dynamics_batch
from puzzle_jepa.object_dynamics.domain import ObjectSpec, SceneSpec
from puzzle_jepa.object_dynamics.generator import ObjectDynamicsGenerator, ObjectDynamicsSpec
from puzzle_jepa.object_dynamics.losses import sigreg_regularizer
from puzzle_jepa.object_dynamics.model import ObjectDynamicsJEPA
from puzzle_jepa.object_dynamics.probes import run_object_dynamics_probes


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
    script = Path(__file__).resolve().parents[1] / "scripts" / "experiments" / "submit_object_dynamics_phase1.sh"
    text = script.read_text()
    assert "completion" in text
    assert "transform_identity" in text


def test_phase_sweep_requires_explicit_prestage_selection() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "experiments" / "submit_object_dynamics_phase1.sh"
    text = script.read_text()
    assert "PRESTAGE_SELECTION_CONFIRMED" in text
    assert 'LEARNING_RATE="${LEARNING_RATE}"' in text
    assert 'MAX_STEPS="${MAX_STEPS}"' in text
    assert "phase3_h8_ldad" in text
    assert "phase3_h4_ldad" not in text


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
