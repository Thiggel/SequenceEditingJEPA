from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from puzzle_jepa.object_dynamics.domain import ObjectTrajectory
from puzzle_jepa.object_dynamics.generator import ObjectDynamicsGenerator
from puzzle_jepa.object_dynamics.shapes import SHAPE_TYPES


RELATION_NAMES = ("same_color", "same_shape", "touching", "left_of", "inside")
PROCESS_NAMES = ("unknown", "build", "complete", "transform", "repair_fill", "trim", "recolor", "random")


@dataclass(frozen=True, slots=True)
class ObjectDynamicsBatch:
    states: torch.Tensor
    actions: torch.Tensor
    futures: torch.Tensor
    object_count: torch.Tensor
    scene_object_count: torch.Tensor
    future_object_count: torch.Tensor
    current_object_id: torch.Tensor
    next_object_id: torch.Tensor
    valid_state: torch.Tensor
    trajectory_category: torch.Tensor
    completion: torch.Tensor
    future_completion: torch.Tensor
    object_present: torch.Tensor
    future_object_present: torch.Tensor
    object_colors: torch.Tensor
    object_bboxes: torch.Tensor
    future_object_bboxes: torch.Tensor
    object_centroids: torch.Tensor
    object_areas: torch.Tensor
    object_shapes: torch.Tensor
    object_part_counts: torch.Tensor
    object_missing: torch.Tensor
    future_object_missing: torch.Tensor
    object_overgrowth: torch.Tensor
    future_object_overgrowth: torch.Tensor
    object_wrong_color: torch.Tensor
    future_object_wrong_color: torch.Tensor
    object_map: torch.Tensor
    future_object_map: torch.Tensor
    action_object_ids: torch.Tensor
    action_object_shapes: torch.Tensor
    action_process_types: torch.Tensor
    continues_object: torch.Tensor
    relation_present: torch.Tensor
    relations: torch.Tensor


def sample_object_dynamics_batch(
    generator: ObjectDynamicsGenerator,
    rng: np.random.Generator,
    *,
    batch_size: int,
    horizon: int,
    device: str | torch.device = "cpu",
) -> ObjectDynamicsBatch:
    states = []
    actions = []
    futures = []
    object_count = []
    scene_object_count = []
    future_object_count = []
    current_object_id = []
    next_object_id = []
    valid_state = []
    trajectory_category = []
    completion = []
    future_completion = []
    object_present = []
    future_object_present = []
    object_colors = []
    object_bboxes = []
    future_object_bboxes = []
    object_centroids = []
    object_areas = []
    object_shapes = []
    object_part_counts = []
    object_missing = []
    future_object_missing = []
    object_overgrowth = []
    future_object_overgrowth = []
    object_wrong_color = []
    future_object_wrong_color = []
    object_maps = []
    future_object_maps = []
    action_object_ids = []
    action_object_shapes = []
    action_process_types = []
    continues_object = []
    relation_present = []
    relations = []
    max_objects = generator.spec.max_objects

    for _ in range(batch_size):
        trajectory = generator.sample_trajectory(rng, min_actions=horizon)
        starts = trajectory.sample_start_indices(horizon)
        if not starts.size:
            raise RuntimeError(f"Trajectory {trajectory.kind!r} has no {horizon}-step {trajectory.category.name} segment.")
        start = int(starts[int(rng.integers(0, len(starts)))])
        states.append(trajectory.states[start])
        actions.append(np.stack([action.as_array() for action in trajectory.actions[start : start + horizon]]))
        futures.append(trajectory.states[start + 1 : start + horizon + 1])
        metadata = _object_metadata(trajectory, start, max_objects=max_objects)
        future_metadata = [
            _object_metadata(trajectory, start + step, max_objects=max_objects) for step in range(1, horizon + 1)
        ]
        object_count.append(int(np.count_nonzero(metadata[0])))
        scene_object_count.append(min(trajectory.scene.object_count, max_objects))
        future_object_count.append([int(np.count_nonzero(item[0])) for item in future_metadata])
        previous_id = trajectory.action_object_ids[start - 1] if start > 0 else -1
        next_id = trajectory.action_object_ids[start]
        current_object_id.append(_visible_slot_id(trajectory, start, previous_id, max_objects=max_objects))
        next_object_id.append(_action_slot_id(trajectory, start, next_id, max_objects=max_objects))
        valid_state.append(float(trajectory.state_validity[start]))
        trajectory_category.append(int(trajectory.category))
        completion.append(_completion_vector(trajectory, start, max_objects=max_objects))
        future_completion.append(
            np.stack(
                [_completion_vector(trajectory, start + step, max_objects=max_objects) for step in range(1, horizon + 1)]
            )
        )
        object_present.append(metadata[0])
        future_object_present.append(np.stack([item[0] for item in future_metadata]))
        object_colors.append(metadata[1])
        object_bboxes.append(metadata[2])
        future_object_bboxes.append(np.stack([item[2] for item in future_metadata]))
        object_centroids.append(metadata[3])
        object_areas.append(metadata[4])
        object_shapes.append(metadata[5])
        object_part_counts.append(metadata[9])
        object_missing.append(metadata[6])
        future_object_missing.append(np.stack([item[6] for item in future_metadata]))
        object_overgrowth.append(metadata[7])
        future_object_overgrowth.append(np.stack([item[7] for item in future_metadata]))
        object_wrong_color.append(metadata[8])
        future_object_wrong_color.append(np.stack([item[8] for item in future_metadata]))
        object_maps.append(_visible_object_map(trajectory, start, max_objects=max_objects))
        future_object_maps.append(
            np.stack(
                [
                    _visible_object_map(trajectory, start + step, max_objects=max_objects)
                    for step in range(1, horizon + 1)
                ]
            )
        )
        sampled_object_ids = np.asarray(trajectory.action_object_ids[start : start + horizon], dtype=np.int64)
        action_object_ids.append(
            np.asarray(
                [
                    _action_slot_id(trajectory, start + step, int(object_id), max_objects=max_objects)
                    for step, object_id in enumerate(sampled_object_ids)
                ],
                dtype=np.int64,
            )
        )
        action_object_shapes.append(
            np.asarray(
                [
                    _shape_index(trajectory, int(object_id))
                    for object_id in sampled_object_ids
                ],
                dtype=np.int64,
            )
        )
        action_process_types.append(
            np.asarray(
                [_process_index(trajectory, start + step) for step in range(horizon)],
                dtype=np.int64,
            )
        )
        continues_object.append(float(previous_id >= 0 and previous_id == trajectory.action_object_ids[start]))
        pair_present, pair_relations = _object_relations(trajectory, start, max_objects=max_objects)
        relation_present.append(pair_present)
        relations.append(pair_relations)

    return ObjectDynamicsBatch(
        states=torch.as_tensor(np.stack(states), dtype=torch.long, device=device),
        actions=torch.as_tensor(np.stack(actions), dtype=torch.long, device=device),
        futures=torch.as_tensor(np.stack(futures), dtype=torch.long, device=device),
        object_count=torch.as_tensor(object_count, dtype=torch.long, device=device),
        scene_object_count=torch.as_tensor(scene_object_count, dtype=torch.long, device=device),
        future_object_count=torch.as_tensor(np.stack(future_object_count), dtype=torch.long, device=device),
        current_object_id=torch.as_tensor(current_object_id, dtype=torch.long, device=device),
        next_object_id=torch.as_tensor(next_object_id, dtype=torch.long, device=device),
        valid_state=torch.as_tensor(valid_state, dtype=torch.float32, device=device),
        trajectory_category=torch.as_tensor(trajectory_category, dtype=torch.long, device=device),
        completion=torch.as_tensor(np.stack(completion), dtype=torch.float32, device=device),
        future_completion=torch.as_tensor(np.stack(future_completion), dtype=torch.float32, device=device),
        object_present=torch.as_tensor(np.stack(object_present), dtype=torch.bool, device=device),
        future_object_present=torch.as_tensor(np.stack(future_object_present), dtype=torch.bool, device=device),
        object_colors=torch.as_tensor(np.stack(object_colors), dtype=torch.long, device=device),
        object_bboxes=torch.as_tensor(np.stack(object_bboxes), dtype=torch.float32, device=device),
        future_object_bboxes=torch.as_tensor(np.stack(future_object_bboxes), dtype=torch.float32, device=device),
        object_centroids=torch.as_tensor(np.stack(object_centroids), dtype=torch.float32, device=device),
        object_areas=torch.as_tensor(np.stack(object_areas), dtype=torch.float32, device=device),
        object_shapes=torch.as_tensor(np.stack(object_shapes), dtype=torch.long, device=device),
        object_part_counts=torch.as_tensor(np.stack(object_part_counts), dtype=torch.long, device=device),
        object_missing=torch.as_tensor(np.stack(object_missing), dtype=torch.float32, device=device),
        future_object_missing=torch.as_tensor(np.stack(future_object_missing), dtype=torch.float32, device=device),
        object_overgrowth=torch.as_tensor(np.stack(object_overgrowth), dtype=torch.float32, device=device),
        future_object_overgrowth=torch.as_tensor(np.stack(future_object_overgrowth), dtype=torch.float32, device=device),
        object_wrong_color=torch.as_tensor(np.stack(object_wrong_color), dtype=torch.float32, device=device),
        future_object_wrong_color=torch.as_tensor(
            np.stack(future_object_wrong_color), dtype=torch.float32, device=device
        ),
        object_map=torch.as_tensor(np.stack(object_maps), dtype=torch.long, device=device),
        future_object_map=torch.as_tensor(np.stack(future_object_maps), dtype=torch.long, device=device),
        action_object_ids=torch.as_tensor(np.stack(action_object_ids), dtype=torch.long, device=device),
        action_object_shapes=torch.as_tensor(np.stack(action_object_shapes), dtype=torch.long, device=device),
        action_process_types=torch.as_tensor(np.stack(action_process_types), dtype=torch.long, device=device),
        continues_object=torch.as_tensor(continues_object, dtype=torch.float32, device=device),
        relation_present=torch.as_tensor(np.stack(relation_present), dtype=torch.bool, device=device),
        relations=torch.as_tensor(np.stack(relations), dtype=torch.float32, device=device),
    )


def _completion_vector(trajectory: ObjectTrajectory, state_index: int, *, max_objects: int) -> np.ndarray:
    state = trajectory.states[state_index]
    values = np.zeros((max_objects,), dtype=np.float32)
    for object_id, _ in _visible_objects(trajectory, state_index, max_objects=max_objects):
        slot = object_id
        obj = trajectory.scene.objects[object_id]
        target_cells = obj.mask
        correct = np.count_nonzero((state == obj.color) & target_cells)
        values[slot] = correct / max(1, obj.area)
    return values


def _object_metadata(
    trajectory: ObjectTrajectory,
    state_index: int,
    *,
    max_objects: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    present = np.zeros((max_objects,), dtype=bool)
    colors = np.zeros((max_objects,), dtype=np.int64)
    bboxes = np.zeros((max_objects, 4), dtype=np.float32)
    centroids = np.zeros((max_objects, 2), dtype=np.float32)
    areas = np.zeros((max_objects,), dtype=np.float32)
    shapes = np.zeros((max_objects,), dtype=np.int64)
    missing = np.zeros((max_objects,), dtype=np.float32)
    overgrowth = np.zeros((max_objects,), dtype=np.float32)
    wrong_color = np.zeros((max_objects,), dtype=np.float32)
    part_counts = np.zeros((max_objects,), dtype=np.int64)
    grid_size = trajectory.scene.grid.shape[0]
    state = trajectory.states[state_index]
    for object_id, visible_mask in _visible_objects(trajectory, state_index, max_objects=max_objects):
        slot = object_id
        obj = trajectory.scene.objects[object_id]
        present[slot] = True
        visible_colors = state[visible_mask]
        colors[slot] = int(np.bincount(visible_colors).argmax())
        rows, cols = np.where(visible_mask)
        bbox = (int(rows.min()), int(cols.min()), int(rows.max()) + 1, int(cols.max()) + 1)
        bboxes[slot] = np.asarray(bbox, dtype=np.float32) / grid_size
        centroids[slot] = np.asarray((rows.mean(), cols.mean()), dtype=np.float32) / max(1, grid_size - 1)
        areas[slot] = np.count_nonzero(visible_mask) / trajectory.scene.grid.size
        shapes[slot] = _shape_index(trajectory, object_id)
        target_area = max(1, obj.area)
        missing[slot] = np.count_nonzero(obj.mask & (state != obj.color)) / target_area
        overgrowth[slot] = np.count_nonzero(visible_mask & ~obj.mask) / target_area
        wrong_color[slot] = np.count_nonzero(visible_mask & obj.mask & (state != obj.color)) / target_area
        part_counts[slot] = len(obj.parts)
    return present, colors, bboxes, centroids, areas, shapes, missing, overgrowth, wrong_color, part_counts


def _object_relations(
    trajectory: ObjectTrajectory,
    state_index: int,
    *,
    max_objects: int,
) -> tuple[np.ndarray, np.ndarray]:
    pair_count = max_objects * (max_objects - 1) // 2
    present = np.zeros((pair_count,), dtype=bool)
    relations = np.zeros((pair_count, len(RELATION_NAMES)), dtype=np.float32)
    state = trajectory.states[state_index]
    visible = dict(_visible_objects(trajectory, state_index, max_objects=max_objects))
    pair_index = 0
    for left_id in range(max_objects):
        for right_id in range(left_id + 1, max_objects):
            if left_id in visible and right_id in visible:
                left_mask = visible[left_id]
                right_mask = visible[right_id]
                left = trajectory.scene.objects[left_id]
                right = trajectory.scene.objects[right_id]
                left_color = int(np.bincount(state[left_mask]).argmax())
                right_color = int(np.bincount(state[right_mask]).argmax())
                left_centroid = np.argwhere(left_mask).mean(axis=0)
                right_centroid = np.argwhere(right_mask).mean(axis=0)
                present[pair_index] = True
                relations[pair_index] = np.asarray(
                    [
                        left_color == right_color,
                        left.shape_type == right.shape_type,
                        _masks_touch(left_mask, right_mask),
                        left_centroid[1] < right_centroid[1],
                        _mask_inside(left.mask, right.mask) or _mask_inside(right.mask, left.mask),
                    ],
                    dtype=np.float32,
                )
            pair_index += 1
    return present, relations


def _visible_objects(
    trajectory: ObjectTrajectory,
    state_index: int,
    *,
    max_objects: int,
) -> list[tuple[int, np.ndarray]]:
    state_object_map = trajectory.object_maps[state_index]
    visible = []
    for object_id in range(trajectory.scene.object_count):
        mask = state_object_map == object_id
        if bool(np.any(mask)):
            visible.append((object_id, mask))
    return visible[:max_objects]


def _visible_object_map(trajectory: ObjectTrajectory, state_index: int, *, max_objects: int) -> np.ndarray:
    output = np.zeros(trajectory.scene.grid.shape, dtype=np.int64)
    for object_id, mask in _visible_objects(trajectory, state_index, max_objects=max_objects):
        output[mask] = object_id + 1
    return output


def _visible_slot_id(
    trajectory: ObjectTrajectory,
    state_index: int,
    object_id: int,
    *,
    max_objects: int,
) -> int:
    if object_id < 0:
        return 0
    for candidate, _ in _visible_objects(trajectory, state_index, max_objects=max_objects):
        if candidate == object_id:
            return candidate + 1
    return 0


def _action_slot_id(
    trajectory: ObjectTrajectory,
    state_index: int,
    object_id: int,
    *,
    max_objects: int,
) -> int:
    if 0 <= object_id < min(trajectory.scene.object_count, max_objects):
        return object_id + 1
    return 0


def _shape_index(trajectory: ObjectTrajectory, object_id: int) -> int:
    if not (0 <= object_id < trajectory.scene.object_count):
        return 0
    shape = trajectory.scene.objects[object_id].shape_type.removeprefix("transformed_")
    return SHAPE_TYPES.index(shape) + 1 if shape in SHAPE_TYPES else 0


def _process_index(trajectory: ObjectTrajectory, action_index: int) -> int:
    action = trajectory.actions[action_index]
    object_id = trajectory.action_object_ids[action_index]
    kind = trajectory.kind.removeprefix("counterfactual_").removeprefix("wrong_")
    if object_id < 0 or kind == "random_off_manifold":
        return PROCESS_NAMES.index("random")
    if action.op.name == "ERASE":
        return PROCESS_NAMES.index("trim")
    if action.op.name == "RECOLOR":
        return PROCESS_NAMES.index("recolor")
    if kind == "completion":
        return PROCESS_NAMES.index("complete")
    if kind == "transform_identity":
        return PROCESS_NAMES.index("transform")
    if kind == "noisy_repair":
        return PROCESS_NAMES.index("repair_fill")
    return PROCESS_NAMES.index("build")


def _mask_inside(inner: np.ndarray, container: np.ndarray) -> bool:
    if bool(np.any(inner & container)):
        return False
    exterior = np.zeros_like(container, dtype=bool)
    stack = [
        (row, col)
        for row in range(container.shape[0])
        for col in range(container.shape[1])
        if (row in {0, container.shape[0] - 1} or col in {0, container.shape[1] - 1}) and not container[row, col]
    ]
    for cell in stack:
        exterior[cell] = True
    while stack:
        row, col = stack.pop()
        for row_offset, col_offset in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor = (row + row_offset, col + col_offset)
            if not (0 <= neighbor[0] < container.shape[0] and 0 <= neighbor[1] < container.shape[1]):
                continue
            if not container[neighbor] and not exterior[neighbor]:
                exterior[neighbor] = True
                stack.append(neighbor)
    enclosed = ~container & ~exterior
    return bool(np.any(inner)) and bool(np.all(enclosed[inner]))


def _masks_touch(left: np.ndarray, right: np.ndarray) -> bool:
    for row, col in np.argwhere(left):
        row0, row1 = max(0, int(row) - 1), min(left.shape[0], int(row) + 2)
        col0, col1 = max(0, int(col) - 1), min(left.shape[1], int(col) + 2)
        if bool(np.any(right[row0:row1, col0:col1])):
            return True
    return False
