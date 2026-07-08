from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from puzzle_jepa.object_dynamics.domain import ActionOp, LowLevelAction, ObjectSpec, ObjectTrajectory, SceneSpec, apply_low_level_action
from puzzle_jepa.object_dynamics.shapes import SHAPE_TYPES, frontier_order, place_mask, sample_shape_mask, transform_mask


TRAJECTORY_KINDS = (
    "object_blocked",
    "frontier_build",
    "random_within_object",
    "interleaved_build",
    "completion",
    "transform_identity",
    "noisy_repair",
    "global_random",
    "semantic_mix",
)


@dataclass(frozen=True, slots=True)
class ObjectDynamicsSpec:
    grid_size: int = 16
    num_colors: int = 10
    min_objects: int = 1
    max_objects: int = 4
    max_shape_extent: int = 6
    trajectory_kind: str = "frontier_build"
    counterfactual_ratio: float = 0.15
    wrong_ratio: float = 0.05
    max_scene_retries: int = 128


class ObjectDynamicsGenerator:
    def __init__(self, spec: ObjectDynamicsSpec):
        if spec.trajectory_kind not in TRAJECTORY_KINDS:
            raise ValueError(f"Unknown trajectory_kind {spec.trajectory_kind!r}.")
        self.spec = spec

    def sample_scene(self, rng: np.random.Generator) -> SceneSpec:
        shape = (self.spec.grid_size, self.spec.grid_size)
        occupied = np.zeros(shape, dtype=bool)
        grid = np.zeros(shape, dtype=np.int64)
        objects: list[ObjectSpec] = []
        object_count = int(rng.integers(self.spec.min_objects, self.spec.max_objects + 1))

        for object_id in range(object_count):
            placed = False
            for _ in range(self.spec.max_scene_retries):
                shape_type = str(rng.choice(SHAPE_TYPES))
                local_mask = sample_shape_mask(rng, shape_type, max_extent=self.spec.max_shape_extent)
                if local_mask.shape[0] >= shape[0] or local_mask.shape[1] >= shape[1]:
                    continue
                row = int(rng.integers(0, shape[0] - local_mask.shape[0] + 1))
                col = int(rng.integers(0, shape[1] - local_mask.shape[1] + 1))
                mask = place_mask(local_mask, row, col, shape)
                expanded = _dilate(mask)
                if bool(np.any(occupied & expanded)):
                    continue
                color = int(rng.integers(1, self.spec.num_colors))
                grid[mask] = color
                occupied |= mask
                objects.append(ObjectSpec(object_id=object_id, shape_type=shape_type, color=color, mask=mask))
                placed = True
                break
            if not placed and not objects:
                raise RuntimeError("Could not place any object in synthetic scene.")
        return SceneSpec(grid=grid, objects=tuple(objects))

    def sample_trajectory(self, rng: np.random.Generator, *, min_actions: int = 1) -> ObjectTrajectory:
        for _ in range(self.spec.max_scene_retries):
            scene = self.sample_scene(rng)
            kind = self._resolve_kind(rng)
            if rng.random() < self.spec.wrong_ratio:
                trajectory = self._sample_wrong_trajectory(scene, rng, base_kind=kind)
            else:
                if kind == "frontier_build":
                    trajectory = self._frontier_build(scene, rng)
                elif kind == "object_blocked":
                    trajectory = self._object_blocked(scene, rng)
                elif kind == "random_within_object":
                    trajectory = self._random_within_object(scene, rng)
                elif kind == "interleaved_build":
                    trajectory = self._interleaved_build(scene, rng)
                elif kind == "completion":
                    trajectory = self._completion(scene, rng)
                elif kind == "transform_identity":
                    trajectory = self._transform_identity(scene, rng)
                elif kind == "noisy_repair":
                    trajectory = self._noisy_repair(scene, rng)
                elif kind == "global_random":
                    trajectory = self._global_random(scene, rng)
                else:
                    raise ValueError(f"Unsupported resolved trajectory kind {kind!r}.")
            if len(trajectory.actions) >= min_actions:
                if self.spec.counterfactual_ratio > 0.0 and rng.random() < self.spec.counterfactual_ratio:
                    return self._inject_counterfactual_suffix(trajectory, rng)
                return trajectory
        raise RuntimeError(f"Could not sample trajectory with at least {min_actions} actions.")

    def _resolve_kind(self, rng: np.random.Generator) -> str:
        if self.spec.trajectory_kind != "semantic_mix":
            return self.spec.trajectory_kind
        return str(
            rng.choice(
                [
                    "object_blocked",
                    "frontier_build",
                    "random_within_object",
                    "interleaved_build",
                    "completion",
                    "transform_identity",
                    "noisy_repair",
                ],
                p=np.asarray([0.15, 0.20, 0.15, 0.15, 0.15, 0.10, 0.10]),
            )
        )

    def _object_blocked(self, scene: SceneSpec, rng: np.random.Generator) -> ObjectTrajectory:
        actions: list[LowLevelAction] = []
        object_ids: list[int] = []
        for obj in _shuffle_objects(scene.objects, rng):
            for row, col in sorted(tuple(int(x) for x in cell) for cell in np.argwhere(obj.mask)):
                actions.append(LowLevelAction(ActionOp.PAINT, row, col, obj.color))
                object_ids.append(obj.object_id)
        return _rollout(np.zeros_like(scene.grid), actions, object_ids, scene, kind="object_blocked", semantic=True)

    def _frontier_build(self, scene: SceneSpec, rng: np.random.Generator) -> ObjectTrajectory:
        actions: list[LowLevelAction] = []
        object_ids: list[int] = []
        for obj in _shuffle_objects(scene.objects, rng):
            for row, col in frontier_order(obj.mask, rng):
                actions.append(LowLevelAction(ActionOp.PAINT, row, col, obj.color))
                object_ids.append(obj.object_id)
        return _rollout(np.zeros_like(scene.grid), actions, object_ids, scene, kind="frontier_build", semantic=True)

    def _random_within_object(self, scene: SceneSpec, rng: np.random.Generator) -> ObjectTrajectory:
        actions: list[LowLevelAction] = []
        object_ids: list[int] = []
        for obj in _shuffle_objects(scene.objects, rng):
            cells = [tuple(int(x) for x in cell) for cell in np.argwhere(obj.mask)]
            rng.shuffle(cells)
            for row, col in cells:
                actions.append(LowLevelAction(ActionOp.PAINT, row, col, obj.color))
                object_ids.append(obj.object_id)
        return _rollout(np.zeros_like(scene.grid), actions, object_ids, scene, kind="random_within_object", semantic=True)

    def _interleaved_build(self, scene: SceneSpec, rng: np.random.Generator) -> ObjectTrajectory:
        remaining = {obj.object_id: [tuple(int(x) for x in cell) for cell in np.argwhere(obj.mask)] for obj in scene.objects}
        colors = {obj.object_id: obj.color for obj in scene.objects}
        for cells in remaining.values():
            rng.shuffle(cells)
        actions: list[LowLevelAction] = []
        object_ids: list[int] = []
        while any(remaining.values()):
            active = sorted(object_id for object_id, cells in remaining.items() if cells)
            object_id = active[int(rng.integers(0, len(active)))]
            row, col = remaining[object_id].pop()
            actions.append(LowLevelAction(ActionOp.PAINT, row, col, colors[object_id]))
            object_ids.append(object_id)
        return _rollout(np.zeros_like(scene.grid), actions, object_ids, scene, kind="interleaved_build", semantic=True)

    def _global_random(self, scene: SceneSpec, rng: np.random.Generator) -> ObjectTrajectory:
        actions: list[LowLevelAction] = []
        object_ids: list[int] = []
        items = []
        for obj in scene.objects:
            for row, col in np.argwhere(obj.mask):
                items.append((int(row), int(col), obj.color, obj.object_id))
        rng.shuffle(items)
        for row, col, color, object_id in items:
            actions.append(LowLevelAction(ActionOp.PAINT, row, col, int(color)))
            object_ids.append(int(object_id))
        return _rollout(np.zeros_like(scene.grid), actions, object_ids, scene, kind="global_random", semantic=True)

    def _completion(self, scene: SceneSpec, rng: np.random.Generator) -> ObjectTrajectory:
        start = scene.grid.copy()
        actions: list[LowLevelAction] = []
        object_ids: list[int] = []
        for obj in scene.objects:
            cells = [tuple(int(x) for x in cell) for cell in np.argwhere(obj.mask)]
            rng.shuffle(cells)
            remove_count = max(1, int(round(0.35 * len(cells))))
            for row, col in cells[:remove_count]:
                start[row, col] = 0
                actions.append(LowLevelAction(ActionOp.PAINT, row, col, obj.color))
                object_ids.append(obj.object_id)
        return _rollout(start, actions, object_ids, scene, kind="completion", semantic=True)

    def _transform_identity(self, scene: SceneSpec, rng: np.random.Generator) -> ObjectTrajectory:
        obj = scene.objects[int(rng.integers(0, len(scene.objects)))]
        source = np.zeros_like(scene.grid)
        source[obj.mask] = obj.color
        target_mask = _place_transformed_object(obj.mask, rng)
        target_color = int(rng.integers(1, self.spec.num_colors))
        target = np.zeros_like(scene.grid)
        target[target_mask] = target_color
        transformed_scene = SceneSpec(
            grid=target,
            objects=(ObjectSpec(object_id=0, shape_type=f"transformed_{obj.shape_type}", color=target_color, mask=target_mask),),
        )
        actions, object_ids = _repair_actions(source, target, object_id=0, rng=rng)
        return _rollout(source, actions, object_ids, transformed_scene, kind="transform_identity", semantic=True)

    def _noisy_repair(self, scene: SceneSpec, rng: np.random.Generator) -> ObjectTrajectory:
        start = scene.grid.copy()
        for obj in scene.objects:
            cells = [tuple(int(x) for x in cell) for cell in np.argwhere(obj.mask)]
            rng.shuffle(cells)
            for row, col in cells[: max(1, len(cells) // 5)]:
                start[row, col] = 0 if bool(rng.integers(0, 2)) else int(rng.integers(1, self.spec.num_colors))
            row0, col0, row1, col1 = obj.bbox
            candidates = []
            for row in range(max(0, row0 - 1), min(scene.grid.shape[0], row1 + 1)):
                for col in range(max(0, col0 - 1), min(scene.grid.shape[1], col1 + 1)):
                    if not obj.mask[row, col]:
                        candidates.append((row, col))
            if candidates:
                row, col = candidates[int(rng.integers(0, len(candidates)))]
                start[row, col] = obj.color
        actions, object_ids = _repair_actions(start, scene.grid, object_id=-1, rng=rng, object_map=scene.object_map)
        return _rollout(start, actions, object_ids, scene, kind="noisy_repair", semantic=True)

    def _sample_wrong_trajectory(self, scene: SceneSpec, rng: np.random.Generator, *, base_kind: str) -> ObjectTrajectory:
        trajectory = self._global_random(scene, rng) if base_kind == "global_random" else self._frontier_build(scene, rng)
        state = trajectory.states[-1].copy()
        actions: list[LowLevelAction] = []
        object_ids: list[int] = []
        for _ in range(max(1, len(trajectory.actions) // 6)):
            row = int(rng.integers(0, state.shape[0]))
            col = int(rng.integers(0, state.shape[1]))
            color = int(rng.integers(1, self.spec.num_colors))
            actions.append(LowLevelAction(ActionOp.RECOLOR, row, col, color))
            object_ids.append(-1)
        wrong = _rollout(state, actions, object_ids, scene, kind=f"wrong_{base_kind}", semantic=False)
        states = np.concatenate([trajectory.states, wrong.states[1:]], axis=0)
        return ObjectTrajectory(
            states=states,
            actions=(*trajectory.actions, *wrong.actions),
            action_object_ids=(*trajectory.action_object_ids, *wrong.action_object_ids),
            scene=scene,
            kind=f"wrong_{base_kind}",
            semantic=False,
        )

    def _inject_counterfactual_suffix(self, trajectory: ObjectTrajectory, rng: np.random.Generator) -> ObjectTrajectory:
        if not trajectory.actions:
            return trajectory
        prefix_len = int(rng.integers(0, len(trajectory.actions)))
        state = trajectory.states[prefix_len].copy()
        actions = list(trajectory.actions[:prefix_len])
        object_ids = list(trajectory.action_object_ids[:prefix_len])
        for _ in range(max(1, min(4, len(trajectory.actions) - prefix_len))):
            row = int(rng.integers(0, state.shape[0]))
            col = int(rng.integers(0, state.shape[1]))
            color = int(rng.integers(0, self.spec.num_colors))
            op = ActionOp.ERASE if color == 0 else ActionOp.PAINT
            action = LowLevelAction(op, row, col, color)
            actions.append(action)
            object_ids.append(int(trajectory.scene.object_map[row, col]))
            state = apply_low_level_action(state, action)
        return _rollout(trajectory.states[0], actions, object_ids, trajectory.scene, kind=f"counterfactual_{trajectory.kind}", semantic=trajectory.semantic)


def _rollout(
    initial: np.ndarray,
    actions: list[LowLevelAction],
    object_ids: list[int],
    scene: SceneSpec,
    *,
    kind: str,
    semantic: bool,
) -> ObjectTrajectory:
    states = [np.asarray(initial, dtype=np.int64).copy()]
    current = states[0]
    for action in actions:
        current = apply_low_level_action(current, action)
        states.append(current)
    return ObjectTrajectory(
        states=np.stack(states),
        actions=tuple(actions),
        action_object_ids=tuple(int(x) for x in object_ids),
        scene=scene,
        kind=kind,
        semantic=semantic,
    )


def _shuffle_objects(objects: tuple[ObjectSpec, ...], rng: np.random.Generator) -> list[ObjectSpec]:
    shuffled = list(objects)
    rng.shuffle(shuffled)
    return shuffled


def _dilate(mask: np.ndarray) -> np.ndarray:
    output = mask.copy()
    for row, col in np.argwhere(mask):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = int(row) + dr, int(col) + dc
                if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1]:
                    output[rr, cc] = True
    return output


def _place_transformed_object(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    local = transform_mask(_crop_mask(mask), rng)
    shape = mask.shape
    row = int(rng.integers(0, shape[0] - local.shape[0] + 1))
    col = int(rng.integers(0, shape[1] - local.shape[1] + 1))
    return place_mask(local, row, col, shape)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    return mask[int(rows.min()) : int(rows.max()) + 1, int(cols.min()) : int(cols.max()) + 1]


def _repair_actions(
    start: np.ndarray,
    target: np.ndarray,
    *,
    object_id: int,
    rng: np.random.Generator,
    object_map: np.ndarray | None = None,
) -> tuple[list[LowLevelAction], list[int]]:
    actions: list[LowLevelAction] = []
    object_ids: list[int] = []
    cells = [tuple(int(x) for x in cell) for cell in np.argwhere(start != target)]
    rng.shuffle(cells)
    for row, col in cells:
        color = int(target[row, col])
        op = ActionOp.ERASE if color == 0 else (ActionOp.PAINT if start[row, col] == 0 else ActionOp.RECOLOR)
        actions.append(LowLevelAction(op, row, col, color))
        object_ids.append(int(object_map[row, col]) if object_map is not None else object_id)
    return actions, object_ids
