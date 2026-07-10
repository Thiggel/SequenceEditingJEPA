from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from puzzle_jepa.object_dynamics.domain import (
    ActionOp,
    LowLevelAction,
    ObjectSpec,
    ObjectTrajectory,
    SceneSpec,
    TrajectoryCategory,
    apply_low_level_action,
)
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
    "random_off_manifold",
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
    duplicate_shape_ratio: float = 0.25
    same_color_ratio: float = 0.25
    max_scene_retries: int = 128


class ObjectDynamicsGenerator:
    def __init__(self, spec: ObjectDynamicsSpec):
        if spec.trajectory_kind not in TRAJECTORY_KINDS:
            raise ValueError(f"Unknown trajectory_kind {spec.trajectory_kind!r}.")
        if not (0.0 <= spec.counterfactual_ratio <= 1.0 and 0.0 <= spec.wrong_ratio <= 1.0):
            raise ValueError("counterfactual_ratio and wrong_ratio must lie in [0, 1].")
        if spec.counterfactual_ratio + spec.wrong_ratio > 1.0:
            raise ValueError("counterfactual_ratio + wrong_ratio must not exceed 1.")
        if not (0.0 <= spec.duplicate_shape_ratio <= 1.0 and 0.0 <= spec.same_color_ratio <= 1.0):
            raise ValueError("Scene relation ratios must lie in [0, 1].")
        if not (1 <= spec.min_objects <= spec.max_objects):
            raise ValueError("Object count bounds must satisfy 1 <= min_objects <= max_objects.")
        if spec.num_colors < 2:
            raise ValueError("num_colors must include background and at least one foreground color.")
        self.spec = spec

    def sample_scene(self, rng: np.random.Generator) -> SceneSpec:
        shape = (self.spec.grid_size, self.spec.grid_size)
        for _ in range(self.spec.max_scene_retries):
            occupied = np.zeros(shape, dtype=bool)
            grid = np.zeros(shape, dtype=np.int64)
            objects: list[ObjectSpec] = []
            object_count = int(rng.integers(self.spec.min_objects, self.spec.max_objects + 1))

            for object_id in range(object_count):
                placed = False
                for _ in range(max(16, self.spec.max_scene_retries)):
                    duplicate_source = None
                    if objects and rng.random() < self.spec.duplicate_shape_ratio:
                        duplicate_source = objects[int(rng.integers(0, len(objects)))]
                    if duplicate_source is None:
                        shape_type = str(rng.choice(SHAPE_TYPES))
                        local_mask = sample_shape_mask(rng, shape_type, max_extent=self.spec.max_shape_extent)
                    else:
                        shape_type = duplicate_source.shape_type
                        local_mask = _crop_mask(duplicate_source.mask)
                    if local_mask.shape[0] > shape[0] or local_mask.shape[1] > shape[1]:
                        continue
                    row = int(rng.integers(0, shape[0] - local_mask.shape[0] + 1))
                    col = int(rng.integers(0, shape[1] - local_mask.shape[1] + 1))
                    mask = place_mask(local_mask, row, col, shape)
                    if bool(np.any(occupied & mask)):
                        continue
                    if objects and rng.random() < self.spec.same_color_ratio:
                        color = objects[int(rng.integers(0, len(objects)))].color
                    else:
                        color = int(rng.integers(1, self.spec.num_colors))
                        if duplicate_source is not None and self.spec.num_colors > 2:
                            while color == duplicate_source.color:
                                color = int(rng.integers(1, self.spec.num_colors))
                    touching_colors = {obj.color for obj in objects if _masks_touch(mask, obj.mask)}
                    if color in touching_colors:
                        choices = [candidate for candidate in range(1, self.spec.num_colors) if candidate not in touching_colors]
                        if not choices:
                            continue
                        color = int(choices[int(rng.integers(0, len(choices)))])
                    grid[mask] = color
                    occupied |= mask
                    objects.append(ObjectSpec(object_id=object_id, shape_type=shape_type, color=color, mask=mask))
                    placed = True
                    break
                if not placed:
                    break
            if len(objects) == object_count:
                return SceneSpec(grid=grid, objects=_canonicalize_objects(objects))
        raise RuntimeError(
            f"Could not place {self.spec.min_objects}-{self.spec.max_objects} objects "
            f"after {self.spec.max_scene_retries} scene attempts."
        )

    def sample_trajectory(self, rng: np.random.Generator, *, min_actions: int = 1) -> ObjectTrajectory:
        for _ in range(self.spec.max_scene_retries):
            scene = self.sample_scene(rng)
            kind = self._resolve_kind(rng)
            if kind == "random_off_manifold":
                trajectory = self._random_off_manifold(scene, rng)
                if trajectory.sample_start_indices(min_actions).size:
                    return trajectory
                continue
            base = self._sample_semantic_trajectory(scene, rng, kind=kind)
            if len(base.actions) < min_actions:
                continue
            category_draw = float(rng.random())
            if category_draw < self.spec.wrong_ratio:
                trajectory = self._sample_wrong_trajectory(base, rng, min_actions=min_actions)
            elif category_draw < self.spec.wrong_ratio + self.spec.counterfactual_ratio:
                trajectory = self._inject_counterfactual_suffix(base, rng, min_actions=min_actions)
            else:
                trajectory = base
            if trajectory.sample_start_indices(min_actions).size:
                return trajectory
        raise RuntimeError(f"Could not sample trajectory with at least {min_actions} actions.")

    def _sample_semantic_trajectory(
        self,
        scene: SceneSpec,
        rng: np.random.Generator,
        *,
        kind: str,
    ) -> ObjectTrajectory:
        if kind == "frontier_build":
            return self._frontier_build(scene, rng)
        if kind == "object_blocked":
            return self._object_blocked(scene, rng)
        if kind == "random_within_object":
            return self._random_within_object(scene, rng)
        if kind == "interleaved_build":
            return self._interleaved_build(scene, rng)
        if kind == "completion":
            return self._completion(scene, rng)
        if kind == "transform_identity":
            return self._transform_identity(scene, rng)
        if kind == "noisy_repair":
            return self._noisy_repair(scene, rng)
        if kind == "global_random":
            return self._global_random(scene, rng)
        raise ValueError(f"Unsupported resolved trajectory kind {kind!r}.")

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

    def _random_off_manifold(self, scene: SceneSpec, rng: np.random.Generator) -> ObjectTrajectory:
        initial = np.zeros_like(scene.grid)
        noise_mask = rng.random(initial.shape) < 0.3
        initial[noise_mask] = rng.integers(1, self.spec.num_colors, size=int(np.count_nonzero(noise_mask)))
        current = initial.copy()
        actions: list[LowLevelAction] = []
        object_ids: list[int] = []
        action_count = max(32, 2 * self.spec.grid_size)
        for _ in range(action_count):
            row = int(rng.integers(0, current.shape[0]))
            col = int(rng.integers(0, current.shape[1]))
            if current[row, col] == 0:
                op = ActionOp.PAINT
                color = int(rng.integers(1, self.spec.num_colors))
            elif bool(rng.integers(0, 2)):
                op = ActionOp.ERASE
                color = 0
            else:
                op = ActionOp.RECOLOR
                choices = [color for color in range(1, self.spec.num_colors) if color != int(current[row, col])]
                color = int(choices[int(rng.integers(0, len(choices)))])
            action = LowLevelAction(op, row, col, color)
            actions.append(action)
            object_ids.append(-1)
            current = apply_low_level_action(current, action)
        return _rollout(
            initial,
            actions,
            object_ids,
            scene,
            kind="random_off_manifold",
            semantic=False,
            category=TrajectoryCategory.WRONG,
            initial_object_map=np.full(initial.shape, -1, dtype=np.int64),
        )

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
        source_object_map = np.full(source.shape, -1, dtype=np.int64)
        source_object_map[obj.mask] = 0
        return _rollout(
            source,
            actions,
            object_ids,
            transformed_scene,
            kind="transform_identity",
            semantic=True,
            initial_object_map=source_object_map,
        )

    def _noisy_repair(self, scene: SceneSpec, rng: np.random.Generator) -> ObjectTrajectory:
        start = scene.grid.copy()
        start_object_map = scene.object_map.copy()
        for obj in scene.objects:
            cells = [tuple(int(x) for x in cell) for cell in np.argwhere(obj.mask)]
            rng.shuffle(cells)
            damage_count = max(1, int(round(0.4 * len(cells))))
            for damage_index, (row, col) in enumerate(cells[:damage_count]):
                if damage_index % 2 == 0:
                    start[row, col] = 0
                    start_object_map[row, col] = -1
                else:
                    choices = [color for color in range(1, self.spec.num_colors) if color != obj.color]
                    start[row, col] = int(choices[int(rng.integers(0, len(choices)))])
            row0, col0, row1, col1 = obj.bbox
            candidates = []
            for row in range(max(0, row0 - 1), min(scene.grid.shape[0], row1 + 1)):
                for col in range(max(0, col0 - 1), min(scene.grid.shape[1], col1 + 1)):
                    if not obj.mask[row, col] and scene.grid[row, col] == 0:
                        candidates.append((row, col))
            rng.shuffle(candidates)
            overgrowth_count = min(len(candidates), max(1, int(round(0.2 * len(cells)))))
            for row, col in candidates[:overgrowth_count]:
                start[row, col] = obj.color
                start_object_map[row, col] = obj.object_id
        actions, object_ids = _repair_actions(
            start,
            scene.grid,
            object_id=-1,
            rng=rng,
            object_map=scene.object_map,
            start_object_map=start_object_map,
        )
        return _rollout(
            start,
            actions,
            object_ids,
            scene,
            kind="noisy_repair",
            semantic=True,
            initial_object_map=start_object_map,
        )

    def _sample_wrong_trajectory(
        self,
        trajectory: ObjectTrajectory,
        rng: np.random.Generator,
        *,
        min_actions: int,
    ) -> ObjectTrajectory:
        state = trajectory.states[-1].copy()
        scene = trajectory.scene
        actions: list[LowLevelAction] = []
        object_ids: list[int] = []
        candidates = _hard_wrong_candidates(scene)
        rng.shuffle(candidates)
        wrong_action_count = max(1, 2 * min_actions + 1)
        for action_index in range(wrong_action_count):
            row, col, object_id = candidates[action_index % len(candidates)]
            target_color = int(scene.grid[row, col])
            if target_color == 0:
                base_color = _nearest_object_color(scene, row, col)
                color = 1 + ((base_color - 1 + action_index // len(candidates)) % (self.spec.num_colors - 1))
                op = ActionOp.PAINT
            else:
                choices = [color for color in range(1, self.spec.num_colors) if color != target_color]
                color = int(choices[(action_index + int(rng.integers(0, len(choices)))) % len(choices)])
                op = ActionOp.RECOLOR
            actions.append(LowLevelAction(op, row, col, color))
            object_ids.append(object_id)
        wrong = _rollout(
            state,
            actions,
            object_ids,
            scene,
            kind=f"wrong_{trajectory.kind}",
            semantic=False,
            category=TrajectoryCategory.WRONG,
            initial_object_map=trajectory.object_maps[-1],
        )
        states = np.concatenate([trajectory.states, wrong.states[1:]], axis=0)
        return ObjectTrajectory(
            states=states,
            object_maps=np.concatenate([trajectory.object_maps, wrong.object_maps[1:]], axis=0),
            actions=(*trajectory.actions, *wrong.actions),
            action_object_ids=(*trajectory.action_object_ids, *wrong.action_object_ids),
            scene=scene,
            kind=f"wrong_{trajectory.kind}",
            semantic=False,
            category=TrajectoryCategory.WRONG,
            transition_categories=(*trajectory.transition_categories, *wrong.transition_categories),
            state_validity=np.concatenate([trajectory.state_validity, wrong.state_validity[1:]]),
        )

    def _inject_counterfactual_suffix(
        self,
        trajectory: ObjectTrajectory,
        rng: np.random.Generator,
        *,
        min_actions: int,
    ) -> ObjectTrajectory:
        if len(trajectory.actions) < min_actions:
            return trajectory
        max_prefix = len(trajectory.actions) - min_actions
        prefix_len = int(rng.integers(0, max_prefix + 1))
        references = list(
            zip(trajectory.actions[prefix_len:], trajectory.action_object_ids[prefix_len:], strict=True)
        )
        state = trajectory.states[prefix_len].copy()
        counterfactual_actions = []
        counterfactual_object_ids = []
        for reference, object_id in references:
            action = _sample_structured_counterfactual(
                state,
                trajectory.scene,
                reference,
                object_id,
                self.spec.num_colors,
                rng,
            )
            counterfactual_actions.append(action)
            counterfactual_object_ids.append(object_id)
            state = apply_low_level_action(state, action)

        actions = [*trajectory.actions[:prefix_len], *counterfactual_actions]
        object_ids = [*trajectory.action_object_ids[:prefix_len], *counterfactual_object_ids]
        categories = [TrajectoryCategory.SEMANTIC] * prefix_len + [
            TrajectoryCategory.COUNTERFACTUAL
        ] * len(counterfactual_actions)
        state_validity = np.asarray(
            [True] * (prefix_len + 1) + [False] * len(counterfactual_actions), dtype=bool
        )
        return _rollout(
            trajectory.states[0],
            actions,
            object_ids,
            trajectory.scene,
            kind=f"counterfactual_{trajectory.kind}",
            semantic=True,
            category=TrajectoryCategory.COUNTERFACTUAL,
            transition_categories=categories,
            initial_object_map=trajectory.object_maps[0],
            state_validity=state_validity,
        )


def _sample_structured_counterfactual(
    state: np.ndarray,
    scene: SceneSpec,
    reference: LowLevelAction,
    object_id: int,
    num_colors: int,
    rng: np.random.Generator,
) -> LowLevelAction:
    candidates: list[LowLevelAction] = []
    row, col = reference.row, reference.col
    current_color = int(state[row, col])
    target_color = int(scene.grid[row, col])
    object_color = (
        scene.objects[object_id].color
        if 0 <= object_id < scene.object_count
        else max(1, reference.color)
    )

    wrong_colors = [
        color
        for color in range(1, num_colors)
        if color != current_color and color != target_color
    ]
    if wrong_colors:
        color = int(wrong_colors[int(rng.integers(0, len(wrong_colors)))])
        op = ActionOp.PAINT if current_color == 0 else ActionOp.RECOLOR
        candidates.append(LowLevelAction(op, row, col, color))

    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr, cc = row + dr, col + dc
            if dr == dc == 0 or not (0 <= rr < state.shape[0] and 0 <= cc < state.shape[1]):
                continue
            if scene.grid[rr, cc] != 0:
                continue
            color = object_color
            if state[rr, cc] == color:
                alternatives = [candidate for candidate in range(1, num_colors) if candidate != color]
                color = int(alternatives[int(rng.integers(0, len(alternatives)))])
            op = ActionOp.PAINT if state[rr, cc] == 0 else ActionOp.RECOLOR
            candidates.append(LowLevelAction(op, rr, cc, color))

    if 0 <= object_id < scene.object_count:
        visible_cells = np.argwhere(scene.objects[object_id].mask & (state != 0))
        if visible_cells.size:
            rr, cc = visible_cells[int(rng.integers(0, len(visible_cells)))]
            candidates.append(LowLevelAction(ActionOp.ERASE, int(rr), int(cc), 0))

        unfinished = np.argwhere(scene.objects[object_id].mask & (state != scene.grid))
        if unfinished.size:
            rr, cc = unfinished[int(rng.integers(0, len(unfinished)))]
            color = int(scene.grid[rr, cc])
            op = ActionOp.PAINT if state[rr, cc] == 0 else ActionOp.RECOLOR
            candidates.append(LowLevelAction(op, int(rr), int(cc), color))

    candidates = [
        action
        for action in candidates
        if action != reference and not np.array_equal(apply_low_level_action(state, action), state)
    ]
    if candidates:
        return candidates[int(rng.integers(0, len(candidates)))]

    if current_color != 0 and reference.op != ActionOp.ERASE:
        return LowLevelAction(ActionOp.ERASE, row, col, 0)
    fallback_colors = [color for color in range(1, num_colors) if color != current_color]
    color = int(fallback_colors[int(rng.integers(0, len(fallback_colors)))])
    op = ActionOp.PAINT if current_color == 0 else ActionOp.RECOLOR
    return LowLevelAction(op, row, col, color)


def _rollout(
    initial: np.ndarray,
    actions: list[LowLevelAction],
    object_ids: list[int],
    scene: SceneSpec,
    *,
    kind: str,
    semantic: bool,
    category: TrajectoryCategory = TrajectoryCategory.SEMANTIC,
    transition_categories: list[TrajectoryCategory] | None = None,
    initial_object_map: np.ndarray | None = None,
    state_validity: np.ndarray | None = None,
) -> ObjectTrajectory:
    states = [np.asarray(initial, dtype=np.int64).copy()]
    if initial_object_map is None:
        current_object_map = np.full(initial.shape, -1, dtype=np.int64)
        for obj in scene.objects:
            current_object_map[obj.mask & (initial != 0)] = obj.object_id
    else:
        current_object_map = np.asarray(initial_object_map, dtype=np.int64).copy()
        if current_object_map.shape != initial.shape:
            raise ValueError("initial_object_map must match the grid shape.")
    object_maps = [current_object_map]
    current = states[0]
    for action, object_id in zip(actions, object_ids, strict=True):
        current = apply_low_level_action(current, action)
        states.append(current)
        current_object_map = current_object_map.copy()
        if action.op == ActionOp.ERASE or current[action.row, action.col] == 0:
            current_object_map[action.row, action.col] = -1
        elif object_id >= 0:
            current_object_map[action.row, action.col] = object_id
        object_maps.append(current_object_map)
    state_array = np.stack(states)
    if state_validity is None:
        if category == TrajectoryCategory.WRONG:
            validity = np.asarray(
                [_is_build_manifold_state(state, scene.grid) for state in state_array], dtype=bool
            )
        else:
            validity = np.ones((len(states),), dtype=bool)
    else:
        validity = np.asarray(state_validity, dtype=bool)
        if validity.shape != (len(states),):
            raise ValueError("state_validity must contain one value per rollout state.")
    categories = transition_categories or [category] * len(actions)
    return ObjectTrajectory(
        states=state_array,
        object_maps=np.stack(object_maps),
        actions=tuple(actions),
        action_object_ids=tuple(int(x) for x in object_ids),
        scene=scene,
        kind=kind,
        semantic=semantic,
        category=category,
        transition_categories=tuple(categories),
        state_validity=validity,
    )


def _shuffle_objects(objects: tuple[ObjectSpec, ...], rng: np.random.Generator) -> list[ObjectSpec]:
    shuffled = list(objects)
    rng.shuffle(shuffled)
    return shuffled


def _canonicalize_objects(objects: list[ObjectSpec]) -> tuple[ObjectSpec, ...]:
    ordered = sorted(objects, key=lambda obj: (*obj.bbox, obj.color, obj.shape_type))
    return tuple(
        ObjectSpec(object_id=object_id, shape_type=obj.shape_type, color=obj.color, mask=obj.mask)
        for object_id, obj in enumerate(ordered)
    )


def _masks_touch(left: np.ndarray, right: np.ndarray) -> bool:
    for row, col in np.argwhere(left):
        row0, row1 = max(0, int(row) - 1), min(left.shape[0], int(row) + 2)
        col0, col1 = max(0, int(col) - 1), min(left.shape[1], int(col) + 2)
        if bool(np.any(right[row0:row1, col0:col1])):
            return True
    return False


def _dilate(mask: np.ndarray) -> np.ndarray:
    output = mask.copy()
    for row, col in np.argwhere(mask):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = int(row) + dr, int(col) + dc
                if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1]:
                    output[rr, cc] = True
    return output


def _hard_wrong_candidates(scene: SceneSpec) -> list[tuple[int, int, int]]:
    candidates: dict[tuple[int, int], int] = {}
    for obj in scene.objects:
        for row, col in np.argwhere(_dilate(obj.mask)):
            cell = (int(row), int(col))
            candidates.setdefault(cell, obj.object_id)
        for row, col in np.argwhere(obj.mask):
            cell = (int(row), int(col))
            candidates[cell] = obj.object_id
    return [(row, col, object_id) for (row, col), object_id in candidates.items()]


def _nearest_object_color(scene: SceneSpec, row: int, col: int) -> int:
    distances = []
    for obj in scene.objects:
        cells = np.argwhere(obj.mask)
        distance = int(np.abs(cells - np.asarray([row, col])).sum(axis=1).min())
        distances.append((distance, obj.object_id, obj.color))
    return int(min(distances)[2])


def _is_build_manifold_state(state: np.ndarray, target: np.ndarray) -> bool:
    return bool(np.all((state == 0) | (state == target)))


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
    start_object_map: np.ndarray | None = None,
) -> tuple[list[LowLevelAction], list[int]]:
    actions: list[LowLevelAction] = []
    object_ids: list[int] = []
    cells = [tuple(int(x) for x in cell) for cell in np.argwhere(start != target)]
    rng.shuffle(cells)
    for row, col in cells:
        color = int(target[row, col])
        op = ActionOp.ERASE if color == 0 else (ActionOp.PAINT if start[row, col] == 0 else ActionOp.RECOLOR)
        actions.append(LowLevelAction(op, row, col, color))
        target_object_id = int(object_map[row, col]) if object_map is not None else object_id
        if target_object_id < 0 and start_object_map is not None:
            target_object_id = int(start_object_map[row, col])
        object_ids.append(target_object_id)
    return actions, object_ids
