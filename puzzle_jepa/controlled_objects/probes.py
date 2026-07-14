from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from puzzle_jepa.controlled_objects.generator import (
    TRANSFORM_NAMES,
    ControlledObjectGenerator,
)
from puzzle_jepa.controlled_objects.model import ControlledObjectJEPA


SHAPE_FAMILY_BY_ID = (0, 1, 1, 2, 3, 4)


@dataclass(slots=True)
class _ProbeLabels:
    states: torch.Tensor
    future_states: dict[int, torch.Tensor]
    actions: torch.Tensor
    object_count: torch.Tensor
    selected_colors: torch.Tensor
    shape_families: torch.Tensor
    motion_policies: torch.Tensor
    object_present: torch.Tensor
    positions: torch.Tensor
    areas: torch.Tensor
    future_positions: dict[int, torch.Tensor]
    relations: torch.Tensor
    future_relations: dict[int, torch.Tensor]
    relation_present: torch.Tensor


@dataclass(slots=True)
class _ProbeFeatures:
    latent: torch.Tensor
    rollouts: dict[str, torch.Tensor]
    rollout_horizons: dict[str, int]
    delta: torch.Tensor
    predicted_delta: torch.Tensor
    raw: torch.Tensor
    raw_delta: torch.Tensor


def run_controlled_object_probes(
    model: ControlledObjectJEPA,
    generator: ControlledObjectGenerator,
    *,
    seed: int,
    train_samples: int,
    eval_samples: int,
    batch_size: int,
    device: torch.device,
    steps: int,
    learning_rate: float,
    object_counts: tuple[int, ...] | None = None,
) -> dict[str, float | str]:
    if min(train_samples, eval_samples, batch_size, steps) < 1:
        raise ValueError("Probe sample, batch, and step counts must be positive.")
    was_training = model.training
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    try:
        model.eval()
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed + 31)
            train_labels = _collect_labels(
                generator,
                seed=seed,
                samples=train_samples,
                device=device,
                horizons=_rollout_horizons(model),
                object_counts=object_counts or (1, 2, 4, 8),
            )
            eval_labels = _collect_labels(
                generator,
                seed=seed + 1,
                samples=eval_samples,
                device=device,
                horizons=_rollout_horizons(model),
                object_counts=object_counts or (1, 2, 4, 8),
            )
            train = _encode_features(model, train_labels, batch_size=batch_size)
            evaluate = _encode_features(model, eval_labels, batch_size=batch_size)
            metrics = _fit_probe_suite(
                train,
                train_labels,
                evaluate,
                eval_labels,
                num_colors=generator.spec.num_colors,
                steps=steps,
                learning_rate=learning_rate,
            )
    finally:
        model.train(was_training)
    return {
        "probe_schema": "controlled_objects_v4",
        "probe_object_counts": ",".join(
            str(value) for value in (object_counts or (1, 2, 4, 8))
        ),
        "probe_motion_policy_interpretation": (
            "unobservable_single_frame_negative_control"
        ),
        **metrics,
    }


def _collect_labels(
    generator: ControlledObjectGenerator,
    *,
    seed: int,
    samples: int,
    device: torch.device,
    horizons: tuple[int, ...],
    object_counts: tuple[int, ...],
) -> _ProbeLabels:
    rng = np.random.default_rng(seed)
    color_slots = generator.spec.num_colors - 1
    pair_ids = [(left, right) for left in range(color_slots) for right in range(left + 1, color_slots)]
    states = []
    actions = []
    object_count_labels = []
    selected_colors = []
    shape_families = []
    motion_policies = []
    object_present = []
    positions = []
    areas = []
    relations = []
    relation_present = []
    future_states = {horizon: [] for horizon in horizons}
    future_positions = {horizon: [] for horizon in horizons}
    future_relations = {horizon: [] for horizon in horizons}
    max_horizon = max(horizons)
    if max_horizon > generator.spec.trajectory_length:
        raise ValueError(
            f"Probe requires horizon {max_horizon}, but data has "
            f"{generator.spec.trajectory_length}."
        )
    invalid_counts = set(object_counts) - {1, 2, 4, 8}
    if not object_counts or invalid_counts:
        raise ValueError(f"Probe object counts must come from 1,2,4,8; got {object_counts}.")
    object_count_generators = {
        object_count: ControlledObjectGenerator(
            replace(generator.spec, object_count=object_count)
        )
        for object_count in object_counts
    }
    for sample in range(samples):
        object_count = object_counts[sample % len(object_counts)]
        trajectory = object_count_generators[object_count].sample_trajectory(
            rng, horizon=max_horizon
        )
        state = trajectory.states[0]
        action = trajectory.actions[0]
        present = np.zeros(color_slots, dtype=bool)
        shapes = np.full(color_slots, -1, dtype=np.int64)
        motions = np.full(color_slots, -1, dtype=np.int64)
        for color, shape_id, motion_id in zip(
            trajectory.scene.colors,
            trajectory.scene.shape_ids,
            trajectory.scene.motion_ids,
            strict=True,
        ):
            slot = int(color) - 1
            present[slot] = True
            shapes[slot] = SHAPE_FAMILY_BY_ID[int(shape_id)]
            motions[slot] = int(motion_id)
        current_positions = _color_positions(state, color_slots)
        current_areas = _color_areas(state, color_slots)
        current_relations = np.stack(
            [current_positions[right] - current_positions[left] for left, right in pair_ids]
        )
        present_relations = np.asarray(
            [present[left] and present[right] for left, right in pair_ids],
            dtype=bool,
        )
        states.append(state)
        actions.append(trajectory.actions)
        object_count_labels.append((1, 2, 4, 8).index(object_count))
        selected_colors.append(int(state[int(action[0]), int(action[1])]))
        shape_families.append(shapes)
        motion_policies.append(motions)
        object_present.append(present)
        positions.append(current_positions)
        areas.append(current_areas[:, None])
        relations.append(current_relations)
        relation_present.append(present_relations)
        for horizon in horizons:
            future = trajectory.states[horizon]
            endpoint_positions = _color_positions(future, color_slots)
            future_states[horizon].append(future)
            future_positions[horizon].append(endpoint_positions)
            future_relations[horizon].append(
                np.stack(
                    [
                        endpoint_positions[right] - endpoint_positions[left]
                        for left, right in pair_ids
                    ]
                )
            )
    return _ProbeLabels(
        states=torch.as_tensor(np.stack(states), dtype=torch.long, device=device),
        future_states={
            horizon: torch.as_tensor(np.stack(items), dtype=torch.long, device=device)
            for horizon, items in future_states.items()
        },
        actions=torch.as_tensor(np.stack(actions), dtype=torch.long, device=device),
        object_count=torch.as_tensor(
            object_count_labels, dtype=torch.long, device=device
        ),
        selected_colors=torch.as_tensor(selected_colors, dtype=torch.long, device=device),
        shape_families=torch.as_tensor(
            np.stack(shape_families), dtype=torch.long, device=device
        ),
        motion_policies=torch.as_tensor(
            np.stack(motion_policies), dtype=torch.long, device=device
        ),
        object_present=torch.as_tensor(
            np.stack(object_present), dtype=torch.bool, device=device
        ),
        positions=torch.as_tensor(
            np.stack(positions), dtype=torch.float32, device=device
        ),
        areas=torch.as_tensor(np.stack(areas), dtype=torch.float32, device=device),
        future_positions={
            horizon: torch.as_tensor(
                np.stack(items), dtype=torch.float32, device=device
            )
            for horizon, items in future_positions.items()
        },
        relations=torch.as_tensor(
            np.stack(relations), dtype=torch.float32, device=device
        ),
        future_relations={
            horizon: torch.as_tensor(
                np.stack(items), dtype=torch.float32, device=device
            )
            for horizon, items in future_relations.items()
        },
        relation_present=torch.as_tensor(
            np.stack(relation_present), dtype=torch.bool, device=device
        ),
    )


def _rollout_horizons(model: ControlledObjectJEPA) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                span * rollout_step
                for level, span in enumerate(model.level_spans)
                for rollout_step in range(1, model.level_rollout_steps(level) + 1)
            }
        )
    )


def _color_positions(state: np.ndarray, color_slots: int) -> np.ndarray:
    scale = max(1, state.shape[0] - 1)
    positions = np.zeros((color_slots, 2), dtype=np.float32)
    for slot in range(color_slots):
        cells = np.argwhere(state == slot + 1)
        if len(cells):
            positions[slot] = cells.mean(axis=0) / scale
    return positions


def _color_areas(state: np.ndarray, color_slots: int) -> np.ndarray:
    scale = float(state.shape[0] * state.shape[1])
    return np.asarray(
        [np.count_nonzero(state == slot + 1) / scale for slot in range(color_slots)],
        dtype=np.float32,
    )


@torch.no_grad()
def _encode_features(
    model: ControlledObjectJEPA,
    labels: _ProbeLabels,
    *,
    batch_size: int,
) -> _ProbeFeatures:
    parts: dict[str, list[torch.Tensor]] = {
        "latent": [],
        "delta": [],
        "predicted_delta": [],
        "raw": [],
        "raw_delta": [],
    }
    rollout_specs = [
        (f"level{level}_rollout{rollout_step}", level, rollout_step, span)
        for level, span in enumerate(model.level_spans)
        for rollout_step in range(1, model.level_rollout_steps(level) + 1)
    ]
    rollout_parts = {name: [] for name, _, _, _ in rollout_specs}
    rollout_horizons = {
        name: span * rollout_step
        for name, _, rollout_step, span in rollout_specs
    }
    num_colors = model.num_colors
    for start in range(0, len(labels.states), batch_size):
        stop = min(start + batch_size, len(labels.states))
        states = labels.states[start:stop]
        futures = labels.future_states[1][start:stop]
        actions = labels.actions[start:stop]
        latent = model.encode(states)
        future = model.encode(futures)
        for level, span in enumerate(model.level_spans):
            rollout = model.rollout_level(
                level,
                latent,
                actions[:, : span * model.level_rollout_steps(level)],
            )
            for rollout_step in range(1, model.level_rollout_steps(level) + 1):
                rollout_parts[f"level{level}_rollout{rollout_step}"].append(
                    rollout[:, rollout_step - 1].detach()
                )
        raw = model.encoder.render_rgb(states).flatten(1)
        raw_future = model.encoder.render_rgb(futures).flatten(1)
        parts["latent"].append(latent.detach())
        parts["delta"].append((future - latent).detach())
        parts["predicted_delta"].append(
            (rollout_parts["level0_rollout1"][-1] - latent).detach()
        )
        parts["raw"].append(raw)
        parts["raw_delta"].append(raw_future - raw)
    return _ProbeFeatures(
        **{name: torch.cat(items) for name, items in parts.items()},
        rollouts={name: torch.cat(items) for name, items in rollout_parts.items()},
        rollout_horizons=rollout_horizons,
    )


def _fit_probe_suite(
    train: _ProbeFeatures,
    train_y: _ProbeLabels,
    evaluate: _ProbeFeatures,
    eval_y: _ProbeLabels,
    *,
    num_colors: int,
    steps: int,
    learning_rate: float,
) -> dict[str, float]:
    transfer_presence_y = {
        name: eval_y.object_present for name in evaluate.rollouts
    }
    transfer_shape_y = {
        name: eval_y.shape_families for name in evaluate.rollouts
    }
    transfer_positions_y = {
        name: eval_y.future_positions[evaluate.rollout_horizons[name]]
        for name in evaluate.rollouts
    }
    transfer_relations_y = {
        name: eval_y.future_relations[evaluate.rollout_horizons[name]]
        for name in evaluate.rollouts
    }
    transfer_relation_masks = {
        name: eval_y.relation_present for name in evaluate.rollouts
    }
    transfer_states_y = {
        name: eval_y.future_states[evaluate.rollout_horizons[name]]
        for name in evaluate.rollouts
    }
    presence, rollout_presence = _fit_slot_binary_classifier(
        train.latent,
        train_y.object_present,
        evaluate.latent,
        eval_y.object_present,
        transfer_x=evaluate.rollouts,
        transfer_y=transfer_presence_y,
        steps=steps,
        learning_rate=learning_rate,
    )
    raw_presence, _ = _fit_slot_binary_classifier(
        train.raw,
        train_y.object_present,
        evaluate.raw,
        eval_y.object_present,
        transfer_x=None,
        transfer_y=None,
        steps=steps,
        learning_rate=learning_rate,
        standardize_inputs=False,
    )
    object_count, rollout_object_count = _fit_classifier_many(
        train.latent,
        train_y.object_count,
        evaluate.latent,
        eval_y.object_count,
        num_classes=4,
        transfer_x=evaluate.rollouts,
        transfer_y={name: eval_y.object_count for name in evaluate.rollouts},
        steps=steps,
        learning_rate=learning_rate,
    )
    raw_object_count, _ = _fit_classifier(
        train.raw,
        train_y.object_count,
        evaluate.raw,
        eval_y.object_count,
        num_classes=4,
        transfer_x=None,
        transfer_y=None,
        steps=steps,
        learning_rate=learning_rate,
        standardize_inputs=False,
    )
    shape, rollout_shape = _fit_slot_classifier(
        train.latent,
        train_y.shape_families,
        train_y.object_present,
        evaluate.latent,
        eval_y.shape_families,
        eval_y.object_present,
        num_classes=len(set(SHAPE_FAMILY_BY_ID)),
        transfer_x=evaluate.rollouts,
        transfer_y=transfer_shape_y,
        transfer_mask=transfer_presence_y,
        steps=steps,
        learning_rate=learning_rate,
    )
    raw_shape, _ = _fit_slot_classifier(
        train.raw,
        train_y.shape_families,
        train_y.object_present,
        evaluate.raw,
        eval_y.shape_families,
        eval_y.object_present,
        num_classes=len(set(SHAPE_FAMILY_BY_ID)),
        transfer_x=None,
        transfer_y=None,
        transfer_mask=None,
        steps=steps,
        learning_rate=learning_rate,
        standardize_inputs=False,
    )
    motion, rollout_motion = _fit_slot_classifier(
        train.latent,
        train_y.motion_policies,
        train_y.object_present,
        evaluate.latent,
        eval_y.motion_policies,
        eval_y.object_present,
        num_classes=len(TRANSFORM_NAMES) - 1,
        transfer_x=evaluate.rollouts,
        transfer_y={name: eval_y.motion_policies for name in evaluate.rollouts},
        transfer_mask=transfer_presence_y,
        steps=steps,
        learning_rate=learning_rate,
    )
    raw_motion, _ = _fit_slot_classifier(
        train.raw,
        train_y.motion_policies,
        train_y.object_present,
        evaluate.raw,
        eval_y.motion_policies,
        eval_y.object_present,
        num_classes=len(TRANSFORM_NAMES) - 1,
        transfer_x=None,
        transfer_y=None,
        transfer_mask=None,
        steps=steps,
        learning_rate=learning_rate,
        standardize_inputs=False,
    )
    position, rollout_position = _fit_masked_regressor(
        train.latent,
        train_y.positions,
        train_y.object_present,
        evaluate.latent,
        eval_y.positions,
        eval_y.object_present,
        transfer_x=evaluate.rollouts,
        transfer_y=transfer_positions_y,
        transfer_mask=transfer_presence_y,
        steps=steps,
        learning_rate=learning_rate,
    )
    raw_position, _ = _fit_masked_regressor(
        train.raw,
        train_y.positions,
        train_y.object_present,
        evaluate.raw,
        eval_y.positions,
        eval_y.object_present,
        transfer_x=None,
        transfer_y=None,
        transfer_mask=None,
        steps=steps,
        learning_rate=learning_rate,
        standardize_inputs=False,
    )
    area, rollout_area = _fit_masked_regressor(
        train.latent,
        train_y.areas,
        train_y.object_present,
        evaluate.latent,
        eval_y.areas,
        eval_y.object_present,
        transfer_x=evaluate.rollouts,
        transfer_y={name: eval_y.areas for name in evaluate.rollouts},
        transfer_mask=transfer_presence_y,
        steps=steps,
        learning_rate=learning_rate,
    )
    raw_area, _ = _fit_masked_regressor(
        train.raw,
        train_y.areas,
        train_y.object_present,
        evaluate.raw,
        eval_y.areas,
        eval_y.object_present,
        transfer_x=None,
        transfer_y=None,
        transfer_mask=None,
        steps=steps,
        learning_rate=learning_rate,
        standardize_inputs=False,
    )
    relation, rollout_relation = _fit_masked_regressor(
        train.latent,
        train_y.relations,
        train_y.relation_present,
        evaluate.latent,
        eval_y.relations,
        eval_y.relation_present,
        transfer_x=evaluate.rollouts,
        transfer_y=transfer_relations_y,
        transfer_mask=transfer_relation_masks,
        steps=steps,
        learning_rate=learning_rate,
    )
    raw_relation, _ = _fit_masked_regressor(
        train.raw,
        train_y.relations,
        train_y.relation_present,
        evaluate.raw,
        eval_y.relations,
        eval_y.relation_present,
        transfer_x=None,
        transfer_y=None,
        transfer_mask=None,
        steps=steps,
        learning_rate=learning_rate,
        standardize_inputs=False,
    )
    grid, rollout_grid = _fit_grid_decoder(
        train.latent,
        train_y.states,
        evaluate.latent,
        eval_y.states,
        num_colors=num_colors,
        transfer_x=evaluate.rollouts,
        transfer_y=transfer_states_y,
        steps=steps,
        learning_rate=learning_rate,
    )
    action_row, predicted_action_row = _fit_classifier(
        train.delta,
        train_y.actions[:, 0, 0],
        evaluate.delta,
        eval_y.actions[:, 0, 0],
        num_classes=16,
        transfer_x=evaluate.predicted_delta,
        transfer_y=eval_y.actions[:, 0, 0],
        steps=steps,
        learning_rate=learning_rate,
    )
    raw_action_row, _ = _fit_classifier(
        train.raw_delta,
        train_y.actions[:, 0, 0],
        evaluate.raw_delta,
        eval_y.actions[:, 0, 0],
        num_classes=16,
        transfer_x=None,
        transfer_y=None,
        steps=steps,
        learning_rate=learning_rate,
        standardize_inputs=False,
    )
    action_col, predicted_action_col = _fit_classifier(
        train.delta,
        train_y.actions[:, 0, 1],
        evaluate.delta,
        eval_y.actions[:, 0, 1],
        num_classes=16,
        transfer_x=evaluate.predicted_delta,
        transfer_y=eval_y.actions[:, 0, 1],
        steps=steps,
        learning_rate=learning_rate,
    )
    raw_action_col, _ = _fit_classifier(
        train.raw_delta,
        train_y.actions[:, 0, 1],
        evaluate.raw_delta,
        eval_y.actions[:, 0, 1],
        num_classes=16,
        transfer_x=None,
        transfer_y=None,
        steps=steps,
        learning_rate=learning_rate,
        standardize_inputs=False,
    )
    action_transform, predicted_action_transform = _fit_classifier(
        train.delta,
        train_y.actions[:, 0, 2],
        evaluate.delta,
        eval_y.actions[:, 0, 2],
        num_classes=7,
        transfer_x=evaluate.predicted_delta,
        transfer_y=eval_y.actions[:, 0, 2],
        steps=steps,
        learning_rate=learning_rate,
    )
    raw_action_transform, _ = _fit_classifier(
        train.raw_delta,
        train_y.actions[:, 0, 2],
        evaluate.raw_delta,
        eval_y.actions[:, 0, 2],
        num_classes=7,
        transfer_x=None,
        transfer_y=None,
        steps=steps,
        learning_rate=learning_rate,
        standardize_inputs=False,
    )
    selected, predicted_selected = _fit_classifier(
        train.delta,
        train_y.selected_colors,
        evaluate.delta,
        eval_y.selected_colors,
        num_classes=num_colors,
        transfer_x=evaluate.predicted_delta,
        transfer_y=eval_y.selected_colors,
        steps=steps,
        learning_rate=learning_rate,
    )
    raw_selected, _ = _fit_classifier(
        train.raw_delta,
        train_y.selected_colors,
        evaluate.raw_delta,
        eval_y.selected_colors,
        num_classes=num_colors,
        transfer_x=None,
        transfer_y=None,
        steps=steps,
        learning_rate=learning_rate,
        standardize_inputs=False,
    )
    centered = evaluate.latent - evaluate.latent.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered.float())
    probabilities = singular.square() / singular.square().sum().clamp_min(1.0e-12)
    effective_rank = torch.exp(
        -(probabilities * probabilities.clamp_min(1.0e-12).log()).sum()
    )
    level0 = "level0_rollout1"
    metrics = {
        "probe_object_presence_balanced_acc": presence,
        "raw_probe_object_presence_balanced_acc": raw_presence,
        "probe_rollout_object_presence_balanced_acc": rollout_presence[level0],
        "probe_object_count_balanced_acc": object_count,
        "raw_probe_object_count_balanced_acc": raw_object_count,
        "probe_rollout_object_count_balanced_acc": rollout_object_count[level0],
        "probe_shape_balanced_acc": shape,
        "raw_probe_shape_balanced_acc": raw_shape,
        "probe_rollout_shape_balanced_acc": rollout_shape[level0],
        "probe_motion_policy_balanced_acc": motion,
        "raw_probe_motion_policy_balanced_acc": raw_motion,
        "probe_rollout_motion_policy_balanced_acc": rollout_motion[level0],
        "probe_position_r2": position,
        "raw_probe_position_r2": raw_position,
        "probe_rollout_position_r2": rollout_position[level0],
        "probe_area_r2": area,
        "raw_probe_area_r2": raw_area,
        "probe_rollout_area_r2": rollout_area[level0],
        "probe_relation_r2": relation,
        "raw_probe_relation_r2": raw_relation,
        "probe_rollout_relation_r2": rollout_relation[level0],
        "probe_grid_acc": grid[0],
        "probe_grid_foreground_iou": grid[1],
        "probe_pixel_decoder_acc": grid[0],
        "probe_pixel_decoder_foreground_iou": grid[1],
        "raw_probe_grid_acc": 1.0,
        "raw_probe_grid_foreground_iou": 1.0,
        "probe_rollout_grid_acc": rollout_grid[level0][0],
        "probe_rollout_grid_foreground_iou": rollout_grid[level0][1],
        "probe_rollout_pixel_decoder_acc": rollout_grid[level0][0],
        "probe_rollout_pixel_decoder_foreground_iou": rollout_grid[level0][1],
        "probe_delta_action_row_balanced_acc": action_row,
        "probe_predicted_delta_action_row_balanced_acc": predicted_action_row,
        "raw_probe_delta_action_row_balanced_acc": raw_action_row,
        "probe_delta_action_col_balanced_acc": action_col,
        "probe_predicted_delta_action_col_balanced_acc": predicted_action_col,
        "raw_probe_delta_action_col_balanced_acc": raw_action_col,
        "probe_delta_action_transform_balanced_acc": action_transform,
        "probe_predicted_delta_action_transform_balanced_acc": predicted_action_transform,
        "raw_probe_delta_action_transform_balanced_acc": raw_action_transform,
        "probe_delta_selected_color_balanced_acc": selected,
        "probe_predicted_delta_selected_color_balanced_acc": predicted_selected,
        "raw_probe_delta_selected_color_balanced_acc": raw_selected,
        "probe_latent_std_mean": float(
            evaluate.latent.std(dim=0, unbiased=False).mean()
        ),
        "probe_latent_effective_rank": float(effective_rank),
    }
    for name in evaluate.rollouts:
        prefix = f"probe_{name}"
        metrics.update(
            {
                f"{prefix}_object_presence_balanced_acc": rollout_presence[name],
                f"{prefix}_object_count_balanced_acc": rollout_object_count[name],
                f"{prefix}_shape_balanced_acc": rollout_shape[name],
                f"{prefix}_motion_policy_balanced_acc": rollout_motion[name],
                f"{prefix}_position_r2": rollout_position[name],
                f"{prefix}_area_r2": rollout_area[name],
                f"{prefix}_relation_r2": rollout_relation[name],
                f"{prefix}_grid_acc": rollout_grid[name][0],
                f"{prefix}_grid_foreground_iou": rollout_grid[name][1],
                f"{prefix}_pixel_decoder_acc": rollout_grid[name][0],
                f"{prefix}_pixel_decoder_foreground_iou": rollout_grid[name][1],
            }
        )
    return metrics


def _fit_slot_binary_classifier(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    eval_x: torch.Tensor,
    eval_y: torch.Tensor,
    *,
    transfer_x: dict[str, torch.Tensor] | None,
    transfer_y: dict[str, torch.Tensor] | None,
    steps: int,
    learning_rate: float,
    standardize_inputs: bool = True,
) -> tuple[float, dict[str, float]]:
    if standardize_inputs:
        train_x, eval_x, transfer_x = _standardize_many(
            train_x, eval_x, transfer_x
        )
    else:
        transfer_x = transfer_x or {}
    head = nn.Linear(train_x.shape[1], train_y.shape[1]).to(train_x.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    positives = train_y.sum().clamp_min(1)
    negatives = train_y.numel() - positives
    positive_weight = (negatives / positives).reshape(1)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(
            head(train_x), train_y.float(), pos_weight=positive_weight
        )
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        value = _binary_balanced_accuracy(head(eval_x) >= 0, eval_y)
        transfer = {
            name: _binary_balanced_accuracy(head(values) >= 0, transfer_y[name])
            for name, values in transfer_x.items()
        } if transfer_y is not None else {}
    return value, transfer


def _fit_classifier(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    eval_x: torch.Tensor,
    eval_y: torch.Tensor,
    *,
    num_classes: int,
    transfer_x: torch.Tensor | None,
    transfer_y: torch.Tensor | None,
    steps: int,
    learning_rate: float,
    standardize_inputs: bool = True,
) -> tuple[float, float]:
    if standardize_inputs:
        train_x, eval_x, transfer_x = _standardize(train_x, eval_x, transfer_x)
    head = nn.Linear(train_x.shape[1], num_classes).to(train_x.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    counts = torch.bincount(train_y, minlength=num_classes).clamp_min(1)
    weights = len(train_y) / (num_classes * counts.float())
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(head(train_x), train_y, weight=weights)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        value = _balanced_accuracy(head(eval_x).argmax(dim=1), eval_y)
        transfer = (
            _balanced_accuracy(head(transfer_x).argmax(dim=1), transfer_y)
            if transfer_x is not None and transfer_y is not None
            else float("nan")
        )
    return value, transfer


def _fit_classifier_many(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    eval_x: torch.Tensor,
    eval_y: torch.Tensor,
    *,
    num_classes: int,
    transfer_x: dict[str, torch.Tensor],
    transfer_y: dict[str, torch.Tensor],
    steps: int,
    learning_rate: float,
) -> tuple[float, dict[str, float]]:
    train_x, eval_x, transfer_x = _standardize_many(train_x, eval_x, transfer_x)
    head = nn.Linear(train_x.shape[1], num_classes).to(train_x.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    counts = torch.bincount(train_y, minlength=num_classes).clamp_min(1)
    weights = len(train_y) / (num_classes * counts.float())
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(head(train_x), train_y, weight=weights)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        value = _balanced_accuracy(head(eval_x).argmax(dim=1), eval_y)
        transfer = {
            name: _balanced_accuracy(head(values).argmax(dim=1), transfer_y[name])
            for name, values in transfer_x.items()
        }
    return value, transfer


def _fit_slot_classifier(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_mask: torch.Tensor,
    eval_x: torch.Tensor,
    eval_y: torch.Tensor,
    eval_mask: torch.Tensor,
    *,
    num_classes: int,
    transfer_x: dict[str, torch.Tensor] | None,
    transfer_y: dict[str, torch.Tensor] | None,
    transfer_mask: dict[str, torch.Tensor] | None,
    steps: int,
    learning_rate: float,
    standardize_inputs: bool = True,
) -> tuple[float, dict[str, float]]:
    if standardize_inputs:
        train_x, eval_x, transfer_x = _standardize_many(
            train_x, eval_x, transfer_x
        )
    else:
        transfer_x = transfer_x or {}
    slots = train_y.shape[1]
    head = nn.Linear(train_x.shape[1], slots * num_classes).to(train_x.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    selected = train_y[train_mask]
    counts = torch.bincount(selected, minlength=num_classes).clamp_min(1)
    weights = len(selected) / (num_classes * counts.float())
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = head(train_x).reshape(len(train_x), slots, num_classes)
        loss = F.cross_entropy(logits[train_mask], selected, weight=weights)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        predicted = head(eval_x).reshape(len(eval_x), slots, num_classes).argmax(dim=-1)
        value = _balanced_accuracy(predicted[eval_mask], eval_y[eval_mask])
        transfer = {}
        if transfer_y is not None and transfer_mask is not None:
            for name, values in transfer_x.items():
                transfer_predicted = head(values).reshape(
                    len(values), slots, num_classes
                ).argmax(dim=-1)
                mask = transfer_mask[name]
                transfer[name] = _balanced_accuracy(
                    transfer_predicted[mask], transfer_y[name][mask]
                )
    return value, transfer


def _fit_masked_regressor(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_mask: torch.Tensor,
    eval_x: torch.Tensor,
    eval_y: torch.Tensor,
    eval_mask: torch.Tensor,
    *,
    transfer_x: dict[str, torch.Tensor] | None,
    transfer_y: dict[str, torch.Tensor] | None,
    transfer_mask: dict[str, torch.Tensor] | None,
    steps: int,
    learning_rate: float,
    standardize_inputs: bool = True,
) -> tuple[float, dict[str, float]]:
    if standardize_inputs:
        train_x, eval_x, transfer_x = _standardize_many(
            train_x, eval_x, transfer_x
        )
    else:
        transfer_x = transfer_x or {}
    output_shape = train_y.shape[1:]
    head = nn.Linear(train_x.shape[1], int(np.prod(output_shape))).to(train_x.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    expanded_mask = train_mask.unsqueeze(-1).expand_as(train_y)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        predicted = head(train_x).reshape(len(train_x), *output_shape)
        loss = (predicted[expanded_mask] - train_y[expanded_mask]).square().mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        predicted = head(eval_x).reshape(len(eval_x), *output_shape)
        value = _masked_r2(predicted, eval_y, eval_mask)
        transfer = {}
        if transfer_y is not None and transfer_mask is not None:
            for name, values in transfer_x.items():
                transfer_predicted = head(values).reshape(len(values), *output_shape)
                transfer[name] = _masked_r2(
                    transfer_predicted, transfer_y[name], transfer_mask[name]
                )
    return value, transfer


def _fit_grid_decoder(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    eval_x: torch.Tensor,
    eval_y: torch.Tensor,
    *,
    num_colors: int,
    transfer_x: dict[str, torch.Tensor],
    transfer_y: dict[str, torch.Tensor],
    steps: int,
    learning_rate: float,
) -> tuple[tuple[float, float], dict[str, tuple[float, float]]]:
    train_x, eval_x, transfer_x = _standardize_many(train_x, eval_x, transfer_x)
    pixels = train_y.shape[1] * train_y.shape[2]
    head = nn.Sequential(
        nn.Linear(train_x.shape[1], 2 * train_x.shape[1]),
        nn.GELU(),
        nn.Linear(2 * train_x.shape[1], pixels * num_colors),
    ).to(train_x.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    labels = train_y.flatten(1)
    counts = torch.bincount(labels.flatten(), minlength=num_colors).clamp_min(1)
    weights = labels.numel() / (num_colors * counts.float())
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = head(train_x).reshape(len(train_x), pixels, num_colors)
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), weight=weights)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        predicted = head(eval_x).reshape(len(eval_x), pixels, num_colors).argmax(dim=-1)
        transfer_predicted = {
            name: head(values).reshape(len(values), pixels, num_colors).argmax(dim=-1)
            for name, values in transfer_x.items()
        }
    return _grid_metrics(predicted, eval_y), {
        name: _grid_metrics(values, transfer_y[name])
        for name, values in transfer_predicted.items()
    }


def _standardize(
    train_x: torch.Tensor,
    eval_x: torch.Tensor,
    transfer_x: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1.0e-4)
    return (
        (train_x - mean) / std,
        (eval_x - mean) / std,
        (transfer_x - mean) / std if transfer_x is not None else None,
    )


def _standardize_many(
    train_x: torch.Tensor,
    eval_x: torch.Tensor,
    transfer_x: dict[str, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1.0e-4)
    return (
        (train_x - mean) / std,
        (eval_x - mean) / std,
        {
            name: (values - mean) / std
            for name, values in (transfer_x or {}).items()
        },
    )


def _balanced_accuracy(predicted: torch.Tensor, target: torch.Tensor) -> float:
    recalls = [
        (predicted[target == label] == label).float().mean()
        for label in torch.unique(target)
    ]
    return float(torch.stack(recalls).mean()) if recalls else float("nan")


def _binary_balanced_accuracy(predicted: torch.Tensor, target: torch.Tensor) -> float:
    positive = (predicted[target] == target[target]).float().mean()
    negative = (predicted[~target] == target[~target]).float().mean()
    return float(0.5 * (positive + negative))


def _masked_r2(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    expanded = mask.unsqueeze(-1).expand_as(target)
    selected_prediction = predicted[expanded]
    selected_target = target[expanded]
    residual = (selected_prediction - selected_target).square().sum()
    total = (selected_target - selected_target.mean()).square().sum().clamp_min(1.0e-12)
    return float(1.0 - residual / total)


def _grid_metrics(predicted: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    target = target.flatten(1)
    accuracy = float((predicted == target).float().mean())
    foreground = target > 0
    intersection = ((predicted == target) & foreground).sum().float()
    union = ((predicted > 0) | foreground).sum().float().clamp_min(1.0)
    return accuracy, float(intersection / union)
