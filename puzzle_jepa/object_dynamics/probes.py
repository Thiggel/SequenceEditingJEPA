from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from puzzle_jepa.object_dynamics.batching import PROCESS_NAMES, RELATION_NAMES, sample_object_dynamics_batch
from puzzle_jepa.object_dynamics.generator import ObjectDynamicsGenerator
from puzzle_jepa.object_dynamics.model import ObjectDynamicsJEPA
from puzzle_jepa.object_dynamics.shapes import SHAPE_TYPES


@dataclass(frozen=True, slots=True)
class ProbeDataset:
    features: torch.Tensor
    future_features: torch.Tensor
    predicted_features: torch.Tensor
    hierarchy_predicted_features: torch.Tensor
    spatial_features: torch.Tensor
    future_spatial_features: torch.Tensor
    predicted_spatial_features: torch.Tensor
    hierarchy_predicted_spatial_features: torch.Tensor
    attention_maps: torch.Tensor
    rollout_error: torch.Tensor
    delta_features: torch.Tensor
    chunk_features: torch.Tensor
    states: torch.Tensor
    future_states: torch.Tensor
    hierarchy_goal_states: torch.Tensor
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
    action: torch.Tensor
    action_process_type: torch.Tensor
    actions: torch.Tensor
    action_object_id: torch.Tensor
    continues_object: torch.Tensor
    chunk_object_id: torch.Tensor
    chunk_op: torch.Tensor
    chunk_shape: torch.Tensor
    chunk_correction: torch.Tensor
    relation_present: torch.Tensor
    relations: torch.Tensor


@torch.no_grad()
def collect_probe_dataset(
    model: ObjectDynamicsJEPA,
    generator: ObjectDynamicsGenerator,
    rng: np.random.Generator,
    *,
    samples: int,
    batch_size: int,
    horizon: int,
    device: torch.device,
) -> ProbeDataset:
    model.eval()
    collected: dict[str, list[torch.Tensor]] = {
        field: [] for field in ProbeDataset.__dataclass_fields__
    }
    remaining = int(samples)
    while remaining > 0:
        current_batch = min(batch_size, remaining)
        batch = sample_object_dynamics_batch(
            generator,
            rng,
            batch_size=current_batch,
            horizon=horizon,
            device=device,
        )
        encoded = model.encode(batch.states).float()
        next_encoded = model.encode(batch.futures[:, 0]).float()
        predicted = model.predict_latents(batch.states, batch.actions).float()
        future_encoded = model.encode(batch.futures[:, -1]).float()
        spatial_features = encoded if model.latent_representation == "grid" else encoded.unsqueeze(1)
        predicted_spatial = (
            predicted[:, -1]
            if model.latent_representation == "grid"
            else predicted[:, -1].unsqueeze(1)
        )
        features = model.pool_latents(encoded)
        next_features = model.pool_latents(next_encoded)
        predicted_features = model.pool_latents(predicted[:, -1])
        future_features = model.pool_latents(future_encoded)
        if model.hierarchy_planning:
            hierarchy_predicted_spatial = model.predict_high_level_latents(batch.states, batch.actions).float()
            hierarchy_predicted_features = model.pool_latents(hierarchy_predicted_spatial)
            chunk_actions = batch.actions[:, : model.hierarchy_horizon]
            hierarchy_goal_states = batch.futures[:, model.hierarchy_horizon - 1]
        else:
            hierarchy_predicted_spatial = torch.zeros_like(future_encoded)
            hierarchy_predicted_features = torch.zeros_like(future_features)
            chunk_actions = batch.actions
            hierarchy_goal_states = batch.futures[:, -1]
        chunk_length = chunk_actions.shape[1]
        chunk = model.encode_action_chunk(chunk_actions).float()
        chunk_object_id = batch.action_object_ids[:, :chunk_length].mode(dim=1).values
        chunk_op = batch.actions[:, :chunk_length, 0].mode(dim=1).values
        chunk_shape = batch.action_object_shapes[:, :chunk_length].mode(dim=1).values
        chunk_correction = batch.action_process_types[:, :chunk_length].mode(dim=1).values

        values = {
            "features": features,
            "future_features": future_features,
            "predicted_features": predicted_features,
            "hierarchy_predicted_features": hierarchy_predicted_features,
            "spatial_features": spatial_features,
            "future_spatial_features": future_encoded if model.latent_representation == "grid" else future_encoded.unsqueeze(1),
            "predicted_spatial_features": predicted_spatial,
            "hierarchy_predicted_spatial_features": (
                hierarchy_predicted_spatial
                if model.latent_representation == "grid"
                else hierarchy_predicted_spatial.unsqueeze(1)
            ),
            "attention_maps": model.attention_maps(batch.states).float(),
            "rollout_error": (predicted[:, -1] - future_encoded).square().flatten(1).mean(dim=-1),
            "delta_features": model.delta_probe_features(next_encoded - encoded),
            "chunk_features": chunk,
            "states": batch.states,
            "future_states": batch.futures[:, -1],
            "hierarchy_goal_states": hierarchy_goal_states,
            "object_count": batch.object_count,
            "scene_object_count": batch.scene_object_count,
            "future_object_count": batch.future_object_count[:, -1],
            "current_object_id": batch.current_object_id,
            "next_object_id": batch.next_object_id,
            "valid_state": batch.valid_state,
            "trajectory_category": batch.trajectory_category,
            "completion": batch.completion,
            "future_completion": batch.future_completion[:, -1],
            "object_present": batch.object_present,
            "future_object_present": batch.future_object_present[:, -1],
            "object_colors": batch.object_colors,
            "object_bboxes": batch.object_bboxes,
            "future_object_bboxes": batch.future_object_bboxes[:, -1],
            "object_centroids": batch.object_centroids,
            "object_areas": batch.object_areas,
            "object_shapes": batch.object_shapes,
            "object_part_counts": batch.object_part_counts,
            "object_missing": batch.object_missing,
            "future_object_missing": batch.future_object_missing[:, -1],
            "object_overgrowth": batch.object_overgrowth,
            "future_object_overgrowth": batch.future_object_overgrowth[:, -1],
            "object_wrong_color": batch.object_wrong_color,
            "future_object_wrong_color": batch.future_object_wrong_color[:, -1],
            "object_map": batch.object_map,
            "future_object_map": batch.future_object_map[:, -1],
            "action": batch.actions[:, 0],
            "action_process_type": batch.action_process_types[:, 0],
            "actions": chunk_actions,
            "action_object_id": batch.action_object_ids[:, 0],
            "continues_object": batch.continues_object,
            "chunk_object_id": chunk_object_id,
            "chunk_op": chunk_op,
            "chunk_shape": chunk_shape,
            "chunk_correction": chunk_correction,
            "relation_present": batch.relation_present,
            "relations": batch.relations,
        }
        for name, value in values.items():
            collected[name].append(value.detach().cpu())
        remaining -= current_batch
    return ProbeDataset(**{name: torch.cat(values, dim=0) for name, values in collected.items()})


def run_object_dynamics_probes(
    model: ObjectDynamicsJEPA,
    generator: ObjectDynamicsGenerator,
    rng: np.random.Generator,
    *,
    train_samples: int,
    eval_samples: int,
    batch_size: int,
    horizon: int,
    device: torch.device,
    steps: int = 200,
    learning_rate: float = 1.0e-2,
) -> dict[str, Any]:
    was_training = model.training
    cpu_rng_state = torch.random.get_rng_state()
    try:
        return _run_object_dynamics_probes(
            model,
            generator,
            rng,
            train_samples=train_samples,
            eval_samples=eval_samples,
            batch_size=batch_size,
            horizon=horizon,
            device=device,
            steps=steps,
            learning_rate=learning_rate,
        )
    finally:
        torch.random.set_rng_state(cpu_rng_state)
        model.train(was_training)


def _run_object_dynamics_probes(
    model: ObjectDynamicsJEPA,
    generator: ObjectDynamicsGenerator,
    rng: np.random.Generator,
    *,
    train_samples: int,
    eval_samples: int,
    batch_size: int,
    horizon: int,
    device: torch.device,
    steps: int = 200,
    learning_rate: float = 1.0e-2,
) -> dict[str, Any]:
    train = collect_probe_dataset(
        model,
        generator,
        rng,
        samples=train_samples,
        batch_size=batch_size,
        horizon=horizon,
        device=device,
    )
    eval_data = collect_probe_dataset(
        model,
        generator,
        rng,
        samples=eval_samples,
        batch_size=batch_size,
        horizon=horizon,
        device=device,
    )
    train_x, eval_x = _standardize(train.features, eval_data.features)
    train_delta, eval_delta = _standardize(train.delta_features, eval_data.delta_features)
    train_chunk, eval_chunk = _standardize(train.chunk_features, eval_data.chunk_features)
    train_raw = _raw_grid_features(train.states, generator.spec.num_colors)
    eval_raw = _raw_grid_features(eval_data.states, generator.spec.num_colors)
    train_raw, eval_raw = _standardize(train_raw, eval_raw)
    rollout_x = _standardize_with_train(train.features, eval_data.predicted_features)
    if model.latent_representation == "grid":
        train_grid_x, eval_grid_x = _standardize_spatial(train.spatial_features, eval_data.spatial_features)
        rollout_grid_x = _standardize_spatial_with_train(
            train.spatial_features,
            eval_data.predicted_spatial_features,
        )
    else:
        train_grid_x, eval_grid_x = train_x, eval_x
        rollout_grid_x = rollout_x

    max_objects = int(generator.spec.max_objects)
    grid_size = int(generator.spec.grid_size)
    num_colors = int(generator.spec.num_colors)
    metrics: dict[str, Any] = {
        "probe_fit_version": 4,
        "probe_class_balanced_objectives": 1.0,
        "probe_trajectory_kind": generator.spec.trajectory_kind,
    }

    metrics.update(
        _state_classifier_metrics(
            "probe",
            train_x,
            eval_x,
            train,
            eval_data,
            max_objects=max_objects,
            steps=steps,
            learning_rate=learning_rate,
        )
    )
    metrics.update(
        _state_classifier_metrics(
            "probe_mlp",
            train_x,
            eval_x,
            train,
            eval_data,
            max_objects=max_objects,
            steps=steps,
            learning_rate=learning_rate,
            nonlinear=True,
        )
    )
    metrics.update(
        _state_classifier_metrics(
            "raw_probe",
            train_raw,
            eval_raw,
            train,
            eval_data,
            max_objects=max_objects,
            steps=steps,
            learning_rate=learning_rate,
        )
    )

    metrics["probe_object_color_acc"] = _fit_slot_classifier(
        train_x,
        train.object_colors,
        train.object_present,
        eval_x,
        eval_data.object_colors,
        eval_data.object_present,
        num_classes=num_colors,
        steps=steps,
        learning_rate=learning_rate,
    )
    metrics["probe_object_shape_acc"] = _fit_slot_classifier(
        train_x,
        train.object_shapes,
        train.object_present,
        eval_x,
        eval_data.object_shapes,
        eval_data.object_present,
        num_classes=len(SHAPE_TYPES) + 1,
        steps=steps,
        learning_rate=learning_rate,
    )
    max_part_count = max(
        int(train.object_part_counts.max().item()),
        int(eval_data.object_part_counts.max().item()),
        1,
    )
    metrics["probe_object_part_count_acc"] = _fit_slot_classifier(
        train_x,
        train.object_part_counts,
        train.object_present,
        eval_x,
        eval_data.object_part_counts,
        eval_data.object_present,
        num_classes=max_part_count + 1,
        steps=steps,
        learning_rate=learning_rate,
    )
    metrics["raw_probe_object_color_acc"] = _fit_slot_classifier(
        train_raw,
        train.object_colors,
        train.object_present,
        eval_raw,
        eval_data.object_colors,
        eval_data.object_present,
        num_classes=num_colors,
        steps=steps,
        learning_rate=learning_rate,
    )
    metrics["raw_probe_object_shape_acc"] = _fit_slot_classifier(
        train_raw,
        train.object_shapes,
        train.object_present,
        eval_raw,
        eval_data.object_shapes,
        eval_data.object_present,
        num_classes=len(SHAPE_TYPES) + 1,
        steps=steps,
        learning_rate=learning_rate,
    )

    bbox_mse, rollout_bbox_mse = _fit_masked_regressor(
        train_x,
        train.object_bboxes,
        train.object_present,
        eval_x,
        eval_data.object_bboxes,
        eval_data.object_present,
        transfer_x=rollout_x,
        transfer_y=eval_data.future_object_bboxes,
        transfer_mask=eval_data.future_object_present,
        steps=steps,
        learning_rate=learning_rate,
    )
    metrics["probe_bbox_mse"] = bbox_mse
    metrics["probe_rollout_bbox_mse"] = rollout_bbox_mse
    metrics["probe_centroid_mse"] = _fit_masked_regressor(
        train_x,
        train.object_centroids,
        train.object_present,
        eval_x,
        eval_data.object_centroids,
        eval_data.object_present,
        steps=steps,
        learning_rate=learning_rate,
    )[0]
    metrics["probe_area_mse"] = _fit_masked_regressor(
        train_x,
        train.object_areas.unsqueeze(-1),
        train.object_present,
        eval_x,
        eval_data.object_areas.unsqueeze(-1),
        eval_data.object_present,
        steps=steps,
        learning_rate=learning_rate,
    )[0]
    completion_mse, rollout_completion_mse = _fit_masked_regressor(
        train_x,
        train.completion.unsqueeze(-1),
        train.object_present,
        eval_x,
        eval_data.completion.unsqueeze(-1),
        eval_data.object_present,
        transfer_x=rollout_x,
        transfer_y=eval_data.future_completion.unsqueeze(-1),
        transfer_mask=eval_data.future_object_present,
        steps=steps,
        learning_rate=learning_rate,
    )
    metrics["probe_completion_mse"] = completion_mse
    metrics["probe_rollout_completion_mse"] = rollout_completion_mse
    corruption_targets = (
        ("missing", train.object_missing, eval_data.object_missing, eval_data.future_object_missing),
        ("overgrowth", train.object_overgrowth, eval_data.object_overgrowth, eval_data.future_object_overgrowth),
        (
            "wrong_color",
            train.object_wrong_color,
            eval_data.object_wrong_color,
            eval_data.future_object_wrong_color,
        ),
    )
    for name, train_y, eval_y, future_y in corruption_targets:
        value, rollout_value = _fit_masked_regressor(
            train_x,
            train_y.unsqueeze(-1),
            train.object_present,
            eval_x,
            eval_y.unsqueeze(-1),
            eval_data.object_present,
            transfer_x=rollout_x,
            transfer_y=future_y.unsqueeze(-1),
            transfer_mask=eval_data.future_object_present,
            steps=steps,
            learning_rate=learning_rate,
        )
        metrics[f"probe_{name}_mse"] = value
        metrics[f"probe_rollout_{name}_mse"] = rollout_value
    metrics["raw_probe_bbox_mse"] = _fit_masked_regressor(
        train_raw,
        train.object_bboxes,
        train.object_present,
        eval_raw,
        eval_data.object_bboxes,
        eval_data.object_present,
        steps=steps,
        learning_rate=learning_rate,
    )[0]
    metrics["raw_probe_centroid_mse"] = _fit_masked_regressor(
        train_raw,
        train.object_centroids,
        train.object_present,
        eval_raw,
        eval_data.object_centroids,
        eval_data.object_present,
        steps=steps,
        learning_rate=learning_rate,
    )[0]
    metrics["raw_probe_area_mse"] = _fit_masked_regressor(
        train_raw,
        train.object_areas.unsqueeze(-1),
        train.object_present,
        eval_raw,
        eval_data.object_areas.unsqueeze(-1),
        eval_data.object_present,
        steps=steps,
        learning_rate=learning_rate,
    )[0]
    metrics["raw_probe_completion_mse"] = _fit_masked_regressor(
        train_raw,
        train.completion.unsqueeze(-1),
        train.object_present,
        eval_raw,
        eval_data.completion.unsqueeze(-1),
        eval_data.object_present,
        steps=steps,
        learning_rate=learning_rate,
    )[0]
    raw_corruption_targets = (
        ("missing", train.object_missing, eval_data.object_missing),
        ("overgrowth", train.object_overgrowth, eval_data.object_overgrowth),
        ("wrong_color", train.object_wrong_color, eval_data.object_wrong_color),
    )
    for name, train_y, eval_y in raw_corruption_targets:
        metrics[f"raw_probe_{name}_mse"] = _fit_masked_regressor(
            train_raw,
            train_y.unsqueeze(-1),
            train.object_present,
            eval_raw,
            eval_y.unsqueeze(-1),
            eval_data.object_present,
            steps=steps,
            learning_rate=learning_rate,
        )[0]
    for name, value in _fit_relation_classifier(
        train_x,
        train.relations,
        train.relation_present,
        eval_x,
        eval_data.relations,
        eval_data.relation_present,
        steps=steps,
        learning_rate=learning_rate,
    ).items():
        metrics[f"probe_relation_{name}"] = value
    for name, value in _fit_relation_classifier(
        train_raw,
        train.relations,
        train.relation_present,
        eval_raw,
        eval_data.relations,
        eval_data.relation_present,
        steps=steps,
        learning_rate=learning_rate,
    ).items():
        metrics[f"raw_probe_relation_{name}"] = value

    grid_metrics, rollout_grid_metrics = _fit_grid_decoder(
        train_grid_x,
        train.states,
        eval_grid_x,
        eval_data.states,
        num_classes=num_colors,
        transfer_x=rollout_grid_x,
        transfer_y=eval_data.future_states,
        steps=steps,
        learning_rate=learning_rate,
    )
    for name, value in grid_metrics.items():
        metrics[f"probe_grid_{name}"] = value
    for name, value in rollout_grid_metrics.items():
        metrics[f"probe_rollout_grid_{name}"] = value
    object_map_metrics, rollout_object_map_metrics = _fit_grid_decoder(
        train_grid_x,
        train.object_map,
        eval_grid_x,
        eval_data.object_map,
        num_classes=max_objects + 1,
        transfer_x=rollout_grid_x,
        transfer_y=eval_data.future_object_map,
        steps=steps,
        learning_rate=learning_rate,
    )
    for name, value in object_map_metrics.items():
        metrics[f"probe_object_map_{name}"] = value
    for name, value in rollout_object_map_metrics.items():
        metrics[f"probe_rollout_object_map_{name}"] = value
    raw_object_map_metrics, _ = _fit_grid_decoder(
        train_raw,
        train.object_map,
        eval_raw,
        eval_data.object_map,
        num_classes=max_objects + 1,
        transfer_x=None,
        transfer_y=None,
        steps=steps,
        learning_rate=learning_rate,
    )
    for name, value in raw_object_map_metrics.items():
        metrics[f"raw_probe_object_map_{name}"] = value

    rollout_count_acc, rollout_count_balanced = _fit_classifier_with_balanced_accuracy(
        train_x,
        train.object_count,
        rollout_x,
        eval_data.future_object_count,
        num_classes=max_objects + 1,
        steps=steps,
        learning_rate=learning_rate,
    )
    metrics["probe_rollout_object_count_acc"] = rollout_count_acc
    metrics["probe_rollout_object_count_balanced_acc"] = rollout_count_balanced

    delta_targets = (
        ("action_op", train.action[:, 0], eval_data.action[:, 0], 3),
        ("action_row", train.action[:, 1], eval_data.action[:, 1], grid_size),
        ("action_col", train.action[:, 2], eval_data.action[:, 2], grid_size),
        ("action_color", train.action[:, 3], eval_data.action[:, 3], num_colors),
        ("action_object", train.action_object_id, eval_data.action_object_id, max_objects + 1),
        ("action_process", train.action_process_type, eval_data.action_process_type, len(PROCESS_NAMES)),
        ("action_continues", train.continues_object.long(), eval_data.continues_object.long(), 2),
    )
    for name, train_y, eval_y, num_classes in delta_targets:
        metrics[f"probe_delta_{name}_acc"] = _fit_linear_classifier(
            train_delta,
            train_y,
            eval_delta,
            eval_y,
            num_classes=num_classes,
            steps=steps,
            learning_rate=learning_rate,
        )

    chunk_targets = (
        ("object", train.chunk_object_id, eval_data.chunk_object_id, max_objects + 1),
        ("op", train.chunk_op, eval_data.chunk_op, 3),
        ("shape", train.chunk_shape, eval_data.chunk_shape, len(SHAPE_TYPES) + 1),
        ("correction", train.chunk_correction, eval_data.chunk_correction, len(PROCESS_NAMES)),
        ("category", train.trajectory_category, eval_data.trajectory_category, 3),
    )
    for name, train_y, eval_y, num_classes in chunk_targets:
        metrics[f"probe_chunk_{name}_acc"] = _fit_linear_classifier(
            train_chunk,
            train_y,
            eval_chunk,
            eval_y,
            num_classes=num_classes,
            steps=steps,
            learning_rate=learning_rate,
        )

    metrics.update(_nearest_neighbor_metrics(eval_data))
    metrics.update(_latent_geometry_metrics(eval_data.features))
    metrics.update(_off_manifold_metrics(eval_data))
    metrics.update(_attention_metrics(train, eval_data))
    metrics.update(_hierarchy_planning_metrics(model, eval_data))
    metrics["batch_probe_semantic_rate"] = float((eval_data.trajectory_category == 0).float().mean().item())
    metrics["batch_probe_counterfactual_rate"] = float((eval_data.trajectory_category == 1).float().mean().item())
    metrics["batch_probe_wrong_rate"] = float((eval_data.trajectory_category == 2).float().mean().item())
    return metrics


def _state_classifier_metrics(
    prefix: str,
    train_x: torch.Tensor,
    eval_x: torch.Tensor,
    train: ProbeDataset,
    eval_data: ProbeDataset,
    *,
    max_objects: int,
    steps: int,
    learning_rate: float,
    nonlinear: bool = False,
) -> dict[str, float]:
    targets = (
        ("object_count", train.object_count, eval_data.object_count, max_objects + 1),
        ("scene_object_count", train.scene_object_count, eval_data.scene_object_count, max_objects + 1),
        ("current_object", train.current_object_id, eval_data.current_object_id, max_objects + 1),
        ("next_object", train.next_object_id, eval_data.next_object_id, max_objects + 1),
        ("valid_state", train.valid_state.long(), eval_data.valid_state.long(), 2),
        ("trajectory_category", train.trajectory_category, eval_data.trajectory_category, 3),
    )
    metrics = {}
    for name, train_y, eval_y, num_classes in targets:
        fit = _fit_mlp_classifier_with_balanced_accuracy if nonlinear else _fit_classifier_with_balanced_accuracy
        accuracy, balanced = fit(
            train_x,
            train_y,
            eval_x,
            eval_y,
            num_classes=num_classes,
            steps=steps,
            learning_rate=learning_rate,
        )
        metrics[f"{prefix}_{name}_acc"] = accuracy
        metrics[f"{prefix}_{name}_balanced_acc"] = balanced
        metrics[f"{prefix}_{name}_majority_acc"] = _majority_accuracy(train_y, eval_y)
    return metrics


def _standardize(train_x: torch.Tensor, eval_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1.0e-4)
    return (train_x - mean) / std, (eval_x - mean) / std


def _standardize_with_train(train_x: torch.Tensor, eval_x: torch.Tensor) -> torch.Tensor:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1.0e-4)
    return (eval_x - mean) / std


def _standardize_spatial(train_x: torch.Tensor, eval_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train_x.flatten(0, 1).mean(dim=0, keepdim=True)
    std = train_x.flatten(0, 1).std(dim=0, keepdim=True).clamp_min(1.0e-4)
    return (train_x - mean) / std, (eval_x - mean) / std


def _standardize_spatial_with_train(train_x: torch.Tensor, eval_x: torch.Tensor) -> torch.Tensor:
    mean = train_x.flatten(0, 1).mean(dim=0, keepdim=True)
    std = train_x.flatten(0, 1).std(dim=0, keepdim=True).clamp_min(1.0e-4)
    return (eval_x - mean) / std


def _raw_grid_features(states: torch.Tensor, num_colors: int) -> torch.Tensor:
    return F.one_hot(states.clamp(0, num_colors - 1), num_classes=num_colors).flatten(1).float()


def _fit_linear_classifier(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    eval_x: torch.Tensor,
    eval_y: torch.Tensor,
    *,
    num_classes: int,
    steps: int,
    learning_rate: float,
) -> float:
    return _fit_classifier_with_balanced_accuracy(
        train_x,
        train_y,
        eval_x,
        eval_y,
        num_classes=num_classes,
        steps=steps,
        learning_rate=learning_rate,
    )[0]


def _fit_classifier_with_balanced_accuracy(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    eval_x: torch.Tensor,
    eval_y: torch.Tensor,
    *,
    num_classes: int,
    steps: int,
    learning_rate: float,
) -> tuple[float, float]:
    probe = nn.Linear(train_x.shape[-1], num_classes)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    target = train_y.clamp(0, num_classes - 1)
    class_weights = _class_balanced_weights(target, num_classes=num_classes)
    for _ in range(steps):
        logits = probe(train_x)
        loss = F.cross_entropy(logits, target, weight=class_weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        pred = probe(eval_x).argmax(dim=-1)
        target = eval_y.clamp(0, num_classes - 1)
        accuracy = float((pred == target).float().mean().item())
        recalls = [
            (pred[target == label] == label).float().mean()
            for label in range(num_classes)
            if bool(torch.any(target == label))
        ]
        balanced = float(torch.stack(recalls).mean().item()) if recalls else 0.0
    return accuracy, balanced


def _fit_mlp_classifier_with_balanced_accuracy(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    eval_x: torch.Tensor,
    eval_y: torch.Tensor,
    *,
    num_classes: int,
    steps: int,
    learning_rate: float,
) -> tuple[float, float]:
    hidden = min(256, max(32, 2 * train_x.shape[-1]))
    probe = nn.Sequential(nn.Linear(train_x.shape[-1], hidden), nn.GELU(), nn.Linear(hidden, num_classes))
    optimizer = torch.optim.AdamW(probe.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    target = train_y.clamp(0, num_classes - 1)
    class_weights = _class_balanced_weights(target, num_classes=num_classes)
    for _ in range(steps):
        loss = F.cross_entropy(probe(train_x), target, weight=class_weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        pred = probe(eval_x).argmax(dim=-1)
        target = eval_y.clamp(0, num_classes - 1)
        accuracy = float((pred == target).float().mean().item())
        recalls = [
            (pred[target == label] == label).float().mean()
            for label in range(num_classes)
            if bool(torch.any(target == label))
        ]
        balanced = float(torch.stack(recalls).mean().item()) if recalls else 0.0
    return accuracy, balanced


def _fit_slot_classifier(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_mask: torch.Tensor,
    eval_x: torch.Tensor,
    eval_y: torch.Tensor,
    eval_mask: torch.Tensor,
    *,
    num_classes: int,
    steps: int,
    learning_rate: float,
) -> float:
    if not bool(torch.any(train_mask)) or not bool(torch.any(eval_mask)):
        return float("nan")
    slots = train_y.shape[1]
    probe = nn.Linear(train_x.shape[-1], slots * num_classes)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    target = train_y[train_mask].clamp(0, num_classes - 1)
    class_weights = _class_balanced_weights(target, num_classes=num_classes)
    for _ in range(steps):
        logits = probe(train_x).reshape(-1, slots, num_classes)
        loss = F.cross_entropy(logits[train_mask], target, weight=class_weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        pred = probe(eval_x).reshape(-1, slots, num_classes).argmax(dim=-1)
        return float((pred[eval_mask] == eval_y[eval_mask]).float().mean().item())


def _fit_masked_regressor(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_mask: torch.Tensor,
    eval_x: torch.Tensor,
    eval_y: torch.Tensor,
    eval_mask: torch.Tensor,
    *,
    transfer_x: torch.Tensor | None = None,
    transfer_y: torch.Tensor | None = None,
    transfer_mask: torch.Tensor | None = None,
    steps: int,
    learning_rate: float,
) -> tuple[float, float]:
    if not bool(torch.any(train_mask)) or not bool(torch.any(eval_mask)):
        return float("nan"), float("nan")
    slots, dimensions = train_y.shape[1:]
    probe = nn.Linear(train_x.shape[-1], slots * dimensions)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    for _ in range(steps):
        pred = torch.sigmoid(probe(train_x)).reshape(-1, slots, dimensions)
        loss = _masked_mse(pred, train_y, train_mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        pred = torch.sigmoid(probe(eval_x)).reshape(-1, slots, dimensions)
        eval_mse = float(_masked_mse(pred, eval_y, eval_mask).item())
        transfer_mse = float("nan")
        if (
            transfer_x is not None
            and transfer_y is not None
            and transfer_mask is not None
            and bool(torch.any(transfer_mask))
        ):
            transfer_pred = torch.sigmoid(probe(transfer_x)).reshape(-1, slots, dimensions)
            transfer_mse = float(_masked_mse(transfer_pred, transfer_y, transfer_mask).item())
    return eval_mse, transfer_mse


def _fit_relation_classifier(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_mask: torch.Tensor,
    eval_x: torch.Tensor,
    eval_y: torch.Tensor,
    eval_mask: torch.Tensor,
    *,
    steps: int,
    learning_rate: float,
) -> dict[str, float]:
    pairs, relation_count = train_y.shape[1:]
    if pairs == 0 or not bool(torch.any(train_mask)) or not bool(torch.any(eval_mask)):
        return {
            f"{name}_{metric}": float("nan")
            for name in RELATION_NAMES
            for metric in ("acc", "balanced_acc", "positive_rate")
        }
    probe = nn.Linear(train_x.shape[-1], pairs * relation_count)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    valid_targets = train_y[train_mask]
    positives = valid_targets.sum(dim=0)
    negatives = valid_targets.shape[0] - positives
    pos_weight = torch.where(
        (positives > 0) & (negatives > 0),
        negatives / positives,
        torch.ones_like(positives),
    )
    for _ in range(steps):
        logits = probe(train_x).reshape(-1, pairs, relation_count)
        loss = F.binary_cross_entropy_with_logits(logits[train_mask], valid_targets, pos_weight=pos_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        pred = probe(eval_x).reshape(-1, pairs, relation_count) >= 0.0
        metrics = {}
        for index, name in enumerate(RELATION_NAMES):
            relation_pred = pred[..., index][eval_mask]
            relation_target = eval_y[..., index][eval_mask].bool()
            metrics[f"{name}_acc"] = float((relation_pred == relation_target).float().mean().item())
            recalls = [
                (relation_pred[relation_target == label] == label).float().mean()
                for label in (False, True)
                if bool(torch.any(relation_target == label))
            ]
            metrics[f"{name}_balanced_acc"] = (
                float(torch.stack(recalls).mean().item()) if recalls else float("nan")
            )
            metrics[f"{name}_positive_rate"] = float(relation_target.float().mean().item())
        return metrics


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.unsqueeze(-1).expand_as(pred)
    return (pred[expanded] - target[expanded]).square().mean()


def _fit_grid_decoder(
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
) -> tuple[dict[str, float], dict[str, float]]:
    height, width = train_y.shape[1:]
    tokenwise = train_x.ndim == 3
    if tokenwise and train_x.shape[1] != height * width:
        raise ValueError(f"Spatial probe has {train_x.shape[1]} tokens for a {height}x{width} target.")
    output_size = num_classes if tokenwise else height * width * num_classes
    probe = nn.Linear(train_x.shape[-1], output_size)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    target = train_y.clamp(0, num_classes - 1)
    class_weights = _class_balanced_weights(target, num_classes=num_classes)
    for _ in range(steps):
        logits = _grid_probe_logits(probe, train_x, height=height, width=width, tokenwise=tokenwise)
        loss = F.cross_entropy(logits, target, weight=class_weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        pred = _grid_probe_logits(probe, eval_x, height=height, width=width, tokenwise=tokenwise).argmax(dim=1)
        eval_metrics = _segmentation_metrics(pred, eval_y, num_classes=num_classes)
        transfer_metrics = {
            "cell_acc": float("nan"),
            "balanced_acc": float("nan"),
            "foreground_acc": float("nan"),
            "foreground_miou": float("nan"),
        }
        if transfer_x is not None and transfer_y is not None:
            transfer_pred = _grid_probe_logits(
                probe,
                transfer_x,
                height=height,
                width=width,
                tokenwise=tokenwise,
            ).argmax(dim=1)
            transfer_metrics = _segmentation_metrics(transfer_pred, transfer_y, num_classes=num_classes)
    return eval_metrics, transfer_metrics


def _grid_probe_logits(
    probe: nn.Linear,
    features: torch.Tensor,
    *,
    height: int,
    width: int,
    tokenwise: bool,
) -> torch.Tensor:
    logits = probe(features)
    if tokenwise:
        return logits.transpose(1, 2).reshape(-1, probe.out_features, height, width)
    return logits.reshape(-1, probe.out_features // (height * width), height, width)


def _class_balanced_weights(target: torch.Tensor, *, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(target.long().reshape(-1), minlength=num_classes).float()
    present = counts > 0
    weights = torch.zeros(num_classes, dtype=torch.float32, device=target.device)
    if bool(torch.any(present)):
        total = counts[present].sum()
        weights[present] = total / (present.sum() * counts[present])
    return weights


def _segmentation_metrics(pred: torch.Tensor, target: torch.Tensor, *, num_classes: int) -> dict[str, float]:
    recalls = []
    foreground_ious = []
    for label in range(num_classes):
        target_mask = target == label
        if bool(torch.any(target_mask)):
            recalls.append((pred[target_mask] == label).float().mean())
        if label > 0:
            pred_mask = pred == label
            union = target_mask | pred_mask
            if bool(torch.any(union)):
                foreground_ious.append((target_mask & pred_mask).float().sum() / union.float().sum())
    foreground = target > 0
    return {
        "cell_acc": float((pred == target).float().mean().item()),
        "balanced_acc": float(torch.stack(recalls).mean().item()) if recalls else float("nan"),
        "foreground_acc": (
            float((pred[foreground] == target[foreground]).float().mean().item())
            if bool(torch.any(foreground))
            else float("nan")
        ),
        "foreground_miou": (
            float(torch.stack(foreground_ious).mean().item()) if foreground_ious else float("nan")
        ),
    }


def _majority_accuracy(train_y: torch.Tensor, eval_y: torch.Tensor) -> float:
    majority = int(train_y.long().bincount().argmax())
    return float((eval_y == majority).float().mean().item())


def _nearest_neighbor_metrics(data: ProbeDataset) -> dict[str, float]:
    if data.features.shape[0] < 2:
        return {
            "latent_nn_current_object_acc": float("nan"),
            "pixel_nn_current_object_acc": float("nan"),
            "latent_nn_next_object_acc": float("nan"),
            "pixel_nn_next_object_acc": float("nan"),
        }
    latent = F.normalize(data.features.float(), dim=-1)
    latent_distance = 1.0 - latent @ latent.T
    latent_distance.fill_diagonal_(float("inf"))
    latent_neighbor = latent_distance.argmin(dim=1)

    pixel_distance = _foreground_pixel_distance(data.states)
    pixel_distance.fill_diagonal_(float("inf"))
    pixel_neighbor = pixel_distance.argmin(dim=1)
    current_present = data.current_object_id > 0
    next_present = data.next_object_id > 0
    return {
        "latent_nn_current_object_acc": float(
            (data.current_object_id[latent_neighbor] == data.current_object_id).float().mean().item()
        ),
        "pixel_nn_current_object_acc": float(
            (data.current_object_id[pixel_neighbor] == data.current_object_id).float().mean().item()
        ),
        "latent_nn_current_object_present_acc": _masked_match_accuracy(
            data.current_object_id[latent_neighbor], data.current_object_id, current_present
        ),
        "pixel_nn_current_object_present_acc": _masked_match_accuracy(
            data.current_object_id[pixel_neighbor], data.current_object_id, current_present
        ),
        "latent_nn_next_object_acc": float(
            (data.next_object_id[latent_neighbor] == data.next_object_id).float().mean().item()
        ),
        "pixel_nn_next_object_acc": float(
            (data.next_object_id[pixel_neighbor] == data.next_object_id).float().mean().item()
        ),
        "latent_nn_next_object_present_acc": _masked_match_accuracy(
            data.next_object_id[latent_neighbor], data.next_object_id, next_present
        ),
        "pixel_nn_next_object_present_acc": _masked_match_accuracy(
            data.next_object_id[pixel_neighbor], data.next_object_id, next_present
        ),
        "latent_nn_category_acc": float(
            (data.trajectory_category[latent_neighbor] == data.trajectory_category).float().mean().item()
        ),
        "pixel_nn_category_acc": float(
            (data.trajectory_category[pixel_neighbor] == data.trajectory_category).float().mean().item()
        ),
    }


def _foreground_pixel_distance(states: torch.Tensor) -> torch.Tensor:
    left = states[:, None]
    right = states[None, :]
    union = (left != 0) | (right != 0)
    mismatch = (left != right) & union
    return mismatch.flatten(2).float().sum(dim=-1) / union.flatten(2).sum(dim=-1).clamp_min(1)


def _masked_match_accuracy(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(torch.any(mask)):
        return float("nan")
    return float((pred[mask] == target[mask]).float().mean().item())


def _off_manifold_metrics(data: ProbeDataset) -> dict[str, float]:
    categories = {0: "semantic", 1: "counterfactual", 2: "wrong"}
    metrics = {}
    for label, name in categories.items():
        mask = data.trajectory_category == label
        metrics[f"rollout_error_{name}_mean"] = (
            float(data.rollout_error[mask].mean().item()) if bool(torch.any(mask)) else float("nan")
        )

    normalized = F.normalize(data.features.float(), dim=-1)
    distance = 1.0 - normalized @ normalized.T
    semantic = (data.trajectory_category == 0) & (data.valid_state > 0.5)
    semantic_indices = torch.where(semantic)[0]
    semantic_distance = torch.full((len(data.features),), float("nan"))
    if semantic_indices.numel() > 0:
        candidates = distance[:, semantic_indices].clone()
        for column, sample_index in enumerate(semantic_indices):
            candidates[sample_index, column] = float("inf")
        values = candidates.min(dim=1).values
        semantic_distance = torch.where(torch.isfinite(values), values, torch.full_like(values, float("nan")))

    for label, name in categories.items():
        mask = (data.trajectory_category == label) & torch.isfinite(semantic_distance)
        metrics[f"latent_semantic_distance_{name}_mean"] = (
            float(semantic_distance[mask].mean().item()) if bool(torch.any(mask)) else float("nan")
        )

    invalid = data.valid_state < 0.5
    metrics["rollout_error_invalid_auroc"] = _binary_auroc(data.rollout_error, invalid)
    metrics["latent_semantic_distance_invalid_auroc"] = _binary_auroc(semantic_distance, invalid)
    return metrics


def _attention_metrics(train: ProbeDataset, eval_data: ProbeDataset) -> dict[str, float]:
    current_train = _attention_iou_rows(train, target_kind="current")
    current_eval = _attention_iou_rows(eval_data, target_kind="current")
    current_large_train = _attention_iou_rows(train, target_kind="current", min_cells=4)
    current_large_eval = _attention_iou_rows(eval_data, target_kind="current", min_cells=4)
    future_train = _attention_iou_rows(train, target_kind="future_current")
    future_eval = _attention_iou_rows(eval_data, target_kind="future_current")
    incomplete_train = _attention_iou_rows(train, target_kind="incomplete")
    incomplete_eval = _attention_iou_rows(eval_data, target_kind="incomplete")

    attention = eval_data.attention_maps.float()
    flat_attention = attention.flatten(2)
    entropy = -(flat_attention * flat_attention.clamp_min(1.0e-12).log()).sum(dim=-1)
    normalized_entropy = entropy / np.log(max(2, flat_attention.shape[-1]))

    return {
        "probe_attention_current_object_iou": _train_selected_head_iou(current_train, current_eval),
        "probe_attention_current_object_iou_oracle_head": _oracle_head_iou(current_eval),
        "probe_attention_current_object_iou_ge4": _train_selected_head_iou(
            current_large_train, current_large_eval
        ),
        "probe_attention_current_object_future_iou": _train_selected_head_iou(future_train, future_eval),
        "probe_attention_incomplete_object_iou": _train_selected_head_iou(incomplete_train, incomplete_eval),
        "probe_attention_incomplete_object_iou_oracle_head": _oracle_head_iou(incomplete_eval),
        "probe_attention_current_object_specialization": _head_gap(current_eval),
        "probe_attention_incomplete_object_specialization": _head_gap(incomplete_eval),
        "probe_attention_entropy": float(normalized_entropy.mean().item()),
    }


def _attention_iou_rows(
    data: ProbeDataset,
    *,
    target_kind: str,
    min_cells: int = 1,
) -> torch.Tensor:
    rows = []
    for sample in range(len(data.attention_maps)):
        current_id = int(data.current_object_id[sample])
        if target_kind == "current":
            target = data.object_map[sample] == current_id if current_id > 0 else None
        elif target_kind == "future_current":
            target = data.future_object_map[sample] == current_id if current_id > 0 else None
        elif target_kind == "incomplete":
            incomplete_ids = torch.where(
                data.object_present[sample] & (data.completion[sample] < 1.0 - 1.0e-6)
            )[0] + 1
            target = torch.isin(data.object_map[sample], incomplete_ids) if incomplete_ids.numel() else None
        else:
            raise ValueError(f"Unknown attention target kind {target_kind!r}.")
        if target is None or int(target.sum().item()) < min_cells:
            continue
        scores = _topk_attention_iou(data.attention_maps[sample].float(), target)
        if scores is not None:
            rows.append(scores)
    if rows:
        return torch.stack(rows)
    return torch.empty(0, data.attention_maps.shape[1])


def _train_selected_head_iou(train_scores: torch.Tensor, eval_scores: torch.Tensor) -> float:
    if train_scores.numel() == 0 or eval_scores.numel() == 0:
        return float("nan")
    selected_head = int(train_scores.mean(dim=0).argmax())
    return float(eval_scores[:, selected_head].mean().item())


def _oracle_head_iou(scores: torch.Tensor) -> float:
    return float(scores.max(dim=1).values.mean().item()) if scores.numel() else float("nan")


def _head_gap(scores: torch.Tensor) -> float:
    if scores.numel() == 0:
        return float("nan")
    return float((scores.max(dim=1).values - scores.mean(dim=1)).mean().item())


def _topk_attention_iou(attention: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
    target_size = int(target.sum().item())
    if target_size <= 0:
        return None
    flat = attention.flatten(1)
    selected = flat.topk(min(target_size, flat.shape[-1]), dim=-1).indices
    predicted = torch.zeros_like(flat, dtype=torch.bool)
    predicted.scatter_(1, selected, True)
    target_flat = target.flatten().unsqueeze(0)
    intersection = (predicted & target_flat).sum(dim=-1).float()
    union = (predicted | target_flat).sum(dim=-1).float().clamp_min(1.0)
    return intersection / union


def _hierarchy_planning_metrics(model: ObjectDynamicsJEPA, data: ProbeDataset) -> dict[str, float]:
    if not model.hierarchy_planning:
        return {
            "probe_hierarchy_endpoint_mse": float("nan"),
            "probe_hierarchy_macro_retrieval_acc": float("nan"),
            "probe_hierarchy_low_level_retrieval_acc": float("nan"),
            "probe_hierarchy_level_agreement": float("nan"),
            "probe_hierarchy_optimized_goal_l1": float("nan"),
            "probe_hierarchy_subgoal_reachability_l1": float("nan"),
            "probe_hierarchy_cem_subgoal_l1": float("nan"),
            "probe_hierarchy_cem_goal_l1": float("nan"),
            "probe_hierarchy_retrieval_goal_hamming": float("nan"),
            "probe_hierarchy_retrieval_goal_success": float("nan"),
            "probe_hierarchy_cem_executed_goal_hamming": float("nan"),
            "probe_hierarchy_cem_executed_goal_success": float("nan"),
            "probe_hierarchy_cem_model_bias_l1": float("nan"),
        }
    endpoint_mse = float(
        (data.hierarchy_predicted_spatial_features - data.future_spatial_features).square().mean().item()
    )
    device = next(model.parameters()).device
    high_correct = 0
    low_correct = 0
    agreement = 0
    count = 0
    optimized_goal = []
    subgoal_reachability = []
    cem_subgoal = []
    cem_goal = []
    retrieval_hamming = []
    retrieval_success = []
    cem_executed_hamming = []
    cem_executed_success = []
    cem_model_bias = []
    candidate_count = min(8, len(data.states))
    for start in range(0, len(data.states), candidate_count):
        stop = min(len(data.states), start + candidate_count)
        if stop - start < 2:
            continue
        states = data.states[start:stop].to(device)
        goals = data.hierarchy_goal_states[start:stop].to(device)
        final_goals = data.future_states[start:stop].to(device)
        action_library = data.actions[start:stop].to(device)
        candidates = action_library.unsqueeze(0).expand(stop - start, -1, -1, -1)
        plan = model.plan_macro_actions(states, candidates, goals)
        expected = torch.arange(stop - start, device=device)
        high_correct += int((plan.high_level_indices == expected).sum().item())
        low_correct += int((plan.low_level_indices == expected).sum().item())
        agreement += int((plan.high_level_indices == plan.low_level_indices).sum().item())
        selected_actions = candidates[expected, plan.low_level_indices]
        retrieval_endpoints = _execute_grid_actions(states, selected_actions)
        retrieval_hamming.append((retrieval_endpoints != goals).flatten(1).float().mean(dim=-1).cpu())
        retrieval_success.append((retrieval_endpoints == goals).flatten(1).all(dim=-1).float().cpu())
        continuous = model.optimize_macro_actions(
            states,
            final_goals,
            high_level_steps=model.hierarchy_rollout_steps,
            num_samples=16,
            num_elites=4,
            num_iterations=2,
        )
        _, tracking_scores, _ = model.track_subgoal(
            states,
            candidates,
            continuous.predicted_states[:, 0],
        )
        optimized_goal.append(continuous.goal_scores.detach().cpu())
        subgoal_reachability.append(tracking_scores.min(dim=1).values.detach().cpu())
        primitive = model.optimize_primitive_actions(
            states,
            continuous.predicted_states[:, 0],
            num_samples=16,
            num_elites=4,
            num_iterations=2,
        )
        with torch.no_grad():
            goal_latents = model.encode(goals)
            executed_endpoints = _execute_grid_actions(states, primitive.actions)
            executed_latents = model.encode(executed_endpoints)
        cem_subgoal.append(primitive.subgoal_scores.detach().cpu())
        cem_goal.append(
            (primitive.predicted_endpoints - goal_latents).abs().flatten(1).mean(dim=-1).detach().cpu()
        )
        cem_executed_hamming.append((executed_endpoints != goals).flatten(1).float().mean(dim=-1).cpu())
        cem_executed_success.append((executed_endpoints == goals).flatten(1).all(dim=-1).float().cpu())
        cem_model_bias.append(
            (primitive.predicted_endpoints - executed_latents).abs().flatten(1).mean(dim=-1).cpu()
        )
        count += stop - start
    denominator = max(1, count)
    return {
        "probe_hierarchy_endpoint_mse": endpoint_mse,
        "probe_hierarchy_macro_retrieval_acc": high_correct / denominator,
        "probe_hierarchy_low_level_retrieval_acc": low_correct / denominator,
        "probe_hierarchy_level_agreement": agreement / denominator,
        "probe_hierarchy_optimized_goal_l1": float(torch.cat(optimized_goal).mean().item()),
        "probe_hierarchy_subgoal_reachability_l1": float(torch.cat(subgoal_reachability).mean().item()),
        "probe_hierarchy_cem_subgoal_l1": float(torch.cat(cem_subgoal).mean().item()),
        "probe_hierarchy_cem_goal_l1": float(torch.cat(cem_goal).mean().item()),
        "probe_hierarchy_retrieval_goal_hamming": float(torch.cat(retrieval_hamming).mean().item()),
        "probe_hierarchy_retrieval_goal_success": float(torch.cat(retrieval_success).mean().item()),
        "probe_hierarchy_cem_executed_goal_hamming": float(torch.cat(cem_executed_hamming).mean().item()),
        "probe_hierarchy_cem_executed_goal_success": float(torch.cat(cem_executed_success).mean().item()),
        "probe_hierarchy_cem_model_bias_l1": float(torch.cat(cem_model_bias).mean().item()),
    }


def _execute_grid_actions(states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    output = states.clone()
    batch_indices = torch.arange(len(states), device=states.device)
    for step in range(actions.shape[1]):
        operation = actions[:, step, 0]
        row = actions[:, step, 1].clamp(0, states.shape[1] - 1)
        col = actions[:, step, 2].clamp(0, states.shape[2] - 1)
        color = actions[:, step, 3]
        output[batch_indices, row, col] = torch.where(operation == 1, torch.zeros_like(color), color)
    return output


def _binary_auroc(scores: torch.Tensor, positive: torch.Tensor) -> float:
    finite = torch.isfinite(scores)
    positive_scores = scores[finite & positive]
    negative_scores = scores[finite & ~positive]
    if positive_scores.numel() == 0 or negative_scores.numel() == 0:
        return float("nan")
    comparisons = positive_scores[:, None] - negative_scores[None, :]
    return float(((comparisons > 0).float() + 0.5 * (comparisons == 0).float()).mean().item())


def _latent_geometry_metrics(features: torch.Tensor) -> dict[str, float]:
    values = features.float()
    centered = values - values.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    variance = singular_values.square()
    probabilities = variance / variance.sum().clamp_min(1.0e-12)
    effective_rank = torch.exp(-(probabilities * probabilities.clamp_min(1.0e-12).log()).sum())
    std = values.std(dim=0)
    return {
        "latent_norm_mean": float(values.norm(dim=-1).mean().item()),
        "latent_std_mean": float(std.mean().item()),
        "latent_std_min": float(std.min().item()),
        "latent_effective_rank": float(effective_rank.item()),
    }
