from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from puzzle_jepa.object_dynamics.batching import RELATION_NAMES, sample_object_dynamics_batch
from puzzle_jepa.object_dynamics.generator import ObjectDynamicsGenerator
from puzzle_jepa.object_dynamics.model import ObjectDynamicsJEPA
from puzzle_jepa.object_dynamics.shapes import SHAPE_TYPES


@dataclass(frozen=True, slots=True)
class ProbeDataset:
    features: torch.Tensor
    predicted_features: torch.Tensor
    rollout_error: torch.Tensor
    delta_features: torch.Tensor
    chunk_features: torch.Tensor
    states: torch.Tensor
    future_states: torch.Tensor
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
    object_missing: torch.Tensor
    future_object_missing: torch.Tensor
    object_overgrowth: torch.Tensor
    future_object_overgrowth: torch.Tensor
    object_wrong_color: torch.Tensor
    future_object_wrong_color: torch.Tensor
    object_map: torch.Tensor
    future_object_map: torch.Tensor
    action: torch.Tensor
    action_object_id: torch.Tensor
    continues_object: torch.Tensor
    chunk_object_id: torch.Tensor
    chunk_op: torch.Tensor
    chunk_shape: torch.Tensor
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
        features = model.encode(batch.states).float()
        next_features = model.encode(batch.futures[:, 0]).float()
        predicted = model.predict_latents(batch.states, batch.actions).float()
        future_features = model.encode(batch.futures[:, -1]).float()
        chunk = model.encode_action_chunk(batch.actions).float()
        chunk_object_id = batch.action_object_ids.mode(dim=1).values
        chunk_op = batch.actions[..., 0].mode(dim=1).values
        chunk_shape = batch.action_object_shapes.mode(dim=1).values

        values = {
            "features": features,
            "predicted_features": predicted[:, -1],
            "rollout_error": (predicted[:, -1] - future_features).square().mean(dim=-1),
            "delta_features": next_features - features,
            "chunk_features": chunk,
            "states": batch.states,
            "future_states": batch.futures[:, -1],
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
            "object_missing": batch.object_missing,
            "future_object_missing": batch.future_object_missing[:, -1],
            "object_overgrowth": batch.object_overgrowth,
            "future_object_overgrowth": batch.future_object_overgrowth[:, -1],
            "object_wrong_color": batch.object_wrong_color,
            "future_object_wrong_color": batch.future_object_wrong_color[:, -1],
            "object_map": batch.object_map,
            "future_object_map": batch.future_object_map[:, -1],
            "action": batch.actions[:, 0],
            "action_object_id": batch.action_object_ids[:, 0],
            "continues_object": batch.continues_object,
            "chunk_object_id": chunk_object_id,
            "chunk_op": chunk_op,
            "chunk_shape": chunk_shape,
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

    max_objects = int(generator.spec.max_objects)
    grid_size = int(generator.spec.grid_size)
    num_colors = int(generator.spec.num_colors)
    metrics: dict[str, Any] = {}

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
        transfer_x=_standardize_with_train(train.features, eval_data.predicted_features),
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
        transfer_x=_standardize_with_train(train.features, eval_data.predicted_features),
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
            transfer_x=_standardize_with_train(train.features, eval_data.predicted_features),
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
    for name, accuracy in _fit_relation_classifier(
        train_x,
        train.relations,
        train.relation_present,
        eval_x,
        eval_data.relations,
        eval_data.relation_present,
        steps=steps,
        learning_rate=learning_rate,
    ).items():
        metrics[f"probe_relation_{name}_acc"] = accuracy
    for name, accuracy in _fit_relation_classifier(
        train_raw,
        train.relations,
        train.relation_present,
        eval_raw,
        eval_data.relations,
        eval_data.relation_present,
        steps=steps,
        learning_rate=learning_rate,
    ).items():
        metrics[f"raw_probe_relation_{name}_acc"] = accuracy

    grid_metrics, rollout_grid_metrics = _fit_grid_decoder(
        train_x,
        train.states,
        eval_x,
        eval_data.states,
        num_classes=num_colors,
        transfer_x=_standardize_with_train(train.features, eval_data.predicted_features),
        transfer_y=eval_data.future_states,
        steps=steps,
        learning_rate=learning_rate,
    )
    for name, value in grid_metrics.items():
        metrics[f"probe_grid_{name}"] = value
    for name, value in rollout_grid_metrics.items():
        metrics[f"probe_rollout_grid_{name}"] = value
    object_map_metrics, rollout_object_map_metrics = _fit_grid_decoder(
        train_x,
        train.object_map,
        eval_x,
        eval_data.object_map,
        num_classes=max_objects + 1,
        transfer_x=_standardize_with_train(train.features, eval_data.predicted_features),
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

    delta_targets = (
        ("action_op", train.action[:, 0], eval_data.action[:, 0], 3),
        ("action_row", train.action[:, 1], eval_data.action[:, 1], grid_size),
        ("action_col", train.action[:, 2], eval_data.action[:, 2], grid_size),
        ("action_color", train.action[:, 3], eval_data.action[:, 3], num_colors),
        ("action_object", train.action_object_id, eval_data.action_object_id, max_objects + 1),
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
        accuracy, balanced = _fit_classifier_with_balanced_accuracy(
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
    for _ in range(steps):
        logits = probe(train_x)
        loss = F.cross_entropy(logits, train_y.clamp(0, num_classes - 1))
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
    for _ in range(steps):
        logits = probe(train_x).reshape(-1, slots, num_classes)
        loss = F.cross_entropy(logits[train_mask], train_y[train_mask].clamp(0, num_classes - 1))
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
        return {name: float("nan") for name in RELATION_NAMES}
    probe = nn.Linear(train_x.shape[-1], pairs * relation_count)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    train_mask_expanded = train_mask.unsqueeze(-1).expand_as(train_y)
    for _ in range(steps):
        logits = probe(train_x).reshape(-1, pairs, relation_count)
        loss = F.binary_cross_entropy_with_logits(logits[train_mask_expanded], train_y[train_mask_expanded])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        pred = probe(eval_x).reshape(-1, pairs, relation_count) >= 0.0
        return {
            name: float((pred[..., index][eval_mask] == eval_y[..., index][eval_mask].bool()).float().mean().item())
            for index, name in enumerate(RELATION_NAMES)
        }


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
    probe = nn.Linear(train_x.shape[-1], height * width * num_classes)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    for _ in range(steps):
        logits = probe(train_x).reshape(-1, num_classes, height, width)
        loss = F.cross_entropy(logits, train_y.clamp(0, num_classes - 1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        pred = probe(eval_x).reshape(-1, num_classes, height, width).argmax(dim=1)
        eval_metrics = _segmentation_metrics(pred, eval_y, num_classes=num_classes)
        transfer_metrics = {
            "cell_acc": float("nan"),
            "balanced_acc": float("nan"),
            "foreground_acc": float("nan"),
            "foreground_miou": float("nan"),
        }
        if transfer_x is not None and transfer_y is not None:
            transfer_pred = probe(transfer_x).reshape(-1, num_classes, height, width).argmax(dim=1)
            transfer_metrics = _segmentation_metrics(transfer_pred, transfer_y, num_classes=num_classes)
    return eval_metrics, transfer_metrics


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

    pixels = data.states.flatten(1)
    pixel_distance = (pixels[:, None] != pixels[None, :]).float().mean(dim=-1)
    pixel_distance.fill_diagonal_(float("inf"))
    pixel_neighbor = pixel_distance.argmin(dim=1)
    return {
        "latent_nn_current_object_acc": float(
            (data.current_object_id[latent_neighbor] == data.current_object_id).float().mean().item()
        ),
        "pixel_nn_current_object_acc": float(
            (data.current_object_id[pixel_neighbor] == data.current_object_id).float().mean().item()
        ),
        "latent_nn_next_object_acc": float(
            (data.next_object_id[latent_neighbor] == data.next_object_id).float().mean().item()
        ),
        "pixel_nn_next_object_acc": float(
            (data.next_object_id[pixel_neighbor] == data.next_object_id).float().mean().item()
        ),
        "latent_nn_category_acc": float(
            (data.trajectory_category[latent_neighbor] == data.trajectory_category).float().mean().item()
        ),
        "pixel_nn_category_acc": float(
            (data.trajectory_category[pixel_neighbor] == data.trajectory_category).float().mean().item()
        ),
    }


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
