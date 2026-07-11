from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from puzzle_jepa.moving_objects.batching import MovingObjectBatch, sample_moving_object_batch
from puzzle_jepa.moving_objects.generator import ANGULAR_VELOCITIES, MovingObjectGenerator, SHAPE_NAMES
from puzzle_jepa.moving_objects.model import MovingObjectJEPA


@dataclass(slots=True)
class _ProbeData:
    latents: torch.Tensor
    raw: torch.Tensor
    count: torch.Tensor
    visible_count: torch.Tensor
    future_visible_count: torch.Tensor
    semantic: torch.Tensor
    future_semantic: torch.Tensor
    grid: torch.Tensor
    rollout_latents: torch.Tensor
    object_slots: torch.Tensor
    object_slot_present: torch.Tensor
    future_object_slots: torch.Tensor
    future_object_slot_present: torch.Tensor


def run_moving_object_probes(
    model: MovingObjectJEPA,
    generator: MovingObjectGenerator,
    rng: np.random.Generator,
    *,
    train_samples: int,
    eval_samples: int,
    batch_size: int,
    device: torch.device,
    steps: int,
    learning_rate: float,
) -> dict[str, float | int | str]:
    was_training = model.training
    model.eval()
    train = _collect(model, generator, rng, train_samples, batch_size, device)
    evaluate = _collect(model, generator, rng, eval_samples, batch_size, device)
    max_objects = generator.spec.max_objects
    shape_dim = len(SHAPE_NAMES)

    latent_count = _fit_classifier(
        train.latents,
        train.count,
        evaluate.latents,
        evaluate.count,
        num_classes=max_objects + 1,
        steps=steps,
        learning_rate=learning_rate,
    )
    raw_count = _fit_classifier(
        train.raw,
        train.count,
        evaluate.raw,
        evaluate.count,
        num_classes=max_objects + 1,
        steps=steps,
        learning_rate=learning_rate,
        standardize=False,
    )
    latent_visible_count = _fit_classifier(
        train.latents,
        train.visible_count,
        evaluate.latents,
        evaluate.visible_count,
        num_classes=max_objects + 1,
        steps=steps,
        learning_rate=learning_rate,
        transfer_x=evaluate.rollout_latents,
        transfer_y=evaluate.future_visible_count,
    )
    raw_visible_count = _fit_classifier(
        train.raw,
        train.visible_count,
        evaluate.raw,
        evaluate.visible_count,
        num_classes=max_objects + 1,
        steps=steps,
        learning_rate=learning_rate,
        standardize=False,
    )
    semantic_head, semantic_mae, semantic_r2, semantic_lower, semantic_upper = _fit_regressor(
        train.latents,
        train.semantic,
        evaluate.latents,
        evaluate.semantic,
        steps=steps,
        learning_rate=learning_rate,
    )
    _, raw_semantic_mae, raw_semantic_r2, _, _ = _fit_regressor(
        train.raw,
        train.semantic,
        evaluate.raw,
        evaluate.semantic,
        steps=steps,
        learning_rate=learning_rate,
        standardize=False,
    )
    with torch.no_grad():
        rollout_semantic = semantic_head(_standardize_with_train(train.latents, evaluate.rollout_latents))
        rollout_semantic = rollout_semantic.clamp(semantic_lower, semantic_upper)
    rollout_mae = (rollout_semantic - evaluate.future_semantic).abs().mean(dim=0)
    rollout_r2 = _r2_by_dimension(rollout_semantic, evaluate.future_semantic)
    grid_acc, grid_fg_iou = _fit_grid_decoder(
        train.latents,
        train.grid,
        evaluate.latents,
        evaluate.grid,
        num_colors=generator.spec.num_colors,
        steps=steps,
        learning_rate=learning_rate,
    )
    direct_reconstruction = _direct_reconstruction_metrics(model, evaluate)
    bound_metrics, rollout_bound_metrics = _fit_slot_regressor(
        train.latents,
        train.object_slots,
        train.object_slot_present,
        evaluate.latents,
        evaluate.object_slots,
        evaluate.object_slot_present,
        steps=steps,
        learning_rate=learning_rate,
        transfer_x=evaluate.rollout_latents,
        transfer_y=evaluate.future_object_slots,
        transfer_mask=evaluate.future_object_slot_present,
    )
    raw_bound_metrics, _ = _fit_slot_regressor(
        train.raw,
        train.object_slots,
        train.object_slot_present,
        evaluate.raw,
        evaluate.object_slots,
        evaluate.object_slot_present,
        steps=steps,
        learning_rate=learning_rate,
        standardize=False,
    )
    latent_std = evaluate.latents.std(dim=0)
    covariance = torch.cov(evaluate.latents.T) if len(evaluate.latents) > 1 else torch.zeros(1, device=device)
    eigenvalues = torch.linalg.eigvalsh(covariance.float()).clamp_min(0.0)
    effective_rank = _effective_rank(eigenvalues)
    splits = _semantic_splits(shape_dim, generator.spec.num_colors - 1)
    metrics: dict[str, float | int | str] = {
        "probe_schema": "moving_objects_v4",
        "probe_object_count_acc": latent_visible_count[0],
        "probe_object_count_balanced_acc": latent_visible_count[1],
        "raw_probe_object_count_acc": raw_visible_count[0],
        "raw_probe_object_count_balanced_acc": raw_visible_count[1],
        "probe_visible_object_count_acc": latent_visible_count[0],
        "probe_visible_object_count_balanced_acc": latent_visible_count[1],
        "raw_probe_visible_object_count_acc": raw_visible_count[0],
        "raw_probe_visible_object_count_balanced_acc": raw_visible_count[1],
        "probe_scene_object_count_acc": latent_count[0],
        "probe_scene_object_count_balanced_acc": latent_count[1],
        "raw_probe_scene_object_count_acc": raw_count[0],
        "raw_probe_scene_object_count_balanced_acc": raw_count[1],
        "probe_rollout_object_count_acc": latent_visible_count[2],
        "probe_rollout_object_count_balanced_acc": latent_visible_count[3],
        "probe_grid_acc": grid_acc,
        "probe_grid_foreground_iou": grid_fg_iou,
        **direct_reconstruction,
        "probe_latent_std_mean": float(latent_std.mean()),
        "probe_latent_std_min": float(latent_std.min()),
        "probe_latent_effective_rank": effective_rank,
        **{f"probe_bound_{key}": value for key, value in bound_metrics.items()},
        **{f"raw_probe_bound_{key}": value for key, value in raw_bound_metrics.items()},
        **{f"probe_rollout_bound_{key}": value for key, value in rollout_bound_metrics.items()},
    }
    for name, index in splits.items():
        metrics[f"probe_{name}_mae"] = float(semantic_mae[index].mean())
        metrics[f"probe_{name}_r2"] = float(semantic_r2[index].mean())
        metrics[f"raw_probe_{name}_mae"] = float(raw_semantic_mae[index].mean())
        metrics[f"raw_probe_{name}_r2"] = float(raw_semantic_r2[index].mean())
        metrics[f"probe_rollout_{name}_mae"] = float(rollout_mae[index].mean())
        metrics[f"probe_rollout_{name}_r2"] = float(rollout_r2[index].mean())
    if was_training:
        model.train()
    return metrics


@torch.no_grad()
def run_moving_object_dynamics_diagnostics(
    model: MovingObjectJEPA,
    generator: MovingObjectGenerator,
    rng: np.random.Generator,
    *,
    samples: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    totals = {
        "prediction_squared_error": 0.0,
        "identity_squared_error": 0.0,
        "target_transition_squared_norm": 0.0,
        "predictor_step_squared_norm": 0.0,
        "target_variance": 0.0,
        "pixel_change_rate": 0.0,
    }
    seen = 0
    while seen < samples:
        size = min(batch_size, samples - seen)
        batch = sample_moving_object_batch(generator, rng, batch_size=size, horizon=1, device=device)
        current = model.encode(batch.contexts)
        output = model(batch)
        predicted = output.predictions[:, 0]
        target = output.targets[:, 0]
        totals["prediction_squared_error"] += float((predicted - target).square().mean(dim=1).sum())
        totals["identity_squared_error"] += float((current - target).square().mean(dim=1).sum())
        totals["target_transition_squared_norm"] += float((target - current).square().sum(dim=1).sum())
        totals["predictor_step_squared_norm"] += float((predicted - current).square().sum(dim=1).sum())
        totals["target_variance"] += float(target.var(dim=0, unbiased=False).mean() * size)
        current_grid = batch.contexts[:, -1]
        next_grid = batch.future_contexts[:, 0, -1]
        totals["pixel_change_rate"] += float((current_grid != next_grid).float().mean(dim=(1, 2)).sum())
        seen += size
    metrics = {f"dynamics_{key}": value / seen for key, value in totals.items()}
    identity = metrics["dynamics_identity_squared_error"]
    prediction = metrics["dynamics_prediction_squared_error"]
    metrics["dynamics_prediction_gain"] = identity - prediction
    metrics["dynamics_prediction_gain_fraction"] = 1.0 - prediction / max(identity, 1.0e-12)
    metrics["dynamics_transition_to_variance_ratio"] = identity / max(
        metrics["dynamics_target_variance"], 1.0e-12
    )
    if was_training:
        model.train()
    return metrics


@torch.no_grad()
def _collect(
    model: MovingObjectJEPA,
    generator: MovingObjectGenerator,
    rng: np.random.Generator,
    samples: int,
    batch_size: int,
    device: torch.device,
) -> _ProbeData:
    parts: dict[str, list[torch.Tensor]] = {
        "latents": [], "raw": [], "count": [], "visible_count": [],
        "future_visible_count": [],
        "semantic": [], "future_semantic": [],
        "grid": [], "rollout_latents": [], "object_slots": [],
        "object_slot_present": [], "future_object_slots": [],
        "future_object_slot_present": []
    }
    remaining = samples
    while remaining:
        size = min(batch_size, remaining)
        batch = sample_moving_object_batch(generator, rng, batch_size=size, horizon=1, device=device)
        latents = model.encode(batch.contexts)
        output = model(batch)
        semantic = torch.cat(
            [
                batch.shape_counts,
                batch.color_counts,
                batch.velocity_counts,
                batch.angular_velocity_counts,
                batch.relations,
                batch.completion_features,
            ],
            dim=1,
        )
        future_semantic = torch.cat(
            [
                batch.future_shape_counts,
                batch.future_color_counts,
                batch.future_velocity_counts,
                batch.future_angular_velocity_counts,
                batch.future_relations,
                batch.future_completion_features,
            ],
            dim=1,
        )
        parts["latents"].append(latents.detach())
        parts["raw"].append(
            F.one_hot(batch.contexts, num_classes=generator.spec.num_colors).flatten(1).float()
        )
        parts["count"].append(batch.object_count)
        parts["visible_count"].append(batch.visible_object_count)
        parts["future_visible_count"].append(batch.future_visible_object_count)
        parts["semantic"].append(semantic)
        parts["future_semantic"].append(future_semantic)
        parts["grid"].append(batch.current_grid)
        parts["rollout_latents"].append(output.predictions[:, 0].detach())
        parts["object_slots"].append(batch.object_slot_features)
        parts["object_slot_present"].append(batch.object_slot_present)
        parts["future_object_slots"].append(batch.future_object_slot_features)
        parts["future_object_slot_present"].append(batch.future_object_slot_present)
        remaining -= size
    return _ProbeData(**{name: torch.cat(values) for name, values in parts.items()})


def _fit_classifier(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    eval_x: torch.Tensor,
    eval_y: torch.Tensor,
    *,
    num_classes: int,
    steps: int,
    learning_rate: float,
    standardize: bool = True,
    transfer_x: torch.Tensor | None = None,
    transfer_y: torch.Tensor | None = None,
) -> tuple[float, float, float, float]:
    if standardize:
        mean = train_x.mean(dim=0, keepdim=True)
        std = train_x.std(dim=0, keepdim=True).clamp_min(1.0e-4)
        train_x = (train_x - mean) / std
        eval_x = (eval_x - mean) / std
        if transfer_x is not None:
            transfer_x = (transfer_x - mean) / std
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
        predicted = head(eval_x).argmax(dim=1)
    accuracy = float((predicted == eval_y).float().mean())
    recalls = []
    for label in torch.unique(eval_y):
        mask = eval_y == label
        recalls.append((predicted[mask] == label).float().mean())
    balanced = float(torch.stack(recalls).mean())
    if transfer_x is None or transfer_y is None:
        return accuracy, balanced, float("nan"), float("nan")
    with torch.no_grad():
        transfer_predicted = head(transfer_x).argmax(dim=1)
    transfer_accuracy = float((transfer_predicted == transfer_y).float().mean())
    transfer_recalls = []
    for label in torch.unique(transfer_y):
        mask = transfer_y == label
        transfer_recalls.append((transfer_predicted[mask] == label).float().mean())
    return accuracy, balanced, transfer_accuracy, float(torch.stack(transfer_recalls).mean())


def _fit_regressor(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    eval_x: torch.Tensor,
    eval_y: torch.Tensor,
    *,
    steps: int,
    learning_rate: float,
    standardize: bool = True,
) -> tuple[nn.Linear, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if standardize:
        train_x, eval_x = _standardize(train_x, eval_x)
    head = nn.Linear(train_x.shape[1], train_y.shape[1]).to(train_x.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    scale = train_y.std(dim=0).clamp_min(0.25)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = ((head(train_x) - train_y) / scale).square().mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        lower = train_y.min(dim=0).values
        upper = train_y.max(dim=0).values
        predicted = head(eval_x).clamp(lower, upper)
        mae = (predicted - eval_y).abs().mean(dim=0)
        r2 = _r2_by_dimension(predicted, eval_y)
    return head, mae, r2, lower, upper


def _fit_grid_decoder(
    train_x: torch.Tensor,
    train_grid: torch.Tensor,
    eval_x: torch.Tensor,
    eval_grid: torch.Tensor,
    *,
    num_colors: int,
    steps: int,
    learning_rate: float,
) -> tuple[float, float]:
    train_x, eval_x = _standardize(train_x, eval_x)
    pixels = train_grid.shape[1] * train_grid.shape[2]
    head = nn.Linear(train_x.shape[1], pixels * num_colors).to(train_x.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    labels = train_grid.flatten(1)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = head(train_x).reshape(len(train_x), pixels, num_colors)
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        predicted = head(eval_x).reshape(len(eval_x), pixels, num_colors).argmax(dim=-1)
        target = eval_grid.flatten(1)
        accuracy = float((predicted == target).float().mean())
        foreground = target > 0
        intersection = ((predicted == target) & foreground).sum().float()
        union = ((predicted > 0) | foreground).sum().float().clamp_min(1.0)
    return accuracy, float(intersection / union)


@torch.no_grad()
def _direct_reconstruction_metrics(
    model: MovingObjectJEPA, evaluate: _ProbeData
) -> dict[str, float]:
    if model.reconstruction_decoder is None:
        return {}
    pixels = evaluate.grid.shape[1] * evaluate.grid.shape[2]
    predicted = model.reconstruction_decoder(evaluate.latents).reshape(
        len(evaluate.latents), pixels, model.num_colors
    ).argmax(dim=-1)
    target = evaluate.grid.flatten(1)
    accuracy = float((predicted == target).float().mean())
    foreground = target > 0
    intersection = ((predicted == target) & foreground).sum().float()
    union = ((predicted > 0) | foreground).sum().float().clamp_min(1.0)
    return {
        "model_reconstruction_grid_acc": accuracy,
        "model_reconstruction_foreground_iou": float(intersection / union),
    }


def _fit_slot_regressor(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_mask: torch.Tensor,
    eval_x: torch.Tensor,
    eval_y: torch.Tensor,
    eval_mask: torch.Tensor,
    *,
    steps: int,
    learning_rate: float,
    standardize: bool = True,
    transfer_x: torch.Tensor | None = None,
    transfer_y: torch.Tensor | None = None,
    transfer_mask: torch.Tensor | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    if standardize:
        mean = train_x.mean(dim=0, keepdim=True)
        std = train_x.std(dim=0, keepdim=True).clamp_min(1.0e-4)
        train_x = (train_x - mean) / std
        eval_x = (eval_x - mean) / std
        if transfer_x is not None:
            transfer_x = (transfer_x - mean) / std
    slots, dimensions = train_y.shape[1:]
    head = nn.Linear(train_x.shape[1], slots * dimensions).to(train_x.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    expanded_train_mask = train_mask.unsqueeze(-1).expand_as(train_y)
    observed_train_targets = train_y[train_mask]
    lower = observed_train_targets.min(dim=0).values
    upper = observed_train_targets.max(dim=0).values
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        predicted = head(train_x).reshape(-1, slots, dimensions)
        loss = (predicted - train_y).square()[expanded_train_mask].mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        predicted = head(eval_x).reshape(-1, slots, dimensions).clamp(lower, upper)
        metrics = _bound_metrics(predicted, eval_y, eval_mask)
        transfer_metrics = {}
        if transfer_x is not None and transfer_y is not None and transfer_mask is not None:
            transfer_predicted = head(transfer_x).reshape(-1, slots, dimensions).clamp(
                lower, upper
            )
            transfer_metrics = _bound_metrics(transfer_predicted, transfer_y, transfer_mask)
    return metrics, transfer_metrics


def _bound_metrics(
    predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> dict[str, float]:
    shape_dim = len(SHAPE_NAMES)
    groups = {
        "shape": slice(0, shape_dim),
        "velocity": slice(shape_dim, shape_dim + 2),
        "angular_velocity": slice(shape_dim + 2, shape_dim + 3),
        "position": slice(shape_dim + 3, shape_dim + 5),
        "completion": slice(shape_dim + 5, shape_dim + 6),
    }
    output = {}
    for name, feature_slice in groups.items():
        selected_prediction = predicted[..., feature_slice][mask]
        selected_target = target[..., feature_slice][mask]
        mae = (selected_prediction - selected_target).abs().mean()
        r2 = _r2_by_dimension(selected_prediction, selected_target).mean()
        output[f"{name}_mae"] = float(mae)
        output[f"{name}_r2"] = float(r2)
    shape_prediction = predicted[..., :shape_dim][mask].argmax(dim=-1)
    shape_target = target[..., :shape_dim][mask].argmax(dim=-1)
    output["shape_acc"] = float((shape_prediction == shape_target).float().mean())
    completion = target[..., shape_dim + 5]
    for name, threshold in (("half_complete", 0.5), ("complete", 1.0 - 1.0e-6)):
        selected = mask & (completion >= threshold)
        output[f"{name}_slots"] = float(selected.sum())
        if not selected.any():
            output[f"shape_acc_{name}"] = 0.0
            output[f"position_r2_{name}"] = 0.0
            continue
        selected_shape_prediction = predicted[..., :shape_dim][selected].argmax(dim=-1)
        selected_shape_target = target[..., :shape_dim][selected].argmax(dim=-1)
        output[f"shape_acc_{name}"] = float(
            (selected_shape_prediction == selected_shape_target).float().mean()
        )
        position_slice = groups["position"]
        output[f"position_r2_{name}"] = float(
            _r2_by_dimension(
                predicted[..., position_slice][selected], target[..., position_slice][selected]
            ).mean()
        )
    return output


def _standardize(train: torch.Tensor, evaluate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train.mean(dim=0, keepdim=True)
    std = train.std(dim=0, keepdim=True).clamp_min(1.0e-4)
    return (train - mean) / std, (evaluate - mean) / std


def _standardize_with_train(train: torch.Tensor, evaluate: torch.Tensor) -> torch.Tensor:
    return (evaluate - train.mean(dim=0, keepdim=True)) / train.std(dim=0, keepdim=True).clamp_min(1.0e-4)


def _semantic_splits(shape_dim: int, color_dim: int) -> dict[str, slice]:
    shape = slice(0, shape_dim)
    color = slice(shape.stop, shape.stop + color_dim)
    velocity = slice(color.stop, color.stop + 24)
    angular_velocity = slice(velocity.stop, velocity.stop + len(ANGULAR_VELOCITIES))
    relations = slice(angular_velocity.stop, angular_velocity.stop + 5)
    completion = slice(relations.stop, relations.stop + 5)
    return {
        "shape_count": shape,
        "color_count": color,
        "velocity_count": velocity,
        "angular_velocity_count": angular_velocity,
        "relations": relations,
        "completion": completion,
    }


def _effective_rank(eigenvalues: torch.Tensor) -> float:
    total = eigenvalues.sum()
    if float(total) <= 0.0:
        return 0.0
    probabilities = eigenvalues / total
    entropy = -(probabilities * probabilities.clamp_min(1.0e-12).log()).sum()
    return float(entropy.exp())


def _r2_by_dimension(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    residual = (predicted - target).square().sum(dim=0)
    total = (target - target.mean(dim=0, keepdim=True)).square().sum(dim=0)
    return torch.where(total > 1.0e-8, 1.0 - residual / total.clamp_min(1.0e-8), torch.zeros_like(total))
