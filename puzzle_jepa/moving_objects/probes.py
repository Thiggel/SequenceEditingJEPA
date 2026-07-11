from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from puzzle_jepa.moving_objects.batching import MovingObjectBatch, sample_moving_object_batch
from puzzle_jepa.moving_objects.generator import MovingObjectGenerator
from puzzle_jepa.moving_objects.model import MovingObjectJEPA


@dataclass(slots=True)
class _ProbeData:
    latents: torch.Tensor
    raw: torch.Tensor
    count: torch.Tensor
    semantic: torch.Tensor
    future_semantic: torch.Tensor
    grid: torch.Tensor
    rollout_latents: torch.Tensor


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
    shape_dim = train.semantic.shape[1] - (generator.spec.num_colors - 1) - 24 - 5

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
    latent_std = evaluate.latents.std(dim=0)
    covariance = torch.cov(evaluate.latents.T) if len(evaluate.latents) > 1 else torch.zeros(1, device=device)
    eigenvalues = torch.linalg.eigvalsh(covariance.float()).clamp_min(0.0)
    effective_rank = _effective_rank(eigenvalues)
    splits = _semantic_splits(shape_dim, generator.spec.num_colors - 1)
    metrics: dict[str, float | int | str] = {
        "probe_schema": "moving_objects_v1",
        "probe_object_count_acc": latent_count[0],
        "probe_object_count_balanced_acc": latent_count[1],
        "raw_probe_object_count_acc": raw_count[0],
        "raw_probe_object_count_balanced_acc": raw_count[1],
        "probe_grid_acc": grid_acc,
        "probe_grid_foreground_iou": grid_fg_iou,
        "probe_latent_std_mean": float(latent_std.mean()),
        "probe_latent_std_min": float(latent_std.min()),
        "probe_latent_effective_rank": effective_rank,
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
        "latents": [], "raw": [], "count": [], "semantic": [], "future_semantic": [],
        "grid": [], "rollout_latents": []
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
                batch.relations,
            ],
            dim=1,
        )
        future_semantic = torch.cat(
            [
                batch.shape_counts,
                batch.color_counts,
                batch.future_velocity_counts,
                batch.future_relations,
            ],
            dim=1,
        )
        parts["latents"].append(latents.detach())
        parts["raw"].append(
            F.one_hot(batch.contexts, num_classes=generator.spec.num_colors).flatten(1).float()
        )
        parts["count"].append(batch.object_count)
        parts["semantic"].append(semantic)
        parts["future_semantic"].append(future_semantic)
        parts["grid"].append(batch.current_grid)
        parts["rollout_latents"].append(output.predictions[:, 0].detach())
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
) -> tuple[float, float]:
    if standardize:
        train_x, eval_x = _standardize(train_x, eval_x)
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
    return accuracy, float(torch.stack(recalls).mean())


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
    relations = slice(velocity.stop, velocity.stop + 5)
    return {"shape_count": shape, "color_count": color, "velocity_count": velocity, "relations": relations}


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
