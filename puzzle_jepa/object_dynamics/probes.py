from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from puzzle_jepa.object_dynamics.batching import sample_object_dynamics_batch
from puzzle_jepa.object_dynamics.generator import ObjectDynamicsGenerator
from puzzle_jepa.object_dynamics.model import ObjectDynamicsJEPA


@dataclass(frozen=True, slots=True)
class ProbeDataset:
    features: torch.Tensor
    object_count: torch.Tensor
    next_object_id: torch.Tensor
    valid_state: torch.Tensor
    completion: torch.Tensor


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
    features = []
    object_count = []
    next_object_id = []
    valid_state = []
    completion = []
    remaining = int(samples)
    while remaining > 0:
        current_batch = min(batch_size, remaining)
        batch = sample_object_dynamics_batch(generator, rng, batch_size=current_batch, horizon=horizon, device=device)
        features.append(model.encode(batch.states).detach().float().cpu())
        object_count.append(batch.object_count.detach().cpu())
        next_object_id.append(batch.next_object_id.detach().cpu())
        valid_state.append(batch.valid_state.detach().cpu())
        completion.append(batch.completion.detach().cpu())
        remaining -= current_batch
    return ProbeDataset(
        features=torch.cat(features, dim=0),
        object_count=torch.cat(object_count, dim=0),
        next_object_id=torch.cat(next_object_id, dim=0),
        valid_state=torch.cat(valid_state, dim=0),
        completion=torch.cat(completion, dim=0),
    )


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
    train = collect_probe_dataset(model, generator, rng, samples=train_samples, batch_size=batch_size, horizon=horizon, device=device)
    eval_data = collect_probe_dataset(model, generator, rng, samples=eval_samples, batch_size=batch_size, horizon=horizon, device=device)
    train_x, eval_x = _standardize(train.features, eval_data.features)
    max_objects = int(generator.spec.max_objects)
    metrics = {
        "probe_object_count_acc": _fit_linear_classifier(
            train_x,
            train.object_count,
            eval_x,
            eval_data.object_count,
            num_classes=max_objects + 1,
            steps=steps,
            learning_rate=learning_rate,
        ),
        "probe_next_object_acc": _fit_linear_classifier(
            train_x,
            train.next_object_id,
            eval_x,
            eval_data.next_object_id,
            num_classes=max_objects + 1,
            steps=steps,
            learning_rate=learning_rate,
        ),
        "probe_valid_state_acc": _fit_linear_classifier(
            train_x,
            train.valid_state.long(),
            eval_x,
            eval_data.valid_state.long(),
            num_classes=2,
            steps=steps,
            learning_rate=learning_rate,
        ),
        "probe_completion_mse": _fit_linear_regressor(
            train_x,
            train.completion,
            eval_x,
            eval_data.completion,
            steps=steps,
            learning_rate=learning_rate,
        ),
        "latent_norm_mean": float(eval_data.features.norm(dim=-1).mean().item()),
        "latent_std_mean": float(eval_data.features.std(dim=0).mean().item()),
    }
    return metrics


def _standardize(train_x: torch.Tensor, eval_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1.0e-4)
    return (train_x - mean) / std, (eval_x - mean) / std


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
        return float((pred == eval_y.clamp(0, num_classes - 1)).float().mean().item())


def _fit_linear_regressor(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    eval_x: torch.Tensor,
    eval_y: torch.Tensor,
    *,
    steps: int,
    learning_rate: float,
) -> float:
    probe = nn.Linear(train_x.shape[-1], train_y.shape[-1])
    optimizer = torch.optim.AdamW(probe.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    for _ in range(steps):
        pred = probe(train_x)
        loss = F.mse_loss(torch.sigmoid(pred), train_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        pred = torch.sigmoid(probe(eval_x))
        return float(F.mse_loss(pred, eval_y).item())
