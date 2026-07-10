from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from puzzle_jepa.object_dynamics.batching import sample_object_dynamics_batch
from puzzle_jepa.object_dynamics.generator import ObjectDynamicsGenerator, ObjectDynamicsSpec
from puzzle_jepa.object_dynamics.initialization import initialize_low_level_from_checkpoint
from puzzle_jepa.object_dynamics.model import ObjectDynamicsJEPA


def generate_object_dynamics_qualitative(
    checkpoint_path: Path,
    *,
    output_dir: Path,
    samples: int = 8,
    device: str = "auto",
) -> dict[str, Any]:
    resolved_device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "auto"
        else torch.device(device)
    )
    payload = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    config = payload["config"]
    data_config = _without_name(dict(config["data"]))
    eval_config = dict(config.get("eval", {}))
    data_config["trajectory_kind"] = str(
        eval_config.get("probe_trajectory_kind", data_config["trajectory_kind"])
    )
    generator = ObjectDynamicsGenerator(ObjectDynamicsSpec(**data_config))
    model_config = _without_name(dict(config["model"]))
    objective_config = _without_name(dict(config["objective"]))
    seed = int(config["seed"])

    torch.manual_seed(seed)
    initial_model = ObjectDynamicsJEPA(
        grid_size=generator.spec.grid_size,
        num_colors=generator.spec.num_colors,
        **model_config,
        **objective_config,
    ).to(resolved_device)
    initialize_low_level_from_checkpoint(
        initial_model,
        dict(config.get("training", {})).get("initial_checkpoint"),
        device=resolved_device,
    )
    trained_model = ObjectDynamicsJEPA(
        grid_size=generator.spec.grid_size,
        num_colors=generator.spec.num_colors,
        **model_config,
        **objective_config,
    ).to(resolved_device)
    trained_model.load_state_dict(payload["model"])
    initial_model.eval()
    trained_model.eval()

    probe_seed = seed + 100_003
    horizon = trained_model.training_horizon
    pool_size = max(128, samples * 8)
    batch = sample_object_dynamics_batch(
        generator,
        np.random.default_rng(probe_seed),
        batch_size=pool_size,
        horizon=horizon,
        device=resolved_device,
    )
    selected = _select_examples(batch.object_map, batch.current_object_id, batch.object_present, samples)
    states = batch.states[selected]
    futures = batch.futures[selected, -1]
    object_maps = batch.object_map[selected]
    current_ids = batch.current_object_id[selected]

    with torch.no_grad():
        initial_attention = initial_model.attention_maps(states).cpu().numpy()
        trained_attention = trained_model.attention_maps(states).cpu().numpy()
        initial_pool_features = initial_model.pool_latents(initial_model.encode(batch.states)).cpu()
        trained_pool_features = trained_model.pool_latents(trained_model.encode(batch.states)).cpu()
        predicted = trained_model.pool_latents(
            trained_model.predict_latents(states, batch.actions[selected])[:, -1]
        ).cpu()
        target = trained_model.pool_latents(trained_model.encode(futures)).cpu()

    state_array = states.cpu().numpy()
    future_array = futures.cpu().numpy()
    object_map_array = object_maps.cpu().numpy()
    current_id_array = current_ids.cpu().numpy()
    attention_metadata = _plot_attention_examples(
        output_dir / "attention_examples.png",
        state_array,
        future_array,
        object_map_array,
        current_id_array,
        initial_attention,
        trained_attention,
        num_colors=generator.spec.num_colors,
    )
    neighbor_metadata = _plot_nearest_neighbors(
        output_dir / "nearest_neighbors.png",
        batch.states.cpu().numpy(),
        batch.current_object_id.cpu().numpy(),
        initial_pool_features,
        trained_pool_features,
        selected.cpu().numpy(),
        num_colors=generator.spec.num_colors,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "qualitative_arrays.npz",
        states=state_array,
        futures=future_array,
        object_maps=object_map_array,
        current_object_ids=current_id_array,
        initial_attention=initial_attention,
        trained_attention=trained_attention,
        pool_states=batch.states.cpu().numpy(),
        pool_current_object_ids=batch.current_object_id.cpu().numpy(),
        initial_pool_features=initial_pool_features.numpy(),
        trained_pool_features=trained_pool_features.numpy(),
        predicted_features=predicted.numpy(),
        target_features=target.numpy(),
    )
    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(payload["step"]),
        "seed": seed,
        "probe_seed": probe_seed,
        "probe_trajectory_kind": generator.spec.trajectory_kind,
        "sample_indices": selected.cpu().tolist(),
        "nearest_neighbor_pool_size": pool_size,
        "latent_rollout_mse": float((predicted - target).square().mean().item()),
        "attention_examples": attention_metadata,
        "attention_iou_initial_mean": float(
            np.nanmean([row["initial_attention_iou"] for row in attention_metadata])
        ),
        "attention_iou_trained_mean": float(
            np.nanmean([row["trained_attention_iou"] for row in attention_metadata])
        ),
        "nearest_neighbors": neighbor_metadata,
        "initial_latent_neighbor_current_object_match": float(
            np.mean([row["initial_latent_current_object_match"] for row in neighbor_metadata])
        ),
        "latent_neighbor_current_object_match": float(
            np.mean([row["latent_current_object_match"] for row in neighbor_metadata])
        ),
        "pixel_neighbor_current_object_match": float(
            np.mean([row["pixel_current_object_match"] for row in neighbor_metadata])
        ),
    }
    (output_dir / "qualitative_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _select_examples(
    object_maps: torch.Tensor,
    current_ids: torch.Tensor,
    object_present: torch.Tensor,
    samples: int,
) -> torch.Tensor:
    large = []
    useful = []
    fallback = []
    for index in range(len(object_maps)):
        fallback.append(index)
        current_id = int(current_ids[index])
        current_cells = int((object_maps[index] == current_id).sum().item()) if current_id > 0 else 0
        if current_cells >= 4:
            large.append(index)
        if current_cells > 0:
            useful.append(index)
        elif bool(torch.any(object_present[index])):
            useful.append(index)
    ordered = list(dict.fromkeys([*large, *useful, *fallback]))[:samples]
    return torch.as_tensor(ordered, dtype=torch.long, device=object_maps.device)


def _plot_attention_examples(
    path: Path,
    states: np.ndarray,
    futures: np.ndarray,
    object_maps: np.ndarray,
    current_ids: np.ndarray,
    initial_attention: np.ndarray,
    trained_attention: np.ndarray,
    *,
    num_colors: int,
) -> list[dict[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(len(states), 5, figsize=(12, 2.25 * len(states)), squeeze=False)
    metadata = []
    for row in range(len(states)):
        target = object_maps[row] == int(current_ids[row]) if int(current_ids[row]) > 0 else object_maps[row] > 0
        initial_head, initial_iou = _best_attention_head(initial_attention[row], target)
        trained_head, trained_iou = _best_attention_head(trained_attention[row], target)
        _plot_grid(axes[row, 0], states[row], num_colors=num_colors, title="state")
        _plot_grid(axes[row, 1], object_maps[row] + 1, num_colors=6, title="hidden objects")
        axes[row, 2].imshow(initial_attention[row, initial_head], cmap="magma")
        axes[row, 2].set_title(f"initial h{initial_head} IoU {initial_iou:.2f}")
        axes[row, 3].imshow(trained_attention[row, trained_head], cmap="magma")
        axes[row, 3].set_title(f"trained h{trained_head} IoU {trained_iou:.2f}")
        _plot_grid(axes[row, 4], futures[row], num_colors=num_colors, title="future")
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
        metadata.append(
            {
                "current_object_id": int(current_ids[row]),
                "target_cells": int(np.count_nonzero(target)),
                "initial_best_head": initial_head,
                "initial_attention_iou": initial_iou,
                "trained_best_head": trained_head,
                "trained_attention_iou": trained_iou,
            }
        )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return metadata


def _plot_nearest_neighbors(
    path: Path,
    states: np.ndarray,
    current_ids: np.ndarray,
    initial_features: torch.Tensor,
    trained_features: torch.Tensor,
    query_indices: np.ndarray,
    *,
    num_colors: int,
) -> list[dict[str, int]]:
    initial_normalized = torch.nn.functional.normalize(initial_features.float(), dim=-1)
    trained_normalized = torch.nn.functional.normalize(trained_features.float(), dim=-1)
    initial_distance = 1.0 - initial_normalized @ initial_normalized.T
    trained_distance = 1.0 - trained_normalized @ trained_normalized.T
    initial_distance.fill_diagonal_(float("inf"))
    trained_distance.fill_diagonal_(float("inf"))
    pixels = torch.as_tensor(states)
    union = (pixels[:, None] != 0) | (pixels[None, :] != 0)
    mismatch = (pixels[:, None] != pixels[None, :]) & union
    pixel_distance = mismatch.flatten(2).float().sum(dim=-1) / union.flatten(2).sum(dim=-1).clamp_min(1)
    pixel_distance.fill_diagonal_(float("inf"))
    query_count = min(4, len(query_indices))
    figure, axes = plt.subplots(query_count, 4, figsize=(10, 2.25 * query_count), squeeze=False)
    metadata = []
    for row in range(query_count):
        query = int(query_indices[row])
        initial_neighbor = int(initial_distance[query].argmin())
        latent_neighbor = int(trained_distance[query].argmin())
        pixel_neighbor = int(pixel_distance[query].argmin())
        _plot_grid(
            axes[row, 0],
            states[query],
            num_colors=num_colors,
            title=f"query {query}, object {current_ids[query]}",
        )
        _plot_grid(
            axes[row, 1],
            states[initial_neighbor],
            num_colors=num_colors,
            title=f"initial NN object {current_ids[initial_neighbor]}",
        )
        _plot_grid(
            axes[row, 2],
            states[latent_neighbor],
            num_colors=num_colors,
            title=f"trained NN object {current_ids[latent_neighbor]}",
        )
        _plot_grid(
            axes[row, 3],
            states[pixel_neighbor],
            num_colors=num_colors,
            title=f"pixel NN object {current_ids[pixel_neighbor]}",
        )
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
        metadata.append(
            {
                "query": row,
                "query_pool_index": query,
                "initial_latent_neighbor": initial_neighbor,
                "latent_neighbor": latent_neighbor,
                "pixel_neighbor": pixel_neighbor,
                "latent_neighbor_current_object_id": int(current_ids[latent_neighbor]),
                "initial_latent_neighbor_current_object_id": int(current_ids[initial_neighbor]),
                "pixel_neighbor_current_object_id": int(current_ids[pixel_neighbor]),
                "latent_current_object_match": int(current_ids[latent_neighbor] == current_ids[query]),
                "initial_latent_current_object_match": int(
                    current_ids[initial_neighbor] == current_ids[query]
                ),
                "pixel_current_object_match": int(current_ids[pixel_neighbor] == current_ids[query]),
            }
        )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return metadata


def _best_attention_head(attention: np.ndarray, target: np.ndarray) -> tuple[int, float]:
    target_size = int(np.count_nonzero(target))
    if target_size == 0:
        return 0, float("nan")
    scores = []
    for head in attention:
        selected = np.argpartition(head.reshape(-1), -target_size)[-target_size:]
        prediction = np.zeros(head.size, dtype=bool)
        prediction[selected] = True
        prediction = prediction.reshape(head.shape)
        union = np.count_nonzero(prediction | target)
        scores.append(np.count_nonzero(prediction & target) / max(1, union))
    index = int(np.argmax(scores))
    return index, float(scores[index])


def _plot_grid(axis: Any, grid: np.ndarray, *, num_colors: int, title: str) -> None:
    axis.imshow(grid, cmap="tab10", vmin=0, vmax=max(1, num_colors - 1), interpolation="nearest")
    axis.set_title(title)


def _without_name(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != "name"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export fixed-batch object-dynamics qualitative diagnostics.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = generate_object_dynamics_qualitative(
        args.checkpoint,
        output_dir=args.output_dir,
        samples=args.samples,
        device=args.device,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "samples": len(result["sample_indices"])}, sort_keys=True))


if __name__ == "__main__":
    main()
