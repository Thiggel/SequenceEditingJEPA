from __future__ import annotations

import csv
import json
import math
from contextlib import contextmanager
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterator

import numpy as np
import torch
from torch import nn

from puzzle_jepa.data.lewm_sudoku import action_to_array, apply_fill_action, legal_fill_actions
from puzzle_jepa.data.worlds import PuzzleExample, WorldAction
from puzzle_jepa.models.lewm import LeWMSudokuModel
from puzzle_jepa.planning.lewm_planner import _rank_immediate_actions, hamming_distance, score_action_sequence


DEFAULT_PROJECTION_HORIZONS = (1, 4, 8, 16, 32, 64)
DEFAULT_RANK_FILL_FRACTIONS = (0.0, 0.25, 0.5, 0.75)


def run_lewm_diagnostic_bundle(
    model: LeWMSudokuModel,
    examples: list[PuzzleExample],
    output_dir: Path,
    *,
    device: torch.device,
    seed: int = 0,
    latent_examples: int = 128,
    trajectory_examples: int = 32,
    rank_examples: int = 16,
    panel_examples: int = 3,
    panel_steps: int = 5,
    panel_actions: int = 6,
    projection_horizons: tuple[int, ...] = DEFAULT_PROJECTION_HORIZONS,
    rank_fill_fractions: tuple[float, ...] = DEFAULT_RANK_FILL_FRACTIONS,
    write_plots: bool = True,
) -> dict[str, Any]:
    """Write detailed LeWM training/eval diagnostics under `output_dir`.

    The returned dictionary is a compact summary that is safe to copy into
    `metrics.json`. Full examples are written as JSONL/CSV files.
    """

    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    model.eval()

    summary: dict[str, Any] = {
        "diagnostics_dir": str(diagnostics_dir),
        "diagnostic_projection_horizons": list(projection_horizons),
    }
    summary.update(
        latent_geometry_diagnostics(
            model,
            examples[:latent_examples],
            diagnostics_dir,
            device=device,
            rng=rng,
            write_plots=write_plots,
        )
    )
    summary.update(
        trajectory_value_diagnostics(
            model,
            examples[:trajectory_examples],
            diagnostics_dir,
            device=device,
        )
    )
    summary.update(
        horizon_action_rank_diagnostics(
            model,
            examples[:rank_examples],
            diagnostics_dir,
            device=device,
            rng=rng,
            horizons=projection_horizons,
            fill_fractions=rank_fill_fractions,
        )
    )
    summary.update(
        projection_panel_diagnostics(
            model,
            examples[:panel_examples],
            diagnostics_dir,
            device=device,
            horizons=projection_horizons,
            panel_steps=panel_steps,
            panel_actions=panel_actions,
        )
    )
    summary.update(
        train_eval_goal_distance_diagnostics(
            model,
            examples[: min(trajectory_examples, 32)],
            diagnostics_dir,
            device=device,
        )
    )
    summary.update(
        predictor_bn_delta_diagnostics(
            model,
            diagnostics_dir,
            device=device,
            seed=seed,
        )
    )
    summary.update(
        history_rank_divergence_diagnostics(
            model,
            examples[:rank_examples],
            diagnostics_dir,
            device=device,
            rng=rng,
            horizons=projection_horizons,
            fill_fractions=rank_fill_fractions,
        )
    )
    summary.update(
        branch_prune_survival_diagnostics(
            model,
            examples[:rank_examples],
            diagnostics_dir,
            device=device,
            fill_fractions=rank_fill_fractions,
        )
    )
    summary.update(
        latent_rollout_symbolic_error_diagnostics(
            model,
            examples[:rank_examples],
            diagnostics_dir,
            device=device,
            horizons=projection_horizons,
            fill_fractions=rank_fill_fractions,
        )
    )
    (diagnostics_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n")
    return summary


@torch.no_grad()
def latent_geometry_diagnostics(
    model: LeWMSudokuModel,
    examples: list[PuzzleExample],
    output_dir: Path,
    *,
    device: torch.device,
    rng: np.random.Generator,
    write_plots: bool,
) -> dict[str, Any]:
    samples = _latent_sample_boards(examples, rng)
    boards = torch.as_tensor(np.stack([sample["board"] for sample in samples]), dtype=torch.long, device=device)
    embeddings = _encode_in_batches(model, boards).cpu().numpy().astype(np.float64)
    projection = _pca_2d(embeddings)
    geometry = _embedding_geometry_summary(embeddings, rng)

    projection_path = output_dir / "latent_projection.csv"
    with projection_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "index",
                "example_index",
                "kind",
                "step",
                "hamming",
                "blank_count",
                "pca_x",
                "pca_y",
            ],
        )
        writer.writeheader()
        for index, (sample, coord) in enumerate(zip(samples, projection, strict=True)):
            writer.writerow(
                {
                    "index": index,
                    "example_index": sample["example_index"],
                    "kind": sample["kind"],
                    "step": sample["step"],
                    "hamming": sample["hamming"],
                    "blank_count": sample["blank_count"],
                    "pca_x": float(coord[0]),
                    "pca_y": float(coord[1]),
                }
            )

    plot_path = None
    optional_manifold = {}
    if write_plots:
        plot_path = output_dir / "latent_projection.svg"
        _write_projection_svg(plot_path, projection, [str(sample["kind"]) for sample in samples])
        optional_manifold = _optional_manifold_projections(embeddings, samples, output_dir, rng)

    geometry_path = output_dir / "latent_geometry.json"
    geometry_payload = {
        **geometry,
        **optional_manifold,
        "sample_count": len(samples),
        "projection_csv": str(projection_path),
        "projection_svg": None if plot_path is None else str(plot_path),
    }
    geometry_path.write_text(json.dumps(_jsonable(geometry_payload), indent=2, sort_keys=True) + "\n")
    return {
        "latent_sample_count": len(samples),
        "latent_mean_abs": geometry["mean_abs"],
        "latent_std_mean": geometry["std_mean"],
        "latent_std_min": geometry["std_min"],
        "latent_cov_offdiag_abs_mean": geometry["cov_offdiag_abs_mean"],
        "latent_effective_rank": geometry["effective_rank"],
        "latent_random_projection_quantile_mae": geometry["random_projection_quantile_mae"],
        "latent_geometry_path": str(geometry_path),
    }


@torch.no_grad()
def trajectory_value_diagnostics(
    model: LeWMSudokuModel,
    examples: list[PuzzleExample],
    output_dir: Path,
    *,
    device: torch.device,
) -> dict[str, Any]:
    path = output_dir / "trajectory_values.jsonl"
    oracle_monotone = []
    predicted_monotone = []
    oracle_drops = []
    predicted_drops = []
    value_targets = []
    value_predictions = []
    with path.open("w") as handle:
        for example_index, example in enumerate(examples):
            boards, actions = _oracle_sequence(example)
            boards_t = torch.as_tensor(np.stack(boards), dtype=torch.long, device=device)
            goal_t = torch.as_tensor(example.goal[None], dtype=torch.long, device=device)
            embeddings = _encode_in_batches(model, boards_t)
            goal_embedding = model.encode_board(goal_t)
            oracle_distances = torch.linalg.vector_norm(embeddings - goal_embedding, dim=-1).cpu().numpy()
            predicted_distances = model.score_value(embeddings).cpu().numpy()
            hamming = [hamming_distance(board, example.goal) for board in boards]
            oracle_monotone.append(_monotone_nonincreasing_fraction(oracle_distances))
            predicted_monotone.append(_monotone_nonincreasing_fraction(predicted_distances))
            oracle_drops.extend((oracle_distances[:-1] - oracle_distances[1:]).tolist())
            predicted_drops.extend((predicted_distances[:-1] - predicted_distances[1:]).tolist())
            value_targets.extend(oracle_distances.tolist())
            value_predictions.extend(predicted_distances.tolist())
            records = []
            for step, board in enumerate(boards):
                next_action = None if step >= len(actions) else action_to_array(actions[step]).tolist()
                records.append(
                    {
                        "step": step,
                        "blank_count": int(np.count_nonzero(board == 0)),
                        "hamming": int(hamming[step]),
                        "oracle_goal_distance": float(oracle_distances[step]),
                        "predicted_goal_distance": float(predicted_distances[step]),
                        "next_oracle_action": next_action,
                    }
                )
            handle.write(json.dumps({"example_index": example_index, "steps": records}, sort_keys=True) + "\n")

    value_targets_np = np.asarray(value_targets, dtype=np.float64)
    value_predictions_np = np.asarray(value_predictions, dtype=np.float64)
    errors = value_predictions_np - value_targets_np
    summary = {
        "trajectory_examples": len(examples),
        "trajectory_oracle_distance_monotone_fraction_mean": _safe_mean(oracle_monotone),
        "trajectory_predicted_distance_monotone_fraction_mean": _safe_mean(predicted_monotone),
        "trajectory_oracle_distance_drop_mean": _safe_mean(oracle_drops),
        "trajectory_predicted_distance_drop_mean": _safe_mean(predicted_drops),
        "trajectory_value_mae": float(np.abs(errors).mean()) if errors.size else 0.0,
        "trajectory_value_rmse": float(np.sqrt(np.square(errors).mean())) if errors.size else 0.0,
        "trajectory_value_corr": _corr(value_targets_np, value_predictions_np),
        "trajectory_values_path": str(path),
    }
    (output_dir / "trajectory_value_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n"
    )
    return summary


@torch.no_grad()
def horizon_action_rank_diagnostics(
    model: LeWMSudokuModel,
    examples: list[PuzzleExample],
    output_dir: Path,
    *,
    device: torch.device,
    rng: np.random.Generator,
    horizons: tuple[int, ...],
    fill_fractions: tuple[float, ...],
) -> dict[str, Any]:
    summary_accumulator: dict[tuple[float, int, str], dict[str, list[float]]] = {}
    examples_path = output_dir / "action_rank_examples.jsonl"
    with examples_path.open("w") as handle:
        for example_index, example in enumerate(examples):
            oracle_boards, _ = _oracle_sequence(example)
            blank_count = max(1, len(oracle_boards) - 1)
            for fraction in fill_fractions:
                step = min(blank_count, int(round(float(fraction) * blank_count)))
                board = oracle_boards[step]
                actions = legal_fill_actions(board, allow_conflicts=True)
                if not actions:
                    continue
                gold_indices = {
                    index
                    for index, action in enumerate(actions)
                    if action.value == int(example.goal[action.row, action.col])
                }
                if not gold_indices:
                    continue
                panel_record: dict[str, Any] | None = None
                for horizon in horizons:
                    costs = _score_first_actions_after_oracle_completion(
                        model,
                        board,
                        example.goal,
                        actions,
                        horizon=horizon,
                        device=device,
                    )
                    for scorer, scorer_costs in costs.items():
                        key = (float(fraction), int(horizon), scorer)
                        bucket = summary_accumulator.setdefault(
                            key,
                            {"best_gold_rank": [], "top_is_gold": [], "pairwise_gold_beats_wrong": []},
                        )
                        bucket["best_gold_rank"].append(float(_best_positive_rank(scorer_costs, gold_indices)))
                        order = np.argsort(np.asarray(scorer_costs, dtype=np.float64))
                        bucket["top_is_gold"].append(float(int(int(order[0]) in gold_indices)))
                        bucket["pairwise_gold_beats_wrong"].append(
                            _pairwise_cost_accuracy(scorer_costs, gold_indices)
                        )
                    if panel_record is None:
                        panel_record = {
                            "example_index": example_index,
                            "fill_fraction": float(fraction),
                            "step": step,
                            "blank_count": int(np.count_nonzero(board == 0)),
                            "horizon": int(horizon),
                            "top_actions": {
                                scorer: _top_action_records(actions, scorer_costs, example.goal, limit=8)
                                for scorer, scorer_costs in costs.items()
                            },
                        }
                if panel_record is not None:
                    handle.write(json.dumps(panel_record, sort_keys=True) + "\n")

    rows = []
    for (fraction, horizon, scorer), values in sorted(summary_accumulator.items()):
        rows.append(
            {
                "fill_fraction": fraction,
                "horizon": horizon,
                "scorer": scorer,
                "states": len(values["best_gold_rank"]),
                "best_gold_rank_mean": _safe_mean(values["best_gold_rank"]),
                "top_is_gold_fraction": _safe_mean(values["top_is_gold"]),
                "pairwise_gold_beats_wrong": _safe_mean(values["pairwise_gold_beats_wrong"]),
            }
        )
    summary_path = output_dir / "action_rank_summary.csv"
    with summary_path.open("w", newline="") as handle:
        fieldnames = [
            "fill_fraction",
            "horizon",
            "scorer",
            "states",
            "best_gold_rank_mean",
            "top_is_gold_fraction",
            "pairwise_gold_beats_wrong",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    compact = _compact_rank_summary(rows)
    compact.update({"action_rank_summary_csv": str(summary_path), "action_rank_examples_path": str(examples_path)})
    (output_dir / "action_rank_summary.json").write_text(json.dumps(_jsonable(compact), indent=2, sort_keys=True) + "\n")
    return compact


@torch.no_grad()
def projection_panel_diagnostics(
    model: LeWMSudokuModel,
    examples: list[PuzzleExample],
    output_dir: Path,
    *,
    device: torch.device,
    horizons: tuple[int, ...],
    panel_steps: int,
    panel_actions: int,
) -> dict[str, Any]:
    path = output_dir / "projection_panel_examples.jsonl"
    records_written = 0
    with path.open("w") as handle:
        for example_index, example in enumerate(examples):
            boards, oracle_actions = _oracle_sequence(example)
            if len(boards) < 2:
                continue
            step_indices = _even_indices(len(boards) - 1, max_count=panel_steps)
            for step in step_indices:
                board = boards[step]
                history_boards = boards[: step + 1]
                history_actions = oracle_actions[:step]
                candidates = _candidate_action_panel(board, example.goal, limit=panel_actions)
                if not candidates:
                    continue
                candidate_records = []
                for label, action in candidates:
                    by_horizon = {}
                    for horizon in horizons:
                        sequence = [action]
                        try:
                            first_leaf = apply_fill_action(board, action, allow_conflicts=True)
                        except ValueError:
                            continue
                        sequence.extend(_oracle_completion_actions(first_leaf, example.goal, max(0, horizon - 1)))
                        scores = {}
                        for transition in ("symbolic_reencode", "latent_rollout"):
                            for score_mode in ("oracle_goal_distance", "predicted_goal_distance"):
                                key = f"{transition}:{score_mode}"
                                try:
                                    scores[key] = score_action_sequence(
                                        model,
                                        board,
                                        example.goal,
                                        sequence,
                                        transition_mode=transition,  # type: ignore[arg-type]
                                        score_mode=score_mode,  # type: ignore[arg-type]
                                        device=device,
                                        position_offset=step,
                                        history_boards=history_boards,
                                        history_actions=history_actions,
                                    ).cost
                                except ValueError as exc:
                                    if "exceed max_history" not in str(exc):
                                        raise
                                    scores[key] = None
                        hamming_score = score_action_sequence(
                            None,
                            board,
                            example.goal,
                            sequence,
                            transition_mode="symbolic_reencode",
                            score_mode="true_hamming_oracle",
                            device=device,
                        )
                        scores["true_hamming_oracle"] = hamming_score.cost
                        by_horizon[str(horizon)] = {
                            "scores": scores,
                            "remaining_hamming": int(hamming_score.cost),
                            "terminal": bool(hamming_score.terminal),
                            "sequence": [action_to_array(item).tolist() for item in sequence],
                        }
                    candidate_records.append(
                        {
                            "label": label,
                            "action": action_to_array(action).tolist(),
                            "writes_goal_digit": bool(action.value == int(example.goal[action.row, action.col])),
                            "horizons": by_horizon,
                        }
                    )
                handle.write(
                    json.dumps(
                        {
                            "example_index": example_index,
                            "oracle_step": int(step),
                            "blank_count": int(np.count_nonzero(board == 0)),
                            "hamming": hamming_distance(board, example.goal),
                            "candidates": candidate_records,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                records_written += 1
    return {"projection_panel_examples": records_written, "projection_panel_examples_path": str(path)}


@torch.no_grad()
def train_eval_goal_distance_diagnostics(
    model: LeWMSudokuModel,
    examples: list[PuzzleExample],
    output_dir: Path,
    *,
    device: torch.device,
) -> dict[str, Any]:
    path = output_dir / "train_eval_goal_distance.json"
    if not examples:
        payload = {"examples": [], "status": "skipped: no examples"}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return {"train_eval_goal_distance_path": str(path), "train_eval_goal_distance_examples": 0}

    used_examples = examples if len(examples) >= 2 else [examples[0], examples[0]]
    boards = torch.as_tensor(
        np.stack([[example.state, example.goal] for example in used_examples]),
        dtype=torch.long,
        device=device,
    )
    actions = torch.zeros((len(used_examples), 2, 3), dtype=torch.long, device=device)
    goals = torch.as_tensor(np.stack([example.goal for example in used_examples]), dtype=torch.long, device=device)
    previous_mode = model.training
    try:
        with _preserved_batchnorm_buffers(model):
            model.train()
            train_distances = model(boards, actions, goals).goal_distances[:, -1].detach().cpu().numpy()
        model.eval()
        eval_distances = model(boards, actions, goals).goal_distances[:, -1].detach().cpu().numpy()
    finally:
        model.train(previous_mode)

    rows = []
    for index, example in enumerate(used_examples):
        rows.append(
            {
                "example_index": int(index),
                "duplicated_single_example": bool(len(examples) == 1),
                "start_hamming": hamming_distance(example.state, example.goal),
                "train_mode_terminal_goal_distance": float(train_distances[index]),
                "eval_mode_terminal_goal_distance": float(eval_distances[index]),
                "absolute_delta": float(abs(train_distances[index] - eval_distances[index])),
            }
        )
    payload = {"examples": rows}
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")
    deltas = np.abs(train_distances - eval_distances)
    return {
        "train_eval_goal_distance_path": str(path),
        "train_eval_goal_distance_examples": int(len(used_examples)),
        "train_mode_terminal_goal_distance_mean": float(np.mean(train_distances)),
        "train_mode_terminal_goal_distance_max": float(np.max(train_distances)),
        "eval_mode_terminal_goal_distance_mean": float(np.mean(eval_distances)),
        "eval_mode_terminal_goal_distance_max": float(np.max(eval_distances)),
        "train_eval_terminal_goal_distance_delta_mean": float(np.mean(deltas)),
        "train_eval_terminal_goal_distance_delta_max": float(np.max(deltas)),
    }


@torch.no_grad()
def predictor_bn_delta_diagnostics(
    model: LeWMSudokuModel,
    output_dir: Path,
    *,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    path = output_dir / "predictor_bn_delta.json"
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) + 17)
    batch, time = 4, min(5, max(3, model.predictor.pos_embedding.shape[1]))
    embeddings = torch.randn(batch, time, model.latent_dim, generator=generator, device=device)
    actions = torch.zeros(batch, time, 3, dtype=torch.long, device=device)
    actions[..., 0] = torch.randint(0, 9, (batch, time), generator=generator, device=device)
    actions[..., 1] = torch.randint(0, 9, (batch, time), generator=generator, device=device)
    actions[..., 2] = torch.randint(1, 10, (batch, time), generator=generator, device=device)
    mask = torch.ones(batch, time, dtype=torch.bool, device=device)
    previous_mode = model.training
    try:
        with _preserved_batchnorm_buffers(model), _temporary_zero_dropout(model):
            model.train()
            full = model.predict_sequence(embeddings, actions, mask=mask)[:, :-1]
            truncated = model.predict_sequence(embeddings[:, :-1], actions[:, :-1], mask=mask[:, :-1])
    finally:
        model.train(previous_mode)
    delta = (full - truncated).abs().detach().cpu().numpy()
    payload = {
        "batch": batch,
        "time": time,
        "max_abs_delta": float(delta.max()) if delta.size else 0.0,
        "mean_abs_delta": float(delta.mean()) if delta.size else 0.0,
        "note": "Compares supervised prefix predictions from full vs truncated predictor calls in train mode with dropout disabled.",
    }
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")
    return {
        "predictor_bn_delta_path": str(path),
        "predictor_bn_max_abs_delta": float(payload["max_abs_delta"]),
        "predictor_bn_mean_abs_delta": float(payload["mean_abs_delta"]),
    }


@torch.no_grad()
def history_rank_divergence_diagnostics(
    model: LeWMSudokuModel,
    examples: list[PuzzleExample],
    output_dir: Path,
    *,
    device: torch.device,
    rng: np.random.Generator,
    horizons: tuple[int, ...],
    fill_fractions: tuple[float, ...],
    max_actions: int = 64,
) -> dict[str, Any]:
    rows = []
    for example_index, example in enumerate(examples):
        boards, oracle_actions = _oracle_sequence(example)
        blank_count = max(1, len(boards) - 1)
        for fraction in fill_fractions:
            step = min(blank_count - 1, int(round(float(fraction) * blank_count)))
            board = boards[step]
            actions = legal_fill_actions(board, allow_conflicts=True)
            if not actions:
                continue
            gold_indices = {
                index
                for index, action in enumerate(actions)
                if action.value == int(example.goal[action.row, action.col])
            }
            selected_actions, selected_gold_indices = _sample_actions_with_gold(actions, gold_indices, rng, max_actions)
            if not selected_actions:
                continue
            history_boards = boards[: step + 1]
            history_actions = oracle_actions[:step]
            for horizon in horizons:
                no_history_costs = []
                full_history_costs = []
                for action in selected_actions:
                    sequence = _action_plus_oracle_completion(board, example.goal, action, int(horizon))
                    try:
                        no_history = score_action_sequence(
                            model,
                            board,
                            example.goal,
                            sequence,
                            transition_mode="latent_rollout",
                            score_mode="oracle_goal_distance",
                            device=device,
                            position_offset=step,
                        ).cost
                        full_history = score_action_sequence(
                            model,
                            board,
                            example.goal,
                            sequence,
                            transition_mode="latent_rollout",
                            score_mode="oracle_goal_distance",
                            device=device,
                            history_boards=history_boards,
                            history_actions=history_actions,
                        ).cost
                    except ValueError as exc:
                        if "exceed max_history" in str(exc):
                            continue
                        raise
                    no_history_costs.append(no_history)
                    full_history_costs.append(full_history)
                if not no_history_costs:
                    continue
                no_history_arr = np.asarray(no_history_costs, dtype=np.float64)
                full_history_arr = np.asarray(full_history_costs, dtype=np.float64)
                rows.append(
                    {
                        "example_index": example_index,
                        "fill_fraction": float(fraction),
                        "step": int(step),
                        "horizon": int(horizon),
                        "actions": int(len(no_history_costs)),
                        "gold_actions": int(len(selected_gold_indices)),
                        "spearman_rank_corr": _spearman_from_costs(no_history_arr, full_history_arr),
                        "top1_agreement": int(int(no_history_arr.argmin()) == int(full_history_arr.argmin())),
                        "no_history_best_gold_rank": _best_positive_rank(no_history_costs, selected_gold_indices),
                        "full_history_best_gold_rank": _best_positive_rank(full_history_costs, selected_gold_indices),
                        "mean_abs_cost_delta": float(np.mean(np.abs(no_history_arr - full_history_arr))),
                    }
                )

    path = output_dir / "history_rank_divergence.csv"
    _write_dict_rows(path, rows)
    spearman = [float(row["spearman_rank_corr"]) for row in rows]
    top1 = [float(row["top1_agreement"]) for row in rows]
    rank_delta = [
        abs(float(row["no_history_best_gold_rank"]) - float(row["full_history_best_gold_rank"]))
        for row in rows
    ]
    summary = {
        "history_rank_divergence_path": str(path),
        "history_rank_divergence_rows": len(rows),
        "history_rank_spearman_mean": _safe_mean(spearman),
        "history_rank_top1_agreement_mean": _safe_mean(top1),
        "history_rank_best_gold_rank_delta_mean": _safe_mean(rank_delta),
    }
    (output_dir / "history_rank_divergence.json").write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n"
    )
    return summary


@torch.no_grad()
def branch_prune_survival_diagnostics(
    model: LeWMSudokuModel,
    examples: list[PuzzleExample],
    output_dir: Path,
    *,
    device: torch.device,
    fill_fractions: tuple[float, ...],
    branch_sizes: tuple[int, ...] = (4, 8, 16, 32),
    max_examples: int = 4,
) -> dict[str, Any]:
    rows = []
    for example_index, example in enumerate(examples[:max_examples]):
        boards, oracle_actions = _oracle_sequence(example)
        blank_count = max(1, len(boards) - 1)
        for fraction in fill_fractions:
            step = min(blank_count - 1, int(round(float(fraction) * blank_count)))
            board = boards[step]
            actions = legal_fill_actions(board, allow_conflicts=True)
            if not actions:
                continue
            gold_set = {
                (action.row, action.col, action.value)
                for action in actions
                if action.value == int(example.goal[action.row, action.col])
            }
            if not gold_set:
                continue
            history_boards = boards[: step + 1]
            history_actions = oracle_actions[:step]
            for score_mode in ("oracle_goal_distance", "predicted_goal_distance"):
                for branch_size in branch_sizes:
                    try:
                        selected = _rank_immediate_actions(
                            model,
                            board,
                            example.goal,
                            actions,
                            branch_size,
                            transition_mode="latent_rollout",
                            score_mode=score_mode,
                            position_offset=step,
                            history_boards=history_boards,
                            history_actions=history_actions,
                            device=device,
                        )
                    except ValueError as exc:
                        if "exceed max_history" in str(exc):
                            continue
                        raise
                    selected_set = {(action.row, action.col, action.value) for action in selected}
                    gold_survived = len(selected_set & gold_set)
                    rows.append(
                        {
                            "example_index": example_index,
                            "fill_fraction": float(fraction),
                            "step": int(step),
                            "score_mode": score_mode,
                            "branch_size": int(branch_size),
                            "total_actions": int(len(actions)),
                            "gold_actions": int(len(gold_set)),
                            "selected_actions": int(len(selected)),
                            "gold_survived": int(gold_survived),
                            "gold_survival": float(gold_survived > 0),
                            "gold_selected_fraction": float(gold_survived / max(1, len(gold_set))),
                        }
                    )

    path = output_dir / "branch_prune_survival.csv"
    _write_dict_rows(path, rows)
    summary: dict[str, Any] = {
        "branch_prune_survival_path": str(path),
        "branch_prune_survival_rows": len(rows),
        "branch_prune_survival_examples": min(len(examples), max_examples),
    }
    for score_mode in ("oracle_goal_distance", "predicted_goal_distance"):
        for branch_size in branch_sizes:
            values = [
                float(row["gold_survival"])
                for row in rows
                if row["score_mode"] == score_mode and int(row["branch_size"]) == int(branch_size)
            ]
            summary[f"branch_prune_{score_mode}_k{branch_size}_gold_survival"] = _safe_mean(values)
    (output_dir / "branch_prune_survival.json").write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n")
    return summary


@torch.no_grad()
def latent_rollout_symbolic_error_diagnostics(
    model: LeWMSudokuModel,
    examples: list[PuzzleExample],
    output_dir: Path,
    *,
    device: torch.device,
    horizons: tuple[int, ...],
    fill_fractions: tuple[float, ...],
) -> dict[str, Any]:
    rows = []
    for example_index, example in enumerate(examples):
        boards, oracle_actions = _oracle_sequence(example)
        blank_count = max(1, len(boards) - 1)
        goal_t = torch.as_tensor(example.goal[None], dtype=torch.long, device=device)
        goal_emb = model.encode_board(goal_t)
        for fraction in fill_fractions:
            step = min(blank_count - 1, int(round(float(fraction) * blank_count)))
            board = boards[step]
            history_boards = boards[: step + 1]
            history_actions = oracle_actions[:step]
            available = blank_count - step
            for horizon in horizons:
                if available <= 0:
                    continue
                sequence = oracle_actions[step : step + min(int(horizon), available)]
                if not sequence:
                    continue
                symbolic_board, valid = _apply_sequence_or_skip(board, sequence)
                if not valid:
                    continue
                symbolic_t = torch.as_tensor(symbolic_board[None], dtype=torch.long, device=device)
                symbolic_emb = model.encode_board(symbolic_t)
                try:
                    latent_emb = _latent_rollout_leaf_embedding(
                        model,
                        board,
                        sequence,
                        device=device,
                        position_offset=step,
                        history_boards=history_boards,
                        history_actions=history_actions,
                    )
                except ValueError as exc:
                    if "exceed max_history" in str(exc):
                        continue
                    raise
                diff = (latent_emb - symbolic_emb).float()
                latent_goal = torch.linalg.vector_norm(latent_emb - goal_emb, dim=-1).item()
                symbolic_goal = torch.linalg.vector_norm(symbolic_emb - goal_emb, dim=-1).item()
                rows.append(
                    {
                        "example_index": int(example_index),
                        "fill_fraction": float(fraction),
                        "step": int(step),
                        "horizon": int(horizon),
                        "actual_actions": int(len(sequence)),
                        "mse": float(diff.square().mean().item()),
                        "l2": float(torch.linalg.vector_norm(diff, dim=-1).item()),
                        "latent_goal_distance": float(latent_goal),
                        "symbolic_goal_distance": float(symbolic_goal),
                        "goal_distance_abs_error": float(abs(latent_goal - symbolic_goal)),
                        "remaining_hamming": hamming_distance(symbolic_board, example.goal),
                    }
                )

    path = output_dir / "latent_rollout_symbolic_error.csv"
    _write_dict_rows(path, rows)
    summary: dict[str, Any] = {
        "latent_rollout_symbolic_error_path": str(path),
        "latent_rollout_symbolic_rows": len(rows),
    }
    for horizon in horizons:
        horizon_rows = [row for row in rows if int(row["horizon"]) == int(horizon)]
        summary[f"latent_rollout_symbolic_mse_h{horizon}_mean"] = _safe_mean(
            [float(row["mse"]) for row in horizon_rows]
        )
        summary[f"latent_rollout_symbolic_l2_h{horizon}_mean"] = _safe_mean(
            [float(row["l2"]) for row in horizon_rows]
        )
        summary[f"latent_rollout_symbolic_goal_error_h{horizon}_mean"] = _safe_mean(
            [float(row["goal_distance_abs_error"]) for row in horizon_rows]
        )
    (output_dir / "latent_rollout_symbolic_error.json").write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n"
    )
    return summary


def _latent_sample_boards(examples: list[PuzzleExample], rng: np.random.Generator) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for example_index, example in enumerate(examples):
        boards, _ = _oracle_sequence(example)
        for step in _even_indices(len(boards), max_count=8):
            board = boards[step]
            samples.append(
                {
                    "example_index": example_index,
                    "kind": "oracle_path",
                    "step": step,
                    "hamming": hamming_distance(board, example.goal),
                    "blank_count": int(np.count_nonzero(board == 0)),
                    "board": board,
                }
            )
        wrong = _random_wrong_terminal(example, rng)
        samples.append(
            {
                "example_index": example_index,
                "kind": "wrong_terminal",
                "step": len(boards) - 1,
                "hamming": hamming_distance(wrong, example.goal),
                "blank_count": int(np.count_nonzero(wrong == 0)),
                "board": wrong,
            }
        )
    return samples


def _oracle_sequence(example: PuzzleExample) -> tuple[list[np.ndarray], list[WorldAction]]:
    board = example.state.copy()
    boards = [board.copy()]
    actions: list[WorldAction] = []
    for row, col in np.argwhere(board == 0):
        action = WorldAction(int(row), int(col), int(example.goal[int(row), int(col)]))
        board = apply_fill_action(board, action, allow_conflicts=True)
        actions.append(action)
        boards.append(board.copy())
    return boards, actions


def _oracle_completion_actions(board: np.ndarray, goal: np.ndarray, count: int) -> list[WorldAction]:
    actions = []
    current = board.copy()
    for row, col in np.argwhere(current == 0):
        if len(actions) >= count:
            break
        action = WorldAction(int(row), int(col), int(goal[int(row), int(col)]))
        current = apply_fill_action(current, action, allow_conflicts=True)
        actions.append(action)
    return actions


def _score_first_actions_after_oracle_completion(
    model: LeWMSudokuModel,
    board: np.ndarray,
    goal: np.ndarray,
    actions: list[WorldAction],
    *,
    horizon: int,
    device: torch.device,
) -> dict[str, list[float]]:
    leaves = []
    hamming_costs = []
    for action in actions:
        leaf = apply_fill_action(board, action, allow_conflicts=True)
        for followup in _oracle_completion_actions(leaf, goal, max(0, horizon - 1)):
            leaf = apply_fill_action(leaf, followup, allow_conflicts=True)
        leaves.append(leaf)
        hamming_costs.append(float(hamming_distance(leaf, goal)))
    leaves_t = torch.as_tensor(np.stack(leaves), dtype=torch.long, device=device)
    goal_t = torch.as_tensor(goal[None], dtype=torch.long, device=device)
    leaf_embeddings = _encode_in_batches(model, leaves_t)
    goal_embedding = model.encode_board(goal_t)
    oracle_costs = torch.linalg.vector_norm(leaf_embeddings - goal_embedding, dim=-1).cpu().numpy().tolist()
    predicted_costs = model.score_value(leaf_embeddings).cpu().numpy().tolist()
    return {
        "true_hamming_oracle": hamming_costs,
        "oracle_goal_distance": [float(item) for item in oracle_costs],
        "predicted_goal_distance": [float(item) for item in predicted_costs],
    }


def _action_plus_oracle_completion(
    board: np.ndarray,
    goal: np.ndarray,
    action: WorldAction,
    horizon: int,
) -> list[WorldAction]:
    sequence = [action]
    try:
        leaf = apply_fill_action(board, action, allow_conflicts=True)
    except ValueError:
        return sequence
    sequence.extend(_oracle_completion_actions(leaf, goal, max(0, int(horizon) - 1)))
    return sequence


def _sample_actions_with_gold(
    actions: list[WorldAction],
    gold_indices: set[int],
    rng: np.random.Generator,
    max_actions: int,
) -> tuple[list[WorldAction], set[int]]:
    if len(actions) <= max_actions:
        return actions, set(gold_indices)
    chosen = set(gold_indices)
    remaining_slots = max(0, int(max_actions) - len(chosen))
    non_gold = [index for index in range(len(actions)) if index not in chosen]
    if remaining_slots > 0 and non_gold:
        selected = rng.choice(non_gold, size=min(remaining_slots, len(non_gold)), replace=False)
        chosen.update(int(index) for index in selected)
    ordered = sorted(chosen)
    remap = {old_index: new_index for new_index, old_index in enumerate(ordered)}
    return [actions[index] for index in ordered], {remap[index] for index in gold_indices if index in remap}


def _apply_sequence_or_skip(board: np.ndarray, actions: list[WorldAction]) -> tuple[np.ndarray, bool]:
    current = board.copy()
    for action in actions:
        try:
            current = apply_fill_action(current, action, allow_conflicts=True)
        except ValueError:
            return current, False
    return current, True


def _latent_rollout_leaf_embedding(
    model: LeWMSudokuModel,
    board: np.ndarray,
    actions: list[WorldAction],
    *,
    device: torch.device,
    position_offset: int = 0,
    history_boards: list[np.ndarray] | None = None,
    history_actions: list[WorldAction] | None = None,
) -> torch.Tensor:
    action_t = torch.as_tensor(
        np.asarray([[action_to_array(action) for action in actions]], dtype=np.int64),
        dtype=torch.long,
        device=device,
    )
    if history_boards is not None:
        if history_actions is None:
            history_actions = []
        if len(history_boards) != len(history_actions) + 1:
            raise ValueError("history_boards must contain exactly one more item than history_actions.")
        history_t = torch.as_tensor(np.stack(history_boards), dtype=torch.long, device=device)[None]
        prefix_emb = model.encode_sequence(history_t)
        prefix_actions_t = (
            torch.as_tensor(
                np.asarray([[action_to_array(action) for action in history_actions]], dtype=np.int64),
                dtype=torch.long,
                device=device,
            )
            if history_actions
            else torch.zeros((1, 0, 3), dtype=torch.long, device=device)
        )
        return model.rollout_latent(
            prefix_emb[:, -1],
            action_t,
            prefix_embeddings=prefix_emb,
            prefix_actions=prefix_actions_t,
        )[:, -1]
    board_t = torch.as_tensor(board[None], dtype=torch.long, device=device)
    start_emb = model.encode_board(board_t)
    return model.rollout_latent(start_emb, action_t, position_offset=position_offset)[:, -1]


def _candidate_action_panel(board: np.ndarray, goal: np.ndarray, *, limit: int) -> list[tuple[str, WorldAction]]:
    empties = [tuple(int(x) for x in item) for item in np.argwhere(board == 0)]
    if not empties:
        return []
    anchor = empties[0]
    candidates: list[tuple[str, WorldAction]] = []

    def add(label: str, row: int, col: int, value: int) -> None:
        action = WorldAction(row, col, value)
        if board[row, col] != 0:
            return
        key = (action.row, action.col, action.value)
        if key not in {(item.row, item.col, item.value) for _, item in candidates}:
            candidates.append((label, action))

    row, col = anchor
    gold_value = int(goal[row, col])
    add("anchor_gold", row, col, gold_value)
    add("anchor_wrong_digit", row, col, _wrong_digit(gold_value))

    other = _nearest_other_empty(anchor, empties)
    if other is not None:
        other_row, other_col = other
        other_gold = int(goal[other_row, other_col])
        add("other_cell_gold", other_row, other_col, other_gold)
        add("other_cell_wrong_digit", other_row, other_col, _wrong_digit(other_gold))

    far = _farthest_other_empty(anchor, empties)
    if far is not None:
        far_row, far_col = far
        far_gold = int(goal[far_row, far_col])
        add("far_cell_gold", far_row, far_col, far_gold)
        add("far_cell_wrong_digit", far_row, far_col, _wrong_digit(far_gold))
    return candidates[:limit]


def _random_wrong_terminal(example: PuzzleExample, rng: np.random.Generator) -> np.ndarray:
    board = example.state.copy()
    for row, col in np.argwhere(board == 0):
        row_i, col_i = int(row), int(col)
        gold = int(example.goal[row_i, col_i])
        value = int(rng.integers(1, 10))
        if value == gold:
            value = _wrong_digit(gold)
        board[row_i, col_i] = value
    return board


def _encode_in_batches(model: LeWMSudokuModel, boards: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
    chunks = []
    for start in range(0, boards.shape[0], batch_size):
        chunks.append(model.encode_board(boards[start : start + batch_size]))
    return torch.cat(chunks, dim=0)


def _embedding_geometry_summary(embeddings: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be [samples, dim].")
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    std = embeddings.std(axis=0)
    cov = centered.T @ centered / max(1, embeddings.shape[0] - 1)
    eigvals = np.linalg.eigvalsh(cov).clip(min=0.0)
    eigsum = float(eigvals.sum())
    eigprob = eigvals / eigsum if eigsum > 0 else np.full_like(eigvals, 1.0 / max(1, eigvals.size))
    effective_rank = float(np.exp(-(eigprob * np.log(eigprob + 1.0e-12)).sum()))
    offdiag = cov - np.diag(np.diag(cov))
    projection_stats = _random_projection_normality(centered, rng)
    return {
        "mean_abs": float(np.abs(embeddings.mean(axis=0)).mean()),
        "std_mean": float(std.mean()),
        "std_min": float(std.min()),
        "std_max": float(std.max()),
        "norm_mean": float(np.linalg.norm(embeddings, axis=1).mean()),
        "cov_diag_mean": float(np.diag(cov).mean()),
        "cov_offdiag_abs_mean": float(np.abs(offdiag).mean()),
        "effective_rank": effective_rank,
        **projection_stats,
    }


def _random_projection_normality(centered: np.ndarray, rng: np.random.Generator, projections: int = 128) -> dict[str, float]:
    if centered.shape[0] < 2:
        return {
            "random_projection_skew_abs_mean": 0.0,
            "random_projection_excess_kurtosis_abs_mean": 0.0,
            "random_projection_quantile_mae": 0.0,
        }
    proj = rng.normal(size=(centered.shape[1], projections))
    proj /= np.linalg.norm(proj, axis=0, keepdims=True).clip(min=1.0e-12)
    values = centered @ proj
    values = (values - values.mean(axis=0, keepdims=True)) / values.std(axis=0, keepdims=True).clip(min=1.0e-12)
    skew = np.mean(values**3, axis=0)
    kurtosis = np.mean(values**4, axis=0) - 3.0
    probs = np.asarray([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    empirical = np.quantile(values, probs, axis=0)
    normal = np.asarray([NormalDist().inv_cdf(float(prob)) for prob in probs])[:, None]
    return {
        "random_projection_skew_abs_mean": float(np.abs(skew).mean()),
        "random_projection_excess_kurtosis_abs_mean": float(np.abs(kurtosis).mean()),
        "random_projection_quantile_mae": float(np.abs(empirical - normal).mean()),
    }


def _pca_2d(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    if embeddings.shape[0] == 1:
        return np.zeros((1, 2), dtype=np.float64)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vh[:2].T
    if coords.shape[1] == 1:
        coords = np.concatenate([coords, np.zeros_like(coords)], axis=1)
    return coords[:, :2]


def _write_projection_svg(path: Path, coords: np.ndarray, labels: list[str]) -> None:
    if coords.size == 0:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"640\" height=\"480\" />\n")
        return
    colors = {
        "oracle_path": "#2563eb",
        "wrong_terminal": "#dc2626",
    }
    x = coords[:, 0]
    y = coords[:, 1]
    width, height, pad = 640, 480, 32
    x_span = float(x.max() - x.min()) or 1.0
    y_span = float(y.max() - y.min()) or 1.0
    sx = pad + (x - x.min()) / x_span * (width - 2 * pad)
    sy = height - pad - (y - y.min()) / y_span * (height - 2 * pad)
    circles = "\n".join(
        f'<circle cx="{float(cx):.2f}" cy="{float(cy):.2f}" r="2.5" fill="{colors.get(label, "#111827")}" fill-opacity="0.75" />'
        for cx, cy, label in zip(sx, sy, labels, strict=True)
    )
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
        f'<rect width="{width}" height="{height}" fill="white" />\n{circles}\n</svg>\n'
    )


def _optional_manifold_projections(
    embeddings: np.ndarray,
    samples: list[dict[str, Any]],
    output_dir: Path,
    rng: np.random.Generator,
    *,
    max_points: int = 512,
) -> dict[str, Any]:
    if embeddings.shape[0] < 4:
        return {"latent_tsne_status": "skipped: fewer than 4 samples", "latent_umap_status": "skipped: fewer than 4 samples"}
    if embeddings.shape[0] > max_points:
        indices = np.sort(rng.choice(embeddings.shape[0], size=max_points, replace=False))
    else:
        indices = np.arange(embeddings.shape[0])
    subset = embeddings[indices]
    subset_samples = [samples[int(index)] for index in indices]
    result: dict[str, Any] = {"latent_manifold_sample_count": int(len(indices))}
    try:
        from sklearn.manifold import TSNE  # type: ignore

        perplexity = max(2, min(30, (len(indices) - 1) // 3))
        coords = TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
            random_state=0,
        ).fit_transform(subset)
        path = output_dir / "latent_tsne.csv"
        _write_projection_csv(path, subset_samples, coords, "tsne_x", "tsne_y")
        result["latent_tsne_status"] = "written"
        result["latent_tsne_csv"] = str(path)
    except Exception as exc:  # pragma: no cover - optional dependency path
        result["latent_tsne_status"] = f"skipped: {type(exc).__name__}: {exc}"
    try:
        import umap  # type: ignore

        coords = umap.UMAP(n_components=2, random_state=0).fit_transform(subset)
        path = output_dir / "latent_umap.csv"
        _write_projection_csv(path, subset_samples, coords, "umap_x", "umap_y")
        result["latent_umap_status"] = "written"
        result["latent_umap_csv"] = str(path)
    except Exception as exc:  # pragma: no cover - optional dependency path
        result["latent_umap_status"] = f"skipped: {type(exc).__name__}: {exc}"
    return result


def _write_projection_csv(
    path: Path,
    samples: list[dict[str, Any]],
    coords: np.ndarray,
    x_name: str,
    y_name: str,
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "index",
                "example_index",
                "kind",
                "step",
                "hamming",
                "blank_count",
                x_name,
                y_name,
            ],
        )
        writer.writeheader()
        for index, (sample, coord) in enumerate(zip(samples, coords, strict=True)):
            writer.writerow(
                {
                    "index": index,
                    "example_index": sample["example_index"],
                    "kind": sample["kind"],
                    "step": sample["step"],
                    "hamming": sample["hamming"],
                    "blank_count": sample["blank_count"],
                    x_name: float(coord[0]),
                    y_name: float(coord[1]),
                }
            )


def _write_dict_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        if not fieldnames:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _top_action_records(
    actions: list[WorldAction],
    costs: list[float],
    goal: np.ndarray,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    order = np.argsort(np.asarray(costs, dtype=np.float64))[:limit]
    return [
        {
            "rank": int(rank),
            "action": action_to_array(actions[int(index)]).tolist(),
            "cost": float(costs[int(index)]),
            "writes_goal_digit": bool(actions[int(index)].value == int(goal[actions[int(index)].row, actions[int(index)].col])),
        }
        for rank, index in enumerate(order, start=1)
    ]


def _compact_rank_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    compact: dict[str, Any] = {"action_rank_rows": len(rows)}
    for row in rows:
        prefix = (
            f"rank_{row['scorer']}_h{row['horizon']}_f{str(row['fill_fraction']).replace('.', 'p')}"
        )
        compact[f"{prefix}_best_gold_rank_mean"] = float(row["best_gold_rank_mean"])
        compact[f"{prefix}_top_is_gold_fraction"] = float(row["top_is_gold_fraction"])
        compact[f"{prefix}_pairwise_gold_beats_wrong"] = float(row["pairwise_gold_beats_wrong"])
    return compact


def _best_positive_rank(costs: list[float], positive_indices: set[int]) -> int:
    order = np.argsort(np.asarray(costs, dtype=np.float64)).tolist()
    for rank, index in enumerate(order, start=1):
        if index in positive_indices:
            return rank
    return len(costs)


def _pairwise_cost_accuracy(costs: list[float], positive_indices: set[int]) -> float:
    positives = [float(costs[index]) for index in positive_indices]
    negatives = [float(cost) for index, cost in enumerate(costs) if index not in positive_indices]
    if not positives or not negatives:
        return 0.0
    wins = 0
    total = 0
    for pos in positives:
        for neg in negatives:
            wins += int(pos < neg) + 0.5 * int(pos == neg)
            total += 1
    return float(wins / max(1, total))


def _spearman_from_costs(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2 or a.size != b.size:
        return 0.0
    return _corr(_rank_positions(a), _rank_positions(b))


def _rank_positions(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    return ranks


def _monotone_nonincreasing_fraction(values: np.ndarray) -> float:
    if values.size < 2:
        return 1.0
    return float(np.mean(values[1:] <= values[:-1]))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2 or float(a.std()) == 0.0 or float(b.std()) == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _safe_mean(values: list[float] | list[int]) -> float:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(clean)) if clean else 0.0


def _even_indices(length: int, *, max_count: int) -> list[int]:
    if length <= 0 or max_count <= 0:
        return []
    count = min(length, max_count)
    return sorted(set(int(x) for x in np.linspace(0, length - 1, count)))


def _wrong_digit(gold: int) -> int:
    return 1 if int(gold) != 1 else 2


def _nearest_other_empty(anchor: tuple[int, int], empties: list[tuple[int, int]]) -> tuple[int, int] | None:
    others = [item for item in empties if item != anchor]
    if not others:
        return None
    return min(others, key=lambda item: abs(item[0] - anchor[0]) + abs(item[1] - anchor[1]))


def _farthest_other_empty(anchor: tuple[int, int], empties: list[tuple[int, int]]) -> tuple[int, int] | None:
    others = [item for item in empties if item != anchor]
    if not others:
        return None
    return max(others, key=lambda item: abs(item[0] - anchor[0]) + abs(item[1] - anchor[1]))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@contextmanager
def _preserved_batchnorm_buffers(model: LeWMSudokuModel) -> Iterator[None]:
    buffers = {
        name: buffer.detach().clone()
        for name, buffer in model.named_buffers()
        if name.endswith("running_mean") or name.endswith("running_var") or name.endswith("num_batches_tracked")
    }
    try:
        yield
    finally:
        named_buffers = dict(model.named_buffers())
        for name, value in buffers.items():
            named_buffers[name].copy_(value)


@contextmanager
def _temporary_zero_dropout(model: LeWMSudokuModel) -> Iterator[None]:
    modules = [(module, module.p) for module in model.modules() if isinstance(module, nn.Dropout)]
    try:
        for module, _ in modules:
            module.p = 0.0
        yield
    finally:
        for module, probability in modules:
            module.p = probability
