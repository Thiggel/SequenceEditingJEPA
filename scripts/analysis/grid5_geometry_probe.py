from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from puzzle_jepa.data import SudokuWorld, sample_oracle_partial_transition
from puzzle_jepa.eval.grid5_diagnostics import candidate_actions, load_model, score_candidate_states
from puzzle_jepa.models import SigRegActionJEPA
from puzzle_jepa.train.grid0 import _build_world, _load_examples


def safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def safe_median(values: list[float]) -> float:
    return float(np.median(values)) if values else 0.0


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return 0.0
    rx = rankdata(np.asarray(x, dtype=np.float64))
    ry = rankdata(np.asarray(y, dtype=np.float64))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    return float(rx.dot(ry) / denom) if denom > 0.0 else 0.0


@torch.no_grad()
def encode_states(model: SigRegActionJEPA, states: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(states, dtype=torch.long, device=device)
    return model.encode(tensor).detach().float()


def one_cell_terminal_corruptions(
    world: SudokuWorld,
    goal: np.ndarray,
    clue_mask: np.ndarray,
    *,
    max_candidates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    candidates = []
    mutable = np.argwhere(~world.validate_clue_mask(clue_mask))
    rng.shuffle(mutable)
    for row, col in mutable:
        true_value = int(goal[int(row), int(col)])
        wrong_values = [value for value in range(1, 10) if value != true_value]
        rng.shuffle(wrong_values)
        for value in wrong_values:
            board = goal.copy()
            board[int(row), int(col)] = int(value)
            candidates.append(board)
            if len(candidates) >= max_candidates:
                return np.stack(candidates)
    return np.stack(candidates) if candidates else goal[None].copy()


@torch.no_grad()
def terminal_corruption_summary(
    model: SigRegActionJEPA,
    world: SudokuWorld,
    examples,
    rng: np.random.Generator,
    *,
    count: int,
    corruptions_per_board: int,
    device: torch.device,
) -> dict[str, float]:
    latent_margins = []
    latent_corrupt_distance = []
    energy_true_scores = []
    energy_corrupt_scores = []
    energy_true_top1 = []
    energy_true_ranks = []
    for _ in range(max(1, count)):
        example = examples[int(rng.integers(0, len(examples)))]
        clue_mask = world.clue_mask_from_puzzle(example.state)
        corruptions = one_cell_terminal_corruptions(
            world,
            example.goal,
            clue_mask,
            max_candidates=max(1, corruptions_per_board),
            rng=rng,
        )
        boards = np.concatenate([example.goal[None], corruptions], axis=0)
        latents = encode_states(model, boards, device)
        goal_latent = latents[0:1]
        latent_scores = F.mse_loss(latents, goal_latent.expand_as(latents), reduction="none").mean(dim=-1)
        energy_scores = score_candidate_states(
            model,
            boards,
            example.goal,
            example.state,
            score_mode="goal_energy",
            device=device,
        )
        latent_values = latent_scores.cpu().numpy()
        corrupt_values = latent_values[1:]
        latent_corrupt_distance.extend(float(value) for value in corrupt_values)
        latent_margins.append(float(np.min(corrupt_values) - latent_values[0]))
        order = np.argsort(energy_scores)
        true_rank = int(np.where(order == 0)[0][0]) + 1
        energy_true_ranks.append(float(true_rank))
        energy_true_top1.append(float(true_rank == 1))
        energy_true_scores.append(float(energy_scores[0]))
        energy_corrupt_scores.extend(float(value) for value in energy_scores[1:])
    return {
        "count": float(count),
        "corruptions_per_board": float(corruptions_per_board),
        "latent_min_corrupt_margin_mean": safe_mean(latent_margins),
        "latent_min_corrupt_margin_median": safe_median(latent_margins),
        "latent_corrupt_distance_mean": safe_mean(latent_corrupt_distance),
        "latent_corrupt_distance_p10": float(np.quantile(latent_corrupt_distance, 0.1)) if latent_corrupt_distance else 0.0,
        "goal_energy_true_top1_rate": safe_mean(energy_true_top1),
        "goal_energy_true_rank_mean": safe_mean(energy_true_ranks),
        "goal_energy_true_score_mean": safe_mean(energy_true_scores),
        "goal_energy_corrupt_score_mean": safe_mean(energy_corrupt_scores),
    }


def oracle_partial_board(example, world: SudokuWorld, rng: np.random.Generator) -> np.ndarray:
    clue_mask = world.clue_mask_from_puzzle(example.state)
    board = example.state.copy()
    mutable = np.argwhere(~world.validate_clue_mask(clue_mask))
    reveal_count = int(rng.integers(0, len(mutable) + 1))
    if reveal_count > 0:
        chosen = mutable[rng.choice(len(mutable), size=reveal_count, replace=False)]
        for row, col in chosen:
            board[int(row), int(col)] = int(example.goal[int(row), int(col)])
    return board


def random_mutable_board(example, world: SudokuWorld, rng: np.random.Generator) -> np.ndarray:
    clue_mask = world.clue_mask_from_puzzle(example.state)
    board = example.state.copy()
    mutable = np.argwhere(~world.validate_clue_mask(clue_mask))
    fill_count = int(rng.integers(0, len(mutable) + 1))
    if fill_count > 0:
        chosen = mutable[rng.choice(len(mutable), size=fill_count, replace=False)]
        for row, col in chosen:
            board[int(row), int(col)] = int(rng.integers(1, 10))
    return board


@torch.no_grad()
def nearest_neighbor_summary(
    model: SigRegActionJEPA,
    world: SudokuWorld,
    examples,
    rng: np.random.Generator,
    *,
    query_count: int,
    pool_count: int,
    device: torch.device,
) -> dict[str, float]:
    pool_examples = [examples[int(rng.integers(0, len(examples)))] for _ in range(max(2, pool_count))]
    pool_boards = []
    for example in pool_examples:
        pool_boards.append(oracle_partial_board(example, world, rng))
        pool_boards.append(random_mutable_board(example, world, rng))
    pool = np.stack(pool_boards)
    pool_latents = encode_states(model, pool, device)
    nn_hamming = []
    random_hamming = []
    pair_latent = []
    pair_hamming = []
    for _ in range(max(1, query_count)):
        example = examples[int(rng.integers(0, len(examples)))]
        query = oracle_partial_board(example, world, rng)
        query_latent = encode_states(model, query[None], device)
        distances = F.mse_loss(pool_latents, query_latent.expand_as(pool_latents), reduction="none").mean(dim=-1)
        order = torch.argsort(distances).detach().cpu().numpy()
        nearest = int(order[0])
        random_index = int(rng.integers(0, len(pool)))
        nn_hamming.append(float(np.not_equal(query, pool[nearest]).sum()))
        random_hamming.append(float(np.not_equal(query, pool[random_index]).sum()))
        sample_size = min(64, len(pool))
        for index in rng.choice(len(pool), size=sample_size, replace=False):
            pair_latent.append(float(distances[int(index)].cpu()))
            pair_hamming.append(float(np.not_equal(query, pool[int(index)]).sum()))
    return {
        "query_count": float(query_count),
        "pool_count": float(len(pool)),
        "nearest_hamming_mean": safe_mean(nn_hamming),
        "nearest_hamming_median": safe_median(nn_hamming),
        "random_hamming_mean": safe_mean(random_hamming),
        "latent_hamming_spearman": spearman(pair_latent, pair_hamming),
    }


@torch.no_grad()
def action_geometry_summary(
    model: SigRegActionJEPA,
    world: SudokuWorld,
    examples,
    rng: np.random.Generator,
    *,
    count: int,
    device: torch.device,
) -> dict[str, float]:
    gold_goal_cos = []
    best_wrong_goal_cos = []
    gold_delta_norm = []
    wrong_delta_norm = []
    for _ in range(max(1, count)):
        transition = sample_oracle_partial_transition(
            world,
            examples[int(rng.integers(0, len(examples)))],
            rng,
        )
        clue_mask = transition.clue_mask if transition.clue_mask is not None else world.clue_mask_from_puzzle(
            transition.state
        )
        actions = candidate_actions(world, transition.state, clue_mask)
        next_states = np.stack(
            [
                world.apply(
                    transition.state,
                    action,
                    clue_mask=clue_mask,
                    allow_overwrite=True,
                    allow_conflicts=True,
                )
                for action in actions
            ]
        )
        state_latent = encode_states(model, transition.state[None], device)
        goal_latent = encode_states(model, transition.goal[None], device)
        next_latents = encode_states(model, next_states, device)
        deltas = next_latents - state_latent
        goal_direction = goal_latent - state_latent
        cosines = F.cosine_similarity(deltas, goal_direction.expand_as(deltas), dim=-1).detach().cpu().numpy()
        norms = deltas.norm(dim=-1).detach().cpu().numpy()
        gold_index = next(
            i
            for i, action in enumerate(actions)
            if action.row == transition.action.row
            and action.col == transition.action.col
            and action.value == transition.action.value
        )
        wrong_indices = [i for i, action in enumerate(actions) if action.value != int(transition.goal[action.row, action.col])]
        gold_goal_cos.append(float(cosines[gold_index]))
        gold_delta_norm.append(float(norms[gold_index]))
        if wrong_indices:
            best_wrong = max(wrong_indices, key=lambda index: float(cosines[index]))
            best_wrong_goal_cos.append(float(cosines[best_wrong]))
            wrong_delta_norm.append(float(norms[best_wrong]))
    return {
        "count": float(count),
        "gold_goal_cosine_mean": safe_mean(gold_goal_cos),
        "best_wrong_goal_cosine_mean": safe_mean(best_wrong_goal_cos),
        "wrong_beats_gold_cosine_rate": safe_mean(
            [float(wrong > gold) for wrong, gold in zip(best_wrong_goal_cos, gold_goal_cos, strict=False)]
        ),
        "gold_delta_norm_mean": safe_mean(gold_delta_norm),
        "best_wrong_delta_norm_mean": safe_mean(wrong_delta_norm),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Grid 5 Geometry Probe",
        "",
        f"- Run: `{payload['run_root']}`",
        f"- Device: `{payload['device']}`",
        f"- Seed: `{payload['seed']}`",
        "",
        "## Terminal Corruptions",
    ]
    for key, value in payload["terminal_corruptions"].items():
        lines.append(f"- `{key}`: {value:.6g}")
    lines.extend(["", "## Nearest Neighbors"])
    for key, value in payload["nearest_neighbors"].items():
        lines.append(f"- `{key}`: {value:.6g}")
    lines.extend(["", "## Action Geometry"])
    for key, value in payload["action_geometry"].items():
        lines.append(f"- `{key}`: {value:.6g}")
    path.write_text("\n".join(lines) + "\n")


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(int(args.seed))
    model, config = load_model(args.run_root, device)
    world = _build_world(dict(config["task"]))
    if not isinstance(world, SudokuWorld):
        raise ValueError("Grid 5 geometry probe currently supports Sudoku only.")
    examples = _load_examples(dict(config["task"]), "eval")
    payload = {
        "run_root": str(args.run_root),
        "device": str(device),
        "seed": int(args.seed),
        "config": dict(config["model"]),
        "terminal_corruptions": terminal_corruption_summary(
            model,
            world,
            examples,
            rng,
            count=int(args.terminal_examples),
            corruptions_per_board=int(args.corruptions_per_board),
            device=device,
        ),
        "nearest_neighbors": nearest_neighbor_summary(
            model,
            world,
            examples,
            rng,
            query_count=int(args.nearest_queries),
            pool_count=int(args.nearest_pool),
            device=device,
        ),
        "action_geometry": action_geometry_summary(
            model,
            world,
            examples,
            rng,
            count=int(args.action_examples),
            device=device,
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    write_markdown(args.output_dir / "summary.md", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--terminal-examples", type=int, default=16)
    parser.add_argument("--corruptions-per-board", type=int, default=128)
    parser.add_argument("--nearest-queries", type=int, default=32)
    parser.add_argument("--nearest-pool", type=int, default=128)
    parser.add_argument("--action-examples", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(run_probe(args), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
