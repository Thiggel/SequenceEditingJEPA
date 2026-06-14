from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from puzzle_jepa.data.hf_puzzles import HFPuzzleColumns, iter_hf_examples
from puzzle_jepa.data.lewm_sudoku import action_to_array, apply_fill_action, legal_fill_actions
from puzzle_jepa.data.worlds import PuzzleExample, SudokuWorld, WorldAction
from puzzle_jepa.eval.lewm_diagnostics import DEFAULT_PROJECTION_HORIZONS, run_lewm_diagnostic_bundle
from puzzle_jepa.models.lewm import LeWMSudokuModel
from puzzle_jepa.planning.lewm_planner import (
    PlannerName,
    ScoreMode,
    TransitionMode,
    hamming_distance,
    run_mpc,
)


PLANNERS: tuple[PlannerName, ...] = (
    "greedy",
    "beam",
    "best_first",
    "categorical_cem",
    "local_search",
    "mcts",
    "exact",
)
TRANSITIONS: tuple[TransitionMode, ...] = ("symbolic_reencode", "latent_rollout")
SCORES: tuple[ScoreMode, ...] = ("true_hamming_oracle", "oracle_goal_distance", "predicted_goal_distance")
DEPTHS: tuple[int, ...] = (4, 8, 16, 32, 64)


def model_from_config(config: dict[str, Any]) -> LeWMSudokuModel:
    return LeWMSudokuModel(**dict(config["model"]))


def load_checkpoint(path: Path, device: torch.device) -> tuple[LeWMSudokuModel, dict[str, Any]]:
    payload = torch.load(path, map_location=device)
    config = dict(payload["config"])
    model = model_from_config(config).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, config


def load_eval_examples(config: dict[str, Any], limit: int | None = None) -> list[PuzzleExample]:
    task_cfg = dict(config["task"])
    world = SudokuWorld()
    columns = HFPuzzleColumns(
        puzzle=str(task_cfg.get("puzzle_column", "question")),
        solution=str(task_cfg.get("solution_column", "answer")),
    )
    split = str(task_cfg.get("eval_split", "test[:128]"))
    repo_id = str(task_cfg["repo_id"])
    return list(iter_hf_examples(repo_id, split, world, columns, limit=limit))


@torch.no_grad()
def latent_statistics(
    model: LeWMSudokuModel,
    examples: list[PuzzleExample],
    *,
    device: torch.device,
    max_examples: int = 128,
) -> dict[str, float]:
    boards = torch.as_tensor(np.stack([item.state for item in examples[:max_examples]]), dtype=torch.long, device=device)
    emb = model.encode_board(boards).float()
    centered = emb - emb.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / max(1, emb.shape[0] - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return {
        "latent_mean_abs": float(emb.mean(dim=0).abs().mean().cpu()),
        "latent_std_mean": float(emb.std(dim=0).mean().cpu()),
        "latent_std_min": float(emb.std(dim=0).min().cpu()),
        "latent_cov_diag_mean": float(torch.diag(cov).mean().cpu()),
        "latent_cov_offdiag_abs_mean": float(off_diag.abs().mean().cpu()),
        "latent_norm_mean": float(torch.linalg.vector_norm(emb, dim=-1).mean().cpu()),
    }


@torch.no_grad()
def oracle_distance_diagnostics(
    model: LeWMSudokuModel,
    examples: list[PuzzleExample],
    *,
    device: torch.device,
    max_examples: int = 32,
) -> dict[str, float]:
    drops: list[float] = []
    monotone_steps = 0
    total_steps = 0
    final_distances: list[float] = []
    for example in examples[:max_examples]:
        board = example.state.copy()
        sequence = [board.copy()]
        for row, col in np.argwhere(board == 0):
            action = WorldAction(int(row), int(col), int(example.goal[row, col]))
            board = apply_fill_action(board, action, allow_conflicts=True)
            sequence.append(board.copy())
        boards = torch.as_tensor(np.stack(sequence), dtype=torch.long, device=device)
        goal = torch.as_tensor(example.goal[None], dtype=torch.long, device=device)
        emb = model.encode_board(boards)
        goal_emb = model.encode_board(goal)
        distances = torch.linalg.vector_norm(emb - goal_emb, dim=-1).cpu().numpy()
        diffs = distances[:-1] - distances[1:]
        drops.extend(float(x) for x in diffs)
        monotone_steps += int((diffs >= 0.0).sum())
        total_steps += len(diffs)
        final_distances.append(float(distances[-1]))
    return {
        "oracle_path_distance_drop_mean": float(np.mean(drops)) if drops else 0.0,
        "oracle_path_distance_monotone_fraction": float(monotone_steps / max(1, total_steps)),
        "oracle_path_goal_distance_mean": float(np.mean(final_distances)) if final_distances else 0.0,
    }


@torch.no_grad()
def local_action_rank_diagnostics(
    model: LeWMSudokuModel,
    examples: list[PuzzleExample],
    *,
    device: torch.device,
    max_examples: int = 32,
) -> dict[str, float]:
    oracle_ranks: list[int] = []
    value_ranks: list[int] = []
    hamming_ranks: list[int] = []
    for example in examples[:max_examples]:
        board = example.state.copy()
        empty = np.argwhere(board == 0)
        if len(empty) < 2:
            continue
        reveal_count = len(empty) // 2
        for row, col in empty[:reveal_count]:
            board[int(row), int(col)] = int(example.goal[int(row), int(col)])
        actions = legal_fill_actions(board, allow_conflicts=True)
        if not actions:
            continue
        goal_actions = {
            idx
            for idx, action in enumerate(actions)
            if int(example.goal[action.row, action.col]) == action.value
        }
        if not goal_actions:
            continue
        goal_t = torch.as_tensor(example.goal[None], dtype=torch.long, device=device)
        goal_emb = model.encode_board(goal_t)
        oracle_costs: list[float] = []
        value_costs: list[float] = []
        hamming_costs: list[float] = []
        for action in actions:
            leaf = apply_fill_action(board, action, allow_conflicts=True)
            leaf_t = torch.as_tensor(leaf[None], dtype=torch.long, device=device)
            leaf_emb = model.encode_board(leaf_t)
            oracle_costs.append(float(torch.linalg.vector_norm(leaf_emb - goal_emb, dim=-1).item()))
            value_costs.append(float(model.score_value(leaf_emb).item()))
            hamming_costs.append(float(hamming_distance(leaf, example.goal)))
        oracle_ranks.append(_best_positive_rank(oracle_costs, goal_actions))
        value_ranks.append(_best_positive_rank(value_costs, goal_actions))
        hamming_ranks.append(_best_positive_rank(hamming_costs, goal_actions))
    return {
        "local_oracle_goal_distance_gold_rank_mean": float(np.mean(oracle_ranks)) if oracle_ranks else 0.0,
        "local_predicted_goal_distance_gold_rank_mean": float(np.mean(value_ranks)) if value_ranks else 0.0,
        "local_true_hamming_gold_rank_mean": float(np.mean(hamming_ranks)) if hamming_ranks else 0.0,
        "local_rank_examples": float(len(oracle_ranks)),
    }


def run_planner_matrix(
    model: LeWMSudokuModel,
    examples: list[PuzzleExample],
    *,
    output_path: Path,
    device: torch.device,
    seed: int = 0,
    planners: tuple[PlannerName, ...] = PLANNERS,
    transitions: tuple[TransitionMode, ...] = TRANSITIONS,
    scores: tuple[ScoreMode, ...] = SCORES,
    depths: tuple[int, ...] = DEPTHS,
    max_examples: int = 8,
    max_steps: int = 81,
    fast: bool = False,
) -> list[dict[str, Any]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    budgets = _planner_budgets(fast=fast)
    with output_path.open("w") as handle:
        for planner in planners:
            if planner == "exact":
                combos = [("symbolic_reencode", "true_hamming_oracle", 0)]
            else:
                combos = []
                for score in scores:
                    score_transitions = ("symbolic_reencode",) if score == "true_hamming_oracle" else transitions
                    combos.extend((transition, score, depth) for transition in score_transitions for depth in depths)
            for transition, score, depth in combos:
                solved = 0
                remaining: list[int] = []
                steps: list[int] = []
                action_evals: list[int] = []
                elapsed_seconds: list[float] = []
                for example in examples[:max_examples]:
                    result = run_mpc(
                        model,
                        example.state,
                        example.goal,
                        planner=planner,
                        transition_mode=transition,
                        score_mode=score,
                        horizon=max(1, depth),
                        max_steps=max_steps,
                        rng=rng,
                        device=device,
                        **budgets,
                    )
                    solved += int(result.solved)
                    remaining.append(result.remaining_hamming)
                    steps.append(result.steps)
                    action_evals.append(result.action_evals)
                    elapsed_seconds.append(result.elapsed_seconds)
                elapsed_total = float(np.sum(elapsed_seconds)) if elapsed_seconds else 0.0
                action_evals_total = int(np.sum(action_evals)) if action_evals else 0
                record = {
                    "planner": planner,
                    "transition_mode": transition,
                    "score_mode": score,
                    "horizon": depth,
                    "examples": min(max_examples, len(examples)),
                    "solved": solved,
                    "solve_rate": solved / max(1, min(max_examples, len(examples))),
                    "remaining_hamming_mean": float(np.mean(remaining)) if remaining else 0.0,
                    "steps_mean": float(np.mean(steps)) if steps else 0.0,
                    "action_evals_mean": float(np.mean(action_evals)) if action_evals else 0.0,
                    "action_evals_total": action_evals_total,
                    "elapsed_seconds_mean": float(np.mean(elapsed_seconds)) if elapsed_seconds else 0.0,
                    "elapsed_seconds_total": elapsed_total,
                    "seconds_per_action_eval": (
                        elapsed_total / float(action_evals_total) if action_evals_total > 0 else 0.0
                    ),
                }
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                records.append(record)
    return records


def _best_positive_rank(costs: list[float], positive_indices: set[int]) -> int:
    order = np.argsort(np.asarray(costs, dtype=np.float64)).tolist()
    for rank, index in enumerate(order, start=1):
        if index in positive_indices:
            return rank
    return len(costs)


def _planner_budgets(*, fast: bool) -> dict[str, Any]:
    if fast:
        return {
            "beam_width": 2,
            "branch_size": 4,
            "best_first_expansions": 16,
            "cem_candidates": 8,
            "cem_elites": 2,
            "cem_iterations": 1,
            "local_candidates": 8,
            "local_iterations": 8,
            "mcts_simulations": 8,
            "mcts_branch_size": 8,
            "mcts_progressive_c": 1.5,
            "mcts_progressive_alpha": 0.5,
        }
    return {
        "beam_width": 8,
        "branch_size": 24,
        "best_first_expansions": 256,
        "cem_candidates": 128,
        "cem_elites": 16,
        "cem_iterations": 4,
        "local_candidates": 64,
        "local_iterations": 128,
        "mcts_simulations": 256,
        "mcts_branch_size": 32,
        "mcts_progressive_c": 2.0,
        "mcts_progressive_alpha": 0.5,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--planners", type=str, default=",".join(PLANNERS))
    parser.add_argument("--transitions", type=str, default=",".join(TRANSITIONS))
    parser.add_argument("--scores", type=str, default=",".join(SCORES))
    parser.add_argument("--depths", type=str, default=",".join(str(item) for item in DEPTHS))
    parser.add_argument("--latent-examples", type=int, default=128)
    parser.add_argument("--trajectory-examples", type=int, default=32)
    parser.add_argument("--rank-examples", type=int, default=16)
    parser.add_argument("--panel-examples", type=int, default=3)
    parser.add_argument("--panel-steps", type=int, default=5)
    parser.add_argument("--panel-actions", type=int, default=6)
    parser.add_argument("--projection-horizons", type=str, default=",".join(str(item) for item in DEFAULT_PROJECTION_HORIZONS))
    parser.add_argument("--no-diagnostic-plots", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_checkpoint(args.checkpoint, device)
    examples = load_eval_examples(config, limit=max(args.examples, 128))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = run_lewm_diagnostic_bundle(
        model,
        examples,
        args.output_dir,
        device=device,
        seed=args.seed + 500,
        latent_examples=args.latent_examples,
        trajectory_examples=args.trajectory_examples,
        rank_examples=args.rank_examples,
        panel_examples=args.panel_examples,
        panel_steps=args.panel_steps,
        panel_actions=args.panel_actions,
        projection_horizons=tuple(int(item) for item in args.projection_horizons.split(",") if item),
        write_plots=not args.no_diagnostic_plots,
    )
    (args.output_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True))
    run_planner_matrix(
        model,
        examples,
        output_path=args.output_dir / "planner_matrix.jsonl",
        device=device,
        seed=args.seed,
        max_examples=args.examples,
        fast=args.fast,
        planners=_parse_names(args.planners, PLANNERS),
        transitions=_parse_names(args.transitions, TRANSITIONS),
        scores=_parse_names(args.scores, SCORES),
        depths=tuple(int(item) for item in args.depths.split(",") if item),
    )


def _parse_names(text: str, allowed: tuple[Any, ...]) -> tuple[Any, ...]:
    names = tuple(item for item in text.split(",") if item)
    unknown = sorted(set(names) - {str(item) for item in allowed})
    if unknown:
        raise ValueError(f"Unknown names {unknown}; allowed: {allowed}.")
    return names


if __name__ == "__main__":
    main()
