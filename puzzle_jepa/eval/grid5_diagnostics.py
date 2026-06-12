from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from puzzle_jepa.data import (
    SudokuWorld,
    WorldAction,
    collate_rollouts,
    sample_oracle_partial_transition,
    sample_oracle_rollout_transition,
)
from puzzle_jepa.models import SigRegActionJEPA, sigreg_loss
from puzzle_jepa.train.grid0 import _build_world, _load_examples


@dataclass(slots=True)
class Grid5Diagnostics:
    latent: dict[str, float]
    trajectory: dict[str, float]
    action_rank: dict[str, float]
    plan: dict[str, float]


def load_model(run_root: Path, device: torch.device) -> tuple[SigRegActionJEPA, dict[str, Any]]:
    checkpoint = torch.load(run_root / "checkpoint.pt", map_location=device)
    config = dict(checkpoint["config"])
    task_cfg = dict(config["task"])
    world = _build_world(task_cfg)
    model = SigRegActionJEPA(vocab_size=world.vocab_size, **dict(config["model"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, config


def run_grid5_diagnostics(
    run_root: Path,
    output_dir: Path,
    *,
    seed: int = 0,
    latent_examples: int = 64,
    trajectory_examples: int = 32,
    rank_examples: int = 32,
    detail_examples: int = 4,
    plan_examples: int = 16,
    plan_beam_size: int = 4,
    plan_branch_size: int = 8,
    max_plan_steps: int = 96,
) -> Grid5Diagnostics:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model(run_root, device)
    task_cfg = dict(config["task"])
    world = _build_world(task_cfg)
    if not isinstance(world, SudokuWorld):
        raise ValueError("Grid 5 diagnostics currently support Sudoku only.")
    examples = _load_examples(task_cfg, "eval")
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    latent = latent_space_summary(model, world, examples, rng, latent_examples, device)
    trajectory, trajectory_records = trajectory_summary(model, world, examples, rng, trajectory_examples, device)
    action_rank, rank_records, detail_records = action_rank_summary(
        model,
        world,
        examples,
        rng,
        rank_examples,
        detail_examples,
        device,
    )
    plan = {}
    for score_mode in ("latent_goal", "goal_energy"):
        plan.update(
            {
                f"{score_mode}_{key}": value
                for key, value in beam_plan_summary(
                    model,
                    world,
                    examples,
                    rng,
                    plan_examples,
                    score_mode,
                    plan_beam_size,
                    plan_branch_size,
                    max_plan_steps,
                    device,
                ).items()
            }
        )

    summary = Grid5Diagnostics(latent=latent, trajectory=trajectory, action_rank=action_rank, plan=plan)
    (output_dir / "diagnostics.json").write_text(json.dumps(asdict(summary), indent=2, sort_keys=True))
    write_jsonl(output_dir / "trajectory_records.jsonl", trajectory_records)
    write_jsonl(output_dir / "action_rank_records.jsonl", rank_records)
    write_jsonl(output_dir / "action_rank_examples.jsonl", detail_records)
    return summary


@torch.no_grad()
def latent_space_summary(
    model: SigRegActionJEPA,
    world: SudokuWorld,
    examples,
    rng: np.random.Generator,
    count: int,
    device: torch.device,
) -> dict[str, float]:
    states = []
    for _ in range(max(1, count)):
        rollout = sample_oracle_rollout_transition(
            world,
            examples[int(rng.integers(0, len(examples)))],
            rng,
            steps=min(4, int(model.max_rollout_steps)),
        )
        states.append(rollout.state)
        states.extend(rollout.target_states)
    tensor = torch.as_tensor(np.stack(states), dtype=torch.long, device=device)
    latents = model.encode(tensor)
    centered = latents - latents.mean(dim=0, keepdim=True)
    std = latents.std(dim=0, unbiased=False)
    cov = centered.t() @ centered / max(1, latents.shape[0] - 1)
    diag = torch.diagonal(cov)
    offdiag = cov - torch.diag(diag)
    pairwise = torch.pdist(latents.float()) if latents.shape[0] > 1 else latents.new_zeros(1)
    return {
        "count": float(latents.shape[0]),
        "latent_dim": float(latents.shape[1]),
        "mean_abs": float(latents.mean(dim=0).abs().mean().cpu()),
        "std_mean": float(std.mean().cpu()),
        "std_min": float(std.min().cpu()),
        "std_max": float(std.max().cpu()),
        "cov_diag_mean": float(diag.mean().cpu()),
        "cov_offdiag_abs_mean": float(offdiag.abs().mean().cpu()),
        "norm_mean": float(latents.norm(dim=-1).mean().cpu()),
        "norm_std": float(latents.norm(dim=-1).std(unbiased=False).cpu()),
        "pairwise_distance_mean": float(pairwise.mean().cpu()),
        "pairwise_distance_std": float(pairwise.std(unbiased=False).cpu()),
        "sigreg_eval": float(
            sigreg_loss(
                latents,
                projections=model.sigreg_projections,
                knots=model.sigreg_knots,
                knot_max=model.sigreg_knot_max,
            ).cpu()
        ),
    }


@torch.no_grad()
def trajectory_summary(
    model: SigRegActionJEPA,
    world: SudokuWorld,
    examples,
    rng: np.random.Generator,
    count: int,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    latent_monotone = []
    energy_monotone = []
    latent_deltas = []
    energy_deltas = []
    energy_errors = []
    steps = min(16, int(model.max_rollout_steps))
    for index in range(max(1, count)):
        rollout = sample_oracle_rollout_transition(
            world,
            examples[int(rng.integers(0, len(examples)))],
            rng,
            steps=steps,
        )
        states_np = np.stack([rollout.state, *rollout.target_states])
        states = torch.as_tensor(states_np, dtype=torch.long, device=device)
        goal = torch.as_tensor(rollout.goal[None], dtype=torch.long, device=device).expand(states.shape[0], -1, -1)
        initial = torch.as_tensor(rollout.state[None], dtype=torch.long, device=device).expand(states.shape[0], -1, -1)
        latents = model.encode(states)
        goal_latents = model.encode(goal)
        true_energy = F.mse_loss(latents, goal_latents, reduction="none").mean(dim=-1)
        pred_energy = model.predict_goal_energy(states, initial)
        true_values = true_energy.detach().cpu().tolist()
        pred_values = pred_energy.detach().cpu().tolist()
        true_delta = np.diff(true_values)
        pred_delta = np.diff(pred_values)
        latent_monotone.extend((true_delta <= 1.0e-8).astype(float).tolist())
        energy_monotone.extend((pred_delta <= 1.0e-8).astype(float).tolist())
        latent_deltas.extend(true_delta.tolist())
        energy_deltas.extend(pred_delta.tolist())
        energy_errors.extend((np.asarray(pred_values) - np.asarray(true_values)).tolist())
        records.append(
            {
                "example_index": index,
                "true_energy": true_values,
                "predicted_energy": pred_values,
                "true_delta": true_delta.tolist(),
                "predicted_delta": pred_delta.tolist(),
            }
        )
    return (
        {
            "count": float(count),
            "latent_monotone_rate": safe_mean(latent_monotone),
            "goal_energy_monotone_rate": safe_mean(energy_monotone),
            "latent_delta_mean": safe_mean(latent_deltas),
            "goal_energy_delta_mean": safe_mean(energy_deltas),
            "goal_energy_abs_error_mean": safe_mean([abs(x) for x in energy_errors]),
            "goal_energy_error_mean": safe_mean(energy_errors),
        },
        records,
    )


@torch.no_grad()
def action_rank_summary(
    model: SigRegActionJEPA,
    world: SudokuWorld,
    examples,
    rng: np.random.Generator,
    count: int,
    detail_examples: int,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, Any]], list[dict[str, Any]]]:
    records = []
    details = []
    for index in range(max(1, count)):
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
        true_scores = score_candidate_states(
            model,
            next_states,
            transition.goal,
            transition.state,
            score_mode="latent_goal",
            device=device,
        )
        pred_scores = score_candidate_states(
            model,
            next_states,
            transition.goal,
            transition.state,
            score_mode="goal_energy",
            device=device,
        )
        gold_index = next(
            i
            for i, action in enumerate(actions)
            if action.row == transition.action.row
            and action.col == transition.action.col
            and action.value == transition.action.value
        )
        true_order = np.argsort(true_scores)
        pred_order = np.argsort(pred_scores)
        true_rank = int(np.where(true_order == gold_index)[0][0]) + 1
        pred_rank = int(np.where(pred_order == gold_index)[0][0]) + 1
        true_top = actions[int(true_order[0])]
        pred_top = actions[int(pred_order[0])]
        record = {
            "example_index": index,
            "candidate_count": len(actions),
            "gold_row": transition.action.row,
            "gold_col": transition.action.col,
            "gold_value": transition.action.value,
            "latent_gold_rank": true_rank,
            "goal_energy_gold_rank": pred_rank,
            "latent_gold_top1": float(true_rank == 1),
            "goal_energy_gold_top1": float(pred_rank == 1),
            "latent_top_goal_value": float(true_top.value == int(transition.goal[true_top.row, true_top.col])),
            "goal_energy_top_goal_value": float(pred_top.value == int(transition.goal[pred_top.row, pred_top.col])),
            "latent_gold_score": float(true_scores[gold_index]),
            "goal_energy_gold_score": float(pred_scores[gold_index]),
            "latent_best_score": float(true_scores[int(true_order[0])]),
            "goal_energy_best_score": float(pred_scores[int(pred_order[0])]),
        }
        records.append(record)
        if index < detail_examples:
            details.append(
                detail_action_example(
                    transition,
                    actions,
                    true_scores,
                    pred_scores,
                    gold_index,
                )
            )
    return (
        {
            "count": float(len(records)),
            "latent_gold_top1_rate": safe_mean([item["latent_gold_top1"] for item in records]),
            "goal_energy_gold_top1_rate": safe_mean([item["goal_energy_gold_top1"] for item in records]),
            "latent_gold_rank_mean": safe_mean([item["latent_gold_rank"] for item in records]),
            "goal_energy_gold_rank_mean": safe_mean([item["goal_energy_gold_rank"] for item in records]),
            "latent_top_goal_value_rate": safe_mean([item["latent_top_goal_value"] for item in records]),
            "goal_energy_top_goal_value_rate": safe_mean([item["goal_energy_top_goal_value"] for item in records]),
        },
        records,
        details,
    )


def candidate_actions(world: SudokuWorld, state: np.ndarray, clue_mask: np.ndarray) -> list[WorldAction]:
    actions = []
    mutable = np.argwhere(~world.validate_clue_mask(clue_mask))
    for row, col in mutable:
        current = int(state[int(row), int(col)])
        for value in range(1, 10):
            if current != value:
                actions.append(WorldAction(int(row), int(col), value))
    return actions


@torch.no_grad()
def score_candidate_states(
    model: SigRegActionJEPA,
    states_np: np.ndarray,
    goal_np: np.ndarray,
    initial_np: np.ndarray,
    *,
    score_mode: str,
    device: torch.device,
) -> np.ndarray:
    states = torch.as_tensor(states_np, dtype=torch.long, device=device)
    goal = torch.as_tensor(goal_np[None], dtype=torch.long, device=device).expand(states.shape[0], -1, -1)
    initial = torch.as_tensor(initial_np[None], dtype=torch.long, device=device).expand(states.shape[0], -1, -1)
    if score_mode == "latent_goal":
        scores = model.score_states_to_goal(states, goal)
    elif score_mode == "goal_energy":
        scores = model.predict_goal_energy(states, initial)
    else:
        raise ValueError(f"unknown score_mode {score_mode!r}.")
    return scores.detach().float().cpu().numpy()


def detail_action_example(transition, actions, true_scores, pred_scores, gold_index: int) -> dict[str, Any]:
    gold = actions[gold_index]
    selected = {gold_index}
    candidates = [gold_index]
    same_cell = [
        i for i, action in enumerate(actions) if action.row == gold.row and action.col == gold.col and i != gold_index
    ]
    nearby = [
        i
        for i, action in enumerate(actions)
        if abs(action.row - gold.row) + abs(action.col - gold.col) <= 2
        and (action.row, action.col) != (gold.row, gold.col)
    ]
    far = [
        i
        for i, action in enumerate(actions)
        if abs(action.row - gold.row) + abs(action.col - gold.col) > 4
    ]
    for pool in (same_cell[:3], nearby[:3], far[:3]):
        for item in pool:
            if item not in selected:
                candidates.append(item)
                selected.add(item)
    return {
        "gold": {"row": gold.row, "col": gold.col, "value": gold.value},
        "candidates": [
            {
                "row": actions[i].row,
                "col": actions[i].col,
                "value": actions[i].value,
                "is_gold": i == gold_index,
                "is_goal_value": actions[i].value == int(transition.goal[actions[i].row, actions[i].col]),
                "latent_goal_score": float(true_scores[i]),
                "goal_energy_score": float(pred_scores[i]),
            }
            for i in candidates
        ],
    }


@torch.no_grad()
def beam_plan_summary(
    model: SigRegActionJEPA,
    world: SudokuWorld,
    examples,
    rng: np.random.Generator,
    count: int,
    score_mode: str,
    beam_size: int,
    branch_size: int,
    max_steps: int,
    device: torch.device,
) -> dict[str, float]:
    solved = []
    remaining = []
    terminals = []
    for _ in range(max(1, count)):
        example = examples[int(rng.integers(0, len(examples)))]
        clue_mask = world.clue_mask_from_puzzle(example.state)
        beam = [(example.state.copy(), 0.0)]
        best_board = example.state.copy()
        for _step in range(max_steps):
            if any(world.is_goal(board, example.goal) for board, _score in beam):
                best_board = next(board for board, _score in beam if world.is_goal(board, example.goal))
                break
            expanded = []
            for board, _score in beam:
                actions = candidate_actions(world, board, clue_mask)
                if not actions:
                    expanded.append((board, 0.0))
                    continue
                next_states = np.stack(
                    [
                        world.apply(
                            board,
                            action,
                            clue_mask=clue_mask,
                            allow_overwrite=True,
                            allow_conflicts=True,
                        )
                        for action in actions
                    ]
                )
                scores = score_candidate_states(
                    model,
                    next_states,
                    example.goal,
                    example.state,
                    score_mode=score_mode,
                    device=device,
                )
                order = np.argsort(scores)[: max(1, branch_size)]
                expanded.extend((next_states[int(index)], float(scores[int(index)])) for index in order)
            expanded.sort(key=lambda item: item[1])
            beam = expanded[: max(1, beam_size)]
            best_board = beam[0][0]
        solved.append(float(world.is_goal(best_board, example.goal)))
        remaining.append(float(np.not_equal(best_board, example.goal).sum()))
        terminals.append(float(np.count_nonzero(best_board == 0) == 0))
    return {
        "solve_rate": safe_mean(solved),
        "terminal_rate": safe_mean(terminals),
        "mean_remaining_hamming": safe_mean(remaining),
        "count": float(count),
    }


def safe_mean(values) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--latent-examples", type=int, default=64)
    parser.add_argument("--trajectory-examples", type=int, default=32)
    parser.add_argument("--rank-examples", type=int, default=32)
    parser.add_argument("--detail-examples", type=int, default=4)
    parser.add_argument("--plan-examples", type=int, default=16)
    parser.add_argument("--plan-beam-size", type=int, default=4)
    parser.add_argument("--plan-branch-size", type=int, default=8)
    parser.add_argument("--max-plan-steps", type=int, default=96)
    args = parser.parse_args()
    summary = run_grid5_diagnostics(
        args.run_root,
        args.output_dir,
        seed=args.seed,
        latent_examples=args.latent_examples,
        trajectory_examples=args.trajectory_examples,
        rank_examples=args.rank_examples,
        detail_examples=args.detail_examples,
        plan_examples=args.plan_examples,
        plan_beam_size=args.plan_beam_size,
        plan_branch_size=args.plan_branch_size,
        max_plan_steps=args.max_plan_steps,
    )
    print(json.dumps(asdict(summary), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
