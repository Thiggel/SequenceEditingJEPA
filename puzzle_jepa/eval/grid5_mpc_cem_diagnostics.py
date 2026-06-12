from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from puzzle_jepa.data import SudokuWorld, WorldAction, sample_oracle_partial_transition
from puzzle_jepa.eval.grid5_diagnostics import candidate_actions, load_model
from puzzle_jepa.models import SigRegActionJEPA
from puzzle_jepa.train.grid0 import _build_world, _load_examples


@dataclass(slots=True)
class CemConfig:
    horizons: tuple[int, ...]
    plan_examples: int
    max_mpc_steps: int
    candidates: int
    elites: int
    iterations: int
    smoothing: float
    seed: int


def run_grid5_mpc_cem_diagnostics(
    run_root: Path,
    output_dir: Path,
    *,
    seed: int = 0,
    horizons: tuple[int, ...] = (4, 8, 16, 32, 64),
    plan_examples: int = 4,
    max_mpc_steps: int = 64,
    candidates: int = 300,
    elites: int = 30,
    iterations: int = 10,
    smoothing: float = 0.7,
    qualitative_examples: int = 3,
    qualitative_candidates: int = 6,
    qualitative_cem_candidates: int = 128,
    qualitative_cem_iterations: int = 5,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model(run_root, device)
    task_cfg = dict(config["task"])
    world = _build_world(task_cfg)
    if not isinstance(world, SudokuWorld):
        raise ValueError("Grid 5 MPC-CEM diagnostics currently support Sudoku only.")
    examples = _load_examples(task_cfg, "eval")
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "run_root": str(run_root),
        "model": {
            "encoder_type": config["model"]["encoder_type"],
            "predictor_type": config["model"]["predictor_type"],
            "predict_delta": bool(config["model"].get("predict_delta", False)),
            "latent_size": int(config["model"]["latent_size"]),
        },
        "cem": asdict(
            CemConfig(
                horizons=tuple(int(h) for h in horizons),
                plan_examples=int(plan_examples),
                max_mpc_steps=int(max_mpc_steps),
                candidates=int(candidates),
                elites=int(elites),
                iterations=int(iterations),
                smoothing=float(smoothing),
                seed=int(seed),
            )
        ),
        "horizons": {},
    }
    planning_records: list[dict[str, Any]] = []
    root_records: list[dict[str, Any]] = []
    qualitative_records: list[dict[str, Any]] = []

    eval_indices = [int(rng.integers(0, len(examples))) for _ in range(max(1, plan_examples))]
    for horizon in horizons:
        for score_mode in ("latent_goal", "goal_energy"):
            records = []
            for local_index, example_index in enumerate(eval_indices):
                result = mpc_cem_plan(
                    model,
                    world,
                    examples[example_index],
                    horizon=int(horizon),
                    score_mode=score_mode,
                    max_steps=max_mpc_steps,
                    candidates=candidates,
                    elites=elites,
                    iterations=iterations,
                    smoothing=smoothing,
                    rng=rng,
                    device=device,
                )
                record = {
                    "example_index": example_index,
                    "local_index": local_index,
                    "horizon": int(horizon),
                    "score_mode": score_mode,
                    **result,
                }
                records.append(record)
                planning_records.append(record)
                if result.get("root_action") is not None:
                    root_records.append(
                        {
                            "example_index": example_index,
                            "local_index": local_index,
                            "horizon": int(horizon),
                            "score_mode": score_mode,
                            **result["root_action"],
                        }
                    )
            prefix = f"h{int(horizon)}_{score_mode}"
            summary["horizons"][prefix] = summarize_planning_records(records)

    detail_indices = [int(rng.integers(0, len(examples))) for _ in range(max(1, qualitative_examples))]
    for example_index in detail_indices:
        transition = sample_oracle_partial_transition(world, examples[example_index], rng)
        clue_mask = transition.clue_mask if transition.clue_mask is not None else world.clue_mask_from_puzzle(
            transition.state
        )
        actions = select_detail_actions(world, transition, clue_mask, max_actions=qualitative_candidates)
        for horizon in horizons:
            qualitative_records.append(
                compare_fixed_first_actions(
                    model,
                    world,
                    transition.state,
                    transition.goal,
                    clue_mask,
                    actions,
                    gold_action=transition.action,
                    horizon=int(horizon),
                    candidates=qualitative_cem_candidates,
                    elites=max(1, min(elites, qualitative_cem_candidates)),
                    iterations=qualitative_cem_iterations,
                    smoothing=smoothing,
                    rng=rng,
                    device=device,
                )
            )

    write_jsonl(output_dir / "mpc_cem_records.jsonl", planning_records)
    write_jsonl(output_dir / "mpc_cem_root_actions.jsonl", root_records)
    write_jsonl(output_dir / "mpc_cem_lookahead_examples.jsonl", qualitative_records)
    (output_dir / "mpc_cem_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


@torch.no_grad()
def mpc_cem_plan(
    model: SigRegActionJEPA,
    world: SudokuWorld,
    example,
    *,
    horizon: int,
    score_mode: str,
    max_steps: int,
    candidates: int,
    elites: int,
    iterations: int,
    smoothing: float,
    rng: np.random.Generator,
    device: torch.device,
) -> dict[str, Any]:
    clue_mask = world.clue_mask_from_puzzle(example.state)
    board = example.state.copy()
    root_action_record: dict[str, Any] | None = None
    best_score = float("inf")
    for step in range(max_steps):
        if world.is_goal(board, example.goal):
            break
        actions = candidate_actions(world, board, clue_mask)
        if not actions:
            break
        plan = cem_optimize_action_sequence(
            model,
            world,
            board,
            example.goal,
            example.state,
            actions,
            horizon=horizon,
            score_mode=score_mode,
            candidates=candidates,
            elites=elites,
            iterations=iterations,
            smoothing=smoothing,
            rng=rng,
            device=device,
        )
        first_action = actions[int(plan["indices"][0])]
        best_score = float(plan["score"])
        if step == 0:
            root_action_record = {
                "row": first_action.row,
                "col": first_action.col,
                "value": first_action.value,
                "is_goal_value": bool(first_action.value == int(example.goal[first_action.row, first_action.col])),
                "score": best_score,
            }
        board = world.apply(board, first_action, clue_mask=clue_mask, allow_overwrite=True, allow_conflicts=True)
    remaining_hamming = int(np.not_equal(board, example.goal).sum())
    return {
        "solved": bool(world.is_goal(board, example.goal)),
        "terminal": bool(np.count_nonzero(board == 0) == 0),
        "steps": int(step + 1 if not world.is_goal(board, example.goal) else step),
        "remaining_hamming": remaining_hamming,
        "best_score": best_score,
        "root_action": root_action_record,
    }


@torch.no_grad()
def cem_optimize_action_sequence(
    model: SigRegActionJEPA,
    world: SudokuWorld,
    state_np: np.ndarray,
    goal_np: np.ndarray,
    initial_np: np.ndarray,
    actions: list[WorldAction],
    *,
    horizon: int,
    score_mode: str,
    candidates: int,
    elites: int,
    iterations: int,
    smoothing: float,
    rng: np.random.Generator,
    device: torch.device,
    fixed_first_action: WorldAction | None = None,
) -> dict[str, Any]:
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if not actions:
        raise ValueError("CEM needs at least one action.")
    action_options = torch.as_tensor(
        np.stack([action.as_array(world.task_id) for action in actions]),
        dtype=torch.long,
        device=device,
    )
    action_count = len(actions)
    suffix_horizon = horizon - 1 if fixed_first_action is not None else horizon
    probs = torch.full(
        (max(1, suffix_horizon), action_count),
        1.0 / float(action_count),
        dtype=torch.float32,
        device=device,
    )
    best_indices = torch.zeros(horizon, dtype=torch.long, device=device)
    best_score = torch.tensor(float("inf"), dtype=torch.float32, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(rng.integers(0, 2**31 - 1)))

    for _ in range(max(1, iterations)):
        if suffix_horizon > 0:
            sampled_suffix = torch.stack(
                [
                    torch.multinomial(probs[t], num_samples=candidates, replacement=True, generator=generator)
                    for t in range(suffix_horizon)
                ],
                dim=1,
            )
        else:
            sampled_suffix = torch.empty((candidates, 0), dtype=torch.long, device=device)
        if fixed_first_action is None:
            sampled = sampled_suffix
            sampled_actions = action_options[sampled]
        else:
            first = torch.as_tensor(
                fixed_first_action.as_array(world.task_id),
                dtype=torch.long,
                device=device,
            ).view(1, 1, 4).expand(candidates, 1, 4)
            sampled = torch.cat(
                [torch.zeros((candidates, 1), dtype=torch.long, device=device), sampled_suffix],
                dim=1,
            )
            sampled_actions = torch.cat([first, action_options[sampled_suffix]], dim=1)
        scores = score_latent_rollouts(
            model,
            state_np,
            goal_np,
            initial_np,
            sampled_actions,
            score_mode=score_mode,
            device=device,
        )
        elite_count = max(1, min(int(elites), candidates))
        elite_scores, elite_pos = torch.topk(scores, k=elite_count, largest=False)
        if elite_scores[0] < best_score:
            best_score = elite_scores[0]
            best_indices = sampled[elite_pos[0]].detach().clone()
        if suffix_horizon > 0:
            elite_suffix = sampled_suffix[elite_pos]
            updated = []
            for t in range(suffix_horizon):
                counts = torch.bincount(elite_suffix[:, t], minlength=action_count).float()
                new_probs = (counts + 1.0e-3) / (counts.sum() + 1.0e-3 * action_count)
                updated.append(new_probs)
            new_probs = torch.stack(updated, dim=0)
            probs = float(smoothing) * new_probs + (1.0 - float(smoothing)) * probs
            probs = probs / probs.sum(dim=-1, keepdim=True)
    return {"indices": best_indices.detach().cpu().tolist(), "score": float(best_score.detach().cpu())}


@torch.no_grad()
def score_latent_rollouts(
    model: SigRegActionJEPA,
    state_np: np.ndarray,
    goal_np: np.ndarray,
    initial_np: np.ndarray,
    action_sequences: torch.Tensor,
    *,
    score_mode: str,
    device: torch.device,
) -> torch.Tensor:
    batch, horizon, _ = action_sequences.shape
    state = torch.as_tensor(state_np[None], dtype=torch.long, device=device).expand(batch, -1, -1)
    goal = torch.as_tensor(goal_np[None], dtype=torch.long, device=device).expand(batch, -1, -1)
    initial = torch.as_tensor(initial_np[None], dtype=torch.long, device=device).expand(batch, -1, -1)
    current = model.encode(state)
    initial_latent = model.encode(initial)
    goal_latent = model.encode(goal)
    history: list[torch.Tensor] = [current]
    max_history = max(1, int(model.max_rollout_steps))
    for step in range(horizon):
        if model.predictor_type == "ar_transformer":
            start = max(0, len(history) - max_history)
            latent_window = torch.stack(history[start:], dim=1)
            action_start = max(0, step + 1 - latent_window.shape[1])
            action_window = action_sequences[:, action_start : step + 1]
            current = model.predict_sequence(latent_window, action_window)[:, -1]
        else:
            current = model.predict_next(current, action_sequences[:, step])
        history.append(current)
    if score_mode == "latent_goal":
        return F.mse_loss(current, goal_latent, reduction="none").mean(dim=-1)
    if score_mode == "goal_energy":
        return model.predict_goal_energy_from_latents(current, initial_latent)
    raise ValueError(f"unknown score_mode {score_mode!r}.")


def summarize_planning_records(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {
            "count": 0.0,
            "solve_rate": 0.0,
            "terminal_rate": 0.0,
            "mean_remaining_hamming": 0.0,
            "root_goal_value_rate": 0.0,
        }
    roots = [record["root_action"] for record in records if record.get("root_action") is not None]
    return {
        "count": float(len(records)),
        "solve_rate": float(np.mean([record["solved"] for record in records])),
        "terminal_rate": float(np.mean([record["terminal"] for record in records])),
        "mean_remaining_hamming": float(np.mean([record["remaining_hamming"] for record in records])),
        "root_goal_value_rate": float(np.mean([root["is_goal_value"] for root in roots])) if roots else 0.0,
        "mean_steps": float(np.mean([record["steps"] for record in records])),
    }


def select_detail_actions(
    world: SudokuWorld,
    transition,
    clue_mask: np.ndarray,
    *,
    max_actions: int,
) -> list[WorldAction]:
    actions = candidate_actions(world, transition.state, clue_mask)
    selected: list[WorldAction] = [transition.action]
    gold = transition.action
    pools = [
        [action for action in actions if action.row == gold.row and action.col == gold.col and action.value != gold.value],
        [
            action
            for action in actions
            if abs(action.row - gold.row) + abs(action.col - gold.col) <= 2
            and (action.row, action.col) != (gold.row, gold.col)
        ],
        [action for action in actions if abs(action.row - gold.row) + abs(action.col - gold.col) > 4],
    ]
    seen = {(gold.row, gold.col, gold.value)}
    for pool in pools:
        for action in pool:
            key = (action.row, action.col, action.value)
            if key in seen:
                continue
            selected.append(action)
            seen.add(key)
            if len(selected) >= max_actions:
                return selected
    return selected


@torch.no_grad()
def compare_fixed_first_actions(
    model: SigRegActionJEPA,
    world: SudokuWorld,
    state_np: np.ndarray,
    goal_np: np.ndarray,
    clue_mask: np.ndarray,
    detail_actions: list[WorldAction],
    *,
    gold_action: WorldAction,
    horizon: int,
    candidates: int,
    elites: int,
    iterations: int,
    smoothing: float,
    rng: np.random.Generator,
    device: torch.device,
) -> dict[str, Any]:
    all_actions = candidate_actions(world, state_np, clue_mask)
    candidates_out = []
    for action in detail_actions:
        plan = cem_optimize_action_sequence(
            model,
            world,
            state_np,
            goal_np,
            state_np,
            all_actions,
            horizon=horizon,
            score_mode="latent_goal",
            candidates=candidates,
            elites=elites,
            iterations=iterations,
            smoothing=smoothing,
            rng=rng,
            device=device,
            fixed_first_action=action,
        )
        symbolic = apply_symbolic_sequence(world, state_np, clue_mask, action, all_actions, plan["indices"][1:])
        candidates_out.append(
            {
                "row": action.row,
                "col": action.col,
                "value": action.value,
                "is_gold": bool(
                    action.row == gold_action.row and action.col == gold_action.col and action.value == gold_action.value
                ),
                "is_goal_value": bool(action.value == int(goal_np[action.row, action.col])),
                "latent_goal_score": float(plan["score"]),
                "symbolic_remaining_hamming": int(np.not_equal(symbolic, goal_np).sum()),
                "symbolic_terminal": bool(np.count_nonzero(symbolic == 0) == 0),
            }
        )
    candidates_out.sort(key=lambda item: item["latent_goal_score"])
    return {
        "gold": {"row": gold_action.row, "col": gold_action.col, "value": gold_action.value},
        "horizon": int(horizon),
        "candidates": candidates_out,
    }


def apply_symbolic_sequence(
    world: SudokuWorld,
    state_np: np.ndarray,
    clue_mask: np.ndarray,
    first_action: WorldAction,
    all_actions: list[WorldAction],
    suffix_indices: list[int],
) -> np.ndarray:
    board = world.apply(state_np, first_action, clue_mask=clue_mask, allow_overwrite=True, allow_conflicts=True)
    for index in suffix_indices:
        board = world.apply(board, all_actions[int(index)], clue_mask=clue_mask, allow_overwrite=True, allow_conflicts=True)
    return board


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizons", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    parser.add_argument("--plan-examples", type=int, default=4)
    parser.add_argument("--max-mpc-steps", type=int, default=64)
    parser.add_argument("--candidates", type=int, default=300)
    parser.add_argument("--elites", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--smoothing", type=float, default=0.7)
    parser.add_argument("--qualitative-examples", type=int, default=3)
    parser.add_argument("--qualitative-candidates", type=int, default=6)
    parser.add_argument("--qualitative-cem-candidates", type=int, default=128)
    parser.add_argument("--qualitative-cem-iterations", type=int, default=5)
    args = parser.parse_args()
    summary = run_grid5_mpc_cem_diagnostics(
        args.run_root,
        args.output_dir,
        seed=args.seed,
        horizons=tuple(args.horizons),
        plan_examples=args.plan_examples,
        max_mpc_steps=args.max_mpc_steps,
        candidates=args.candidates,
        elites=args.elites,
        iterations=args.iterations,
        smoothing=args.smoothing,
        qualitative_examples=args.qualitative_examples,
        qualitative_candidates=args.qualitative_candidates,
        qualitative_cem_candidates=args.qualitative_cem_candidates,
        qualitative_cem_iterations=args.qualitative_cem_iterations,
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
