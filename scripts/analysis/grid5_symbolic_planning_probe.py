from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from puzzle_jepa.data import SudokuWorld, WorldAction, sample_random_mutable_rollout_transition
from puzzle_jepa.eval.grid5_diagnostics import candidate_actions, load_model
from puzzle_jepa.models import SigRegActionJEPA
from puzzle_jepa.train.grid0 import _build_world, _load_examples


@dataclass(frozen=True, slots=True)
class RunSpec:
    name: str
    root: Path


def parse_run_specs(values: list[str]) -> list[RunSpec]:
    specs = []
    for value in values:
        if "=" in value:
            name, raw_path = value.split("=", 1)
            root = Path(raw_path)
        else:
            root = Path(value)
            name = root.name
        specs.append(RunSpec(name=name, root=root))
    return specs


def fill_empty_actions(world: SudokuWorld, state: np.ndarray, clue_mask: np.ndarray) -> list[WorldAction]:
    return world.legal_actions(
        state,
        clue_mask=clue_mask,
        allow_overwrite=False,
        allow_conflicts=True,
    )


def mutable_overwrite_actions(world: SudokuWorld, state: np.ndarray, clue_mask: np.ndarray) -> list[WorldAction]:
    return candidate_actions(world, state, clue_mask)


def action_pool(world: SudokuWorld, state: np.ndarray, clue_mask: np.ndarray, action_space: str) -> list[WorldAction]:
    if action_space == "fill_empty":
        return fill_empty_actions(world, state, clue_mask)
    if action_space == "overwrite":
        return mutable_overwrite_actions(world, state, clue_mask)
    raise ValueError(f"unknown action_space {action_space!r}.")


@torch.no_grad()
def score_states(
    model: SigRegActionJEPA,
    states_np: np.ndarray,
    goal_np: np.ndarray,
    initial_np: np.ndarray,
    score_mode: str,
    device: torch.device,
) -> np.ndarray:
    if score_mode == "true_hamming":
        return np.not_equal(states_np, goal_np[None]).reshape(states_np.shape[0], -1).mean(axis=1).astype(np.float32)
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


def apply_sequences_vectorized(
    state_np: np.ndarray,
    actions: list[WorldAction],
    sampled: np.ndarray,
) -> np.ndarray:
    boards = np.repeat(state_np[None], sampled.shape[0], axis=0)
    if sampled.shape[1] == 0:
        return boards
    rows = np.asarray([action.row for action in actions], dtype=np.int64)
    cols = np.asarray([action.col for action in actions], dtype=np.int64)
    values = np.asarray([action.value for action in actions], dtype=np.int64)
    batch_index = np.arange(sampled.shape[0])
    for step in range(sampled.shape[1]):
        indices = sampled[:, step]
        boards[batch_index, rows[indices], cols[indices]] = values[indices]
    return boards


@torch.no_grad()
def symbolic_cem_optimize(
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
) -> dict[str, Any]:
    action_count = len(actions)
    if action_count <= 0:
        raise ValueError("symbolic CEM needs at least one action.")
    horizon = max(1, int(horizon))
    probs = np.full((horizon, action_count), 1.0 / float(action_count), dtype=np.float64)
    best_indices = np.zeros(horizon, dtype=np.int64)
    best_score = float("inf")
    best_remaining = int(np.not_equal(state_np, goal_np).sum())
    elite_count = max(1, min(int(elites), int(candidates)))

    for _ in range(max(1, int(iterations))):
        sampled = np.empty((int(candidates), horizon), dtype=np.int64)
        for step in range(horizon):
            sampled[:, step] = rng.choice(action_count, size=int(candidates), replace=True, p=probs[step])
        boards = apply_sequences_vectorized(state_np, actions, sampled)
        scores = score_states(model, boards, goal_np, initial_np, score_mode, device)
        order = np.argsort(scores)[:elite_count]
        if float(scores[order[0]]) < best_score:
            best_score = float(scores[order[0]])
            best_indices = sampled[order[0]].copy()
            best_remaining = int(np.not_equal(boards[order[0]], goal_np).sum())
        elite = sampled[order]
        for step in range(horizon):
            counts = np.bincount(elite[:, step], minlength=action_count).astype(np.float64)
            new_probs = (counts + 1.0e-3) / (counts.sum() + 1.0e-3 * action_count)
            probs[step] = float(smoothing) * new_probs + (1.0 - float(smoothing)) * probs[step]
            probs[step] /= probs[step].sum()

    first = actions[int(best_indices[0])]
    return {
        "indices": best_indices.tolist(),
        "score": best_score,
        "sequence_remaining_hamming": best_remaining,
        "root_action": {
            "row": int(first.row),
            "col": int(first.col),
            "value": int(first.value),
            "is_goal_value": bool(first.value == int(goal_np[first.row, first.col])),
        },
    }


@torch.no_grad()
def symbolic_mpc_plan(
    model: SigRegActionJEPA,
    world: SudokuWorld,
    example,
    *,
    horizon: int | str,
    score_mode: str,
    action_space: str,
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
    start_hamming = int(np.not_equal(board, example.goal).sum())
    root_action = None
    first_sequence_remaining = None
    best_score = float("inf")
    steps_taken = 0

    for step in range(max_steps):
        if world.is_goal(board, example.goal):
            break
        actions = action_pool(world, board, clue_mask, action_space)
        if not actions:
            break
        blanks = int(np.count_nonzero(board == 0))
        if horizon == "full":
            plan_horizon = max(1, blanks)
        else:
            plan_horizon = int(horizon)
            if action_space == "fill_empty":
                plan_horizon = max(1, min(plan_horizon, blanks))
        plan = symbolic_cem_optimize(
            model,
            world,
            board,
            example.goal,
            example.state,
            actions,
            horizon=plan_horizon,
            score_mode=score_mode,
            candidates=candidates,
            elites=elites,
            iterations=iterations,
            smoothing=smoothing,
            rng=rng,
            device=device,
        )
        first = actions[int(plan["indices"][0])]
        if step == 0:
            root_action = plan["root_action"]
            first_sequence_remaining = int(plan["sequence_remaining_hamming"])
        board = world.apply(
            board,
            first,
            clue_mask=clue_mask,
            allow_overwrite=(action_space == "overwrite"),
            allow_conflicts=True,
        )
        best_score = float(plan["score"])
        steps_taken = step + 1

    return {
        "start_hamming": start_hamming,
        "remaining_hamming": int(np.not_equal(board, example.goal).sum()),
        "terminal": bool(np.count_nonzero(board == 0) == 0),
        "solved": bool(world.is_goal(board, example.goal)),
        "steps": int(steps_taken),
        "best_score": best_score,
        "root_action": root_action,
        "root_goal_value": bool(root_action["is_goal_value"]) if root_action is not None else False,
        "first_sequence_remaining_hamming": first_sequence_remaining,
    }


def summarize_plan_records(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {}
    return {
        "count": float(len(records)),
        "start_hamming_mean": float(np.mean([r["start_hamming"] for r in records])),
        "solve_rate": float(np.mean([r["solved"] for r in records])),
        "solves": float(np.sum([r["solved"] for r in records])),
        "terminal_rate": float(np.mean([r["terminal"] for r in records])),
        "mean_remaining_hamming": float(np.mean([r["remaining_hamming"] for r in records])),
        "root_goal_value_rate": float(np.mean([r["root_goal_value"] for r in records])),
        "mean_first_sequence_remaining_hamming": float(
            np.mean([r["first_sequence_remaining_hamming"] for r in records if r["first_sequence_remaining_hamming"] is not None])
        ),
        "mean_steps": float(np.mean([r["steps"] for r in records])),
    }


@torch.no_grad()
def random_rollout_drift(
    model: SigRegActionJEPA,
    world: SudokuWorld,
    examples,
    *,
    horizons: list[int],
    count: int,
    rng: np.random.Generator,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    max_horizon = max(horizons)
    rollouts = [
        sample_random_mutable_rollout_transition(
            world,
            examples[int(rng.integers(0, len(examples)))],
            rng,
            steps=max_horizon,
        )
        for _ in range(max(1, count))
    ]
    starts = torch.as_tensor(np.stack([r.state for r in rollouts]), dtype=torch.long, device=device)
    actions = torch.as_tensor(
        np.stack([[action.as_array(world.task_id) for action in r.actions] for r in rollouts]),
        dtype=torch.long,
        device=device,
    )
    targets = torch.as_tensor(
        np.stack([np.stack(r.target_states) for r in rollouts]),
        dtype=torch.long,
        device=device,
    )
    goals = torch.as_tensor(np.stack([r.goal for r in rollouts]), dtype=torch.long, device=device)
    target_latents = model.encode(targets.reshape(-1, *targets.shape[-2:])).reshape(
        targets.shape[0],
        targets.shape[1],
        model.latent_size,
    )
    goal_latents = model.encode(goals)
    current = model.encode(starts)
    history = [current]
    summaries: dict[str, dict[str, float]] = {}
    horizon_set = set(int(h) for h in horizons)
    max_history = max(1, int(model.max_rollout_steps))
    for step in range(max_horizon):
        if model.predictor_type == "ar_transformer":
            start = max(0, len(history) - max_history)
            latent_window = torch.stack(history[start:], dim=1)
            action_start = max(0, step + 1 - latent_window.shape[1])
            action_window = actions[:, action_start : step + 1]
            current = model.predict_sequence(latent_window, action_window)[:, -1]
        else:
            current = model.predict_next(current, actions[:, step])
        history.append(current)
        horizon = step + 1
        if horizon not in horizon_set:
            continue
        target = target_latents[:, step]
        diff = current - target
        mse = diff.pow(2).mean(dim=-1)
        l2 = diff.norm(dim=-1)
        target_goal_mse = F.mse_loss(target, goal_latents, reduction="none").mean(dim=-1)
        pred_goal_mse = F.mse_loss(current, goal_latents, reduction="none").mean(dim=-1)
        summaries[str(horizon)] = {
            "count": float(len(rollouts)),
            "latent_drift_mse_mean": float(mse.mean().cpu()),
            "latent_drift_mse_p90": float(torch.quantile(mse.float(), 0.9).cpu()),
            "latent_drift_l2_mean": float(l2.mean().cpu()),
            "latent_drift_l2_p90": float(torch.quantile(l2.float(), 0.9).cpu()),
            "target_goal_mse_mean": float(target_goal_mse.mean().cpu()),
            "pred_goal_mse_mean": float(pred_goal_mse.mean().cpu()),
        }
    return summaries


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(value) for value in row) + " |")
    return "\n".join(lines)


def format_cell(value: Any) -> str:
    if isinstance(value, float):
        if abs(value) >= 100:
            return f"{value:.1f}"
        if abs(value) >= 10:
            return f"{value:.2f}"
        return f"{value:.4f}"
    return str(value)


def write_markdown(output_dir: Path, payload: dict[str, Any]) -> None:
    plan_rows = []
    for item in payload["symbolic_planning"]:
        summary = item["summary"]
        plan_rows.append(
            [
                item["run_name"],
                item["action_space"],
                item["score_mode"],
                item["horizon"],
                summary["solves"],
                summary["count"],
                summary["solve_rate"],
                summary["terminal_rate"],
                summary["mean_remaining_hamming"],
                summary["root_goal_value_rate"],
                summary["mean_first_sequence_remaining_hamming"],
            ]
        )
    drift_rows = []
    for item in payload["drift"]:
        for horizon, summary in item["summary"].items():
            drift_rows.append(
                [
                    item["run_name"],
                    horizon,
                    summary["latent_drift_mse_mean"],
                    summary["latent_drift_l2_mean"],
                    summary["latent_drift_mse_p90"],
                    summary["target_goal_mse_mean"],
                    summary["pred_goal_mse_mean"],
                ]
            )
    overview_rows = []
    for item in payload["run_overview"]:
        overview_rows.append(
            [
                item["run_name"],
                item["encoder_type"],
                item["predictor_type"],
                item["predict_delta"],
                item["latent_size"],
                item["param_count"],
                item["start_hamming_mean"],
            ]
        )
    text = "\n\n".join(
        [
            "# Grid 5 Symbolic Planning Probe",
            "## Run Overview",
            markdown_table(
                ["run", "encoder", "predictor", "delta", "z", "params", "start Hamming"],
                overview_rows,
            ),
            "## Symbolic MPC-CEM",
            markdown_table(
                [
                    "run",
                    "actions",
                    "score",
                    "horizon",
                    "solves",
                    "n",
                    "solve rate",
                    "terminal rate",
                    "remaining Hamming",
                    "root goal value",
                    "first seq Hamming",
                ],
                plan_rows,
            ),
            "## Random Rollout Drift",
            markdown_table(
                [
                    "run",
                    "K",
                    "drift MSE",
                    "drift L2",
                    "drift MSE p90",
                    "true z_K-goal MSE",
                    "pred z_K-goal MSE",
                ],
                drift_rows,
            ),
        ]
    )
    (output_dir / "summary.md").write_text(text + "\n")


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(int(args.seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "device": str(device),
        "seed": int(args.seed),
        "run_overview": [],
        "symbolic_planning": [],
        "drift": [],
    }
    for spec in parse_run_specs(args.run_root):
        model, config = load_model(spec.root, device)
        world = _build_world(dict(config["task"]))
        if not isinstance(world, SudokuWorld):
            raise ValueError("This probe currently supports Sudoku Grid 5 runs only.")
        examples = _load_examples(dict(config["task"]), "eval")
        eval_indices = [int(rng.integers(0, len(examples))) for _ in range(max(1, int(args.plan_examples)))]
        start_hamming = [int(np.not_equal(examples[index].state, examples[index].goal).sum()) for index in eval_indices]
        payload["run_overview"].append(
            {
                "run_name": spec.name,
                "run_root": str(spec.root),
                "encoder_type": config["model"]["encoder_type"],
                "predictor_type": config["model"]["predictor_type"],
                "predict_delta": bool(config["model"].get("predict_delta", False)),
                "latent_size": int(config["model"]["latent_size"]),
                "param_count": int(sum(param.numel() for param in model.parameters())),
                "start_hamming_mean": float(np.mean(start_hamming)),
            }
        )
        for action_space in args.action_space:
            for score_mode in args.score_mode:
                for raw_horizon in args.horizon:
                    horizon: int | str = "full" if str(raw_horizon).lower() == "full" else int(raw_horizon)
                    records = []
                    local_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
                    for index in eval_indices:
                        records.append(
                            symbolic_mpc_plan(
                                model,
                                world,
                                examples[index],
                                horizon=horizon,
                                score_mode=score_mode,
                                action_space=action_space,
                                max_steps=int(args.max_mpc_steps),
                                candidates=int(args.candidates),
                                elites=int(args.elites),
                                iterations=int(args.iterations),
                                smoothing=float(args.smoothing),
                                rng=local_rng,
                                device=device,
                            )
                        )
                    payload["symbolic_planning"].append(
                        {
                            "run_name": spec.name,
                            "action_space": action_space,
                            "score_mode": score_mode,
                            "horizon": str(horizon),
                            "records": records,
                            "summary": summarize_plan_records(records),
                        }
                    )
        drift_summary = random_rollout_drift(
            model,
            world,
            examples,
            horizons=[int(h) for h in args.drift_horizon],
            count=int(args.drift_examples),
            rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
            device=device,
        )
        payload["drift"].append({"run_name": spec.name, "summary": drift_summary})
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    write_markdown(output_dir, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", action="append", required=True, help="Run root or name=/path. Repeatable.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--plan-examples", type=int, default=8)
    parser.add_argument("--horizon", nargs="+", default=["8", "16", "32", "64", "full"])
    parser.add_argument("--score-mode", nargs="+", default=["latent_goal", "goal_energy"])
    parser.add_argument("--action-space", nargs="+", default=["fill_empty"])
    parser.add_argument("--max-mpc-steps", type=int, default=64)
    parser.add_argument("--candidates", type=int, default=256)
    parser.add_argument("--elites", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--smoothing", type=float, default=0.7)
    parser.add_argument("--drift-horizon", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--drift-examples", type=int, default=64)
    args = parser.parse_args()
    payload = run_probe(args)
    print(json.dumps({key: payload[key] for key in ("device", "seed", "run_overview")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
