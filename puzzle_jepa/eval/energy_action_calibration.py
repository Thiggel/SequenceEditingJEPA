from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from puzzle_jepa.data.worlds import PuzzleExample, SudokuWorld, WorldAction
from puzzle_jepa.eval.diagnostics import (
    apply_planning_action,
    build_world,
    clue_mask_for_planning,
    legal_planning_actions,
    load_examples,
    load_model_checkpoint_state,
    mean,
    oracle_action_sequence,
    score_symbolic_states_to_goal,
)
from puzzle_jepa.models import ActionConditionedWorldModel


def run_energy_action_calibration(
    run_root: Path,
    *,
    output_dir: Path | None,
    num_examples: int,
    max_steps: int,
    qualitative_examples: int,
    qualitative_actions: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    checkpoint_path = run_root / "checkpoint.pt"
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    task_cfg = dict(config["task"])
    world = build_world(task_cfg)
    if not isinstance(world, SudokuWorld):
        raise ValueError("Energy action calibration currently expects Sudoku.")
    examples = load_examples(task_cfg, "eval")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActionConditionedWorldModel(vocab_size=world.vocab_size, **dict(config["model"])).to(device)
    load_model_checkpoint_state(model, checkpoint["model"])
    model.eval()

    selected = rng.choice(len(examples), size=min(num_examples, len(examples)), replace=False)
    step_records: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    qualitative: list[dict[str, Any]] = []
    for local_index, example_index in enumerate(selected.tolist()):
        example = examples[int(example_index)]
        records, candidates, examples_out = analyze_example(
            model,
            world,
            example,
            example_index=int(example_index),
            local_index=local_index,
            rng=rng,
            max_steps=max_steps,
            qualitative=local_index < qualitative_examples,
            qualitative_actions=qualitative_actions,
        )
        step_records.extend(records)
        candidate_records.extend(candidates)
        qualitative.extend(examples_out)

    summary = summarize(step_records)
    result = {
        "run_root": str(run_root),
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "num_examples": int(num_examples),
        "max_steps": int(max_steps),
        "records": summary,
    }
    destination = output_dir or (run_root / "energy_action_calibration")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (destination / "qualitative_examples.json").write_text(json.dumps(qualitative, indent=2, sort_keys=True) + "\n")
    with (destination / "step_records.jsonl").open("w") as handle:
        for record in step_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    with (destination / "candidate_records.jsonl").open("w") as handle:
        for record in candidate_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


@torch.no_grad()
def analyze_example(
    model: ActionConditionedWorldModel,
    world: SudokuWorld,
    example: PuzzleExample,
    *,
    example_index: int,
    local_index: int,
    rng: np.random.Generator,
    max_steps: int,
    qualitative: bool,
    qualitative_actions: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    start = world.validate_state(example.state).copy()
    goal = world.validate_state(example.goal)
    clue_mask = clue_mask_for_planning(world, start)
    current = start.copy()
    oracle_actions = oracle_action_sequence(world, example, rng)
    step_records: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    qualitative_records: list[dict[str, Any]] = []
    for step, gold_action in enumerate(oracle_actions[: max(0, int(max_steps))]):
        if world.is_goal(current, goal):
            break
        actions, states = enumerate_successors(world, current, clue_mask)
        if not actions:
            break
        predicted_scores = np.asarray(
            score_symbolic_states_to_goal(model, world, states, goal, start, planning_score="goal_energy"),
            dtype=np.float64,
        )
        true_scores = np.asarray(
            score_symbolic_states_to_goal(model, world, states, goal, start, planning_score="latent_goal"),
            dtype=np.float64,
        )
        predicted_energy = -predicted_scores
        true_energy = -true_scores
        errors = predicted_energy - true_energy
        order = np.argsort(predicted_energy).tolist()
        best_index = int(order[0])
        gold_index = next((index for index, action in enumerate(actions) if action == gold_action), None)
        best_true_index = int(np.argsort(true_energy)[0])
        record = {
            "example_index": example_index,
            "local_index": local_index,
            "step": int(step),
            "remaining_before": int(np.not_equal(current, goal).sum()),
            "legal_actions": int(len(actions)),
            "best_predicted_action": action_dict(actions[best_index]),
            "best_predicted_energy": float(predicted_energy[best_index]),
            "best_predicted_true_energy": float(true_energy[best_index]),
            "best_predicted_abs_error": float(abs(errors[best_index])),
            "best_true_action": action_dict(actions[best_true_index]),
            "best_true_energy": float(true_energy[best_true_index]),
            "all_mean_abs_error": float(np.abs(errors).mean()),
            "all_median_abs_error": float(np.median(np.abs(errors))),
            "all_max_abs_error": float(np.abs(errors).max()),
            "pearson_pred_true": pearson(predicted_energy, true_energy),
        }
        if gold_index is not None:
            predicted_rank = np.argsort(predicted_energy).tolist().index(gold_index) + 1
            true_rank = np.argsort(true_energy).tolist().index(gold_index) + 1
            record.update(
                {
                    "gold_action": action_dict(gold_action),
                    "gold_predicted_energy": float(predicted_energy[gold_index]),
                    "gold_true_energy": float(true_energy[gold_index]),
                    "gold_abs_error": float(abs(errors[gold_index])),
                    "gold_predicted_rank": int(predicted_rank),
                    "gold_true_rank": int(true_rank),
                }
            )
        step_records.append(record)
        for index, action in enumerate(actions):
            candidate_records.append(
                {
                    "example_index": example_index,
                    "local_index": local_index,
                    "step": int(step),
                    "action": action_dict(action),
                    "category": candidate_category(action, gold_action, current, goal),
                    "predicted_energy": float(predicted_energy[index]),
                    "true_energy": float(true_energy[index]),
                    "error": float(errors[index]),
                    "abs_error": float(abs(errors[index])),
                    "predicted_rank": int(np.argsort(predicted_energy).tolist().index(index) + 1),
                    "true_rank": int(np.argsort(true_energy).tolist().index(index) + 1),
                }
            )
        if qualitative:
            qualitative_records.append(
                qualitative_step_record(
                    example_index=example_index,
                    local_index=local_index,
                    step=step,
                    current=current,
                    goal=goal,
                    actions=actions,
                    gold_action=gold_action,
                    best_index=best_index,
                    predicted_energy=predicted_energy,
                    true_energy=true_energy,
                    rng=rng,
                    max_extra_actions=qualitative_actions,
                )
            )
        current = apply_planning_action(world, current, gold_action, clue_mask)
    return step_records, candidate_records, qualitative_records


def enumerate_successors(
    world: SudokuWorld,
    current: np.ndarray,
    clue_mask: np.ndarray | None,
) -> tuple[list[WorldAction], list[np.ndarray]]:
    actions: list[WorldAction] = []
    states: list[np.ndarray] = []
    for action in legal_planning_actions(world, current, clue_mask):
        try:
            states.append(apply_planning_action(world, current, action, clue_mask))
            actions.append(action)
        except ValueError:
            continue
    return actions, states


def qualitative_step_record(
    *,
    example_index: int,
    local_index: int,
    step: int,
    current: np.ndarray,
    goal: np.ndarray,
    actions: list[WorldAction],
    gold_action: WorldAction,
    best_index: int,
    predicted_energy: np.ndarray,
    true_energy: np.ndarray,
    rng: np.random.Generator,
    max_extra_actions: int,
) -> dict[str, Any]:
    chosen = [best_index]
    gold_index = next((index for index, action in enumerate(actions) if action == gold_action), None)
    if gold_index is not None:
        chosen.append(gold_index)
    chosen.extend(sample_neighbor_indices(actions, best_index, rng, max_extra_actions=max_extra_actions))
    deduped: list[int] = []
    for index in chosen:
        if index not in deduped:
            deduped.append(index)
    return {
        "example_index": example_index,
        "local_index": local_index,
        "step": int(step),
        "remaining_before": int(np.not_equal(current, goal).sum()),
        "board": current.tolist(),
        "goal": goal.tolist(),
        "gold_action": action_dict(gold_action),
        "actions": [
            {
                "role": action_role(index, best_index, gold_index),
                "action": action_dict(actions[index]),
                "category": candidate_category(actions[index], gold_action, current, goal),
                "predicted_energy": float(predicted_energy[index]),
                "true_energy": float(true_energy[index]),
                "error": float(predicted_energy[index] - true_energy[index]),
                "abs_error": float(abs(predicted_energy[index] - true_energy[index])),
            }
            for index in deduped
        ],
    }


def sample_neighbor_indices(
    actions: list[WorldAction],
    anchor_index: int,
    rng: np.random.Generator,
    *,
    max_extra_actions: int,
) -> list[int]:
    anchor = actions[anchor_index]
    same_cell = [
        index
        for index, action in enumerate(actions)
        if index != anchor_index and action.row == anchor.row and action.col == anchor.col
    ]
    adjacent = [
        index
        for index, action in enumerate(actions)
        if index != anchor_index
        and max(abs(action.row - anchor.row), abs(action.col - anchor.col)) == 1
    ]
    far = [
        index
        for index, action in enumerate(actions)
        if index != anchor_index and max(abs(action.row - anchor.row), abs(action.col - anchor.col)) > 1
    ]
    sampled: list[int] = []
    for pool, count in ((same_cell, 2), (adjacent, 2), (far, 2)):
        if not pool:
            continue
        size = min(count, len(pool), max(0, max_extra_actions - len(sampled)))
        if size <= 0:
            break
        sampled.extend(rng.choice(pool, size=size, replace=False).astype(int).tolist())
    return sampled


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"count": 0}
    by_step: dict[str, dict[str, float]] = {}
    for step in sorted({int(item["step"]) for item in records}):
        items = [item for item in records if int(item["step"]) == step]
        by_step[str(step)] = {
            "count": float(len(items)),
            "mean_best_predicted_abs_error": mean([item["best_predicted_abs_error"] for item in items]),
            "mean_all_abs_error": mean([item["all_mean_abs_error"] for item in items]),
            "mean_gold_abs_error": mean([item.get("gold_abs_error", 0.0) for item in items]),
            "mean_pearson_pred_true": mean([item["pearson_pred_true"] for item in items]),
        }
    return {
        "count": len(records),
        "mean_best_predicted_abs_error": mean([item["best_predicted_abs_error"] for item in records]),
        "mean_all_abs_error": mean([item["all_mean_abs_error"] for item in records]),
        "mean_gold_abs_error": mean([item.get("gold_abs_error", 0.0) for item in records]),
        "mean_pearson_pred_true": mean([item["pearson_pred_true"] for item in records]),
        "by_step": by_step,
    }


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2:
        return 0.0
    left_std = float(left.std())
    right_std = float(right.std())
    if left_std <= 1.0e-12 or right_std <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def candidate_category(action: WorldAction, gold_action: WorldAction, current: np.ndarray, goal: np.ndarray) -> str:
    if action == gold_action:
        return "gold"
    if action.row == gold_action.row and action.col == gold_action.col:
        return "same_cell_wrong_value"
    if int(current[action.row, action.col]) != int(goal[action.row, action.col]) and action.value == int(
        goal[action.row, action.col]
    ):
        return "other_cell_goal_value"
    return "other_cell_wrong_value"


def action_role(index: int, best_index: int, gold_index: int | None) -> str:
    if index == best_index and index == gold_index:
        return "best_predicted_and_gold"
    if index == best_index:
        return "best_predicted"
    if index == gold_index:
        return "gold"
    return "sampled_context"


def action_dict(action: WorldAction) -> dict[str, int]:
    return {"row": int(action.row), "col": int(action.col), "value": int(action.value)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare learned scalar energy against true latent goal energy.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--examples", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--qualitative-examples", type=int, default=3)
    parser.add_argument("--qualitative-actions", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_energy_action_calibration(
        args.run_root,
        output_dir=args.output_dir,
        num_examples=args.examples,
        max_steps=args.max_steps,
        qualitative_examples=args.qualitative_examples,
        qualitative_actions=args.qualitative_actions,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
