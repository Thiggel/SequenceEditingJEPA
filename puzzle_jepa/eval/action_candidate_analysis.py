from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from puzzle_jepa.data.worlds import PuzzleExample, PuzzleWorld, SudokuWorld, WorldAction
from puzzle_jepa.eval.diagnostics import (
    apply_planning_action,
    build_world,
    clue_mask_for_planning,
    legal_planning_actions,
    load_examples,
    load_model_checkpoint_state,
    mean,
    median,
    oracle_action_sequence,
    score_symbolic_states_to_goal,
)
from puzzle_jepa.models import ActionConditionedWorldModel


def run_action_candidate_analysis(
    run_root: Path,
    *,
    output_dir: Path | None,
    num_examples: int,
    max_steps: int,
    score_mode: str,
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
        raise ValueError("Action candidate analysis currently expects Sudoku.")
    examples = load_examples(task_cfg, "eval")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActionConditionedWorldModel(vocab_size=world.vocab_size, **dict(config["model"])).to(device)
    load_model_checkpoint_state(model, checkpoint["model"])
    model.eval()

    records: list[dict[str, Any]] = []
    selected_indices = rng.choice(len(examples), size=min(num_examples, len(examples)), replace=False)
    for local_index, example_index in enumerate(selected_indices.tolist()):
        example = examples[int(example_index)]
        records.extend(
            analyze_example(
                model,
                world,
                example,
                example_index=int(example_index),
                local_index=local_index,
                rng=rng,
                max_steps=max_steps,
                score_mode=score_mode,
            )
        )

    summary = summarize_records(records)
    result = {
        "run_root": str(run_root),
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "score_mode": score_mode,
        "num_examples": int(num_examples),
        "max_steps": int(max_steps),
        "records": summary,
    }
    destination = output_dir or (run_root / f"action_candidate_analysis_{score_mode}")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (destination / "records.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


@torch.no_grad()
def analyze_example(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    example: PuzzleExample,
    *,
    example_index: int,
    local_index: int,
    rng: np.random.Generator,
    max_steps: int,
    score_mode: str,
) -> list[dict[str, Any]]:
    start = world.validate_state(example.state).copy()
    goal = world.validate_state(example.goal)
    clue_mask = clue_mask_for_planning(world, start)
    oracle_actions = oracle_action_sequence(world, example, rng)
    current = start.copy()
    records: list[dict[str, Any]] = []
    initial_blanks = int((start == 0).sum())
    for step, gold_action in enumerate(oracle_actions[: max(0, int(max_steps))]):
        if world.is_goal(current, goal):
            break
        legal = legal_planning_actions(world, current, clue_mask)
        candidate_states: list[np.ndarray] = []
        candidate_actions: list[WorldAction] = []
        categories: list[str] = []
        gold_index: int | None = None
        for action in legal:
            try:
                next_state = apply_planning_action(world, current, action, clue_mask)
            except ValueError:
                continue
            category = categorize_action(action, gold_action, current, goal)
            if action == gold_action:
                gold_index = len(candidate_actions)
                category = "gold"
            candidate_actions.append(action)
            candidate_states.append(next_state)
            categories.append(category)
        if gold_index is None:
            break
        scores = score_symbolic_states_to_goal(
            model,
            world,
            candidate_states,
            goal,
            start,
            planning_score=score_mode,
        )
        score_array = np.asarray(scores, dtype=np.float64)
        order = np.argsort(score_array)[::-1].tolist()
        gold_rank = order.index(gold_index) + 1
        gold_score = float(score_array[gold_index])
        best_index = int(order[0])
        category_stats = summarize_categories(score_array, categories, gold_score)
        records.append(
            {
                "example_index": example_index,
                "local_index": local_index,
                "step": int(step),
                "remaining_before": int(np.not_equal(current, goal).sum()),
                "initial_blanks": initial_blanks,
                "gold_action": action_dict(gold_action),
                "gold_rank": int(gold_rank),
                "legal_actions": int(len(candidate_actions)),
                "gold_score": gold_score,
                "best_score": float(score_array[best_index]),
                "best_action": action_dict(candidate_actions[best_index]),
                "best_category": categories[best_index],
                "best_minus_gold": float(score_array[best_index] - gold_score),
                "gold_beats_all_negatives": float(gold_rank == 1),
                "same_cell_wrong": category_stats.get("same_cell_wrong_value", empty_category_summary()),
                "other_cell_goal": category_stats.get("other_cell_goal_value", empty_category_summary()),
                "other_cell_wrong": category_stats.get("other_cell_wrong_value", empty_category_summary()),
                "all_negatives": category_stats.get("all_negatives", empty_category_summary()),
            }
        )
        current = apply_planning_action(world, current, gold_action, clue_mask)
    return records


def categorize_action(action: WorldAction, gold_action: WorldAction, current: np.ndarray, goal: np.ndarray) -> str:
    if action.row == gold_action.row and action.col == gold_action.col:
        return "same_cell_wrong_value"
    if int(current[action.row, action.col]) != int(goal[action.row, action.col]) and action.value == int(
        goal[action.row, action.col]
    ):
        return "other_cell_goal_value"
    return "other_cell_wrong_value"


def summarize_categories(score_array: np.ndarray, categories: list[str], gold_score: float) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for category in sorted(set(categories)):
        if category == "gold":
            continue
        indices = [index for index, item in enumerate(categories) if item == category]
        output[category] = category_summary(score_array, indices, gold_score)
    negative_indices = [index for index, item in enumerate(categories) if item != "gold"]
    output["all_negatives"] = category_summary(score_array, negative_indices, gold_score)
    return output


def category_summary(score_array: np.ndarray, indices: list[int], gold_score: float) -> dict[str, Any]:
    if not indices:
        return empty_category_summary()
    values = score_array[indices]
    best = float(values.max())
    return {
        "count": int(len(indices)),
        "best_score": best,
        "mean_score": float(values.mean()),
        "best_minus_gold": float(best - gold_score),
        "gold_margin": float(gold_score - best),
        "gold_beats_category": float(gold_score > best),
        "num_above_gold": int((values > gold_score).sum()),
        "frac_above_gold": float((values > gold_score).mean()),
    }


def empty_category_summary() -> dict[str, Any]:
    return {
        "count": 0,
        "best_score": None,
        "mean_score": None,
        "best_minus_gold": None,
        "gold_margin": None,
        "gold_beats_category": None,
        "num_above_gold": 0,
        "frac_above_gold": None,
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"count": 0}
    summary = {
        "count": len(records),
        "top1": mean([item["gold_beats_all_negatives"] for item in records]),
        "top5": mean([float(item["gold_rank"] <= 5) for item in records]),
        "mrr": mean([1.0 / float(item["gold_rank"]) for item in records]),
        "mean_rank": mean([item["gold_rank"] for item in records]),
        "median_rank": median([item["gold_rank"] for item in records]),
        "mean_legal_actions": mean([item["legal_actions"] for item in records]),
        "mean_best_minus_gold": mean([item["best_minus_gold"] for item in records]),
    }
    for category in ("same_cell_wrong", "other_cell_goal", "other_cell_wrong", "all_negatives"):
        available = [item[category] for item in records if item[category]["count"]]
        summary[category] = {
            "mean_count": mean([item["count"] for item in available]) if available else 0.0,
            "gold_beats_rate": mean([item["gold_beats_category"] for item in available]) if available else 0.0,
            "mean_gold_margin": mean([item["gold_margin"] for item in available]) if available else 0.0,
            "mean_frac_above_gold": mean([item["frac_above_gold"] for item in available]) if available else 0.0,
        }
    by_step: dict[str, dict[str, float]] = {}
    step_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        step_groups[int(record["step"])].append(record)
    for step in sorted(step_groups)[:81]:
        items = step_groups[step]
        by_step[str(step)] = {
            "count": float(len(items)),
            "top1": mean([item["gold_beats_all_negatives"] for item in items]),
            "mean_rank": mean([item["gold_rank"] for item in items]),
            "mean_best_minus_gold": mean([item["best_minus_gold"] for item in items]),
        }
    summary["by_step"] = by_step
    return summary


def action_dict(action: WorldAction) -> dict[str, int]:
    return {"row": int(action.row), "col": int(action.col), "value": int(action.value)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze gold-action ranking against all local Sudoku alternatives.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=81)
    parser.add_argument("--score-mode", choices=["goal_energy", "latent_goal"], default="goal_energy")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_action_candidate_analysis(
        args.run_root,
        output_dir=args.output_dir,
        num_examples=args.examples,
        max_steps=args.max_steps,
        score_mode=args.score_mode,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
