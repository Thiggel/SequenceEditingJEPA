from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from puzzle_jepa.data.hf_puzzles import HFPuzzleColumns, iter_hf_examples
from puzzle_jepa.data.worlds import PuzzleExample, SudokuWorld
from puzzle_jepa.eval.grid_goal_diagnostics import run_grid_goal_diagnostics
from puzzle_jepa.models.grid_goal_jepa import GridTokenGoalJEPA
from puzzle_jepa.planning.grid_goal_planner import run_beam_mpc


BEAM_DEPTHS = (8, 16, 32, 64)
BEAM_WIDTHS = (1, 4, 16, 64)
SCORES = ("oracle_goal_distance", "predicted_goal_distance")
TRANSITIONS = ("symbolic_reencode", "latent_rollout")


def load_checkpoint(path: Path, device: torch.device) -> tuple[GridTokenGoalJEPA, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = dict(payload["config"])
    model = GridTokenGoalJEPA(**dict(config["model"])).to(device)
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
    return list(
        iter_hf_examples(
            str(task_cfg["repo_id"]),
            str(task_cfg.get("eval_split", "test[:128]")),
            world,
            columns,
            limit=limit,
        )
    )


def run_planner_matrix(
    model: GridTokenGoalJEPA,
    examples: list[PuzzleExample],
    *,
    output_path: Path,
    device: torch.device,
    beam_widths: tuple[int, ...] = BEAM_WIDTHS,
    beam_depths: tuple[int, ...] = BEAM_DEPTHS,
    scores: tuple[str, ...] = SCORES,
    transitions: tuple[str, ...] = TRANSITIONS,
    max_examples: int = 16,
    max_steps: int = 81,
) -> list[dict[str, Any]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with output_path.open("w") as handle:
        for transition in transitions:
            for score in scores:
                for beam_width in beam_widths:
                    for beam_depth in beam_depths:
                        solved = 0
                        remaining = []
                        steps = []
                        action_evals = []
                        elapsed = []
                        for example in examples[:max_examples]:
                            result = run_beam_mpc(
                                model,
                                example.state,
                                example.goal,
                                score_mode=score,  # type: ignore[arg-type]
                                transition_mode=transition,  # type: ignore[arg-type]
                                beam_width=beam_width,
                                beam_depth=beam_depth,
                                max_steps=max_steps,
                                device=device,
                            )
                            solved += int(result.solved)
                            remaining.append(result.remaining_hamming)
                            steps.append(result.steps)
                            action_evals.append(result.action_evals)
                            elapsed.append(result.elapsed_seconds)
                        record = {
                            "planner": "mpc_beam",
                            "transition_mode": transition,
                            "score_mode": score,
                            "beam_width": beam_width,
                            "beam_depth": beam_depth,
                            "examples": min(max_examples, len(examples)),
                            "solved": solved,
                            "solve_rate": solved / max(1, min(max_examples, len(examples))),
                            "remaining_hamming_mean": float(np.mean(remaining)) if remaining else 0.0,
                            "steps_mean": float(np.mean(steps)) if steps else 0.0,
                            "action_evals_mean": float(np.mean(action_evals)) if action_evals else 0.0,
                            "elapsed_seconds_mean": float(np.mean(elapsed)) if elapsed else 0.0,
                        }
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
                        handle.flush()
                        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=81)
    parser.add_argument("--beam-widths", default="1,4,16,64")
    parser.add_argument("--beam-depths", default="8,16,32,64")
    parser.add_argument("--scores", default="oracle_goal_distance,predicted_goal_distance")
    parser.add_argument("--transitions", default="symbolic_reencode,latent_rollout")
    parser.add_argument("--skip-diagnostics", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_checkpoint(args.checkpoint, device)
    examples = load_eval_examples(config, limit=args.examples)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_diagnostics:
        diagnostics = run_grid_goal_diagnostics(model, examples, args.output_dir, device=device, seed=int(config.get("seed", 0)) + 900)
        print(json.dumps({"diagnostics": diagnostics}, sort_keys=True), flush=True)
    records = run_planner_matrix(
        model,
        examples,
        output_path=args.output_dir / "planner_matrix.jsonl",
        device=device,
        beam_widths=_parse_ints(args.beam_widths),
        beam_depths=_parse_ints(args.beam_depths),
        scores=tuple(x for x in args.scores.split(",") if x),
        transitions=tuple(x for x in args.transitions.split(",") if x),
        max_examples=args.examples,
        max_steps=args.max_steps,
    )
    print(json.dumps({"records": len(records), "output": str(args.output_dir)}, sort_keys=True), flush=True)


def _parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(item) for item in text.split(",") if item)


if __name__ == "__main__":
    main()
