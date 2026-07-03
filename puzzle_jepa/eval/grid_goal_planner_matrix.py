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
from puzzle_jepa.planning.grid_goal_planner import (
    run_beam_mpc,
    run_categorical_cem_mpc,
    run_hierarchical_beam_mpc,
    run_hierarchical_cem_mpc,
)


BEAM_DEPTHS = (8, 16, 32, 64)
BEAM_WIDTHS = (1, 4, 16, 64)
SCORES = ("oracle_goal_distance", "predicted_goal_distance")
TRANSITIONS = ("symbolic_reencode", "latent_rollout")
PLANNERS = ("mpc_beam",)


def load_checkpoint(path: Path, device: torch.device) -> tuple[GridTokenGoalJEPA, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = dict(payload["config"])
    model = GridTokenGoalJEPA(**dict(config["model"])).to(device)
    incompatible = model.load_state_dict(payload["model"], strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if missing or unexpected:
        model_cfg = dict(config["model"])
        delta_weight = float(model_cfg.get("delta_action_weight", 0.0) or 0.0)
        allowed_missing = {key for key in missing if key.startswith("delta_action_decoder.") and delta_weight <= 0.0}
        legacy_optional_prefixes = (
            "bad_state_head.",
            "metric_src_projector.",
            "metric_goal_projector.",
            "metric_success_tokens",
            "metric_value_head.",
            "target_metric_src_projector.",
            "target_metric_goal_projector.",
        )
        allowed_missing.update(
            key for key in missing if key.startswith(legacy_optional_prefixes)
        )
        if missing != allowed_missing or unexpected:
            raise RuntimeError(
                "Incompatible checkpoint state_dict: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
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
    planners: tuple[str, ...] = PLANNERS,
    max_examples: int = 16,
    max_steps: int = 81,
    cem_samples: int = 128,
    cem_iters: int = 4,
    cem_elites: int = 16,
    cem_momentum: float = 0.7,
    high_cem_samples: int = 128,
    high_cem_iters: int = 4,
    high_cem_elites: int = 16,
    high_cem_momentum: float = 0.7,
    high_cem_std: float = 1.0,
    high_cem_optimizer: str = "cem",
    high_cem_temperature: float = 1.0,
    high_cem_codebook: str = "none",
    high_cem_codebook_size: int = 0,
) -> list[dict[str, Any]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _read_completed_matrix_keys(output_path)
    records: list[dict[str, Any]] = []
    _ensure_append_starts_on_new_line(output_path)
    with output_path.open("a") as handle:
        for planner in planners:
            for transition in transitions:
                for score in scores:
                    for beam_width in beam_widths:
                        for beam_depth in beam_depths:
                            key = (planner, transition, score, beam_width, beam_depth)
                            if key in completed:
                                continue
                            solved = 0
                            remaining = []
                            steps = []
                            action_evals = []
                            elapsed = []
                            for example_index, example in enumerate(examples[:max_examples]):
                                if planner == "mpc_beam":
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
                                elif planner == "categorical_cem":
                                    result = run_categorical_cem_mpc(
                                        model,
                                        example.state,
                                        example.goal,
                                        score_mode=score,  # type: ignore[arg-type]
                                        transition_mode=transition,  # type: ignore[arg-type]
                                        beam_width=beam_width,
                                        beam_depth=beam_depth,
                                        max_steps=max_steps,
                                        device=device,
                                        cem_samples=cem_samples,
                                        cem_iters=cem_iters,
                                        cem_elites=cem_elites,
                                        cem_momentum=cem_momentum,
                                        seed=example_index,
                                    )
                                elif planner == "hierarchical_cem":
                                    result = run_hierarchical_cem_mpc(
                                        model,
                                        example.state,
                                        example.goal,
                                        score_mode=score,  # type: ignore[arg-type]
                                        transition_mode=transition,  # type: ignore[arg-type]
                                        beam_width=beam_width,
                                        beam_depth=beam_depth,
                                        max_steps=max_steps,
                                        device=device,
                                        cem_samples=cem_samples,
                                        cem_iters=cem_iters,
                                        cem_elites=cem_elites,
                                        cem_momentum=cem_momentum,
                                        high_cem_samples=high_cem_samples,
                                        high_cem_iters=high_cem_iters,
                                        high_cem_elites=high_cem_elites,
                                        high_cem_momentum=high_cem_momentum,
                                        high_cem_std=high_cem_std,
                                        high_cem_optimizer=high_cem_optimizer,
                                        high_cem_temperature=high_cem_temperature,
                                        high_cem_codebook=high_cem_codebook,
                                        high_cem_codebook_size=high_cem_codebook_size,
                                        seed=example_index,
                                    )
                                elif planner == "hierarchical_beam":
                                    result = run_hierarchical_beam_mpc(
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
                                else:
                                    raise ValueError(f"Unknown planner {planner!r}.")
                                solved += int(result.solved)
                                remaining.append(result.remaining_hamming)
                                steps.append(result.steps)
                                action_evals.append(result.action_evals)
                                elapsed.append(result.elapsed_seconds)
                            record = {
                                "planner": planner,
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
                                "high_cem_optimizer": high_cem_optimizer if planner == "hierarchical_cem" else "",
                                "high_cem_codebook": high_cem_codebook if planner == "hierarchical_cem" else "",
                                "high_cem_codebook_size": int(high_cem_codebook_size) if planner == "hierarchical_cem" else 0,
                            }
                            handle.write(json.dumps(record, sort_keys=True) + "\n")
                            handle.flush()
                            records.append(record)
                            completed.add(key)
    return records


def _ensure_append_starts_on_new_line(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        return
    with path.open("rb") as handle:
        handle.seek(-1, 2)
        last = handle.read(1)
    if last != b"\n":
        with path.open("ab") as handle:
            handle.write(b"\n")


def _read_completed_matrix_keys(path: Path) -> set[tuple[str, str, str, int, int]]:
    if not path.is_file():
        return set()
    completed: set[tuple[str, str, str, int, int]] = set()
    with path.open() as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
                completed.add(_matrix_key(record))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return completed


def _matrix_key(record: dict[str, Any]) -> tuple[str, str, str, int, int]:
    return (
        str(record["planner"]),
        str(record["transition_mode"]),
        str(record["score_mode"]),
        int(record["beam_width"]),
        int(record["beam_depth"]),
    )


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
    parser.add_argument("--planners", default="mpc_beam")
    parser.add_argument("--cem-samples", type=int, default=128)
    parser.add_argument("--cem-iters", type=int, default=4)
    parser.add_argument("--cem-elites", type=int, default=16)
    parser.add_argument("--cem-momentum", type=float, default=0.7)
    parser.add_argument("--high-cem-samples", type=int, default=128)
    parser.add_argument("--high-cem-iters", type=int, default=4)
    parser.add_argument("--high-cem-elites", type=int, default=16)
    parser.add_argument("--high-cem-momentum", type=float, default=0.7)
    parser.add_argument("--high-cem-std", type=float, default=1.0)
    parser.add_argument("--high-cem-optimizer", choices=("cem", "mppi"), default="cem")
    parser.add_argument("--high-cem-temperature", type=float, default=1.0)
    parser.add_argument("--high-cem-codebook", choices=("none", "init"), default="none")
    parser.add_argument("--high-cem-codebook-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-diagnostics", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_checkpoint(args.checkpoint, device)
    examples = load_eval_examples(config, limit=args.examples)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_diagnostics:
        diagnostic_seed = int(config.get("seed", 0)) + 900 if args.seed is None else int(args.seed) + 900
        diagnostics = run_grid_goal_diagnostics(model, examples, args.output_dir, device=device, seed=diagnostic_seed)
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
        planners=tuple(x for x in args.planners.split(",") if x),
        max_examples=args.examples,
        max_steps=args.max_steps,
        cem_samples=args.cem_samples,
        cem_iters=args.cem_iters,
        cem_elites=args.cem_elites,
        cem_momentum=args.cem_momentum,
        high_cem_samples=args.high_cem_samples,
        high_cem_iters=args.high_cem_iters,
        high_cem_elites=args.high_cem_elites,
        high_cem_momentum=args.high_cem_momentum,
        high_cem_std=args.high_cem_std,
        high_cem_optimizer=args.high_cem_optimizer,
        high_cem_temperature=args.high_cem_temperature,
        high_cem_codebook=args.high_cem_codebook,
        high_cem_codebook_size=args.high_cem_codebook_size,
    )
    print(json.dumps({"records": len(records), "output": str(args.output_dir)}, sort_keys=True), flush=True)


def _parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(item) for item in text.split(",") if item)


if __name__ == "__main__":
    main()
