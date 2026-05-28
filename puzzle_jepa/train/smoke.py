from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

from puzzle_jepa.data import MazeWorld, PuzzleExample, SudokuWorld, collate_transitions, sample_oracle_transition
from puzzle_jepa.models import ActionConditionedWorldModel, HRMReasoner, PTRMSampler, TRMReasoner


_SUDOKU_PUZZLE = (
    "530070000"
    "600195000"
    "098000060"
    "800060003"
    "400803001"
    "700020006"
    "060000280"
    "000419005"
    "000080079"
)
_SUDOKU_SOLUTION = (
    "534678912"
    "672195348"
    "198342567"
    "859761423"
    "426853791"
    "713924856"
    "961537284"
    "287419635"
    "345286179"
)


def run_smoke_experiment(config: Mapping) -> dict[str, float | str]:
    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    task = dict(config.get("task", {}))
    world = _build_world(task)
    example = _build_example(world)
    transitions = [sample_oracle_transition(world, example, rng) for _ in range(int(config.get("batch_size", 4)))]
    batch = collate_transitions(transitions)
    model_cfg = dict(config.get("model", {}))
    model_type = str(model_cfg.pop("type", "jepa"))
    if model_type == "jepa":
        model = ActionConditionedWorldModel(vocab_size=world.vocab_size, **model_cfg)
        output = model(batch.states, batch.actions, batch.next_states)
    elif model_type == "hrm":
        model = HRMReasoner(vocab_size=world.vocab_size, **model_cfg)
        output = model(batch.states, labels=batch.goals, task_ids=batch.actions[:, 0])
    elif model_type == "trm":
        model = TRMReasoner(vocab_size=world.vocab_size, **model_cfg)
        output = model(batch.states, labels=batch.goals, task_ids=batch.actions[:, 0])
    elif model_type == "ptrm":
        base = TRMReasoner(vocab_size=world.vocab_size, **dict(model_cfg.pop("base")))
        model = PTRMSampler(base, **model_cfg)
        output = model(batch.states, task_ids=batch.actions[:, 0])
    else:
        raise ValueError(f"Unknown smoke model type {model_type!r}.")
    loss = getattr(output, "loss", None)
    if loss is not None:
        loss.backward()
    return {
        "task": world.name,
        "model_type": model_type,
        "loss": float(loss.detach().item()) if loss is not None else 0.0,
        "num_transitions": float(len(transitions)),
    }


def _build_world(task: Mapping) -> SudokuWorld | MazeWorld:
    name = str(task.get("name", "sudoku"))
    if name == "sudoku":
        return SudokuWorld()
    if name == "maze":
        return MazeWorld(height=int(task.get("height", 5)), width=int(task.get("width", 5)))
    raise ValueError(f"Unknown task {name!r}.")


def _build_example(world: SudokuWorld | MazeWorld):
    if isinstance(world, SudokuWorld):
        return world.example_from_strings(_SUDOKU_PUZZLE, _SUDOKU_SOLUTION)
    puzzle = world.from_lines(["S   #", "### #", "#   #", "# ###", "#   G"])
    goal = world.from_lines(["Sooo#", "###o#", "#ooo#", "#o###", "#oooG"])
    return PuzzleExample(puzzle, goal)
