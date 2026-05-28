from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from datasets import load_dataset

from puzzle_jepa.data.worlds import MazeWorld, PuzzleExample, PuzzleWorld, SudokuWorld


@dataclass(frozen=True, slots=True)
class HFPuzzleColumns:
    puzzle: str = "question"
    solution: str = "answer"


def example_from_strings(world: PuzzleWorld, puzzle: str, solution: str) -> PuzzleExample:
    if isinstance(world, SudokuWorld):
        return world.example_from_strings(puzzle, solution)
    if isinstance(world, MazeWorld):
        side = int(math.isqrt(len(puzzle)))
        if side * side != len(puzzle) or len(solution) != len(puzzle):
            raise ValueError("Flattened maze puzzle and solution must be square strings of equal length.")
        maze = MazeWorld(height=side, width=side)
        return PuzzleExample(
            maze.from_lines(_chunk(puzzle, side)),
            maze.from_lines(_chunk(solution, side)),
        )
    raise TypeError(f"Unsupported world type {type(world).__name__}.")


def iter_hf_examples(
    repo_id: str,
    split: str,
    world: PuzzleWorld,
    columns: HFPuzzleColumns = HFPuzzleColumns(),
    limit: int | None = None,
) -> Iterable[PuzzleExample]:
    dataset = load_dataset(repo_id, split=split)
    for index, row in enumerate(dataset):
        if limit is not None and index >= limit:
            break
        yield example_from_strings(world, str(row[columns.puzzle]), str(row[columns.solution]))


def _chunk(text: str, width: int) -> list[str]:
    return [text[start : start + width] for start in range(0, len(text), width)]
