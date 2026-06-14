from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from puzzle_jepa.data.worlds import PuzzleExample, SudokuWorld, WorldAction


PAD_ACTION = np.asarray([0, 0, 0], dtype=np.int64)


@dataclass(frozen=True, slots=True)
class SudokuTrajectory:
    boards: np.ndarray
    actions: np.ndarray
    goal: np.ndarray
    is_oracle: bool


@dataclass(frozen=True, slots=True)
class SudokuTrajectoryBatch:
    boards: torch.Tensor
    actions: torch.Tensor
    goals: torch.Tensor
    oracle_mask: torch.Tensor


def action_to_array(action: WorldAction) -> np.ndarray:
    return np.asarray([action.row, action.col, action.value], dtype=np.int64)


def array_to_action(values: np.ndarray | torch.Tensor | list[int] | tuple[int, int, int]) -> WorldAction:
    row, col, value = [int(x) for x in values]
    return WorldAction(row=row, col=col, value=value)


def action_id(action: WorldAction) -> int:
    return (action.row * 9 + action.col) * 9 + (action.value - 1)


def action_from_id(index: int) -> WorldAction:
    index = int(index)
    if not 0 <= index < 729:
        raise ValueError(f"Sudoku action id must be in [0, 729), got {index}.")
    cell, value_offset = divmod(index, 9)
    row, col = divmod(cell, 9)
    return WorldAction(row=row, col=col, value=value_offset + 1)


def legal_fill_actions(board: np.ndarray, *, allow_conflicts: bool = True) -> list[WorldAction]:
    world = SudokuWorld()
    arr = world.validate_state(board)
    actions: list[WorldAction] = []
    for row, col in np.argwhere(arr == 0):
        for value in range(1, 10):
            action = WorldAction(int(row), int(col), value)
            if allow_conflicts or world.is_value_allowed_after_write(arr, action.row, action.col, action.value):
                actions.append(action)
    return actions


def apply_fill_action(board: np.ndarray, action: WorldAction, *, allow_conflicts: bool = True) -> np.ndarray:
    world = SudokuWorld()
    return world.apply(board, action, allow_overwrite=False, allow_conflicts=allow_conflicts)


def sample_sudoku_trajectory(
    example: PuzzleExample,
    rng: np.random.Generator,
    *,
    num_frames: int,
    oracle_probability: float = 0.5,
    allow_conflicts: bool = True,
) -> SudokuTrajectory:
    """Sample a LeWM-style subtrajectory with no overwrites.

    `boards[t]` is the current board and `actions[t]` maps `boards[t]` to
    `boards[t + 1]` for every supervised step. The final action is padding so
    shapes match LeWM's `(B, T, *)` pseudocode.
    """

    if num_frames < 2:
        raise ValueError("num_frames must be at least 2.")
    if not 0.0 <= oracle_probability <= 1.0:
        raise ValueError("oracle_probability must be in [0, 1].")

    world = SudokuWorld()
    puzzle = world.validate_state(example.state)
    goal = world.validate_state(example.goal)
    empty_positions = np.argwhere(puzzle == 0)
    supervised_steps = num_frames - 1
    if len(empty_positions) < supervised_steps:
        raise ValueError(
            f"Sudoku example has {len(empty_positions)} empty cells, fewer than {supervised_steps} supervised steps."
        )

    is_oracle = bool(rng.random() < oracle_probability)
    board = puzzle.copy()

    max_prefix = len(empty_positions) - supervised_steps
    prefix_len = int(rng.integers(0, max_prefix + 1))
    if prefix_len:
        prefix_indices = rng.choice(len(empty_positions), size=prefix_len, replace=False)
        for row, col in empty_positions[prefix_indices]:
            row_i, col_i = int(row), int(col)
            if is_oracle:
                value = int(goal[row_i, col_i])
            else:
                value = int(rng.integers(1, 10))
            board[row_i, col_i] = value

    boards = [board.copy()]
    actions: list[np.ndarray] = []
    for _ in range(supervised_steps):
        remaining = np.argwhere(board == 0)
        if len(remaining) == 0:
            raise ValueError("Sampled a solved board before trajectory completion.")
        row, col = (int(x) for x in remaining[int(rng.integers(0, len(remaining)))])
        value = int(goal[row, col]) if is_oracle else int(rng.integers(1, 10))
        action = WorldAction(row=row, col=col, value=value)
        board = apply_fill_action(board, action, allow_conflicts=allow_conflicts)
        actions.append(action_to_array(action))
        boards.append(board.copy())
    actions.append(PAD_ACTION.copy())

    return SudokuTrajectory(
        boards=np.asarray(boards, dtype=np.int64),
        actions=np.asarray(actions, dtype=np.int64),
        goal=goal.copy(),
        is_oracle=is_oracle,
    )


def collate_sudoku_trajectories(
    trajectories: list[SudokuTrajectory],
    *,
    device: str | torch.device = "cpu",
) -> SudokuTrajectoryBatch:
    if not trajectories:
        raise ValueError("Cannot collate an empty trajectory list.")
    num_frames = trajectories[0].boards.shape[0]
    if any(item.boards.shape != (num_frames, 9, 9) for item in trajectories):
        raise ValueError("All Sudoku trajectories must have the same number of 9x9 boards.")
    if any(item.actions.shape != (num_frames, 3) for item in trajectories):
        raise ValueError("All Sudoku trajectories must have action shape [num_frames, 3].")
    return SudokuTrajectoryBatch(
        boards=torch.as_tensor(np.stack([item.boards for item in trajectories]), dtype=torch.long, device=device),
        actions=torch.as_tensor(np.stack([item.actions for item in trajectories]), dtype=torch.long, device=device),
        goals=torch.as_tensor(np.stack([item.goal for item in trajectories]), dtype=torch.long, device=device),
        oracle_mask=torch.as_tensor([item.is_oracle for item in trajectories], dtype=torch.bool, device=device),
    )
