from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from puzzle_jepa.data.worlds import PuzzleExample, SudokuWorld, WorldAction


PAD_ACTION = np.asarray([0, 0, 0], dtype=np.int64)


@dataclass(frozen=True, slots=True)
class GridGoalSudokuTrajectory:
    boards: np.ndarray
    actions: np.ndarray
    context: np.ndarray
    clue_mask: np.ndarray
    editable_mask: np.ndarray
    active_mask: np.ndarray
    goal: np.ndarray
    is_oracle: bool


@dataclass(frozen=True, slots=True)
class GridGoalSudokuBatch:
    boards: torch.Tensor
    actions: torch.Tensor
    context: torch.Tensor
    clue_mask: torch.Tensor
    editable_mask: torch.Tensor
    active_mask: torch.Tensor
    goals: torch.Tensor
    masks: torch.Tensor
    oracle_mask: torch.Tensor


def action_to_array(action: WorldAction) -> np.ndarray:
    return np.asarray([action.row, action.col, action.value], dtype=np.int64)


def array_to_action(values: np.ndarray | torch.Tensor | list[int] | tuple[int, int, int]) -> WorldAction:
    row, col, value = [int(x) for x in values]
    return WorldAction(row=row, col=col, value=value)


def legal_fill_actions(board: np.ndarray, *, allow_conflicts: bool = True) -> list[WorldAction]:
    return SudokuWorld().legal_actions(board, allow_overwrite=False, allow_conflicts=allow_conflicts)


def apply_fill_action(board: np.ndarray, action: WorldAction, *, allow_conflicts: bool = True) -> np.ndarray:
    return SudokuWorld().apply(board, action, allow_overwrite=False, allow_conflicts=allow_conflicts)


def corrupt_terminal(goal: np.ndarray, rng: np.random.Generator, *, min_cells: int = 1, max_cells: int = 5) -> np.ndarray:
    corrupted = np.asarray(goal, dtype=np.int64).copy()
    count = int(rng.integers(min_cells, max_cells + 1))
    indices = rng.choice(81, size=count, replace=False)
    for flat in indices:
        row, col = divmod(int(flat), 9)
        current = int(corrupted[row, col])
        choices = [value for value in range(1, 10) if value != current]
        corrupted[row, col] = int(choices[int(rng.integers(0, len(choices)))])
    return corrupted


def sample_grid_goal_sudoku_trajectory(
    example: PuzzleExample,
    rng: np.random.Generator,
    *,
    oracle_probability: float = 0.5,
    allow_conflicts: bool = True,
) -> GridGoalSudokuTrajectory:
    del oracle_probability
    return _sample_grid_goal_sudoku_trajectory(example, rng, is_oracle=True, allow_conflicts=allow_conflicts)


def sample_random_grid_goal_sudoku_trajectory(
    example: PuzzleExample,
    rng: np.random.Generator,
    *,
    allow_conflicts: bool = True,
) -> GridGoalSudokuTrajectory:
    return _sample_grid_goal_sudoku_trajectory(example, rng, is_oracle=False, allow_conflicts=allow_conflicts)


def _sample_grid_goal_sudoku_trajectory(
    example: PuzzleExample,
    rng: np.random.Generator,
    *,
    is_oracle: bool,
    allow_conflicts: bool,
) -> GridGoalSudokuTrajectory:
    world = SudokuWorld()
    puzzle = world.validate_state(example.state)
    goal = world.validate_state(example.goal)
    clue_mask = puzzle != 0
    editable_mask = ~clue_mask
    active_mask = np.ones_like(clue_mask, dtype=bool)
    empty_positions = np.argwhere(editable_mask)
    order = rng.permutation(len(empty_positions))
    board = puzzle.copy()
    boards = [board.copy()]
    actions: list[np.ndarray] = []
    for index in order:
        row, col = (int(x) for x in empty_positions[int(index)])
        value = int(goal[row, col]) if is_oracle else int(rng.integers(1, 10))
        action = WorldAction(row=row, col=col, value=value)
        board = apply_fill_action(board, action, allow_conflicts=allow_conflicts)
        actions.append(action_to_array(action))
        boards.append(board.copy())
    actions.append(PAD_ACTION.copy())
    return GridGoalSudokuTrajectory(
        boards=np.asarray(boards, dtype=np.int64),
        actions=np.asarray(actions, dtype=np.int64),
        context=puzzle.copy(),
        clue_mask=clue_mask,
        editable_mask=editable_mask,
        active_mask=active_mask,
        goal=goal.copy(),
        is_oracle=is_oracle,
    )


def collate_grid_goal_sudoku_trajectories(
    trajectories: list[GridGoalSudokuTrajectory],
    *,
    device: str | torch.device = "cpu",
) -> GridGoalSudokuBatch:
    if not trajectories:
        raise ValueError("Cannot collate an empty trajectory list.")
    lengths = [int(item.boards.shape[0]) for item in trajectories]
    num_frames = max(lengths)
    padded_boards = []
    padded_actions = []
    masks = []
    for item, length in zip(trajectories, lengths, strict=True):
        boards = np.empty((num_frames, 9, 9), dtype=np.int64)
        actions = np.empty((num_frames, 3), dtype=np.int64)
        boards[:length] = item.boards
        actions[:length] = item.actions
        if length < num_frames:
            boards[length:] = item.boards[-1]
            actions[length:] = PAD_ACTION
        mask = np.zeros((num_frames,), dtype=bool)
        mask[:length] = True
        padded_boards.append(boards)
        padded_actions.append(actions)
        masks.append(mask)
    return GridGoalSudokuBatch(
        boards=torch.as_tensor(np.stack(padded_boards), dtype=torch.long, device=device),
        actions=torch.as_tensor(np.stack(padded_actions), dtype=torch.long, device=device),
        context=torch.as_tensor(np.stack([item.context for item in trajectories]), dtype=torch.long, device=device),
        clue_mask=torch.as_tensor(np.stack([item.clue_mask for item in trajectories]), dtype=torch.bool, device=device),
        editable_mask=torch.as_tensor(
            np.stack([item.editable_mask for item in trajectories]), dtype=torch.bool, device=device
        ),
        active_mask=torch.as_tensor(np.stack([item.active_mask for item in trajectories]), dtype=torch.bool, device=device),
        goals=torch.as_tensor(np.stack([item.goal for item in trajectories]), dtype=torch.long, device=device),
        masks=torch.as_tensor(np.stack(masks), dtype=torch.bool, device=device),
        oracle_mask=torch.as_tensor([item.is_oracle for item in trajectories], dtype=torch.bool, device=device),
    )
