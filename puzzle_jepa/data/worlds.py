from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True, slots=True)
class WorldAction:
    row: int
    col: int
    value: int

    def as_array(self, task_id: int) -> np.ndarray:
        return np.asarray([int(task_id), self.row, self.col, self.value], dtype=np.int64)


@dataclass(frozen=True, slots=True)
class PuzzleExample:
    state: np.ndarray
    goal: np.ndarray

    def __post_init__(self) -> None:
        if self.state.shape != self.goal.shape:
            raise ValueError(f"state shape {self.state.shape} does not match goal shape {self.goal.shape}.")


class PuzzleWorld(ABC):
    name: str
    task_id: int
    height: int
    width: int
    vocab_size: int

    @abstractmethod
    def legal_actions(self, state: np.ndarray) -> list[WorldAction]:
        raise NotImplementedError

    @abstractmethod
    def apply(self, state: np.ndarray, action: WorldAction) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def is_goal(self, state: np.ndarray, goal: np.ndarray | None = None) -> bool:
        raise NotImplementedError

    def validate_state(self, state: np.ndarray) -> np.ndarray:
        arr = np.asarray(state, dtype=np.int64)
        if arr.shape != (self.height, self.width):
            raise ValueError(f"{self.name} state must have shape {(self.height, self.width)}, got {arr.shape}.")
        if np.any(arr < 0) or np.any(arr >= self.vocab_size):
            raise ValueError(f"{self.name} state contains token outside [0, {self.vocab_size}).")
        return arr

    def transition(self, state: np.ndarray, action: WorldAction) -> tuple[np.ndarray, np.ndarray]:
        current = self.validate_state(state)
        return current, self.apply(current, action)


class SudokuWorld(PuzzleWorld):
    name = "sudoku"
    task_id = 0
    height = 9
    width = 9
    vocab_size = 10

    @staticmethod
    def from_string(text: str) -> np.ndarray:
        compact = "".join(ch for ch in text if not ch.isspace())
        if len(compact) != 81:
            raise ValueError(f"Sudoku string must contain 81 cells, got {len(compact)}.")
        values = [0 if ch in ".0" else int(ch) for ch in compact]
        return np.asarray(values, dtype=np.int64).reshape(9, 9)

    def example_from_strings(self, puzzle: str, solution: str) -> PuzzleExample:
        return PuzzleExample(self.from_string(puzzle), self.from_string(solution))

    def legal_actions(
        self,
        state: np.ndarray,
        *,
        clue_mask: np.ndarray | None = None,
        allow_overwrite: bool = False,
        allow_conflicts: bool = False,
    ) -> list[WorldAction]:
        arr = self.validate_state(state)
        fixed = None if clue_mask is None else self.validate_clue_mask(clue_mask)
        actions: list[WorldAction] = []
        if allow_overwrite:
            if fixed is None:
                raise ValueError("clue_mask is required when allow_overwrite=True.")
            positions = np.argwhere(~fixed)
        else:
            positions = np.argwhere(arr == 0)
        for row, col in positions:
            for value in range(1, 10):
                action = WorldAction(int(row), int(col), value)
                if fixed is not None and fixed[action.row, action.col]:
                    continue
                if allow_overwrite and arr[action.row, action.col] == value:
                    continue
                if allow_conflicts or self.is_value_allowed_after_write(arr, action.row, action.col, value):
                    actions.append(action)
        return actions

    def apply(
        self,
        state: np.ndarray,
        action: WorldAction,
        *,
        clue_mask: np.ndarray | None = None,
        allow_overwrite: bool = False,
        allow_conflicts: bool = False,
    ) -> np.ndarray:
        arr = self.validate_state(state).copy()
        self._validate_action_shape(action)
        if clue_mask is not None:
            fixed = self.validate_clue_mask(clue_mask)
            if fixed[action.row, action.col]:
                raise ValueError("Sudoku action cannot overwrite a clue cell.")
        if arr[action.row, action.col] != 0 and not allow_overwrite:
            raise ValueError("Sudoku action can only fill an empty cell unless allow_overwrite=True.")
        if not allow_conflicts and not self.is_value_allowed_after_write(arr, action.row, action.col, action.value):
            raise ValueError("Sudoku action violates row, column, or block constraints.")
        arr[action.row, action.col] = action.value
        return arr

    def is_goal(self, state: np.ndarray, goal: np.ndarray | None = None) -> bool:
        arr = self.validate_state(state)
        if goal is not None:
            return np.array_equal(arr, self.validate_state(goal))
        return self.is_valid_solution(arr)

    def is_value_allowed(self, state: np.ndarray, row: int, col: int, value: int) -> bool:
        return self.is_value_allowed_after_write(state, row, col, value)

    def is_value_allowed_after_write(self, state: np.ndarray, row: int, col: int, value: int) -> bool:
        self._validate_action_shape(WorldAction(row, col, value))
        arr = self.validate_state(state).copy()
        arr[row, col] = 0
        row_vals = np.delete(arr[row, :], col)
        col_vals = np.delete(arr[:, col], row)
        block_r = 3 * (row // 3)
        block_c = 3 * (col // 3)
        block = arr[block_r : block_r + 3, block_c : block_c + 3].reshape(-1)
        block_idx = (row - block_r) * 3 + (col - block_c)
        block_vals = np.delete(block, block_idx)
        return value not in row_vals and value not in col_vals and value not in block_vals

    def validate_clue_mask(self, clue_mask: np.ndarray) -> np.ndarray:
        mask = np.asarray(clue_mask, dtype=bool)
        if mask.shape != (9, 9):
            raise ValueError(f"Sudoku clue_mask must have shape {(9, 9)}, got {mask.shape}.")
        return mask

    def clue_mask_from_puzzle(self, puzzle: np.ndarray) -> np.ndarray:
        return self.validate_state(puzzle) != 0

    def is_valid_solution(self, state: np.ndarray) -> bool:
        arr = self.validate_state(state)
        target = set(range(1, 10))
        rows = all(set(arr[row, :].tolist()) == target for row in range(9))
        cols = all(set(arr[:, col].tolist()) == target for col in range(9))
        blocks = all(
            set(arr[row : row + 3, col : col + 3].reshape(-1).tolist()) == target
            for row in range(0, 9, 3)
            for col in range(0, 9, 3)
        )
        return rows and cols and blocks

    def _validate_action_shape(self, action: WorldAction) -> None:
        if not (0 <= action.row < 9 and 0 <= action.col < 9 and 1 <= action.value <= 9):
            raise ValueError(f"Invalid Sudoku action: {action}.")


class MazeWorld(PuzzleWorld):
    name = "maze"
    task_id = 1
    WALL = 0
    EMPTY = 1
    START = 2
    GOAL = 3
    PATH = 4
    vocab_size = 5
    _char_to_token = {"#": WALL, " ": EMPTY, ".": EMPTY, "S": START, "G": GOAL, "o": PATH, "*": PATH}
    _token_to_char = {WALL: "#", EMPTY: " ", START: "S", GOAL: "G", PATH: "o"}

    def __init__(self, height: int = 30, width: int = 30):
        self.height = int(height)
        self.width = int(width)
        if self.height <= 0 or self.width <= 0:
            raise ValueError("Maze height and width must be positive.")

    def from_lines(self, lines: Iterable[str]) -> np.ndarray:
        rows = [line.rstrip("\n") for line in lines if line.rstrip("\n")]
        if len(rows) != self.height or any(len(row) != self.width for row in rows):
            raise ValueError(f"Maze lines must form shape {(self.height, self.width)}.")
        try:
            values = [[self._char_to_token[ch] for ch in row] for row in rows]
        except KeyError as exc:
            raise ValueError(f"Unsupported maze character {exc.args[0]!r}.") from exc
        return np.asarray(values, dtype=np.int64)

    def to_lines(self, state: np.ndarray) -> list[str]:
        arr = self.validate_state(state)
        return ["".join(self._token_to_char[int(token)] for token in row) for row in arr]

    def legal_actions(self, state: np.ndarray) -> list[WorldAction]:
        arr = self.validate_state(state)
        return [
            WorldAction(int(row), int(col), self.PATH)
            for row, col in zip(*np.where(arr == self.EMPTY), strict=True)
        ]

    def apply(self, state: np.ndarray, action: WorldAction) -> np.ndarray:
        arr = self.validate_state(state).copy()
        if not (0 <= action.row < self.height and 0 <= action.col < self.width):
            raise ValueError(f"Maze action coordinate out of range: {action}.")
        if action.value != self.PATH:
            raise ValueError("Maze action value must be PATH.")
        if arr[action.row, action.col] != self.EMPTY:
            raise ValueError("Maze action can only mark an empty cell.")
        arr[action.row, action.col] = self.PATH
        return arr

    def is_goal(self, state: np.ndarray, goal: np.ndarray | None = None) -> bool:
        arr = self.validate_state(state)
        if goal is not None:
            return np.array_equal(arr, self.validate_state(goal))
        return self.has_connected_path(arr)

    def has_connected_path(self, state: np.ndarray) -> bool:
        arr = self.validate_state(state)
        starts = np.argwhere(arr == self.START)
        goals = np.argwhere(arr == self.GOAL)
        if len(starts) != 1 or len(goals) != 1:
            return False
        start = tuple(int(x) for x in starts[0])
        goal = tuple(int(x) for x in goals[0])
        passable = {self.START, self.GOAL, self.PATH}
        queue: deque[tuple[int, int]] = deque([start])
        visited = {start}
        while queue:
            row, col = queue.popleft()
            if (row, col) == goal:
                return True
            for next_row, next_col in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if not (0 <= next_row < self.height and 0 <= next_col < self.width):
                    continue
                if (next_row, next_col) in visited or int(arr[next_row, next_col]) not in passable:
                    continue
                visited.add((next_row, next_col))
                queue.append((next_row, next_col))
        return False
