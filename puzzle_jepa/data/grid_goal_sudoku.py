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
    counterfactual_states: np.ndarray | None = None
    counterfactual_actions: np.ndarray | None = None
    counterfactual_next_boards: np.ndarray | None = None
    counterfactual_action_sequences: np.ndarray | None = None
    counterfactual_future_boards: np.ndarray | None = None
    counterfactual_step_mask: np.ndarray | None = None


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
    counterfactual_states: torch.Tensor | None = None
    counterfactual_actions: torch.Tensor | None = None
    counterfactual_next_boards: torch.Tensor | None = None
    counterfactual_mask: torch.Tensor | None = None
    counterfactual_action_sequences: torch.Tensor | None = None
    counterfactual_future_boards: torch.Tensor | None = None
    counterfactual_step_mask: torch.Tensor | None = None


def action_to_array(action: WorldAction) -> np.ndarray:
    return np.asarray([action.row, action.col, action.value], dtype=np.int64)


def array_to_action(values: np.ndarray | torch.Tensor | list[int] | tuple[int, int, int]) -> WorldAction:
    row, col, value = [int(x) for x in values]
    return WorldAction(row=row, col=col, value=value)


def legal_fill_actions(board: np.ndarray, *, allow_conflicts: bool = True) -> list[WorldAction]:
    return legal_sudoku_actions(board, allow_conflicts=allow_conflicts, allow_overwrite=False)


def legal_sudoku_actions(
    board: np.ndarray,
    *,
    clue_mask: np.ndarray | None = None,
    allow_conflicts: bool = True,
    allow_overwrite: bool = False,
) -> list[WorldAction]:
    return SudokuWorld().legal_actions(
        board,
        clue_mask=clue_mask,
        allow_overwrite=allow_overwrite,
        allow_conflicts=allow_conflicts,
    )


def apply_fill_action(board: np.ndarray, action: WorldAction, *, allow_conflicts: bool = True) -> np.ndarray:
    return apply_sudoku_action(board, action, allow_conflicts=allow_conflicts, allow_overwrite=False)


def apply_sudoku_action(
    board: np.ndarray,
    action: WorldAction,
    *,
    clue_mask: np.ndarray | None = None,
    allow_conflicts: bool = True,
    allow_overwrite: bool = False,
) -> np.ndarray:
    return SudokuWorld().apply(
        board,
        action,
        clue_mask=clue_mask,
        allow_overwrite=allow_overwrite,
        allow_conflicts=allow_conflicts,
    )


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
    allow_overwrite: bool = False,
    editable_noise_probability: float = 0.0,
    counterfactual_branches: int = 0,
    counterfactual_depth: int = 1,
    counterfactual_max_pairs: int = 0,
) -> GridGoalSudokuTrajectory:
    del oracle_probability
    return _sample_grid_goal_sudoku_trajectory(
        example,
        rng,
        is_oracle=True,
        allow_conflicts=allow_conflicts,
        allow_overwrite=allow_overwrite,
        editable_noise_probability=editable_noise_probability,
        counterfactual_branches=counterfactual_branches,
        counterfactual_depth=counterfactual_depth,
        counterfactual_max_pairs=counterfactual_max_pairs,
    )


def sample_random_grid_goal_sudoku_trajectory(
    example: PuzzleExample,
    rng: np.random.Generator,
    *,
    allow_conflicts: bool = True,
    allow_overwrite: bool = False,
    max_steps: int | None = None,
    counterfactual_branches: int = 0,
    counterfactual_depth: int = 1,
    counterfactual_max_pairs: int = 0,
) -> GridGoalSudokuTrajectory:
    return _sample_grid_goal_sudoku_trajectory(
        example,
        rng,
        is_oracle=False,
        allow_conflicts=allow_conflicts,
        allow_overwrite=allow_overwrite,
        random_max_steps=max_steps,
        counterfactual_branches=counterfactual_branches,
        counterfactual_depth=counterfactual_depth,
        counterfactual_max_pairs=counterfactual_max_pairs,
    )


def _sample_grid_goal_sudoku_trajectory(
    example: PuzzleExample,
    rng: np.random.Generator,
    *,
    is_oracle: bool,
    allow_conflicts: bool,
    allow_overwrite: bool = False,
    editable_noise_probability: float = 0.0,
    random_max_steps: int | None = None,
    counterfactual_branches: int = 0,
    counterfactual_depth: int = 1,
    counterfactual_max_pairs: int = 0,
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
    if is_oracle:
        for index in order:
            row, col = (int(x) for x in empty_positions[int(index)])
            if allow_overwrite and editable_noise_probability > 0.0 and rng.random() < editable_noise_probability:
                wrong_choices = [value for value in range(1, 10) if value != int(goal[row, col])]
                wrong = int(wrong_choices[int(rng.integers(0, len(wrong_choices)))])
                wrong_action = WorldAction(row=row, col=col, value=wrong)
                board = apply_sudoku_action(
                    board,
                    wrong_action,
                    clue_mask=clue_mask,
                    allow_conflicts=allow_conflicts,
                    allow_overwrite=allow_overwrite,
                )
                actions.append(action_to_array(wrong_action))
                boards.append(board.copy())
            action = WorldAction(row=row, col=col, value=int(goal[row, col]))
            board = apply_sudoku_action(
                board,
                action,
                clue_mask=clue_mask,
                allow_conflicts=allow_conflicts,
                allow_overwrite=allow_overwrite,
            )
            actions.append(action_to_array(action))
            boards.append(board.copy())
    else:
        steps = int(random_max_steps) if random_max_steps is not None else len(empty_positions)
        for _ in range(max(0, steps)):
            legal = legal_sudoku_actions(
                board,
                clue_mask=clue_mask,
                allow_conflicts=allow_conflicts,
                allow_overwrite=allow_overwrite,
            )
            if not legal:
                break
            action = legal[int(rng.integers(0, len(legal)))]
            board = apply_sudoku_action(
                board,
                action,
                clue_mask=clue_mask,
                allow_conflicts=allow_conflicts,
                allow_overwrite=allow_overwrite,
            )
            actions.append(action_to_array(action))
            boards.append(board.copy())
    actions.append(PAD_ACTION.copy())
    counterfactual = _sample_counterfactual_pairs(
        np.asarray(boards, dtype=np.int64),
        np.asarray(actions, dtype=np.int64),
        clue_mask,
        rng,
        allow_conflicts=allow_conflicts,
        allow_overwrite=allow_overwrite,
        branches=counterfactual_branches,
        depth=counterfactual_depth,
        max_pairs=counterfactual_max_pairs,
    )
    return GridGoalSudokuTrajectory(
        boards=np.asarray(boards, dtype=np.int64),
        actions=np.asarray(actions, dtype=np.int64),
        context=puzzle.copy(),
        clue_mask=clue_mask,
        editable_mask=editable_mask,
        active_mask=active_mask,
        goal=goal.copy(),
        is_oracle=is_oracle,
        counterfactual_states=counterfactual[0],
        counterfactual_actions=counterfactual[1],
        counterfactual_next_boards=counterfactual[2],
        counterfactual_action_sequences=counterfactual[3],
        counterfactual_future_boards=counterfactual[4],
        counterfactual_step_mask=counterfactual[5],
    )


def _sample_counterfactual_pairs(
    boards: np.ndarray,
    actions: np.ndarray,
    clue_mask: np.ndarray,
    rng: np.random.Generator,
    *,
    allow_conflicts: bool,
    allow_overwrite: bool,
    branches: int,
    depth: int,
    max_pairs: int,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    branches = int(branches)
    depth = max(1, int(depth))
    max_pairs = int(max_pairs)
    if branches <= 0 or max_pairs == 0 or boards.shape[0] <= 1:
        return None, None, None, None, None, None
    states: list[np.ndarray] = []
    sampled_actions: list[np.ndarray] = []
    next_boards: list[np.ndarray] = []
    action_sequences: list[np.ndarray] = []
    future_boards: list[np.ndarray] = []
    step_masks: list[np.ndarray] = []

    def add_branch(state: np.ndarray, first_action: WorldAction, *, canonical_next: np.ndarray | None = None) -> None:
        if max_pairs > 0 and len(states) >= max_pairs:
            return
        current = np.asarray(state, dtype=np.int64).copy()
        sequence = np.tile(PAD_ACTION, (depth, 1))
        futures = np.zeros((depth, *state.shape), dtype=np.int64)
        step_mask = np.zeros((depth,), dtype=bool)
        actions_to_try = [first_action]
        for step in range(depth):
            if step > 0:
                legal_next = legal_sudoku_actions(
                    current,
                    clue_mask=clue_mask,
                    allow_conflicts=allow_conflicts,
                    allow_overwrite=allow_overwrite,
                )
                if not legal_next:
                    break
                actions_to_try = [legal_next[int(rng.integers(0, len(legal_next)))]]
            action = actions_to_try[0]
            try:
                current = (
                    np.asarray(canonical_next, dtype=np.int64).copy()
                    if step == 0 and canonical_next is not None
                    else apply_sudoku_action(
                        current,
                        action,
                        clue_mask=clue_mask,
                        allow_conflicts=allow_conflicts,
                        allow_overwrite=allow_overwrite,
                    )
                )
            except ValueError:
                break
            sequence[step] = action_to_array(action)
            futures[step] = current
            step_mask[step] = True
        if not bool(step_mask.any()):
            return
        states.append(np.asarray(state, dtype=np.int64).copy())
        sampled_actions.append(sequence[0].copy())
        next_boards.append(futures[np.nonzero(step_mask)[0][-1]].copy())
        action_sequences.append(sequence)
        future_boards.append(futures)
        step_masks.append(step_mask)

    for frame in range(boards.shape[0] - 1):
        if max_pairs > 0 and len(states) >= max_pairs:
            break
        true_action = array_to_action(actions[frame])
        add_branch(boards[frame], true_action, canonical_next=boards[frame + 1])
        legal = legal_sudoku_actions(
            boards[frame],
            clue_mask=clue_mask,
            allow_conflicts=allow_conflicts,
            allow_overwrite=allow_overwrite,
        )
        if not legal:
            continue
        chosen = rng.choice(len(legal), size=min(branches, len(legal)), replace=False)
        for action_index in np.atleast_1d(chosen):
            add_branch(boards[frame], legal[int(action_index)])
    if not states:
        return None, None, None, None, None, None
    return (
        np.asarray(states, dtype=np.int64),
        np.asarray(sampled_actions, dtype=np.int64),
        np.asarray(next_boards, dtype=np.int64),
        np.asarray(action_sequences, dtype=np.int64),
        np.asarray(future_boards, dtype=np.int64),
        np.asarray(step_masks, dtype=bool),
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
    cf_lengths = [
        0 if item.counterfactual_states is None else int(item.counterfactual_states.shape[0])
        for item in trajectories
    ]
    cf_frames = max(cf_lengths, default=0)
    padded_cf_states = []
    padded_cf_actions = []
    padded_cf_next = []
    padded_cf_action_sequences = []
    padded_cf_future_boards = []
    padded_cf_step_masks = []
    cf_masks = []
    cf_depth = max(
        (
            0 if item.counterfactual_action_sequences is None else int(item.counterfactual_action_sequences.shape[1])
            for item in trajectories
        ),
        default=0,
    )
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
        if cf_frames > 0:
            cf_states = np.zeros((cf_frames, 9, 9), dtype=np.int64)
            cf_actions = np.zeros((cf_frames, 3), dtype=np.int64)
            cf_next = np.zeros((cf_frames, 9, 9), dtype=np.int64)
            cf_action_seq = np.zeros((cf_frames, cf_depth, 3), dtype=np.int64)
            cf_future = np.zeros((cf_frames, cf_depth, 9, 9), dtype=np.int64)
            cf_step_mask = np.zeros((cf_frames, cf_depth), dtype=bool)
            cf_mask = np.zeros((cf_frames,), dtype=bool)
            cf_len = 0 if item.counterfactual_states is None else int(item.counterfactual_states.shape[0])
            if cf_len > 0:
                cf_states[:cf_len] = item.counterfactual_states
                cf_actions[:cf_len] = item.counterfactual_actions
                cf_next[:cf_len] = item.counterfactual_next_boards
                cf_mask[:cf_len] = True
                if item.counterfactual_action_sequences is not None:
                    depth = int(item.counterfactual_action_sequences.shape[1])
                    cf_action_seq[:cf_len, :depth] = item.counterfactual_action_sequences
                    cf_future[:cf_len, :depth] = item.counterfactual_future_boards
                    cf_step_mask[:cf_len, :depth] = item.counterfactual_step_mask
            padded_cf_states.append(cf_states)
            padded_cf_actions.append(cf_actions)
            padded_cf_next.append(cf_next)
            padded_cf_action_sequences.append(cf_action_seq)
            padded_cf_future_boards.append(cf_future)
            padded_cf_step_masks.append(cf_step_mask)
            cf_masks.append(cf_mask)
    counterfactual_states = None
    counterfactual_actions = None
    counterfactual_next_boards = None
    counterfactual_mask = None
    counterfactual_action_sequences = None
    counterfactual_future_boards = None
    counterfactual_step_mask = None
    if cf_frames > 0:
        counterfactual_states = torch.as_tensor(np.stack(padded_cf_states), dtype=torch.long, device=device)
        counterfactual_actions = torch.as_tensor(np.stack(padded_cf_actions), dtype=torch.long, device=device)
        counterfactual_next_boards = torch.as_tensor(np.stack(padded_cf_next), dtype=torch.long, device=device)
        counterfactual_mask = torch.as_tensor(np.stack(cf_masks), dtype=torch.bool, device=device)
        counterfactual_action_sequences = torch.as_tensor(np.stack(padded_cf_action_sequences), dtype=torch.long, device=device)
        counterfactual_future_boards = torch.as_tensor(np.stack(padded_cf_future_boards), dtype=torch.long, device=device)
        counterfactual_step_mask = torch.as_tensor(np.stack(padded_cf_step_masks), dtype=torch.bool, device=device)
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
        counterfactual_states=counterfactual_states,
        counterfactual_actions=counterfactual_actions,
        counterfactual_next_boards=counterfactual_next_boards,
        counterfactual_mask=counterfactual_mask,
        counterfactual_action_sequences=counterfactual_action_sequences,
        counterfactual_future_boards=counterfactual_future_boards,
        counterfactual_step_mask=counterfactual_step_mask,
    )
