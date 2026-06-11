from __future__ import annotations

import numpy as np

from puzzle_jepa.data.worlds import MazeWorld, PuzzleExample, PuzzleWorld, SudokuWorld, WorldAction
from puzzle_jepa.data.trajectories import RolloutTransition, Transition


def sample_oracle_partial_transition(
    world: PuzzleWorld,
    example: PuzzleExample,
    rng: np.random.Generator,
) -> Transition:
    if isinstance(world, SudokuWorld):
        return _sample_sudoku_oracle_partial(world, example, rng)
    if isinstance(world, MazeWorld):
        return _sample_maze_oracle_partial(world, example, rng)
    raise TypeError(f"Unsupported world type {type(world).__name__}.")


def sample_curriculum_transition(
    world: PuzzleWorld,
    example: PuzzleExample,
    rng: np.random.Generator,
    oracle_probability: float = 1.0,
) -> Transition:
    if not 0.0 <= oracle_probability <= 1.0:
        raise ValueError("oracle_probability must be in [0, 1].")
    if rng.random() < oracle_probability:
        return sample_oracle_partial_transition(world, example, rng)
    return sample_random_mutable_transition(world, example, rng)


def sample_random_mutable_transition(
    world: PuzzleWorld,
    example: PuzzleExample,
    rng: np.random.Generator,
) -> Transition:
    if isinstance(world, SudokuWorld):
        return _sample_sudoku_random_mutable(world, example, rng)
    if isinstance(world, MazeWorld):
        return _sample_maze_random_mutable(world, example, rng)
    raise TypeError(f"Unsupported world type {type(world).__name__}.")


def sample_curriculum_rollout_transition(
    world: PuzzleWorld,
    example: PuzzleExample,
    rng: np.random.Generator,
    steps: int,
    oracle_probability: float = 1.0,
) -> RolloutTransition:
    if not 0.0 <= oracle_probability <= 1.0:
        raise ValueError("oracle_probability must be in [0, 1].")
    if rng.random() < oracle_probability:
        return sample_oracle_rollout_transition(world, example, rng, steps)
    return sample_random_mutable_rollout_transition(world, example, rng, steps)


def sample_oracle_rollout_transition(
    world: PuzzleWorld,
    example: PuzzleExample,
    rng: np.random.Generator,
    steps: int,
) -> RolloutTransition:
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if isinstance(world, SudokuWorld):
        return _sample_sudoku_oracle_rollout(world, example, rng, int(steps))
    if isinstance(world, MazeWorld):
        return _sample_maze_oracle_rollout(world, example, rng, int(steps))
    raise TypeError(f"Unsupported world type {type(world).__name__}.")


def sample_random_mutable_rollout_transition(
    world: PuzzleWorld,
    example: PuzzleExample,
    rng: np.random.Generator,
    steps: int,
) -> RolloutTransition:
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if isinstance(world, SudokuWorld):
        return _sample_sudoku_random_mutable_rollout(world, example, rng, int(steps))
    if isinstance(world, MazeWorld):
        return _sample_maze_random_mutable_rollout(world, example, rng, int(steps))
    raise TypeError(f"Unsupported world type {type(world).__name__}.")


def _sample_sudoku_oracle_partial(
    world: SudokuWorld,
    example: PuzzleExample,
    rng: np.random.Generator,
) -> Transition:
    puzzle = world.validate_state(example.state)
    goal = world.validate_state(example.goal)
    mutable_positions = np.argwhere(puzzle == 0)
    if len(mutable_positions) == 0:
        raise ValueError("Sudoku example has no mutable cells.")
    reveal_count = int(rng.integers(0, len(mutable_positions)))
    state = puzzle.copy()
    if reveal_count:
        revealed_indices = rng.choice(len(mutable_positions), size=reveal_count, replace=False)
        revealed = mutable_positions[revealed_indices]
        state[revealed[:, 0], revealed[:, 1]] = goal[revealed[:, 0], revealed[:, 1]]
    remaining = np.argwhere((puzzle == 0) & (state == 0))
    if len(remaining) == 0:
        raise ValueError("Sampled solved Sudoku partial state.")
    row, col = (int(x) for x in remaining[int(rng.integers(0, len(remaining)))])
    action = WorldAction(row, col, int(goal[row, col]))
    return Transition(
        state=state,
        action=action,
        next_state=world.apply(state, action, clue_mask=world.clue_mask_from_puzzle(puzzle)),
        goal=goal.copy(),
        task_id=world.task_id,
        clue_mask=world.clue_mask_from_puzzle(puzzle),
    )


def _sample_sudoku_oracle_rollout(
    world: SudokuWorld,
    example: PuzzleExample,
    rng: np.random.Generator,
    steps: int,
) -> RolloutTransition:
    puzzle = world.validate_state(example.state)
    goal = world.validate_state(example.goal)
    mutable_positions = np.argwhere(puzzle == 0)
    if len(mutable_positions) < steps:
        raise ValueError(f"Sudoku example has fewer than {steps} mutable cells.")
    reveal_count = int(rng.integers(0, len(mutable_positions) - steps + 1))
    state = puzzle.copy()
    if reveal_count:
        revealed_indices = rng.choice(len(mutable_positions), size=reveal_count, replace=False)
        revealed = mutable_positions[revealed_indices]
        state[revealed[:, 0], revealed[:, 1]] = goal[revealed[:, 0], revealed[:, 1]]

    clue_mask = world.clue_mask_from_puzzle(puzzle)
    actions: list[WorldAction] = []
    target_states: list[np.ndarray] = []
    current = state.copy()
    for _ in range(steps):
        remaining = np.argwhere((puzzle == 0) & (current != goal))
        if len(remaining) == 0:
            raise ValueError("Sampled solved Sudoku partial state before rollout completed.")
        row, col = (int(x) for x in remaining[int(rng.integers(0, len(remaining)))])
        action = WorldAction(row, col, int(goal[row, col]))
        current = world.apply(current, action, clue_mask=clue_mask, allow_overwrite=True, allow_conflicts=False)
        actions.append(action)
        target_states.append(current.copy())
    return RolloutTransition(
        state=state,
        actions=actions,
        target_states=target_states,
        goal=goal.copy(),
        task_id=world.task_id,
        clue_mask=clue_mask,
    )


def _sample_sudoku_random_mutable_rollout(
    world: SudokuWorld,
    example: PuzzleExample,
    rng: np.random.Generator,
    steps: int,
) -> RolloutTransition:
    puzzle = world.validate_state(example.state)
    goal = world.validate_state(example.goal)
    mutable_positions = np.argwhere(puzzle == 0)
    if len(mutable_positions) == 0:
        raise ValueError("Sudoku example has no mutable cells.")

    state = puzzle.copy()
    fill_count = int(rng.integers(0, len(mutable_positions) + 1))
    if fill_count:
        filled_indices = rng.choice(len(mutable_positions), size=fill_count, replace=False)
        filled = mutable_positions[filled_indices]
        state[filled[:, 0], filled[:, 1]] = rng.integers(1, 10, size=fill_count)

    clue_mask = world.clue_mask_from_puzzle(puzzle)
    actions: list[WorldAction] = []
    target_states: list[np.ndarray] = []
    current = state.copy()
    for _ in range(steps):
        row, col = (int(x) for x in mutable_positions[int(rng.integers(0, len(mutable_positions)))])
        current_value = int(current[row, col])
        if current_value == 0:
            value = int(rng.integers(1, 10))
        else:
            value_choices = [digit for digit in range(1, 10) if digit != current_value]
            value = int(value_choices[int(rng.integers(0, len(value_choices)))])
        action = WorldAction(row, col, value)
        current = world.apply(current, action, clue_mask=clue_mask, allow_overwrite=True, allow_conflicts=True)
        actions.append(action)
        target_states.append(current.copy())
    return RolloutTransition(
        state=state,
        actions=actions,
        target_states=target_states,
        goal=goal.copy(),
        task_id=world.task_id,
        clue_mask=clue_mask,
    )


def _sample_sudoku_random_mutable(
    world: SudokuWorld,
    example: PuzzleExample,
    rng: np.random.Generator,
) -> Transition:
    puzzle = world.validate_state(example.state)
    goal = world.validate_state(example.goal)
    mutable_positions = np.argwhere(puzzle == 0)
    if len(mutable_positions) == 0:
        raise ValueError("Sudoku example has no mutable cells.")

    state = puzzle.copy()
    fill_count = int(rng.integers(0, len(mutable_positions) + 1))
    if fill_count:
        filled_indices = rng.choice(len(mutable_positions), size=fill_count, replace=False)
        filled = mutable_positions[filled_indices]
        state[filled[:, 0], filled[:, 1]] = rng.integers(1, 10, size=fill_count)

    row, col = (int(x) for x in mutable_positions[int(rng.integers(0, len(mutable_positions)))])
    current = int(state[row, col])
    if current == 0:
        value = int(rng.integers(1, 10))
    else:
        value_choices = [digit for digit in range(1, 10) if digit != current]
        value = int(value_choices[int(rng.integers(0, len(value_choices)))])
    action = WorldAction(row, col, value)
    clue_mask = world.clue_mask_from_puzzle(puzzle)
    next_state = world.apply(state, action, clue_mask=clue_mask, allow_overwrite=True, allow_conflicts=True)
    return Transition(
        state=state,
        action=action,
        next_state=next_state,
        goal=goal.copy(),
        task_id=world.task_id,
        clue_mask=clue_mask,
    )


def _sample_maze_oracle_rollout(
    world: MazeWorld,
    example: PuzzleExample,
    rng: np.random.Generator,
    steps: int,
) -> RolloutTransition:
    puzzle = world.validate_state(example.state)
    goal = world.validate_state(example.goal)
    path_positions = np.argwhere((puzzle == world.EMPTY) & (goal == world.PATH))
    if len(path_positions) < steps:
        raise ValueError(f"Maze example has fewer than {steps} oracle path cells.")
    reveal_count = int(rng.integers(0, len(path_positions) - steps + 1))
    state = puzzle.copy()
    if reveal_count:
        revealed_indices = rng.choice(len(path_positions), size=reveal_count, replace=False)
        revealed = path_positions[revealed_indices]
        state[revealed[:, 0], revealed[:, 1]] = world.PATH

    actions: list[WorldAction] = []
    target_states: list[np.ndarray] = []
    current = state.copy()
    for _ in range(steps):
        remaining = np.argwhere((puzzle == world.EMPTY) & (goal == world.PATH) & (current != world.PATH))
        if len(remaining) == 0:
            raise ValueError("Sampled solved Maze partial state before rollout completed.")
        row, col = (int(x) for x in remaining[int(rng.integers(0, len(remaining)))])
        action = WorldAction(row, col, world.PATH)
        current = world.apply(current, action)
        actions.append(action)
        target_states.append(current.copy())
    return RolloutTransition(
        state=state,
        actions=actions,
        target_states=target_states,
        goal=goal.copy(),
        task_id=world.task_id,
    )


def _sample_maze_random_mutable_rollout(
    world: MazeWorld,
    example: PuzzleExample,
    rng: np.random.Generator,
    steps: int,
) -> RolloutTransition:
    puzzle = world.validate_state(example.state)
    goal = world.validate_state(example.goal)
    mutable_positions = np.argwhere(puzzle == world.EMPTY)
    if len(mutable_positions) < steps:
        raise ValueError(f"Maze example has fewer than {steps} mutable empty cells.")

    fill_count = int(rng.integers(0, len(mutable_positions) - steps + 1))
    state = puzzle.copy()
    if fill_count:
        filled_indices = rng.choice(len(mutable_positions), size=fill_count, replace=False)
        filled = mutable_positions[filled_indices]
        state[filled[:, 0], filled[:, 1]] = world.PATH

    actions: list[WorldAction] = []
    target_states: list[np.ndarray] = []
    current = state.copy()
    for _ in range(steps):
        remaining = np.argwhere(current == world.EMPTY)
        if len(remaining) == 0:
            raise ValueError("Sampled filled Maze state before rollout completed.")
        row, col = (int(x) for x in remaining[int(rng.integers(0, len(remaining)))])
        action = WorldAction(row, col, world.PATH)
        current = world.apply(current, action)
        actions.append(action)
        target_states.append(current.copy())
    return RolloutTransition(
        state=state,
        actions=actions,
        target_states=target_states,
        goal=goal.copy(),
        task_id=world.task_id,
    )


def _sample_maze_oracle_partial(
    world: MazeWorld,
    example: PuzzleExample,
    rng: np.random.Generator,
) -> Transition:
    puzzle = world.validate_state(example.state)
    goal = world.validate_state(example.goal)
    path_positions = np.argwhere((puzzle == world.EMPTY) & (goal == world.PATH))
    if len(path_positions) == 0:
        raise ValueError("Maze example has no oracle path cells to reveal.")
    reveal_count = int(rng.integers(0, len(path_positions)))
    state = puzzle.copy()
    if reveal_count:
        revealed_indices = rng.choice(len(path_positions), size=reveal_count, replace=False)
        revealed = path_positions[revealed_indices]
        state[revealed[:, 0], revealed[:, 1]] = world.PATH
    remaining = np.argwhere((puzzle == world.EMPTY) & (goal == world.PATH) & (state != world.PATH))
    if len(remaining) == 0:
        raise ValueError("Sampled solved Maze partial state.")
    row, col = (int(x) for x in remaining[int(rng.integers(0, len(remaining)))])
    action = WorldAction(row, col, world.PATH)
    return Transition(
        state=state,
        action=action,
        next_state=world.apply(state, action),
        goal=goal.copy(),
        task_id=world.task_id,
    )


def _sample_maze_random_mutable(
    world: MazeWorld,
    example: PuzzleExample,
    rng: np.random.Generator,
) -> Transition:
    puzzle = world.validate_state(example.state)
    goal = world.validate_state(example.goal)
    mutable_positions = np.argwhere(puzzle == world.EMPTY)
    if len(mutable_positions) == 0:
        raise ValueError("Maze example has no mutable empty cells.")

    state = puzzle.copy()
    path_positions = np.argwhere((puzzle == world.EMPTY) & (goal == world.PATH))
    if len(path_positions):
        reveal_count = int(rng.integers(0, len(path_positions)))
        if reveal_count:
            revealed_indices = rng.choice(len(path_positions), size=reveal_count, replace=False)
            revealed = path_positions[revealed_indices]
            state[revealed[:, 0], revealed[:, 1]] = world.PATH

    remaining = np.argwhere((puzzle == world.EMPTY) & (state == world.EMPTY))
    if len(remaining) == 0:
        state = puzzle.copy()
        remaining = mutable_positions
    row, col = (int(x) for x in remaining[int(rng.integers(0, len(remaining)))])
    action = WorldAction(row, col, world.PATH)
    next_state = state.copy()
    next_state[row, col] = world.PATH
    return Transition(
        state=state,
        action=action,
        next_state=next_state,
        goal=goal.copy(),
        task_id=world.task_id,
    )
