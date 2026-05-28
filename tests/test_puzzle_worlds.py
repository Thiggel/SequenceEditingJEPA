import numpy as np
import pytest

from puzzle_jepa.data import (
    MazeWorld,
    PuzzleExample,
    SudokuWorld,
    WorldAction,
    collate_transitions,
    example_from_strings,
    sample_curriculum_transition,
    sample_oracle_transition,
    sample_random_mutable_transition,
)
from puzzle_jepa.planning import LatentActionPlanner, SymbolicOracleScorer


SUDOKU_PUZZLE = (
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
SUDOKU_SOLUTION = (
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


@pytest.mark.parametrize("row", range(9))
@pytest.mark.parametrize("col", range(9))
@pytest.mark.parametrize("value", [1, 5, 9])
def test_sudoku_blank_board_accepts_every_cell_value(row, col, value):
    world = SudokuWorld()
    state = np.zeros((9, 9), dtype=np.int64)
    next_state = world.apply(state, WorldAction(row, col, value))
    assert next_state[row, col] == value
    assert state[row, col] == 0


@pytest.mark.parametrize(
    ("action", "message"),
    [
        (WorldAction(0, 2, 5), "violates"),
        (WorldAction(2, 0, 5), "violates"),
        (WorldAction(1, 1, 5), "violates"),
        (WorldAction(0, 0, 4), "empty"),
        (WorldAction(-1, 0, 1), "Invalid"),
        (WorldAction(0, 9, 1), "Invalid"),
        (WorldAction(0, 0, 10), "Invalid"),
    ],
)
def test_sudoku_rejects_invalid_actions(action, message):
    world = SudokuWorld()
    state = world.from_string(SUDOKU_PUZZLE)
    with pytest.raises(ValueError, match=message):
        world.apply(state, action)


def test_sudoku_oracle_transition_moves_toward_solution():
    world = SudokuWorld()
    example = world.example_from_strings(SUDOKU_PUZZLE, SUDOKU_SOLUTION)
    transition = sample_oracle_transition(world, example, np.random.default_rng(0))
    assert transition.next_state[transition.action.row, transition.action.col] == transition.goal[
        transition.action.row, transition.action.col
    ]
    assert np.not_equal(transition.next_state, transition.goal).sum() < np.not_equal(transition.state, transition.goal).sum()


def test_sudoku_random_mutable_transition_keeps_clues_fixed():
    world = SudokuWorld()
    example = world.example_from_strings(SUDOKU_PUZZLE, SUDOKU_SOLUTION)
    transition = sample_random_mutable_transition(world, example, np.random.default_rng(0))
    clue_mask = example.state != 0
    assert np.array_equal(transition.state[clue_mask], example.state[clue_mask])
    assert np.array_equal(transition.next_state[clue_mask], example.state[clue_mask])
    assert example.state[transition.action.row, transition.action.col] == 0
    assert transition.next_state[transition.action.row, transition.action.col] == transition.action.value


def test_sudoku_mutable_actions_allow_non_clue_overwrite_only():
    world = SudokuWorld()
    puzzle = world.from_string(SUDOKU_PUZZLE)
    clue_mask = world.clue_mask_from_puzzle(puzzle)
    state = puzzle.copy()
    state[0, 2] = 4

    overwritten = world.apply(
        state,
        WorldAction(0, 2, 9),
        clue_mask=clue_mask,
        allow_overwrite=True,
        allow_conflicts=True,
    )
    assert overwritten[0, 2] == 9
    assert WorldAction(0, 2, 9) in world.legal_actions(
        state,
        clue_mask=clue_mask,
        allow_overwrite=True,
        allow_conflicts=True,
    )
    with pytest.raises(ValueError, match="clue"):
        world.apply(
            state,
            WorldAction(0, 0, 9),
            clue_mask=clue_mask,
            allow_overwrite=True,
            allow_conflicts=True,
        )


def test_curriculum_transition_rejects_invalid_mix_probability():
    world = SudokuWorld()
    example = world.example_from_strings(SUDOKU_PUZZLE, SUDOKU_SOLUTION)
    with pytest.raises(ValueError, match="oracle_probability"):
        sample_curriculum_transition(world, example, np.random.default_rng(0), oracle_probability=1.5)


def test_sudoku_solution_validation_and_goal_matching():
    world = SudokuWorld()
    solution = world.from_string(SUDOKU_SOLUTION)
    puzzle = world.from_string(SUDOKU_PUZZLE)
    assert world.is_valid_solution(solution)
    assert world.is_goal(solution)
    assert world.is_goal(solution, solution)
    assert not world.is_goal(puzzle, solution)


def test_maze_parser_actions_and_connected_path():
    world = MazeWorld(height=5, width=5)
    state = world.from_lines(["S   #", "### #", "#   #", "# ###", "#   G"])
    goal = world.from_lines(["Sooo#", "###o#", "#ooo#", "#o###", "#oooG"])
    assert world.to_lines(state)[0] == "S   #"
    assert WorldAction(0, 1, world.PATH) in world.legal_actions(state)
    assert not world.has_connected_path(state)
    assert world.has_connected_path(goal)
    assert world.is_goal(goal)


def test_maze_random_mutable_transition_marks_original_empty_cell():
    world = MazeWorld(height=5, width=5)
    state = world.from_lines(["S   #", "### #", "#   #", "# ###", "#   G"])
    goal = world.from_lines(["Sooo#", "###o#", "#ooo#", "#o###", "#oooG"])
    transition = sample_random_mutable_transition(world, PuzzleExample(state, goal), np.random.default_rng(1))
    assert state[transition.action.row, transition.action.col] == world.EMPTY
    assert transition.next_state[transition.action.row, transition.action.col] == world.PATH
    assert transition.action.value == world.PATH


@pytest.mark.parametrize(
    "action",
    [
        WorldAction(0, 0, MazeWorld.PATH),
        WorldAction(0, 4, MazeWorld.PATH),
        WorldAction(0, 1, MazeWorld.GOAL),
        WorldAction(5, 1, MazeWorld.PATH),
    ],
)
def test_maze_rejects_invalid_actions(action):
    world = MazeWorld(height=5, width=5)
    state = world.from_lines(["S   #", "### #", "#   #", "# ###", "#   G"])
    with pytest.raises(ValueError):
        world.apply(state, action)


def test_transition_collation_shapes_and_action_fields():
    world = SudokuWorld()
    example = world.example_from_strings(SUDOKU_PUZZLE, SUDOKU_SOLUTION)
    transitions = [sample_oracle_transition(world, example, np.random.default_rng(i)) for i in range(3)]
    batch = collate_transitions(transitions)
    assert batch.states.shape == (3, 9, 9)
    assert batch.next_states.shape == (3, 9, 9)
    assert batch.goals.shape == (3, 9, 9)
    assert batch.clue_masks is not None
    assert batch.clue_masks.shape == (3, 9, 9)
    assert batch.actions.shape == (3, 4)
    assert batch.actions[:, 0].tolist() == [world.task_id] * 3


def test_symbolic_oracle_planner_solves_one_action_sudoku_gap():
    world = SudokuWorld()
    goal = world.from_string(SUDOKU_SOLUTION)
    state = goal.copy()
    state[0, 2] = 0
    planner = LatentActionPlanner(world, SymbolicOracleScorer(world), beam_size=1, max_steps=2)
    trace = planner.plan(state, goal)
    assert len(trace) == 2
    assert trace[-1].action == WorldAction(0, 2, 4)
    assert world.is_goal(trace[-1].state, goal)


def test_sample_oracle_transition_fails_for_solved_example():
    world = SudokuWorld()
    goal = world.from_string(SUDOKU_SOLUTION)
    with pytest.raises(ValueError, match="No oracle-improving"):
        sample_oracle_transition(world, PuzzleExample(goal, goal), np.random.default_rng(0))


def test_hf_string_adapter_for_sudoku_and_flat_maze():
    sudoku = example_from_strings(SudokuWorld(), SUDOKU_PUZZLE, SUDOKU_SOLUTION)
    assert sudoku.state.shape == (9, 9)
    maze = example_from_strings(MazeWorld(height=2, width=2), "S G#", "SoG#")
    assert maze.state.shape == (2, 2)
    assert maze.goal[0, 1] == MazeWorld.PATH
