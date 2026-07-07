import json

import numpy as np

from puzzle_jepa.data.arc import ARCGrid, grid_distance, iter_leave_one_out_episodes, load_arc_task
from puzzle_jepa.data.arc_actions import apply_arc_action, episode_candidate_shapes, episode_palette, generate_arc_actions
from puzzle_jepa.data.arc_proposals import build_arc_sources, extract_arc_proposals
from puzzle_jepa.eval.arc_oracle_coverage import run_arc_coverage, run_arc_episode_coverage


def test_arc_loader_pads_variable_shape_and_builds_leave_one_out(tmp_path):
    path = tmp_path / "toy.json"
    path.write_text(
        json.dumps(
            {
                "train": [
                    {"input": [[1, 0], [0, 1]], "output": [[1]]},
                    {"input": [[2, 0], [0, 2]], "output": [[2]]},
                    {"input": [[3, 0], [0, 3]], "output": [[3]]},
                ],
                "test": [{"input": [[4, 0], [0, 4]]}],
            }
        ),
        encoding="utf-8",
    )

    task = load_arc_task(path)
    episodes = list(iter_leave_one_out_episodes(task))
    values, active = episodes[0].target_output.padded()

    assert task.task_id == "toy"
    assert len(episodes) == 3
    assert values.shape == (30, 30)
    assert active.sum() == 1
    assert episodes[0].query_input.shape == (2, 2)
    assert episodes[0].target_output.shape == (1, 1)


def test_proposals_define_object_boundaries_for_rotate_and_reflect():
    candidate = ARCGrid(np.asarray([[0, 0, 0], [0, 2, 2], [0, 2, 0]], dtype=np.int64))
    context = ()
    proposals = extract_arc_proposals(context, candidate, candidate)
    shapes = ((3, 3),)
    actions = generate_arc_actions(
        context,
        candidate,
        candidate,
        proposals=proposals,
        candidate_shapes=shapes,
        palette=(0, 2),
        include_cell_actions=False,
    )

    transform_actions = [action for action in actions if action.op in {"rotate", "reflect"}]

    assert transform_actions
    assert all("proposal_id" in action.params for action in transform_actions)
    assert all(proposals[action.params["proposal_id"]].bbox for action in transform_actions)


def test_crop_action_handles_shape_reduction_from_proposed_bbox():
    query = ARCGrid(
        np.asarray(
            [
                [0, 0, 0, 0, 0],
                [0, 7, 7, 7, 0],
                [0, 7, 0, 7, 0],
                [0, 7, 7, 7, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.int64,
        )
    )
    candidate = ARCGrid(np.zeros((3, 3), dtype=np.int64))
    proposals = extract_arc_proposals((), query, candidate)
    sources = build_arc_sources((), query, candidate)
    crop_actions = [
        action
        for action in generate_arc_actions(
            (),
            query,
            candidate,
            proposals=proposals,
            candidate_shapes=((3, 3),),
            palette=(0, 7),
            include_cell_actions=False,
        )
        if action.op == "crop"
    ]

    cropped = [apply_arc_action(candidate, action, proposals=proposals, sources=sources) for action in crop_actions]

    assert any(grid.shape == (3, 3) and int(grid.values.sum()) == 56 for grid in cropped)


def test_partition_map_can_represent_compression_actions():
    query = ARCGrid(
        np.asarray(
            [
                [1, 1, 0, 0, 2, 2],
                [1, 1, 0, 0, 2, 2],
                [0, 0, 3, 3, 0, 0],
                [0, 0, 3, 3, 0, 0],
                [4, 4, 0, 0, 5, 5],
                [4, 4, 0, 0, 5, 5],
            ],
            dtype=np.int64,
        )
    )
    candidate = ARCGrid(np.zeros((3, 3), dtype=np.int64))
    proposals = extract_arc_proposals((), query, candidate)
    sources = build_arc_sources((), query, candidate)
    actions = generate_arc_actions(
        (),
        query,
        candidate,
        proposals=proposals,
        candidate_shapes=((3, 3),),
        palette=(0, 1, 2, 3, 4, 5),
        include_cell_actions=False,
    )
    partition_actions = [action for action in actions if action.op == "partition_map" and action.params["mode"] == "majority"]

    outputs = [apply_arc_action(candidate, action, proposals=proposals, sources=sources) for action in partition_actions]

    assert any(grid.shape == (3, 3) and grid.values[0, 0] == 1 and grid.values[2, 2] == 5 for grid in outputs)


def test_oracle_coverage_solves_missing_corner_with_typed_action_without_cell_fallback():
    task = _task_from_pairs(
        [
            (
                [[2, 2, 0], [2, 0, 2], [2, 2, 2]],
                [[2, 2, 3], [2, 0, 2], [2, 2, 2]],
            ),
            (
                [[2, 2, 0], [2, 0, 2], [2, 2, 2]],
                [[2, 2, 3], [2, 0, 2], [2, 2, 2]],
            ),
        ]
    )
    episode = list(iter_leave_one_out_episodes(task))[0]

    result = run_arc_episode_coverage(
        episode,
        max_depth=1,
        beam_width=4,
        oracle_shape=False,
        include_cell_actions=False,
        max_actions=800,
    )

    assert result.solved
    assert result.solved_depth == 1
    assert not any(step.action.startswith("set_cell") for step in result.best_trace)


def test_oracle_coverage_separates_shape_oracle_from_context_shape():
    task = _task_from_pairs(
        [
            (
                [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 2, 2], [0, 0, 2, 2]],
                [[1, 0], [0, 2]],
            ),
            (
                [[3, 3, 0, 0], [3, 3, 0, 0], [0, 0, 4, 4], [0, 0, 4, 4]],
                [[3, 0], [0, 4]],
            ),
        ]
    )

    summary = run_arc_coverage(
        [task],
        max_depth=1,
        beam_width=4,
        oracle_shape=False,
        include_cell_actions=False,
        max_actions=1500,
    )

    assert summary.num_episodes == 2
    assert summary.solved_episodes == 2
    assert all(row.solved_depth == 1 for row in summary.rows)


def test_grid_distance_penalizes_shape_mismatch():
    assert grid_distance(ARCGrid(np.zeros((7, 7), dtype=np.int64)), ARCGrid(np.zeros((3, 3), dtype=np.int64))) > 0


def _task_from_pairs(pairs):
    from puzzle_jepa.data.arc import ARCExample, ARCTask, grid_from_lists

    examples = tuple(ARCExample(grid_from_lists(inp), grid_from_lists(out)) for inp, out in pairs)
    return ARCTask(task_id="synthetic", train=examples, test=())
