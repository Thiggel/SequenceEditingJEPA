import json

import numpy as np

from puzzle_jepa.data.arc import ARCGrid, grid_distance, iter_leave_one_out_episodes, load_arc_task
from puzzle_jepa.data.arc_actions import apply_arc_action, episode_candidate_shapes, episode_palette, generate_arc_actions
from puzzle_jepa.data.arc_proposals import build_arc_sources, extract_arc_proposals
from puzzle_jepa.eval.arc_oracle_coverage import run_arc_coverage, run_arc_episode_coverage
from puzzle_jepa.data.arc_training import collate_arc_records, episodes_from_tasks, sample_arc_candidate_record
from puzzle_jepa.models.arc_models import ARCCandidateScorer
from puzzle_jepa.train.arc_jepa import run_arc_training


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


def test_full_grid_and_hole_proposals_are_available():
    grid = ARCGrid(np.asarray([[8, 8, 8], [8, 0, 8], [8, 8, 8]], dtype=np.int64))

    proposals = extract_arc_proposals((), grid, grid)
    kinds = {proposal.kind for proposal in proposals.values()}

    assert "full_grid" in kinds
    assert "hole" in kinds


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


def test_scale_and_color_map_actions_render_source_grids():
    query = ARCGrid(np.asarray([[1, 0], [0, 1]], dtype=np.int64))
    candidate = ARCGrid(np.zeros((4, 4), dtype=np.int64))
    proposals = extract_arc_proposals((), query, candidate)
    sources = build_arc_sources((), query, candidate)
    actions = generate_arc_actions(
        (),
        query,
        candidate,
        proposals=proposals,
        candidate_shapes=((4, 4),),
        palette=(0, 1, 2),
        include_cell_actions=False,
    )
    scaled_actions = [action for action in actions if action.op == "scale_source" and action.params["source"] == "query_input"]
    color_maps = [
        action
        for action in actions
        if action.op == "apply_color_map" and action.params["source"] == "query_input" and action.params["from_color"] == 1 and action.params["to_color"] == 2
    ]

    scaled = [apply_arc_action(candidate, action, proposals=proposals, sources=sources) for action in scaled_actions]
    mapped = apply_arc_action(candidate, color_maps[0], proposals=proposals, sources=sources)

    assert any(grid.shape == (4, 4) and int(grid.values.sum()) == 8 for grid in scaled)
    assert mapped.shape == (2, 2)
    assert set(mapped.color_set()) == {0, 2}


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


def test_arc_candidate_scorer_variants_forward():
    task = _task_from_pairs(
        [
            ([[1, 0], [0, 1]], [[1, 1], [1, 1]]),
            ([[2, 0], [0, 2]], [[2, 2], [2, 2]]),
        ]
    )
    episodes = episodes_from_tasks([task])
    rng = np.random.default_rng(0)
    records = [
        sample_arc_candidate_record(episodes, rng, max_actions=80, include_cell_actions=True)
        for _ in range(4)
    ]
    batch = collate_arc_records(records)

    for kwargs in (
        {"use_action_features": False, "use_jepa": False},
        {"use_action_features": True, "use_jepa": False},
        {"use_action_features": True, "use_jepa": True},
    ):
        model = ARCCandidateScorer(d_model=32, **kwargs)
        output = model(batch)
        assert output.loss.ndim == 0
        assert output.logits.shape == (4,)


def test_arc_training_smoke_runs_all_variants(tmp_path):
    data_root = tmp_path / "arc"
    train_dir = data_root / "data" / "training"
    train_dir.mkdir(parents=True)
    for index in range(4):
        color = index + 1
        payload = {
            "train": [
                {"input": [[color, 0], [0, color]], "output": [[color, color], [color, color]]},
                {"input": [[color, 0], [color, 0]], "output": [[color, color], [color, color]]},
            ],
            "test": [{"input": [[color, 0], [0, color]]}],
        }
        (train_dir / f"task_{index}.json").write_text(json.dumps(payload), encoding="utf-8")

    for variant in ("raw_grid_energy", "proposal_energy", "jepa_energy"):
        metrics = run_arc_training(
            {
                "seed": 1,
                "variant": variant,
                "output_dir": str(tmp_path / variant),
                "data": {
                    "data_root": str(data_root),
                    "split": "training",
                    "task_limit": 4,
                    "eval_task_count": 1,
                    "max_context": 2,
                },
                "model": {"d_model": 32},
                "sampler": {
                    "oracle_shape": False,
                    "include_cell_actions": True,
                    "max_actions": 80,
                    "positive_probability": 0.4,
                    "best_action_probability": 0.5,
                },
                "training": {
                    "max_steps": 2,
                    "batch_size": 2,
                    "learning_rate": 1.0e-3,
                    "weight_decay": 0.0,
                    "grad_clip": 1.0,
                    "bf16": False,
                    "eval_every_steps": 1,
                    "save_every_steps": 2,
                },
                "eval": {
                    "episodes": 2,
                    "oracle_shape": False,
                    "include_cell_actions": True,
                    "max_actions": 80,
                    "beam_width": 1,
                },
            }
        )
        assert "eval_pass1" in metrics
        assert (tmp_path / variant / "checkpoint.pt").exists()


def _task_from_pairs(pairs):
    from puzzle_jepa.data.arc import ARCExample, ARCTask, grid_from_lists

    examples = tuple(ARCExample(grid_from_lists(inp), grid_from_lists(out)) for inp, out in pairs)
    return ARCTask(task_id="synthetic", train=examples, test=())
