from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from puzzle_jepa.data.arc import (
    ARCGrid,
    ARCEpisode,
    ARCTask,
    grid_distance,
    grid_exact,
    grid_key,
    grid_to_compact_text,
    iter_leave_one_out_episodes,
    load_arc_tasks,
    make_initial_arc_candidates,
    task_shape_profile,
)
from puzzle_jepa.data.arc_actions import (
    ARCAction,
    apply_arc_action,
    episode_candidate_shapes,
    episode_palette,
    generate_arc_actions,
)
from puzzle_jepa.data.arc_proposals import build_arc_sources, extract_arc_proposals


@dataclass(frozen=True, slots=True)
class ARCStepTrace:
    depth: int
    action: str
    distance: int
    grid: str


@dataclass(frozen=True, slots=True)
class ARCEpisodeCoverage:
    task_id: str
    query_index: int
    input_shape: tuple[int, int]
    target_shape: tuple[int, int]
    best_distance: int
    initial_distance: int
    solved: bool
    solved_depth: int | None
    best_depth: int
    expanded_states: int
    evaluated_actions: int
    best_trace: tuple[ARCStepTrace, ...]
    action_family_counts: dict[str, int]
    failure_reason: str


@dataclass(frozen=True, slots=True)
class ARCCoverageSummary:
    num_tasks: int
    num_episodes: int
    solved_episodes: int
    solved_rate: float
    mean_initial_distance: float
    mean_best_distance: float
    same_shape_tasks: int
    shape_change_tasks: int
    rows: tuple[ARCEpisodeCoverage, ...]
    task_profiles: dict[str, dict[str, int | bool]]


@dataclass(frozen=True, slots=True)
class _BeamNode:
    grid: ARCGrid
    score: int
    trace: tuple[ARCStepTrace, ...]


def run_arc_episode_coverage(
    episode: ARCEpisode,
    *,
    max_depth: int = 2,
    beam_width: int = 8,
    oracle_shape: bool = False,
    include_cell_actions: bool = True,
    max_cell_actions: int = 300,
    max_actions: int = 2500,
) -> ARCEpisodeCoverage:
    target = episode.target_output
    initial_candidates = make_initial_arc_candidates(episode, oracle_shape=oracle_shape)
    if not initial_candidates:
        raise ValueError("ARC coverage requires at least one initial candidate.")

    initial_nodes = [
        _BeamNode(grid=grid, score=grid_distance(grid, target), trace=())
        for grid in initial_candidates
    ]
    initial_nodes.sort(key=lambda node: (node.score, node.grid.shape))
    initial_distance = initial_nodes[0].score
    beam = _dedupe_nodes(initial_nodes, beam_width=beam_width)
    best = beam[0]
    expanded_states = 0
    evaluated_actions = 0
    family_counts: dict[str, int] = {}

    for depth in range(max(0, int(max_depth)) + 1):
        for node in beam:
            if node.score < best.score:
                best = node
            if grid_exact(node.grid, target):
                return _episode_result(
                    episode,
                    best=node,
                    initial_distance=initial_distance,
                    solved_depth=depth,
                    expanded_states=expanded_states,
                    evaluated_actions=evaluated_actions,
                    action_family_counts=family_counts,
                )
        if depth == max_depth:
            break

        next_nodes: list[_BeamNode] = []
        for node in beam:
            expanded_states += 1
            proposals = extract_arc_proposals(episode.context, episode.query_input, node.grid)
            sources = build_arc_sources(episode.context, episode.query_input, node.grid)
            shapes = episode_candidate_shapes(
                episode.context,
                episode.query_input,
                oracle_shape=target.shape if oracle_shape else None,
            )
            palette = episode_palette(episode.context, episode.query_input, node.grid)
            actions = generate_arc_actions(
                episode.context,
                episode.query_input,
                node.grid,
                proposals=proposals,
                candidate_shapes=shapes,
                palette=palette,
                include_cell_actions=include_cell_actions,
                max_cell_actions=max_cell_actions,
                max_actions=max_actions,
            )
            for action in actions:
                evaluated_actions += 1
                family_counts[action.op] = family_counts.get(action.op, 0) + 1
                try:
                    next_grid = apply_arc_action(node.grid, action, proposals=proposals, sources=sources)
                except ValueError:
                    continue
                score = grid_distance(next_grid, target)
                trace = node.trace + (
                    ARCStepTrace(
                        depth=depth + 1,
                        action=action.label,
                        distance=score,
                        grid=grid_to_compact_text(next_grid),
                    ),
                )
                next_nodes.append(_BeamNode(grid=next_grid, score=score, trace=trace))
        if not next_nodes:
            break
        beam = _dedupe_nodes(sorted(next_nodes, key=lambda node: (node.score, len(node.trace), node.grid.shape)), beam_width=beam_width)

    return _episode_result(
        episode,
        best=best,
        initial_distance=initial_distance,
        solved_depth=None,
        expanded_states=expanded_states,
        evaluated_actions=evaluated_actions,
        action_family_counts=family_counts,
    )


def run_arc_coverage(
    tasks: Iterable[ARCTask],
    *,
    max_depth: int = 2,
    beam_width: int = 8,
    oracle_shape: bool = False,
    include_cell_actions: bool = True,
    max_cell_actions: int = 300,
    max_actions: int = 2500,
    max_episodes_per_task: int | None = None,
) -> ARCCoverageSummary:
    task_list = list(tasks)
    rows: list[ARCEpisodeCoverage] = []
    profiles = {task.task_id: task_shape_profile(task) for task in task_list}
    for task in task_list:
        episodes = list(iter_leave_one_out_episodes(task))
        if max_episodes_per_task is not None:
            episodes = episodes[: int(max_episodes_per_task)]
        for episode in episodes:
            rows.append(
                run_arc_episode_coverage(
                    episode,
                    max_depth=max_depth,
                    beam_width=beam_width,
                    oracle_shape=oracle_shape,
                    include_cell_actions=include_cell_actions,
                    max_cell_actions=max_cell_actions,
                    max_actions=max_actions,
                )
            )
    solved = sum(1 for row in rows if row.solved)
    same_shape = sum(1 for item in profiles.values() if bool(item["all_same_shape"]))
    shape_change = sum(1 for item in profiles.values() if bool(item["has_shape_change"]))
    return ARCCoverageSummary(
        num_tasks=len(task_list),
        num_episodes=len(rows),
        solved_episodes=solved,
        solved_rate=0.0 if not rows else solved / len(rows),
        mean_initial_distance=0.0 if not rows else sum(row.initial_distance for row in rows) / len(rows),
        mean_best_distance=0.0 if not rows else sum(row.best_distance for row in rows) / len(rows),
        same_shape_tasks=same_shape,
        shape_change_tasks=shape_change,
        rows=tuple(rows),
        task_profiles=profiles,
    )


def summary_to_jsonable(summary: ARCCoverageSummary) -> dict:
    payload = asdict(summary)
    payload["rows"] = [asdict(row) for row in summary.rows]
    return payload


def write_summary(summary: ARCCoverageSummary, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_to_jsonable(summary), handle, indent=2, sort_keys=True)


def _episode_result(
    episode: ARCEpisode,
    *,
    best: _BeamNode,
    initial_distance: int,
    solved_depth: int | None,
    expanded_states: int,
    evaluated_actions: int,
    action_family_counts: dict[str, int],
) -> ARCEpisodeCoverage:
    solved = solved_depth is not None or grid_exact(best.grid, episode.target_output)
    reason = "solved" if solved else _failure_reason(best.grid, episode.target_output)
    return ARCEpisodeCoverage(
        task_id=episode.task_id,
        query_index=episode.query_index,
        input_shape=episode.query_input.shape,
        target_shape=episode.target_output.shape,
        best_distance=best.score,
        initial_distance=initial_distance,
        solved=solved,
        solved_depth=solved_depth if solved_depth is not None else (len(best.trace) if grid_exact(best.grid, episode.target_output) else None),
        best_depth=len(best.trace),
        expanded_states=expanded_states,
        evaluated_actions=evaluated_actions,
        best_trace=best.trace,
        action_family_counts=dict(sorted(action_family_counts.items())),
        failure_reason=reason,
    )


def _failure_reason(grid: ARCGrid, target: ARCGrid) -> str:
    if grid.shape != target.shape:
        return f"shape_mismatch:{grid.shape}->{target.shape}"
    return "value_mismatch"


def _dedupe_nodes(nodes: list[_BeamNode], *, beam_width: int) -> list[_BeamNode]:
    kept: list[_BeamNode] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for node in nodes:
        key = grid_key(node.grid)
        if key in seen:
            continue
        kept.append(node)
        seen.add(key)
        if len(kept) >= beam_width:
            break
    return kept


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CPU-only ARC DSL oracle coverage.")
    parser.add_argument("--data-root", required=True, help="ARC-AGI root, data root, or split parent.")
    parser.add_argument("--split", default="training")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of task JSON files.")
    parser.add_argument("--max-episodes-per-task", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--max-actions", type=int, default=2500)
    parser.add_argument("--max-cell-actions", type=int, default=300)
    parser.add_argument("--oracle-shape", action="store_true")
    parser.add_argument("--no-cell-actions", action="store_true")
    parser.add_argument("--output", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    tasks = load_arc_tasks(args.data_root, split=args.split, limit=args.limit)
    summary = run_arc_coverage(
        tasks,
        max_depth=args.max_depth,
        beam_width=args.beam_width,
        oracle_shape=args.oracle_shape,
        include_cell_actions=not args.no_cell_actions,
        max_cell_actions=args.max_cell_actions,
        max_actions=args.max_actions,
        max_episodes_per_task=args.max_episodes_per_task,
    )
    if args.output:
        write_summary(summary, args.output)
    print(
        json.dumps(
            {
                "num_tasks": summary.num_tasks,
                "num_episodes": summary.num_episodes,
                "solved_episodes": summary.solved_episodes,
                "solved_rate": summary.solved_rate,
                "mean_initial_distance": summary.mean_initial_distance,
                "mean_best_distance": summary.mean_best_distance,
                "same_shape_tasks": summary.same_shape_tasks,
                "shape_change_tasks": summary.shape_change_tasks,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
