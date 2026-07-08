#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle

from puzzle_jepa.data.arc import ARCGrid, grid_distance, grid_to_compact_text, iter_leave_one_out_episodes, load_arc_task, make_initial_arc_candidates
from puzzle_jepa.data.arc_actions import apply_arc_action, episode_candidate_shapes, episode_palette, generate_arc_actions
from puzzle_jepa.data.arc_proposals import ARCProposal, build_arc_sources, extract_arc_proposals, mask_bbox


ARC_COLORS = ListedColormap(
    [
        "#000000",
        "#0074D9",
        "#FF4136",
        "#2ECC40",
        "#FFDC00",
        "#AAAAAA",
        "#F012BE",
        "#FF851B",
        "#7FDBFF",
        "#870C25",
    ]
)
ARC_NORM = BoundaryNorm(np.arange(-0.5, 10.5, 1.0), ARC_COLORS.N)


@dataclass(frozen=True, slots=True)
class TraceExample:
    task_id: str
    query_index: int
    title: str
    note: str


@dataclass(frozen=True, slots=True)
class ReconstructedTrace:
    task_id: str
    query_index: int
    title: str
    note: str
    context_pairs: tuple[tuple[ARCGrid, ARCGrid], ...]
    query_input: ARCGrid
    current: ARCGrid
    candidate: ARCGrid
    target: ARCGrid
    action_label: str
    action_op: str
    proposal: ARCProposal | None
    proposal_source: ARCGrid | None
    initial_distance: int
    final_distance: int
    solved: bool
    second_candidate: ARCGrid | None = None
    second_action_label: str | None = None


EXAMPLES = (
    TraceExample(
        "11852cab",
        0,
        "Complete missing bbox corners",
        "The proposal is a structured same-shape object; the action fills missing corners.",
    ),
    TraceExample(
        "1b2d62fb",
        0,
        "Crop a relevant sub-grid",
        "The proposal bbox becomes the whole smaller output grid.",
    ),
    TraceExample(
        "0b148d64",
        0,
        "Copy a large source patch",
        "The selected source proposal is copied into the output canvas.",
    ),
    TraceExample(
        "0520fde7",
        0,
        "Rotate a small proposed region",
        "The proposal is transformed before being pasted into the candidate.",
    ),
    TraceExample(
        "1cf80156",
        0,
        "Shape-changing copy",
        "A proposed 4x4 region in the query input becomes the output.",
    ),
    TraceExample(
        "1caeab9d",
        1,
        "Mask-render near miss",
        "A color mask action improves distance but does not solve the task.",
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render ARC proposal/action example diagrams.")
    parser.add_argument("--arc-root", default="/home/atuin/c107fa/c107fa12/datasets/arc-agi")
    parser.add_argument(
        "--coverage",
        default="../sequence-editing-report/assets/arc/arc_agi1_train_limit50_depth1_improved_oracle_shape_no_cell.json",
    )
    parser.add_argument("--output-dir", default="../sequence-editing-report/assets/arc/diagrams")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage = json.loads(Path(args.coverage).read_text(encoding="utf-8"))
    rows = {(row["task_id"], int(row["query_index"])): row for row in coverage["rows"]}

    traces = [
        reconstruct_trace(example, rows[(example.task_id, example.query_index)], Path(args.arc_root))
        for example in EXAMPLES
    ]
    synthetic = synthetic_checkerboard_trace()

    written: list[Path] = []
    overview_path = output_dir / "arc_modeled_examples_overview.pdf"
    with PdfPages(overview_path) as pdf:
        for trace in [*traces, synthetic]:
            figure = render_trace(trace)
            stem = f"{trace.task_id}_q{trace.query_index}_{slugify(trace.title)}"
            png_path = output_dir / f"{stem}.png"
            pdf_path = output_dir / f"{stem}.pdf"
            figure.savefig(png_path, dpi=180, bbox_inches="tight")
            figure.savefig(pdf_path, bbox_inches="tight")
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
            written.extend([png_path, pdf_path])
    written.append(overview_path)

    index_path = output_dir / "README.md"
    index_path.write_text(render_index([*traces, synthetic], written), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "files": [str(path) for path in written]}, indent=2))
    return 0


def reconstruct_trace(example: TraceExample, row: dict, arc_root: Path) -> ReconstructedTrace:
    task = load_arc_task(resolve_training_dir(arc_root) / f"{example.task_id}.json")
    episode = next(item for item in iter_leave_one_out_episodes(task) if item.query_index == example.query_index)
    trace = row["best_trace"]
    if not trace:
        raise ValueError(f"Example {example.task_id} q{example.query_index} has no action trace.")
    action_label = str(trace[0]["action"])
    expected_grid = str(trace[0]["grid"])

    initial_candidates = make_initial_arc_candidates(episode, oracle_shape=True)
    initial_candidates.sort(key=lambda grid: (grid_distance(grid, episode.target_output), grid.shape))
    for current in dedupe_grids(initial_candidates)[:4]:
        proposals = extract_arc_proposals(episode.context, episode.query_input, current)
        sources = build_arc_sources(episode.context, episode.query_input, current)
        actions = generate_arc_actions(
            episode.context,
            episode.query_input,
            current,
            proposals=proposals,
            candidate_shapes=episode_candidate_shapes(episode.context, episode.query_input, oracle_shape=episode.target_output.shape),
            palette=episode_palette(episode.context, episode.query_input, current),
            include_cell_actions=False,
            max_actions=4000,
        )
        for action in actions:
            if action.label != action_label:
                continue
            candidate = apply_arc_action(current, action, proposals=proposals, sources=sources)
            if grid_to_compact_text(candidate) != expected_grid:
                continue
            proposal = proposals.get(str(action.params.get("proposal_id", "")))
            return ReconstructedTrace(
                task_id=example.task_id,
                query_index=example.query_index,
                title=example.title,
                note=example.note,
                context_pairs=tuple(
                    (item.input, item.output)
                    for item in episode.context[:2]
                    if item.output is not None
                ),
                query_input=episode.query_input,
                current=current,
                candidate=candidate,
                target=episode.target_output,
                action_label=action.label,
                action_op=action.op,
                second_candidate=None,
                second_action_label=None,
                proposal=proposal,
                proposal_source=sources[proposal.source] if proposal is not None else None,
                initial_distance=int(row["initial_distance"]),
                final_distance=int(row["best_distance"]),
                solved=bool(row["solved"]),
            )
    raise ValueError(f"Could not reconstruct {example.task_id} q{example.query_index}: {action_label}")


def synthetic_checkerboard_trace() -> ReconstructedTrace:
    query = ARCGrid(
        np.asarray(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 2, 0, 2, 0, 0],
                [0, 0, 2, 0, 0, 0],
                [0, 2, 0, 2, 0, 0],
                [0, 0, 0, 0, 9, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=np.int64,
        )
    )
    current = ARCGrid(np.zeros((6, 6), dtype=np.int64))
    copied = current.values.copy()
    copied[2:5, 2:5] = np.asarray([[2, 0, 2], [0, 2, 0], [2, 0, 2]], dtype=np.int64)
    candidate = ARCGrid(copied)
    target_values = copied.copy()
    target_values[target_values == 2] = 7
    target = ARCGrid(target_values)
    mask = np.zeros(query.shape, dtype=bool)
    mask[1:4, 1:4] = query.values[1:4, 1:4] != 0
    proposal = ARCProposal(
        proposal_id="synthetic_pattern",
        source="query_input",
        kind="component_bbox",
        mask=mask,
        bbox=mask_bbox(mask),
        colors=(2,),
        label="query_input:checkerboard_pattern",
    )
    return ReconstructedTrace(
        task_id="synthetic_checkerboard",
        query_index=0,
        title="Checkerboard copy then recolor",
        note="Illustrative two-step composition: copy_patch pattern, then apply_color_map 2->7. The current first-pass eval only scored one-step candidates.",
        context_pairs=(),
        query_input=query,
        current=current,
        candidate=candidate,
        target=target,
        action_label="copy_patch pattern -> (2,2)",
        action_op="copy_patch",
        second_candidate=target,
        second_action_label="apply_color_map 2->7",
        proposal=proposal,
        proposal_source=query,
        initial_distance=5,
        final_distance=0,
        solved=True,
    )


def render_trace(trace: ReconstructedTrace):
    fig = plt.figure(figsize=(20, 8.5))
    fig.suptitle(
        f"{trace.task_id} q{trace.query_index}: {trace.title}",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    fig.text(0.5, 0.925, trace.note, ha="center", va="top", fontsize=10)
    context_count = max(1, len(trace.context_pairs))
    bottom_cols = 7 if trace.second_candidate is not None else 6
    top_cols = max(bottom_cols, context_count * 3)
    gs = fig.add_gridspec(2, top_cols, height_ratios=[1.0, 1.5], hspace=0.45, wspace=0.75)

    if trace.context_pairs:
        for index, (ctx_input, ctx_output) in enumerate(trace.context_pairs):
            base = index * 3
            draw_grid(fig.add_subplot(gs[0, base]), ctx_input, f"context {index} input")
            draw_arrow(fig.add_subplot(gs[0, base + 1]), "example\nrule")
            draw_grid(fig.add_subplot(gs[0, base + 2]), ctx_output, f"context {index} output")
    else:
        ax = fig.add_subplot(gs[0, :])
        ax.axis("off")
        ax.text(0.5, 0.5, "synthetic illustrative pattern example", ha="center", va="center", fontsize=12)

    draw_grid(fig.add_subplot(gs[1, 0]), trace.query_input, "query input")
    draw_grid(
        fig.add_subplot(gs[1, 1]),
        trace.proposal_source if trace.proposal_source is not None else trace.query_input,
        f"proposal source\n{proposal_label(trace.proposal)}",
        mask=trace.proposal.mask if trace.proposal is not None else None,
    )
    draw_grid(fig.add_subplot(gs[1, 2]), trace.current, f"state: current candidate\nd={trace.initial_distance}")
    draw_arrow(fig.add_subplot(gs[1, 3]), wrap_label(trace.action_label))
    if trace.second_candidate is None:
        draw_grid(fig.add_subplot(gs[1, 4]), trace.candidate, f"successor candidate\nd={trace.final_distance}")
        draw_grid(fig.add_subplot(gs[1, 5]), trace.target, "target output")
    else:
        draw_grid(fig.add_subplot(gs[1, 4]), trace.candidate, "intermediate candidate")
        draw_arrow(fig.add_subplot(gs[1, 5]), wrap_label(trace.second_action_label or "next action"))
        draw_grid(fig.add_subplot(gs[1, 6]), trace.second_candidate, f"successor / target\nd={trace.final_distance}")
    return fig


def draw_grid(ax, grid: ARCGrid, title: str, *, mask: np.ndarray | None = None) -> None:
    values = np.asarray(grid.values)
    ax.imshow(values, cmap=ARC_COLORS, norm=ARC_NORM, interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.set_xticks(np.arange(-0.5, values.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, values.shape[0], 1), minor=True)
    ax.grid(which="minor", color="#d9d9d9", linewidth=0.45)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("#555555")
    if mask is not None:
        clipped = np.asarray(mask, dtype=bool)
        overlay = np.zeros((*values.shape, 4), dtype=float)
        h = min(values.shape[0], clipped.shape[0])
        w = min(values.shape[1], clipped.shape[1])
        overlay[:h, :w, 0] = 1.0
        overlay[:h, :w, 3] = clipped[:h, :w] * 0.28
        ax.imshow(overlay, interpolation="nearest")
        row0, col0, row1, col1 = mask_bbox(clipped)
        ax.add_patch(Rectangle((col0 - 0.5, row0 - 0.5), col1 - col0, row1 - row0, fill=False, edgecolor="#ff1744", linewidth=2.0))


def draw_arrow(ax, label: str) -> None:
    ax.axis("off")
    ax.annotate(
        "",
        xy=(0.9, 0.5),
        xytext=(0.1, 0.5),
        arrowprops={"arrowstyle": "->", "lw": 2.5, "color": "#333333"},
    )
    ax.text(0.5, 0.58, label, ha="center", va="bottom", fontsize=8)


def render_index(traces: list[ReconstructedTrace], files: list[Path]) -> str:
    lines = [
        "# ARC Modeled Example Diagrams",
        "",
        "Generated by `scripts/analysis/render_arc_example_diagrams.py`.",
        "",
        "Each diagram shows context examples, query input, the proposal/mask used by the action, the current candidate state, the action label, the successor candidate, and the target output.",
        "",
        "## Examples",
        "",
    ]
    for trace in traces:
        stem = f"{trace.task_id}_q{trace.query_index}_{slugify(trace.title)}"
        lines.append(f"- `{stem}.png` / `{stem}.pdf`: {trace.note}")
    lines.extend(["", "## Generated Files", ""])
    for path in files:
        lines.append(f"- `{path.name}`")
    lines.append("")
    return "\n".join(lines)


def proposal_label(proposal: ARCProposal | None) -> str:
    if proposal is None:
        return "no proposal"
    return f"{proposal.proposal_id} {proposal.kind} from {proposal.source}"


def wrap_label(label: str) -> str:
    return "\n".join(textwrap.wrap(label, width=28))


def slugify(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text.lower()).strip("_")


def resolve_training_dir(root: Path) -> Path:
    candidates = [root / "data" / "training", root / "training"]
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError(f"Could not resolve ARC training dir under {root}.")


def dedupe_grids(grids: list[ARCGrid]) -> list[ARCGrid]:
    seen = set()
    out = []
    for grid in grids:
        key = (grid.shape, grid.values.tobytes())
        if key in seen:
            continue
        seen.add(key)
        out.append(grid)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
