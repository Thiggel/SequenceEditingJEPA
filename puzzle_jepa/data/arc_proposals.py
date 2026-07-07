from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

from puzzle_jepa.data.arc import ARCExample, ARCGrid


BBox = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class ARCProposal:
    proposal_id: str
    source: str
    kind: str
    mask: np.ndarray
    bbox: BBox
    colors: tuple[int, ...]
    label: str

    @property
    def area(self) -> int:
        return int(np.count_nonzero(self.mask))


def infer_background_color(grid: ARCGrid) -> int:
    counts = Counter(int(x) for x in grid.values.reshape(-1))
    return min((-count, color) for color, count in counts.items())[1]


def build_arc_sources(
    context: tuple[ARCExample, ...],
    query_input: ARCGrid,
    current_output: ARCGrid,
) -> dict[str, ARCGrid]:
    sources: dict[str, ARCGrid] = {"query_input": query_input, "current_output": current_output}
    for index, example in enumerate(context):
        sources[f"context_{index}_input"] = example.input
        if example.output is not None:
            sources[f"context_{index}_output"] = example.output
    return sources


def extract_arc_proposals(
    context: tuple[ARCExample, ...],
    query_input: ARCGrid,
    current_output: ARCGrid,
    *,
    max_components_per_grid: int = 80,
) -> dict[str, ARCProposal]:
    proposals: list[ARCProposal] = []
    for source, grid in build_arc_sources(context, query_input, current_output).items():
        proposals.extend(
            extract_grid_proposals(
                source,
                grid,
                start_index=len(proposals),
                max_components=max_components_per_grid,
            )
        )
    return {proposal.proposal_id: proposal for proposal in proposals}


def extract_grid_proposals(
    source: str,
    grid: ARCGrid,
    *,
    start_index: int = 0,
    max_components: int = 80,
) -> list[ARCProposal]:
    background = infer_background_color(grid)
    proposals: list[ARCProposal] = []
    seen_masks: set[tuple[str, bytes]] = set()

    def add(kind: str, mask: np.ndarray, label: str) -> None:
        nonlocal proposals
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != grid.shape or not bool(mask.any()):
            return
        key = (kind, np.ascontiguousarray(mask).tobytes())
        if key in seen_masks:
            return
        seen_masks.add(key)
        bbox = mask_bbox(mask)
        colors = tuple(sorted(int(x) for x in np.unique(grid.values[mask])))
        proposal_id = f"p{start_index + len(proposals):04d}"
        proposals.append(
            ARCProposal(
                proposal_id=proposal_id,
                source=source,
                kind=kind,
                mask=mask.copy(),
                bbox=bbox,
                colors=colors,
                label=label,
            )
        )

    add("full_grid", np.ones(grid.shape, dtype=bool), f"{source}:full_grid")
    for color in sorted(int(x) for x in np.unique(grid.values)):
        color_mask = grid.values == color
        if color != background:
            add("color_group", color_mask, f"{source}:color={color}")
            for component in connected_components(color_mask)[:max_components]:
                add("component", component, f"{source}:component color={color}")
                add("bbox", bbox_mask(grid.shape, mask_bbox(component)), f"{source}:bbox color={color}")
                if not _touches_border(component):
                    add("hole", component, f"{source}:enclosed color={color}")
        elif color == 0:
            add("background_group", color_mask, f"{source}:background color=0")
        if color == background:
            for component in connected_components(color_mask)[:max_components]:
                if not _touches_border(component):
                    add("hole", component, f"{source}:hole background={background}")

    non_background = grid.values != background
    if bool(non_background.any()):
        add("non_background", non_background, f"{source}:non_background")
        add("bbox", bbox_mask(grid.shape, mask_bbox(non_background)), f"{source}:bbox non_background")

    for row in range(grid.height):
        mask = np.zeros(grid.shape, dtype=bool)
        mask[row, :] = grid.values[row, :] != background
        if bool(mask.any()):
            add("row", mask, f"{source}:row={row}")
    for col in range(grid.width):
        mask = np.zeros(grid.shape, dtype=bool)
        mask[:, col] = grid.values[:, col] != background
        if bool(mask.any()):
            add("col", mask, f"{source}:col={col}")

    for bbox in candidate_rectangles(grid, background):
        add("rectangle", bbox_mask(grid.shape, bbox), f"{source}:rect={bbox}")
    return proposals


def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    arr = np.asarray(mask, dtype=bool)
    seen = np.zeros(arr.shape, dtype=bool)
    components: list[np.ndarray] = []
    for start_row, start_col in np.argwhere(arr):
        if seen[start_row, start_col]:
            continue
        component = np.zeros(arr.shape, dtype=bool)
        queue: deque[tuple[int, int]] = deque([(int(start_row), int(start_col))])
        seen[start_row, start_col] = True
        while queue:
            row, col = queue.popleft()
            component[row, col] = True
            for next_row, next_col in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if not (0 <= next_row < arr.shape[0] and 0 <= next_col < arr.shape[1]):
                    continue
                if seen[next_row, next_col] or not arr[next_row, next_col]:
                    continue
                seen[next_row, next_col] = True
                queue.append((next_row, next_col))
        components.append(component)
    components.sort(key=lambda item: (-int(np.count_nonzero(item)), mask_bbox(item)))
    return components


def mask_bbox(mask: np.ndarray) -> BBox:
    rows, cols = np.where(np.asarray(mask, dtype=bool))
    if len(rows) == 0:
        raise ValueError("Cannot compute bbox of empty mask.")
    return int(rows.min()), int(cols.min()), int(rows.max()) + 1, int(cols.max()) + 1


def bbox_mask(shape: tuple[int, int], bbox: BBox) -> np.ndarray:
    row0, col0, row1, col1 = bbox
    mask = np.zeros(shape, dtype=bool)
    mask[max(0, row0) : min(shape[0], row1), max(0, col0) : min(shape[1], col1)] = True
    return mask


def candidate_rectangles(grid: ARCGrid, background: int) -> list[BBox]:
    values = grid.values
    bboxes: set[BBox] = set()
    for color in sorted(int(x) for x in np.unique(values) if int(x) != background):
        mask = values == color
        for component in connected_components(mask):
            bbox = mask_bbox(component)
            row0, col0, row1, col1 = bbox
            if row1 - row0 >= 2 and col1 - col0 >= 2:
                bboxes.add(bbox)
    return sorted(bboxes)


def proposal_patch(grid: ARCGrid, proposal: ARCProposal) -> tuple[np.ndarray, np.ndarray]:
    row0, col0, row1, col1 = proposal.bbox
    return grid.values[row0:row1, col0:col1].copy(), proposal.mask[row0:row1, col0:col1].copy()


def _touches_border(mask: np.ndarray) -> bool:
    arr = np.asarray(mask, dtype=bool)
    return bool(arr[0, :].any() or arr[-1, :].any() or arr[:, 0].any() or arr[:, -1].any())
