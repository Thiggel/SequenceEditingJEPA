from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from puzzle_jepa.data.arc import ARCExample, ARCGrid
from puzzle_jepa.data.arc_proposals import ARCProposal, BBox, build_arc_sources, infer_background_color, proposal_patch


@dataclass(frozen=True, slots=True)
class ARCAction:
    op: str
    params: dict[str, Any]
    label: str


def apply_arc_action(
    candidate: ARCGrid,
    action: ARCAction,
    *,
    proposals: dict[str, ARCProposal],
    sources: dict[str, ARCGrid],
) -> ARCGrid:
    op = action.op
    params = action.params
    values = candidate.values.copy()

    if op == "set_canvas":
        height = int(params["height"])
        width = int(params["width"])
        fill = int(params.get("fill", 0))
        return ARCGrid(np.full((height, width), fill, dtype=np.int64))

    if op == "set_cell":
        row = int(params["row"])
        col = int(params["col"])
        if 0 <= row < candidate.height and 0 <= col < candidate.width:
            values[row, col] = int(params["color"])
        return ARCGrid(values)

    if op == "fill_bbox":
        row0, col0, row1, col1 = _bbox_from_params(params["bbox"])
        values[max(0, row0) : min(candidate.height, row1), max(0, col0) : min(candidate.width, col1)] = int(params["color"])
        return ARCGrid(values)

    if op == "fill_mask":
        proposal = proposals[str(params["proposal_id"])]
        color = int(params["color"])
        values = _fill_mask(values, proposal.mask, color)
        return ARCGrid(values)

    if op == "complete_bbox_corners":
        proposal = proposals[str(params["proposal_id"])]
        color = int(params["color"])
        background = infer_background_color(candidate)
        for row, col in _bbox_corners(proposal.bbox):
            if 0 <= row < candidate.height and 0 <= col < candidate.width and values[row, col] == background:
                values[row, col] = color
        return ARCGrid(values)

    if op == "complete_rectangle":
        proposal = proposals[str(params["proposal_id"])]
        color = int(params["color"])
        background = infer_background_color(candidate)
        row0, col0, row1, col1 = proposal.bbox
        window = values[row0:row1, col0:col1]
        window[window == background] = color
        values[row0:row1, col0:col1] = window
        return ARCGrid(values)

    if op == "copy_patch":
        proposal = proposals[str(params["proposal_id"])]
        source = sources[proposal.source]
        patch, mask = proposal_patch(source, proposal)
        return ARCGrid(_paste_patch(values, patch, mask, int(params["dst_row"]), int(params["dst_col"])))

    if op == "crop":
        proposal = proposals[str(params["proposal_id"])]
        source = sources[proposal.source]
        row0, col0, row1, col1 = proposal.bbox
        return ARCGrid(source.values[row0:row1, col0:col1])

    if op == "recolor":
        proposal = proposals[str(params["proposal_id"])]
        if proposal.source != "current_output":
            return ARCGrid(values)
        values = _fill_mask(values, proposal.mask, int(params["color"]))
        return ARCGrid(values)

    if op == "translate":
        proposal = proposals[str(params["proposal_id"])]
        if proposal.source != "current_output":
            return ARCGrid(values)
        background = infer_background_color(candidate)
        patch, mask = proposal_patch(candidate, proposal)
        row0, col0, _, _ = proposal.bbox
        values[proposal.mask] = background
        values = _paste_patch(values, patch, mask, row0 + int(params["dr"]), col0 + int(params["dc"]))
        return ARCGrid(values)

    if op == "reflect" or op == "rotate":
        proposal = proposals[str(params["proposal_id"])]
        source = sources[proposal.source]
        patch, mask = proposal_patch(source, proposal)
        if op == "reflect":
            axis = str(params["axis"])
            patch = np.flipud(patch) if axis == "horizontal" else np.fliplr(patch)
            mask = np.flipud(mask) if axis == "horizontal" else np.fliplr(mask)
        else:
            turns = int(params["turns"]) % 4
            patch = np.rot90(patch, k=turns)
            mask = np.rot90(mask, k=turns)
        dst_row = int(params.get("dst_row", proposal.bbox[0]))
        dst_col = int(params.get("dst_col", proposal.bbox[1]))
        if proposal.source == "current_output":
            background = infer_background_color(candidate)
            values[proposal.mask] = background
        return ARCGrid(_paste_patch(values, patch, mask, dst_row, dst_col))

    if op == "partition_map":
        source = sources[str(params["source"])]
        return ARCGrid(_apply_partition_map(source.values, params))

    if op == "scale_source":
        source = sources[str(params["source"])]
        scaled = _scale_array(source.values, int(params["factor"]))
        if "height" in params and "width" in params:
            scaled = _fit_to_shape(scaled, int(params["height"]), int(params["width"]), fill=int(params.get("fill", 0)))
        return ARCGrid(scaled)

    if op == "scale_patch":
        proposal = proposals[str(params["proposal_id"])]
        source = sources[proposal.source]
        patch, mask = proposal_patch(source, proposal)
        scaled_patch = _scale_array(patch, int(params["factor"]))
        scaled_mask = _scale_array(mask.astype(np.int64), int(params["factor"])).astype(bool)
        return ARCGrid(_paste_patch(values, scaled_patch, scaled_mask, int(params["dst_row"]), int(params["dst_col"])))

    if op == "apply_color_map":
        source = sources[str(params["source"])]
        mapped = source.values.copy()
        mapped[mapped == int(params["from_color"])] = int(params["to_color"])
        return ARCGrid(mapped)

    if op == "render_color_mask":
        source = sources[str(params["source"])]
        fg = int(params["foreground"])
        bg = int(params["background"])
        rendered = np.full(source.shape, bg, dtype=np.int64)
        rendered[source.values == int(params["color"])] = fg
        return ARCGrid(rendered)

    raise ValueError(f"Unsupported ARC action op {op!r}.")


def generate_arc_actions(
    context: tuple[ARCExample, ...],
    query_input: ARCGrid,
    candidate: ARCGrid,
    *,
    proposals: dict[str, ARCProposal],
    candidate_shapes: tuple[tuple[int, int], ...],
    palette: tuple[int, ...] | None = None,
    include_cell_actions: bool = True,
    max_cell_actions: int = 500,
    max_copy_destinations: int = 24,
    max_actions: int = 4000,
) -> list[ARCAction]:
    del context
    if palette is None:
        colors = sorted({int(x) for x in np.unique(candidate.values)} | {int(x) for x in np.unique(query_input.values)})
    else:
        colors = list(palette)
    if not colors:
        colors = [0]

    actions: list[ARCAction] = []
    seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()

    def add(op: str, params: dict[str, Any], label: str) -> None:
        key = (op, tuple(sorted((str(k), repr(v)) for k, v in params.items())))
        if key in seen:
            return
        seen.add(key)
        actions.append(ARCAction(op=op, params=params, label=label))

    for height, width in candidate_shapes:
        if (height, width) != candidate.shape:
            for color in colors[:3]:
                add("set_canvas", {"height": height, "width": width, "fill": int(color)}, f"set_canvas {height}x{width} fill={color}")

    for source_name in ("query_input", "current_output"):
        if source_name == "query_input":
            source_grid = query_input
        else:
            source_grid = candidate
        for from_color in colors:
            for to_color in colors:
                if from_color != to_color:
                    add(
                        "apply_color_map",
                        {"source": source_name, "from_color": int(from_color), "to_color": int(to_color)},
                        f"color_map {source_name} {from_color}->{to_color}",
                    )
            for foreground in colors:
                add(
                    "render_color_mask",
                    {
                        "source": source_name,
                        "color": int(from_color),
                        "foreground": int(foreground),
                        "background": 0,
                    },
                    f"render_color_mask {source_name} color={from_color} fg={foreground}",
                )
        for factor in (2, 3, 4):
            scaled_shape = (source_grid.height * factor, source_grid.width * factor)
            if scaled_shape[0] <= 30 and scaled_shape[1] <= 30:
                add(
                    "scale_source",
                    {"source": source_name, "factor": factor},
                    f"scale_source {source_name} x{factor}",
                )
            for height, width in candidate_shapes:
                if height <= 30 and width <= 30:
                    add(
                        "scale_source",
                        {"source": source_name, "factor": factor, "height": height, "width": width, "fill": 0},
                        f"scale_source {source_name} x{factor} fit={height}x{width}",
                    )

    if include_cell_actions:
        cell_budget = max(0, int(max_cell_actions))
        produced = 0
        for row in range(candidate.height):
            for col in range(candidate.width):
                for color in colors:
                    if int(candidate.values[row, col]) == int(color):
                        continue
                    add("set_cell", {"row": row, "col": col, "color": int(color)}, f"set_cell ({row},{col})={color}")
                    produced += 1
                    if produced >= cell_budget:
                        break
                if produced >= cell_budget:
                    break
            if produced >= cell_budget:
                break

    for out_h, out_w in candidate_shapes:
        for mode in ("majority", "presence"):
            for color in colors:
                add(
                    "partition_map",
                    {"source": "query_input", "height": out_h, "width": out_w, "mode": mode, "color": int(color)},
                    f"partition_map query_input -> {out_h}x{out_w} {mode} color={color}",
                )
        for row_offset in (0, 1):
            for col_offset in (0, 1):
                add(
                    "partition_map",
                    {
                        "source": "query_input",
                        "height": out_h,
                        "width": out_w,
                        "mode": "stride_sample",
                        "row_offset": row_offset,
                        "col_offset": col_offset,
                    },
                    f"stride_sample query_input -> {out_h}x{out_w} offset={row_offset},{col_offset}",
                )

    proposal_items = sorted(proposals.values(), key=lambda item: (item.kind, -item.area, item.proposal_id))
    for proposal in proposal_items:
        row0, col0, row1, col1 = proposal.bbox
        if row0 < candidate.height and col0 < candidate.width:
            for color in colors:
                add("fill_bbox", {"bbox": proposal.bbox, "color": int(color)}, f"fill_bbox {proposal.proposal_id} color={color}")
                add("fill_mask", {"proposal_id": proposal.proposal_id, "color": int(color)}, f"fill_mask {proposal.proposal_id} color={color}")
                add(
                    "complete_bbox_corners",
                    {"proposal_id": proposal.proposal_id, "color": int(color)},
                    f"complete_corners {proposal.proposal_id} color={color}",
                )
                if row1 - row0 >= 2 and col1 - col0 >= 2:
                    add(
                        "complete_rectangle",
                        {"proposal_id": proposal.proposal_id, "color": int(color)},
                        f"complete_rectangle {proposal.proposal_id} color={color}",
                    )
        if proposal.source == "current_output":
            for color in colors:
                add("recolor", {"proposal_id": proposal.proposal_id, "color": int(color)}, f"recolor {proposal.proposal_id} color={color}")
            for dr, dc in _small_offsets():
                add("translate", {"proposal_id": proposal.proposal_id, "dr": dr, "dc": dc}, f"translate {proposal.proposal_id} dr={dr} dc={dc}")

        for dst_row, dst_col in _copy_destinations(candidate.shape, proposal.bbox, max_destinations=max_copy_destinations):
            add(
                "copy_patch",
                {"proposal_id": proposal.proposal_id, "dst_row": dst_row, "dst_col": dst_col},
                f"copy_patch {proposal.proposal_id} -> ({dst_row},{dst_col})",
            )
            for axis in ("horizontal", "vertical"):
                add(
                    "reflect",
                    {"proposal_id": proposal.proposal_id, "axis": axis, "dst_row": dst_row, "dst_col": dst_col},
                    f"reflect {proposal.proposal_id} {axis} -> ({dst_row},{dst_col})",
                )
            for turns in (1, 2, 3):
                add(
                    "rotate",
                    {"proposal_id": proposal.proposal_id, "turns": turns, "dst_row": dst_row, "dst_col": dst_col},
                    f"rotate {proposal.proposal_id} k={turns} -> ({dst_row},{dst_col})",
                )
            for factor in (2, 3, 4):
                patch_h = (proposal.bbox[2] - proposal.bbox[0]) * factor
                patch_w = (proposal.bbox[3] - proposal.bbox[1]) * factor
                if patch_h <= 30 and patch_w <= 30:
                    add(
                        "scale_patch",
                        {"proposal_id": proposal.proposal_id, "factor": factor, "dst_row": dst_row, "dst_col": dst_col},
                        f"scale_patch {proposal.proposal_id} x{factor} -> ({dst_row},{dst_col})",
                    )
        add("crop", {"proposal_id": proposal.proposal_id}, f"crop {proposal.proposal_id}")

        if len(actions) >= max_actions:
            return actions[:max_actions]
    return actions[:max_actions]


def episode_palette(context: tuple[ARCExample, ...], query_input: ARCGrid, candidate: ARCGrid) -> tuple[int, ...]:
    colors = set(query_input.color_set()) | set(candidate.color_set())
    for example in context:
        colors.update(example.input.color_set())
        if example.output is not None:
            colors.update(example.output.color_set())
    return tuple(sorted(colors))


def episode_candidate_shapes(
    context: tuple[ARCExample, ...],
    query_input: ARCGrid,
    *,
    oracle_shape: tuple[int, int] | None = None,
) -> tuple[tuple[int, int], ...]:
    shapes = {query_input.shape}
    for example in context:
        if example.output is not None:
            shapes.add(example.output.shape)
    if oracle_shape is not None:
        shapes.add(oracle_shape)
    return tuple(sorted(shapes))


def _paste_patch(values: np.ndarray, patch: np.ndarray, mask: np.ndarray, dst_row: int, dst_col: int) -> np.ndarray:
    output = values.copy()
    for local_row, local_col in np.argwhere(mask):
        row = dst_row + int(local_row)
        col = dst_col + int(local_col)
        if 0 <= row < output.shape[0] and 0 <= col < output.shape[1]:
            output[row, col] = int(patch[local_row, local_col])
    return output


def _fill_mask(values: np.ndarray, mask: np.ndarray, color: int) -> np.ndarray:
    output = values.copy()
    height = min(output.shape[0], mask.shape[0])
    width = min(output.shape[1], mask.shape[1])
    clipped = mask[:height, :width]
    output[:height, :width][clipped] = int(color)
    return output


def _bbox_from_params(value: Any) -> BBox:
    row0, col0, row1, col1 = value
    return int(row0), int(col0), int(row1), int(col1)


def _bbox_corners(bbox: BBox) -> tuple[tuple[int, int], ...]:
    row0, col0, row1, col1 = bbox
    return ((row0, col0), (row0, col1 - 1), (row1 - 1, col0), (row1 - 1, col1 - 1))


def _small_offsets() -> tuple[tuple[int, int], ...]:
    offsets: list[tuple[int, int]] = []
    for dr in (-3, -2, -1, 1, 2, 3):
        offsets.append((dr, 0))
    for dc in (-3, -2, -1, 1, 2, 3):
        offsets.append((0, dc))
    for dr in (-1, 1):
        for dc in (-1, 1):
            offsets.append((dr, dc))
    return tuple(offsets)


def _copy_destinations(shape: tuple[int, int], bbox: BBox, *, max_destinations: int) -> tuple[tuple[int, int], ...]:
    height, width = shape
    row0, col0, row1, col1 = bbox
    patch_h = row1 - row0
    patch_w = col1 - col0
    candidates = {
        (row0, col0),
        (0, 0),
        (0, max(0, width - patch_w)),
        (max(0, height - patch_h), 0),
        (max(0, height - patch_h), max(0, width - patch_w)),
        ((height - patch_h) // 2, (width - patch_w) // 2),
    }
    if height * width <= 100:
        for row in range(max(1, height - patch_h + 1)):
            for col in range(max(1, width - patch_w + 1)):
                candidates.add((row, col))
                if len(candidates) >= max_destinations:
                    break
            if len(candidates) >= max_destinations:
                break
    valid = [(row, col) for row, col in candidates if 0 <= row < height and 0 <= col < width]
    return tuple(sorted(valid)[:max_destinations])


def _apply_partition_map(source: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    out_h = int(params["height"])
    out_w = int(params["width"])
    mode = str(params["mode"])
    output = np.zeros((out_h, out_w), dtype=np.int64)

    if mode == "stride_sample":
        row_offset = int(params.get("row_offset", 0))
        col_offset = int(params.get("col_offset", 0))
        if out_h <= 0 or out_w <= 0:
            return output
        row_step = max(1, (source.shape[0] - row_offset) // out_h)
        col_step = max(1, (source.shape[1] - col_offset) // out_w)
        for row in range(out_h):
            for col in range(out_w):
                src_row = min(source.shape[0] - 1, row_offset + row * row_step)
                src_col = min(source.shape[1] - 1, col_offset + col * col_step)
                output[row, col] = int(source[src_row, src_col])
        return output

    for row in range(out_h):
        row0 = int(round(row * source.shape[0] / out_h))
        row1 = int(round((row + 1) * source.shape[0] / out_h))
        for col in range(out_w):
            col0 = int(round(col * source.shape[1] / out_w))
            col1 = int(round((col + 1) * source.shape[1] / out_w))
            block = source[row0:max(row0 + 1, row1), col0:max(col0 + 1, col1)]
            if mode == "presence":
                color = int(params["color"])
                output[row, col] = color if bool(np.any(block == color)) else 0
            elif mode == "majority":
                values, counts = np.unique(block, return_counts=True)
                output[row, col] = int(values[np.argmax(counts)])
            else:
                raise ValueError(f"Unsupported partition_map mode {mode!r}.")
    return output


def _scale_array(values: np.ndarray, factor: int) -> np.ndarray:
    factor = int(factor)
    if factor <= 0:
        raise ValueError("Scale factor must be positive.")
    return np.repeat(np.repeat(values, factor, axis=0), factor, axis=1)


def _fit_to_shape(values: np.ndarray, height: int, width: int, *, fill: int = 0) -> np.ndarray:
    output = np.full((height, width), int(fill), dtype=np.int64)
    copy_h = min(height, values.shape[0])
    copy_w = min(width, values.shape[1])
    output[:copy_h, :copy_w] = values[:copy_h, :copy_w]
    return output
