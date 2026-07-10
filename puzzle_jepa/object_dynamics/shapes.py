from __future__ import annotations

import numpy as np


SHAPE_TYPES = (
    "solid_rectangle",
    "hollow_rectangle",
    "line",
    "l_shape",
    "t_shape",
    "cross",
    "checkerboard",
    "stripes",
    "multipart",
)


def sample_shape_mask(rng: np.random.Generator, shape_type: str, *, max_extent: int = 6) -> np.ndarray:
    if max_extent < 3:
        raise ValueError("max_extent must be at least 3.")
    if shape_type == "solid_rectangle":
        height = int(rng.integers(2, max_extent + 1))
        width = int(rng.integers(2, max_extent + 1))
        return np.ones((height, width), dtype=bool)

    if shape_type == "hollow_rectangle":
        height = int(rng.integers(3, max_extent + 1))
        width = int(rng.integers(3, max_extent + 1))
        mask = np.zeros((height, width), dtype=bool)
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True
        return mask

    if shape_type == "line":
        length = int(rng.integers(3, max_extent + 1))
        if bool(rng.integers(0, 2)):
            return np.ones((1, length), dtype=bool)
        return np.ones((length, 1), dtype=bool)

    if shape_type == "l_shape":
        height = int(rng.integers(3, max_extent + 1))
        width = int(rng.integers(3, max_extent + 1))
        mask = np.zeros((height, width), dtype=bool)
        mask[:, 0] = True
        mask[-1, :] = True
        return _maybe_rotate(mask, rng)

    if shape_type == "cross":
        sizes = [size for size in range(3, max_extent + 1) if size % 2 == 1]
        size = int(sizes[int(rng.integers(0, len(sizes)))])
        mask = np.zeros((size, size), dtype=bool)
        mid = size // 2
        mask[mid, :] = True
        mask[:, mid] = True
        return mask

    if shape_type == "t_shape":
        height = int(rng.integers(3, max_extent + 1))
        width = int(rng.integers(3, max_extent + 1))
        mask = np.zeros((height, width), dtype=bool)
        mask[0, :] = True
        mask[:, width // 2] = True
        return _maybe_rotate(mask, rng)

    if shape_type == "checkerboard":
        height = int(rng.integers(3, max_extent + 1))
        width = int(rng.integers(3, max_extent + 1))
        rows, cols = np.indices((height, width))
        return (rows + cols) % 2 == 0

    if shape_type == "stripes":
        height = int(rng.integers(3, max_extent + 1))
        width = int(rng.integers(3, max_extent + 1))
        rows, cols = np.indices((height, width))
        return rows % 2 == 0 if bool(rng.integers(0, 2)) else cols % 2 == 0

    if shape_type == "multipart":
        height = int(rng.integers(3, max_extent + 1))
        width = int(rng.integers(3, max_extent + 1))
        mask = np.zeros((height, width), dtype=bool)
        part_height = max(1, min(2, height // 2))
        part_width = max(1, min(2, width // 2))
        mask[:part_height, :part_width] = True
        mask[-part_height:, -part_width:] = True
        return _maybe_rotate(mask, rng)

    raise ValueError(f"Unknown shape_type {shape_type!r}.")


def place_mask(mask: np.ndarray, row: int, col: int, shape: tuple[int, int]) -> np.ndarray:
    output = np.zeros(shape, dtype=bool)
    height, width = mask.shape
    output[row : row + height, col : col + width] = mask
    return output


def transform_mask(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    transformed = np.asarray(mask, dtype=bool)
    op = str(rng.choice(["rot90", "rot180", "flip_h", "flip_v"]))
    if op == "rot90":
        transformed = np.rot90(transformed, k=1)
    elif op == "rot180":
        transformed = np.rot90(transformed, k=2)
    elif op == "flip_h":
        transformed = np.flipud(transformed)
    elif op == "flip_v":
        transformed = np.fliplr(transformed)
    return np.ascontiguousarray(transformed)


def frontier_order(mask: np.ndarray, rng: np.random.Generator, *, eight_connected: bool = True) -> list[tuple[int, int]]:
    cells = {tuple(int(x) for x in cell) for cell in np.argwhere(mask)}
    if not cells:
        return []
    start = list(cells)[int(rng.integers(0, len(cells)))]
    order = [start]
    remaining = set(cells)
    remaining.remove(start)
    frontier = {cell for cell in _neighbors(start, eight_connected=eight_connected) if cell in remaining}
    while remaining:
        if frontier:
            choices = sorted(frontier)
            cell = choices[int(rng.integers(0, len(choices)))]
        else:
            choices = sorted(remaining)
            cell = choices[int(rng.integers(0, len(choices)))]
        order.append(cell)
        remaining.remove(cell)
        frontier.discard(cell)
        frontier.update(neighbor for neighbor in _neighbors(cell, eight_connected=eight_connected) if neighbor in remaining)
    return order


def _maybe_rotate(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.ascontiguousarray(np.rot90(mask, k=int(rng.integers(0, 4))))


def _neighbors(cell: tuple[int, int], *, eight_connected: bool) -> tuple[tuple[int, int], ...]:
    row, col = cell
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if eight_connected:
        offsets.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
    return tuple((row + dr, col + dc) for dr, dc in offsets)
