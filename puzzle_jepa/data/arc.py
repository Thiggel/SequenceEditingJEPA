from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np


ARC_MAX_SIZE = 30
ARC_NUM_COLORS = 10


@dataclass(frozen=True, slots=True)
class ARCGrid:
    values: np.ndarray

    def __post_init__(self) -> None:
        arr = np.asarray(self.values, dtype=np.int64)
        if arr.ndim != 2:
            raise ValueError(f"ARC grid must be rank 2, got shape {arr.shape}.")
        if arr.shape[0] <= 0 or arr.shape[1] <= 0:
            raise ValueError(f"ARC grid must be non-empty, got shape {arr.shape}.")
        if arr.shape[0] > ARC_MAX_SIZE or arr.shape[1] > ARC_MAX_SIZE:
            raise ValueError(f"ARC grid exceeds {ARC_MAX_SIZE}x{ARC_MAX_SIZE}: {arr.shape}.")
        if np.any(arr < 0) or np.any(arr >= ARC_NUM_COLORS):
            raise ValueError("ARC colors must be integer values in [0, 9].")
        object.__setattr__(self, "values", arr.copy())

    @property
    def height(self) -> int:
        return int(self.values.shape[0])

    @property
    def width(self) -> int:
        return int(self.values.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    def padded(self, *, size: int = ARC_MAX_SIZE, fill: int = 0) -> tuple[np.ndarray, np.ndarray]:
        if self.height > size or self.width > size:
            raise ValueError(f"Cannot pad grid of shape {self.shape} to {size}x{size}.")
        values = np.full((size, size), int(fill), dtype=np.int64)
        active = np.zeros((size, size), dtype=bool)
        values[: self.height, : self.width] = self.values
        active[: self.height, : self.width] = True
        return values, active

    def to_lists(self) -> list[list[int]]:
        return self.values.astype(int).tolist()

    def color_set(self) -> set[int]:
        return {int(x) for x in np.unique(self.values)}


@dataclass(frozen=True, slots=True)
class ARCExample:
    input: ARCGrid
    output: ARCGrid | None = None


@dataclass(frozen=True, slots=True)
class ARCTask:
    task_id: str
    train: tuple[ARCExample, ...]
    test: tuple[ARCExample, ...]
    path: Path | None = None


@dataclass(frozen=True, slots=True)
class ARCEpisode:
    task_id: str
    query_index: int
    context: tuple[ARCExample, ...]
    query_input: ARCGrid
    target_output: ARCGrid


def grid_from_lists(values: Iterable[Iterable[int]]) -> ARCGrid:
    rows = [list(row) for row in values]
    if not rows or not rows[0]:
        raise ValueError("ARC grid list must be non-empty.")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("ARC grid rows must have equal length.")
    return ARCGrid(np.asarray(rows, dtype=np.int64))


def load_arc_task(path: str | Path) -> ARCTask:
    task_path = Path(path)
    with task_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    train = tuple(_example_from_json(item, require_output=True) for item in payload.get("train", []))
    test = tuple(_example_from_json(item, require_output=False) for item in payload.get("test", []))
    if not train:
        raise ValueError(f"ARC task {task_path} has no train examples.")
    return ARCTask(task_id=task_path.stem, train=train, test=test, path=task_path)


def load_arc_tasks(root: str | Path, *, split: str = "training", limit: int | None = None) -> list[ARCTask]:
    split_dir = resolve_arc_split_dir(root, split=split)
    paths = sorted(split_dir.glob("*.json"))
    if limit is not None:
        paths = paths[: int(limit)]
    return [load_arc_task(path) for path in paths]


def resolve_arc_split_dir(root: str | Path, *, split: str = "training") -> Path:
    base = Path(root)
    candidates = [
        base / split,
        base / "data" / split,
        base / "ARC-AGI" / "data" / split,
        base / "arc-agi" / "data" / split,
        base / "arc-agi-1" / "data" / split,
    ]
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError(f"Could not find ARC split {split!r} under {base}.")


def iter_leave_one_out_episodes(task: ARCTask) -> Iterator[ARCEpisode]:
    if len(task.train) < 2:
        return
    for index, example in enumerate(task.train):
        if example.output is None:
            continue
        context = tuple(item for j, item in enumerate(task.train) if j != index and item.output is not None)
        yield ARCEpisode(
            task_id=task.task_id,
            query_index=index,
            context=context,
            query_input=example.input,
            target_output=example.output,
        )


def all_episode_grids(episode: ARCEpisode, *, include_target: bool = False) -> list[ARCGrid]:
    grids: list[ARCGrid] = [episode.query_input]
    for example in episode.context:
        grids.append(example.input)
        if example.output is not None:
            grids.append(example.output)
    if include_target:
        grids.append(episode.target_output)
    return grids


def observed_palette(episode: ARCEpisode, *, include_target: bool = False) -> tuple[int, ...]:
    colors: set[int] = set()
    for grid in all_episode_grids(episode, include_target=include_target):
        colors.update(grid.color_set())
    return tuple(sorted(colors))


def observed_output_shapes(episode: ARCEpisode, *, include_target: bool = False) -> tuple[tuple[int, int], ...]:
    shapes = {episode.query_input.shape}
    for example in episode.context:
        if example.output is not None:
            shapes.add(example.output.shape)
    if include_target:
        shapes.add(episode.target_output.shape)
    return tuple(sorted(shapes))


def make_initial_arc_candidates(
    episode: ARCEpisode,
    *,
    oracle_shape: bool = False,
    include_copy_input: bool = True,
) -> list[ARCGrid]:
    shapes = observed_output_shapes(episode, include_target=oracle_shape)
    palette = observed_palette(episode, include_target=False)
    fill_colors = tuple(color for color in palette[:3]) if palette else (0,)
    candidates: list[ARCGrid] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()

    def add(values: np.ndarray) -> None:
        grid = ARCGrid(values)
        key = grid_key(grid)
        if key not in seen:
            candidates.append(grid)
            seen.add(key)

    for height, width in shapes:
        for color in fill_colors:
            add(np.full((height, width), int(color), dtype=np.int64))
    if include_copy_input:
        add(episode.query_input.values)
    return candidates


def grid_key(grid: ARCGrid) -> tuple[tuple[int, int], bytes]:
    return grid.shape, np.ascontiguousarray(grid.values).tobytes()


def grid_exact(grid: ARCGrid, target: ARCGrid) -> bool:
    return grid.shape == target.shape and bool(np.array_equal(grid.values, target.values))


def grid_distance(grid: ARCGrid, target: ARCGrid) -> int:
    height = max(grid.height, target.height)
    width = max(grid.width, target.width)
    left = np.full((height, width), -1, dtype=np.int64)
    right = np.full((height, width), -2, dtype=np.int64)
    left[: grid.height, : grid.width] = grid.values
    right[: target.height, : target.width] = target.values
    return int(np.count_nonzero(left != right))


def grid_to_compact_text(grid: ARCGrid) -> str:
    return "/".join("".join(str(int(x)) for x in row) for row in grid.values)


def task_shape_profile(task: ARCTask) -> dict[str, int | bool]:
    train_pairs = [item for item in task.train if item.output is not None]
    same_shape_pairs = sum(1 for item in train_pairs if item.input.shape == item.output.shape)
    output_shapes = {item.output.shape for item in train_pairs if item.output is not None}
    input_shapes = {item.input.shape for item in train_pairs}
    return {
        "num_train_pairs": len(train_pairs),
        "same_shape_pairs": same_shape_pairs,
        "all_same_shape": same_shape_pairs == len(train_pairs),
        "num_input_shapes": len(input_shapes),
        "num_output_shapes": len(output_shapes),
        "has_shape_change": any(item.input.shape != item.output.shape for item in train_pairs),
    }


def _example_from_json(item: dict, *, require_output: bool) -> ARCExample:
    if "input" not in item:
        raise ValueError("ARC example is missing input grid.")
    output = item.get("output")
    if output is None and require_output:
        raise ValueError("ARC train example is missing output grid.")
    return ARCExample(
        input=grid_from_lists(item["input"]),
        output=None if output is None else grid_from_lists(output),
    )
