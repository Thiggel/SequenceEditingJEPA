from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from puzzle_jepa.data.worlds import PuzzleExample, PuzzleWorld, SudokuWorld, WorldAction
from puzzle_jepa.eval.diagnostics import (
    apply_planning_action,
    board_as_list,
    clue_mask_for_planning,
    is_terminal_state,
    legal_planning_actions,
    load_examples,
    load_model_checkpoint_state,
    summarize_plan_summaries,
    terminal_step_limit,
)
from puzzle_jepa.models import ActionConditionedWorldModel
from puzzle_jepa.train.grid0 import _build_world


@dataclass
class Node:
    state: np.ndarray
    parent: Node | None = None
    action: WorldAction | None = None
    prior: float = 1.0
    visits: int = 0
    value_sum: float = 0.0
    children: list[Node] = field(default_factory=list)
    expanded: bool = False

    @property
    def value(self) -> float:
        return self.value_sum / max(1, self.visits)


@torch.no_grad()
def value_state_scores(
    model: ActionConditionedWorldModel,
    states: list[np.ndarray],
    initial: np.ndarray,
    task_id: int,
) -> np.ndarray:
    if not states:
        return np.asarray([], dtype=np.float32)
    device = next(model.parameters()).device
    state_tensor = torch.as_tensor(np.stack(states), dtype=torch.long, device=device)
    initial_tensor = torch.as_tensor(
        np.repeat(initial[None, ...], len(states), axis=0),
        dtype=torch.long,
        device=device,
    )
    values = model.predict_goal_energy(
        state_tensor,
        initial_tensor,
        torch.full((len(states),), task_id, dtype=torch.long, device=device),
    )
    return values.detach().cpu().numpy().astype(np.float64)


def expand_node(
    node: Node,
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    goal: np.ndarray,
    initial: np.ndarray,
    clue_mask: np.ndarray | None,
    rng: np.random.Generator,
    max_actions: int,
) -> float:
    if is_terminal_state(world, node.state, goal, clue_mask):
        node.expanded = True
        return float(world.is_goal(node.state, goal))
    actions = legal_planning_actions(world, node.state, clue_mask)
    if max_actions > 0 and len(actions) > max_actions:
        indices = rng.choice(len(actions), size=max_actions, replace=False)
        actions = [actions[int(index)] for index in indices]
    next_states: list[np.ndarray] = []
    next_actions: list[WorldAction] = []
    for action in actions:
        try:
            next_states.append(apply_planning_action(world, node.state, action, clue_mask))
            next_actions.append(action)
        except ValueError:
            continue
    values = value_state_scores(model, next_states, initial, world.task_id)
    if len(next_states) == 0:
        node.expanded = True
        return 0.0
    priors = np.ones(len(next_states), dtype=np.float64) / float(len(next_states))
    node.children = [
        Node(state=state, parent=node, action=action, prior=float(prior))
        for state, action, prior in zip(next_states, next_actions, priors, strict=True)
    ]
    node.expanded = True
    return float(np.max(values))


def select_child(node: Node, c_puct: float) -> Node:
    total = math.sqrt(max(1, node.visits))
    return max(
        node.children,
        key=lambda child: child.value + float(c_puct) * child.prior * total / (1 + child.visits),
    )


def run_mcts_step(
    root: Node,
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    goal: np.ndarray,
    initial: np.ndarray,
    clue_mask: np.ndarray | None,
    rng: np.random.Generator,
    simulations: int,
    max_depth: int,
    max_actions: int,
    c_puct: float,
) -> Node | None:
    if not root.expanded:
        expand_node(root, model, world, goal, initial, clue_mask, rng, max_actions)
    for _ in range(max(1, simulations)):
        node = root
        path = [node]
        depth = 0
        while node.expanded and node.children and depth < max_depth:
            node = select_child(node, c_puct)
            path.append(node)
            depth += 1
        value = expand_node(node, model, world, goal, initial, clue_mask, rng, max_actions)
        for item in path:
            item.visits += 1
            item.value_sum += value
    if not root.children:
        return None
    return max(root.children, key=lambda child: (child.visits, child.value))


def value_mcts_plan(
    model: ActionConditionedWorldModel,
    world: PuzzleWorld,
    example: PuzzleExample,
    rng: np.random.Generator,
    *,
    max_steps: int,
    simulations: int,
    max_depth: int,
    max_actions: int,
    c_puct: float,
) -> dict[str, Any]:
    start = world.validate_state(example.state).copy()
    goal = world.validate_state(example.goal)
    clue_mask = clue_mask_for_planning(world, start)
    initial = start.copy()
    current = start.copy()
    trajectory = [board_as_list(current)]
    limit = min(int(max_steps), terminal_step_limit(world, example, max_steps))
    for step in range(limit):
        if world.is_goal(current, goal):
            break
        root = Node(state=current)
        child = run_mcts_step(
            root,
            model,
            world,
            goal,
            initial,
            clue_mask,
            rng,
            simulations=simulations,
            max_depth=max_depth,
            max_actions=max_actions,
            c_puct=c_puct,
        )
        if child is None:
            break
        current = child.state.copy()
        trajectory.append(board_as_list(current))
    return {
        "planner": "value_mcts",
        "solved": float(world.is_goal(current, goal)),
        "terminal": float(is_terminal_state(world, current, goal, clue_mask)),
        "steps": float(len(trajectory) - 1),
        "energy": float(-value_state_scores(model, [current], initial, world.task_id)[0]),
        "remaining_hamming": float(np.not_equal(current, goal).sum()),
        "final_state": board_as_list(current),
        "goal_state": board_as_list(goal),
        "trajectory_states": trajectory,
    }


def run_value_mcts(
    run_root: Path,
    *,
    output_dir: Path,
    num_examples: int,
    seed: int,
    max_steps: int,
    simulations: int,
    max_depth: int,
    max_actions: int,
    c_puct: float,
) -> dict[str, Any]:
    checkpoint_path = run_root / "checkpoint.pt"
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    world = _build_world(dict(config["task"]))
    model = ActionConditionedWorldModel(vocab_size=world.vocab_size, **dict(config["model"]))
    load_model_checkpoint_state(model, checkpoint["model"])
    model.eval()
    examples = load_examples(dict(config["task"]), "eval")
    rng = np.random.default_rng(seed)
    records = [
        value_mcts_plan(
            model,
            world,
            examples[int(rng.integers(0, len(examples)))],
            rng,
            max_steps=max_steps,
            simulations=simulations,
            max_depth=max_depth,
            max_actions=max_actions,
            c_puct=c_puct,
        )
        for _ in range(num_examples)
    ]
    summary = {
        "run_root": str(run_root),
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "value_mcts": summarize_plan_summaries(records),
        "records": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "diagnostics.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run value-head MCTS diagnostics.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-examples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=81)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--max-actions", type=int, default=64)
    parser.add_argument("--c-puct", type=float, default=1.0)
    summary = run_value_mcts(**vars(parser.parse_args()))
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
