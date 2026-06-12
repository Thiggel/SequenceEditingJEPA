from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from puzzle_jepa.data import SudokuWorld, WorldAction
from puzzle_jepa.eval.grid5_diagnostics import candidate_actions, load_model
from puzzle_jepa.models import SigRegActionJEPA
from puzzle_jepa.train.grid0 import _build_world, _load_examples


@dataclass(slots=True)
class SearchNode:
    board: np.ndarray
    latent: torch.Tensor
    latent_history: list[torch.Tensor]
    action_history: list[torch.Tensor]
    first_action: WorldAction | None
    score: float


class MCTSNode:
    def __init__(
        self,
        *,
        board: np.ndarray,
        latent: torch.Tensor,
        latent_history: list[torch.Tensor],
        action_history: list[torch.Tensor],
        first_action: WorldAction | None,
        depth: int,
    ):
        self.board = board
        self.latent = latent
        self.latent_history = latent_history
        self.action_history = action_history
        self.first_action = first_action
        self.depth = int(depth)
        self.children: list[MCTSNode] = []
        self.untried_actions: list[WorldAction] | None = None
        self.visits = 0
        self.value_sum = 0.0
        self.best_leaf_score = float("inf")
        self.best_leaf_remaining_hamming: int | None = None
        self.best_leaf_terminal = False

    @property
    def mean_value(self) -> float:
        return self.value_sum / max(1, self.visits)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def encode_board(model: SigRegActionJEPA, board: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(board[None], dtype=torch.long, device=device)
    return model.encode(tensor)


def score_latent(
    model: SigRegActionJEPA,
    latent: torch.Tensor,
    goal_np: np.ndarray,
    initial_np: np.ndarray,
    score_mode: str,
    device: torch.device,
) -> float:
    goal = torch.as_tensor(goal_np[None], dtype=torch.long, device=device)
    initial = torch.as_tensor(initial_np[None], dtype=torch.long, device=device)
    goal_latent = model.encode(goal)
    initial_latent = model.encode(initial)
    if score_mode == "latent_goal":
        score = F.mse_loss(latent, goal_latent, reduction="none").mean(dim=-1)
    elif score_mode == "goal_energy":
        score = model.predict_goal_energy_from_latents(latent, initial_latent)
    else:
        raise ValueError(f"unknown score_mode {score_mode!r}.")
    return float(score.detach().cpu().item())


def action_tensor(world: SudokuWorld, action: WorldAction, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(action.as_array(world.task_id)[None], dtype=torch.long, device=device)


def action_pool(world: SudokuWorld, board: np.ndarray, clue_mask: np.ndarray, action_mode: str) -> list[WorldAction]:
    actions = candidate_actions(world, board, clue_mask)
    if action_mode == "mutable_overwrite":
        return actions
    if action_mode == "fill_empty":
        return [action for action in actions if int(board[action.row, action.col]) == 0]
    raise ValueError(f"unknown action_mode {action_mode!r}.")


def apply_action(
    world: SudokuWorld,
    board: np.ndarray,
    action: WorldAction,
    *,
    clue_mask: np.ndarray,
    action_mode: str,
) -> np.ndarray:
    return world.apply(
        board,
        action,
        clue_mask=clue_mask,
        allow_overwrite=action_mode == "mutable_overwrite",
        allow_conflicts=True,
    )


def predict_next_from_history(
    model: SigRegActionJEPA,
    latent_history: list[torch.Tensor],
    action_history: list[torch.Tensor],
    next_action: torch.Tensor,
) -> torch.Tensor:
    if model.predictor_type == "mlp":
        return model.predict_next(latent_history[-1], next_action)
    latents = [*latent_history]
    actions = [*action_history, next_action]
    max_history = max(1, int(model.max_rollout_steps))
    latents = latents[-max_history:]
    actions = actions[-len(latents) :]
    latent_window = torch.stack(latents, dim=1)
    action_window = torch.stack(actions, dim=1)
    return model.predict_sequence(latent_window, action_window)[:, -1]


def transition_child(
    model: SigRegActionJEPA,
    world: SudokuWorld,
    node: SearchNode | MCTSNode,
    action: WorldAction,
    *,
    transition_mode: str,
    clue_mask: np.ndarray,
    goal_np: np.ndarray,
    initial_np: np.ndarray,
    score_mode: str,
    action_mode: str,
    device: torch.device,
) -> tuple[np.ndarray, torch.Tensor, list[torch.Tensor], list[torch.Tensor], float]:
    next_board = apply_action(world, node.board, action, clue_mask=clue_mask, action_mode=action_mode)
    action_t = action_tensor(world, action, device)
    if transition_mode == "symbolic_reencode":
        next_latent = encode_board(model, next_board, device)
    elif transition_mode == "latent_rollout":
        next_latent = predict_next_from_history(model, node.latent_history, node.action_history, action_t)
    else:
        raise ValueError(f"unknown transition_mode {transition_mode!r}.")
    latent_history = [*node.latent_history, next_latent]
    action_history = [*node.action_history, action_t]
    score = score_latent(model, next_latent, goal_np, initial_np, score_mode, device)
    return next_board, next_latent, latent_history, action_history, score


def top_actions(
    model: SigRegActionJEPA,
    world: SudokuWorld,
    node: SearchNode | MCTSNode,
    *,
    transition_mode: str,
    clue_mask: np.ndarray,
    goal_np: np.ndarray,
    initial_np: np.ndarray,
    score_mode: str,
    action_mode: str,
    branch_size: int,
    device: torch.device,
) -> list[WorldAction]:
    actions = action_pool(world, node.board, clue_mask, action_mode)
    if len(actions) <= max(1, branch_size):
        return actions
    scored = []
    for action in actions:
        _, _, _, _, score = transition_child(
            model,
            world,
            node,
            action,
            transition_mode=transition_mode,
            clue_mask=clue_mask,
            goal_np=goal_np,
            initial_np=initial_np,
            score_mode=score_mode,
            action_mode=action_mode,
            device=device,
        )
        scored.append((score, action))
    scored.sort(key=lambda item: item[0])
    return [action for _score, action in scored[: max(1, branch_size)]]


@torch.no_grad()
def beam_search_plan_once(
    model: SigRegActionJEPA,
    world: SudokuWorld,
    board: np.ndarray,
    goal_np: np.ndarray,
    initial_np: np.ndarray,
    clue_mask: np.ndarray,
    *,
    horizon: int,
    transition_mode: str,
    score_mode: str,
    action_mode: str,
    beam_width: int,
    branch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    root_latent = encode_board(model, board, device)
    root = SearchNode(
        board=board.copy(),
        latent=root_latent,
        latent_history=[root_latent],
        action_history=[],
        first_action=None,
        score=score_latent(model, root_latent, goal_np, initial_np, score_mode, device),
    )
    frontier = [root]
    best = root
    for _depth in range(max(1, int(horizon))):
        expanded: list[SearchNode] = []
        for node in frontier:
            actions = top_actions(
                model,
                world,
                node,
                transition_mode=transition_mode,
                clue_mask=clue_mask,
                goal_np=goal_np,
                initial_np=initial_np,
                score_mode=score_mode,
                action_mode=action_mode,
                branch_size=branch_size,
                device=device,
            )
            if not actions:
                expanded.append(node)
                continue
            for action in actions:
                next_board, next_latent, latent_history, action_history, score = transition_child(
                    model,
                    world,
                    node,
                    action,
                    transition_mode=transition_mode,
                    clue_mask=clue_mask,
                    goal_np=goal_np,
                    initial_np=initial_np,
                    score_mode=score_mode,
                    action_mode=action_mode,
                    device=device,
                )
                expanded.append(
                    SearchNode(
                        board=next_board,
                        latent=next_latent,
                        latent_history=latent_history,
                        action_history=action_history,
                        first_action=node.first_action or action,
                        score=score,
                    )
                )
        if not expanded:
            break
        expanded.sort(key=lambda node: node.score)
        frontier = expanded[: max(1, beam_width)]
        best = frontier[0]
    action = best.first_action
    return {
        "action": action,
        "leaf_score": float(best.score),
        "leaf_remaining_hamming": int(np.not_equal(best.board, goal_np).sum()),
        "leaf_terminal": bool(np.count_nonzero(best.board == 0) == 0),
    }


def ucb_score(parent: MCTSNode, child: MCTSNode, exploration: float) -> float:
    if child.visits == 0:
        return float("inf")
    return child.mean_value + float(exploration) * math.sqrt(math.log(max(2, parent.visits)) / child.visits)


@torch.no_grad()
def mcts_plan_once(
    model: SigRegActionJEPA,
    world: SudokuWorld,
    board: np.ndarray,
    goal_np: np.ndarray,
    initial_np: np.ndarray,
    clue_mask: np.ndarray,
    *,
    horizon: int,
    transition_mode: str,
    score_mode: str,
    action_mode: str,
    simulations: int,
    branch_size: int,
    exploration: float,
    device: torch.device,
) -> dict[str, Any]:
    root_latent = encode_board(model, board, device)
    root = MCTSNode(
        board=board.copy(),
        latent=root_latent,
        latent_history=[root_latent],
        action_history=[],
        first_action=None,
        depth=0,
    )
    for _ in range(max(1, int(simulations))):
        node = root
        path = [node]
        while node.children and node.depth < horizon:
            node = max(node.children, key=lambda child: ucb_score(path[-1], child, exploration))
            path.append(node)
        if node.depth < horizon and np.count_nonzero(node.board == 0) > 0:
            if node.untried_actions is None:
                node.untried_actions = top_actions(
                    model,
                    world,
                    node,
                    transition_mode=transition_mode,
                    clue_mask=clue_mask,
                    goal_np=goal_np,
                    initial_np=initial_np,
                    score_mode=score_mode,
                    action_mode=action_mode,
                    branch_size=branch_size,
                    device=device,
                )
            if node.untried_actions:
                action = node.untried_actions.pop(0)
                next_board, next_latent, latent_history, action_history, _score = transition_child(
                    model,
                    world,
                    node,
                    action,
                    transition_mode=transition_mode,
                    clue_mask=clue_mask,
                    goal_np=goal_np,
                    initial_np=initial_np,
                    score_mode=score_mode,
                    action_mode=action_mode,
                    device=device,
                )
                child = MCTSNode(
                    board=next_board,
                    latent=next_latent,
                    latent_history=latent_history,
                    action_history=action_history,
                    first_action=node.first_action or action,
                    depth=node.depth + 1,
                )
                node.children.append(child)
                node = child
                path.append(node)
        score = score_latent(model, node.latent, goal_np, initial_np, score_mode, device)
        leaf_remaining = int(np.not_equal(node.board, goal_np).sum())
        leaf_terminal = bool(np.count_nonzero(node.board == 0) == 0)
        reward = -float(score)
        for item in path:
            item.visits += 1
            item.value_sum += reward
            if score < item.best_leaf_score:
                item.best_leaf_score = float(score)
                item.best_leaf_remaining_hamming = leaf_remaining
                item.best_leaf_terminal = leaf_terminal
    if not root.children:
        return {"action": None, "leaf_score": float("inf"), "leaf_remaining_hamming": int(np.not_equal(board, goal_np).sum())}
    best = max(root.children, key=lambda child: (child.visits, child.mean_value))
    return {
        "action": best.first_action,
        "leaf_score": -float(best.mean_value),
        "leaf_remaining_hamming": (
            int(best.best_leaf_remaining_hamming)
            if best.best_leaf_remaining_hamming is not None
            else int(np.not_equal(best.board, goal_np).sum())
        ),
        "leaf_terminal": bool(best.best_leaf_terminal),
        "root_visits": int(root.visits),
        "child_visits": int(best.visits),
    }


def action_embedding_matrix(
    model: SigRegActionJEPA,
    world: SudokuWorld,
    actions: list[WorldAction],
    device: torch.device,
) -> torch.Tensor:
    action_array = np.stack([action.as_array(world.task_id) for action in actions])
    action_t = torch.as_tensor(action_array, dtype=torch.long, device=device)
    return model.action_encoder(action_t)


def decode_nearest_action(
    model: SigRegActionJEPA,
    world: SudokuWorld,
    board: np.ndarray,
    clue_mask: np.ndarray,
    embedding: torch.Tensor,
    *,
    action_mode: str,
    device: torch.device,
) -> WorldAction | None:
    actions = action_pool(world, board, clue_mask, action_mode)
    if not actions:
        return None
    action_embeddings = action_embedding_matrix(model, world, actions, device)
    distances = F.mse_loss(
        action_embeddings,
        embedding.view(1, -1).expand(action_embeddings.shape[0], -1),
        reduction="none",
    ).mean(dim=-1)
    return actions[int(torch.argmin(distances).detach().cpu().item())]


def rollout_decoded_sequence(
    model: SigRegActionJEPA,
    world: SudokuWorld,
    board: np.ndarray,
    goal_np: np.ndarray,
    initial_np: np.ndarray,
    clue_mask: np.ndarray,
    embeddings: torch.Tensor,
    *,
    transition_mode: str,
    score_mode: str,
    action_mode: str,
    device: torch.device,
) -> dict[str, Any]:
    current_board = board.copy()
    root_latent = encode_board(model, current_board, device)
    node = SearchNode(
        board=current_board,
        latent=root_latent,
        latent_history=[root_latent],
        action_history=[],
        first_action=None,
        score=score_latent(model, root_latent, goal_np, initial_np, score_mode, device),
    )
    actions_out: list[WorldAction] = []
    for step in range(embeddings.shape[0]):
        action = decode_nearest_action(
            model,
            world,
            node.board,
            clue_mask,
            embeddings[step],
            action_mode=action_mode,
            device=device,
        )
        if action is None:
            break
        next_board, next_latent, latent_history, action_history, score = transition_child(
            model,
            world,
            node,
            action,
            transition_mode=transition_mode,
            clue_mask=clue_mask,
            goal_np=goal_np,
            initial_np=initial_np,
            score_mode=score_mode,
            action_mode=action_mode,
            device=device,
        )
        actions_out.append(action)
        node = SearchNode(
            board=next_board,
            latent=next_latent,
            latent_history=latent_history,
            action_history=action_history,
            first_action=node.first_action or action,
            score=score,
        )
        if action_mode == "fill_empty" and np.count_nonzero(next_board == 0) == 0:
            break
    return {
        "action": actions_out[0] if actions_out else None,
        "actions": actions_out,
        "leaf_score": float(node.score),
        "leaf_remaining_hamming": int(np.not_equal(node.board, goal_np).sum()),
        "leaf_terminal": bool(np.count_nonzero(node.board == 0) == 0),
    }


@torch.no_grad()
def nearest_neighbor_cem_plan_once(
    model: SigRegActionJEPA,
    world: SudokuWorld,
    board: np.ndarray,
    goal_np: np.ndarray,
    initial_np: np.ndarray,
    clue_mask: np.ndarray,
    *,
    horizon: int,
    transition_mode: str,
    score_mode: str,
    action_mode: str,
    candidates: int,
    elites: int,
    iterations: int,
    smoothing: float,
    seed: int,
    device: torch.device,
    branch_size: int | None = None,
) -> dict[str, Any]:
    root_actions = action_pool(world, board, clue_mask, action_mode)
    if not root_actions:
        return {"action": None, "leaf_score": float("inf"), "leaf_remaining_hamming": int(np.not_equal(board, goal_np).sum())}
    root_embeddings = action_embedding_matrix(model, world, root_actions, device)
    horizon = max(1, int(horizon))
    action_dim = int(root_embeddings.shape[-1])
    mean = root_embeddings.mean(dim=0, keepdim=True).expand(horizon, -1).clone()
    std = root_embeddings.std(dim=0, unbiased=False).clamp_min(0.25).mean().expand(horizon, action_dim).clone()
    best: dict[str, Any] | None = None
    best_score = float("inf")
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))

    for _ in range(max(1, int(iterations))):
        noise = torch.randn((int(candidates), horizon, action_dim), generator=generator, device=device)
        samples = mean.unsqueeze(0) + std.unsqueeze(0) * noise
        scored: list[tuple[float, int, dict[str, Any]]] = []
        for index in range(int(candidates)):
            result = rollout_decoded_sequence(
                model,
                world,
                board,
                goal_np,
                initial_np,
                clue_mask,
                samples[index],
                transition_mode=transition_mode,
                score_mode=score_mode,
                action_mode=action_mode,
                device=device,
            )
            score = float(result["leaf_score"])
            scored.append((score, index, result))
            if score < best_score:
                best_score = score
                best = result
        scored.sort(key=lambda item: item[0])
        elite_count = max(1, min(int(elites), len(scored)))
        elite_indices = torch.as_tensor([index for _score, index, _result in scored[:elite_count]], dtype=torch.long, device=device)
        elite_samples = samples.index_select(0, elite_indices)
        elite_mean = elite_samples.mean(dim=0)
        elite_std = elite_samples.std(dim=0, unbiased=False).clamp_min(0.05)
        mean = float(smoothing) * elite_mean + (1.0 - float(smoothing)) * mean
        std = float(smoothing) * elite_std + (1.0 - float(smoothing)) * std
    if best is None:
        return {"action": None, "leaf_score": float("inf"), "leaf_remaining_hamming": int(np.not_equal(board, goal_np).sum())}
    return best


@torch.no_grad()
def run_closed_loop(
    model: SigRegActionJEPA,
    world: SudokuWorld,
    example,
    *,
    optimizer: str,
    transition_mode: str,
    score_mode: str,
    action_mode: str,
    horizon: int,
    max_steps: int,
    beam_width: int,
    branch_size: int,
    mcts_simulations: int,
    mcts_exploration: float,
    nn_cem_candidates: int,
    nn_cem_elites: int,
    nn_cem_iterations: int,
    nn_cem_smoothing: float,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    clue_mask = world.clue_mask_from_puzzle(example.state)
    board = example.state.copy()
    start_hamming = int(np.not_equal(board, example.goal).sum())
    root_goal_value = False
    root_leaf_remaining = None
    steps = 0
    for step in range(max_steps):
        if world.is_goal(board, example.goal):
            break
        if action_mode == "fill_empty" and np.count_nonzero(board == 0) == 0:
            break
        kwargs = dict(
            model=model,
            world=world,
            board=board,
            goal_np=example.goal,
            initial_np=example.state,
            clue_mask=clue_mask,
            horizon=max(1, min(int(horizon), int(np.count_nonzero(board == 0)))),
            transition_mode=transition_mode,
            score_mode=score_mode,
            action_mode=action_mode,
            branch_size=branch_size,
            device=device,
        )
        if optimizer == "beam":
            plan = beam_search_plan_once(beam_width=beam_width, **kwargs)
        elif optimizer == "mcts":
            plan = mcts_plan_once(simulations=mcts_simulations, exploration=mcts_exploration, **kwargs)
        elif optimizer == "nn_cem":
            plan = nearest_neighbor_cem_plan_once(
                candidates=nn_cem_candidates,
                elites=nn_cem_elites,
                iterations=nn_cem_iterations,
                smoothing=nn_cem_smoothing,
                seed=seed + 1009 * step,
                **kwargs,
            )
        else:
            raise ValueError(f"unknown optimizer {optimizer!r}.")
        action = plan.get("action")
        if action is None:
            break
        if step == 0:
            root_goal_value = bool(action.value == int(example.goal[action.row, action.col]))
            root_leaf_remaining = plan.get("leaf_remaining_hamming")
        board = apply_action(world, board, action, clue_mask=clue_mask, action_mode=action_mode)
        steps = step + 1
    return {
        "start_hamming": start_hamming,
        "remaining_hamming": int(np.not_equal(board, example.goal).sum()),
        "terminal": bool(np.count_nonzero(board == 0) == 0),
        "solved": bool(world.is_goal(board, example.goal)),
        "steps": int(steps),
        "root_goal_value": root_goal_value,
        "root_leaf_remaining_hamming": root_leaf_remaining,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {}
    root_leaf = [record["root_leaf_remaining_hamming"] for record in records if record["root_leaf_remaining_hamming"] is not None]
    return {
        "count": float(len(records)),
        "solves": float(sum(record["solved"] for record in records)),
        "solve_rate": float(np.mean([record["solved"] for record in records])),
        "terminal_rate": float(np.mean([record["terminal"] for record in records])),
        "mean_remaining_hamming": float(np.mean([record["remaining_hamming"] for record in records])),
        "mean_start_hamming": float(np.mean([record["start_hamming"] for record in records])),
        "root_goal_value_rate": float(np.mean([record["root_goal_value"] for record in records])),
        "mean_root_leaf_remaining_hamming": float(np.mean(root_leaf)) if root_leaf else 0.0,
        "mean_steps": float(np.mean([record["steps"] for record in records])),
    }


def run_grid5_planner_matrix(
    run_root: Path,
    output_dir: Path,
    *,
    seed: int,
    plan_examples: int,
    horizons: list[int],
    optimizers: list[str],
    transition_modes: list[str],
    score_modes: list[str],
    action_mode: str,
    max_steps: int,
    beam_width: int,
    branch_size: int,
    mcts_simulations: int,
    mcts_exploration: float,
    nn_cem_candidates: int,
    nn_cem_elites: int,
    nn_cem_iterations: int,
    nn_cem_smoothing: float,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model(run_root, device)
    world = _build_world(dict(config["task"]))
    if not isinstance(world, SudokuWorld):
        raise ValueError("Grid 5 planner matrix supports Sudoku only.")
    examples = _load_examples(dict(config["task"]), "eval")
    rng = np.random.default_rng(seed)
    indices = [int(rng.integers(0, len(examples))) for _ in range(max(1, plan_examples))]
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "run_root": str(run_root),
        "device": str(device),
        "config": {
            "encoder_type": config["model"]["encoder_type"],
            "predictor_type": config["model"]["predictor_type"],
            "predict_delta": bool(config["model"].get("predict_delta", False)),
            "stabilizer_type": config["model"].get("stabilizer_type", "sigreg"),
            "target_encoder_momentum": float(config["model"].get("target_encoder_momentum", 0.0)),
            "latent_size": int(config["model"]["latent_size"]),
        },
        "action_mode": action_mode,
        "modes": {},
    }
    for optimizer in optimizers:
        for transition_mode in transition_modes:
            for score_mode in score_modes:
                for horizon in horizons:
                    mode_records = []
                    for local_index, example_index in enumerate(indices):
                        result = run_closed_loop(
                            model,
                            world,
                            examples[example_index],
                            optimizer=optimizer,
                            transition_mode=transition_mode,
                            score_mode=score_mode,
                            action_mode=action_mode,
                            horizon=int(horizon),
                            max_steps=max_steps,
                            beam_width=beam_width,
                            branch_size=branch_size,
                            mcts_simulations=mcts_simulations,
                            mcts_exploration=mcts_exploration,
                            nn_cem_candidates=nn_cem_candidates,
                            nn_cem_elites=nn_cem_elites,
                            nn_cem_iterations=nn_cem_iterations,
                            nn_cem_smoothing=nn_cem_smoothing,
                            seed=seed + 7919 * local_index + 104729 * int(horizon),
                            device=device,
                        )
                        record = {
                            "example_index": example_index,
                            "local_index": local_index,
                            "optimizer": optimizer,
                            "transition_mode": transition_mode,
                            "score_mode": score_mode,
                            "horizon": int(horizon),
                            **result,
                        }
                        records.append(record)
                        mode_records.append(record)
                    key = f"{optimizer}_{transition_mode}_{score_mode}_h{int(horizon)}"
                    summary["modes"][key] = summarize(mode_records)
    write_jsonl(output_dir / "planner_records.jsonl", records)
    (output_dir / "planner_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plan-examples", type=int, default=4)
    parser.add_argument("--horizons", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--optimizers", nargs="+", default=["beam", "mcts"])
    parser.add_argument("--transition-modes", nargs="+", default=["symbolic_reencode", "latent_rollout"])
    parser.add_argument("--score-modes", nargs="+", default=["latent_goal", "goal_energy"])
    parser.add_argument("--action-mode", choices=["mutable_overwrite", "fill_empty"], default="mutable_overwrite")
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--branch-size", type=int, default=16)
    parser.add_argument("--mcts-simulations", type=int, default=128)
    parser.add_argument("--mcts-exploration", type=float, default=1.0)
    parser.add_argument("--nn-cem-candidates", type=int, default=128)
    parser.add_argument("--nn-cem-elites", type=int, default=16)
    parser.add_argument("--nn-cem-iterations", type=int, default=5)
    parser.add_argument("--nn-cem-smoothing", type=float, default=0.7)
    args = parser.parse_args()
    summary = run_grid5_planner_matrix(
        args.run_root,
        args.output_dir,
        seed=args.seed,
        plan_examples=args.plan_examples,
        horizons=args.horizons,
        optimizers=args.optimizers,
        transition_modes=args.transition_modes,
        score_modes=args.score_modes,
        action_mode=args.action_mode,
        max_steps=args.max_steps,
        beam_width=args.beam_width,
        branch_size=args.branch_size,
        mcts_simulations=args.mcts_simulations,
        mcts_exploration=args.mcts_exploration,
        nn_cem_candidates=args.nn_cem_candidates,
        nn_cem_elites=args.nn_cem_elites,
        nn_cem_iterations=args.nn_cem_iterations,
        nn_cem_smoothing=args.nn_cem_smoothing,
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
