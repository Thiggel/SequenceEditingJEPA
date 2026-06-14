from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch

from puzzle_jepa.data.lewm_sudoku import (
    action_from_id,
    action_id,
    action_to_array,
    apply_fill_action,
    legal_fill_actions,
)
from puzzle_jepa.data.worlds import SudokuWorld, WorldAction
from puzzle_jepa.models.lewm import LeWMSudokuModel


PlannerName = Literal["greedy", "beam", "best_first", "categorical_cem", "local_search", "mcts", "exact"]
TransitionMode = Literal["symbolic_reencode", "latent_rollout"]
ScoreMode = Literal["true_hamming_oracle", "oracle_goal_distance", "predicted_goal_distance"]


@dataclass(frozen=True, slots=True)
class SequenceScore:
    cost: float
    leaf_board: np.ndarray
    terminal: bool


@dataclass(frozen=True, slots=True)
class MPCResult:
    solved: bool
    steps: int
    start_hamming: int
    remaining_hamming: int
    actions: list[WorldAction]
    final_board: np.ndarray
    planner: str
    transition_mode: str
    score_mode: str
    horizon: int


def hamming_distance(board: np.ndarray, goal: np.ndarray) -> int:
    return int(np.not_equal(board, goal).sum())


def score_action_sequence(
    model: LeWMSudokuModel | None,
    board: np.ndarray,
    goal: np.ndarray,
    actions: list[WorldAction],
    *,
    transition_mode: TransitionMode,
    score_mode: ScoreMode,
    device: torch.device,
    position_offset: int = 0,
    history_boards: list[np.ndarray] | None = None,
    history_actions: list[WorldAction] | None = None,
) -> SequenceScore:
    leaf, valid = apply_action_sequence(board, actions)
    if not valid:
        return SequenceScore(float("inf"), leaf, False)
    terminal = bool(np.array_equal(leaf, goal))
    if score_mode == "true_hamming_oracle":
        return SequenceScore(float(hamming_distance(leaf, goal)), leaf, terminal)
    if model is None:
        raise ValueError(f"score_mode={score_mode} requires a LeWMSudokuModel.")

    model.eval()
    with torch.no_grad():
        goal_t = torch.as_tensor(goal[None], dtype=torch.long, device=device)
        goal_emb = model.encode_board(goal_t)
        if transition_mode == "symbolic_reencode" or not actions:
            leaf_t = torch.as_tensor(leaf[None], dtype=torch.long, device=device)
            leaf_emb = model.encode_board(leaf_t)
        else:
            action_t = torch.as_tensor(
                np.asarray([[action_to_array(action) for action in actions]], dtype=np.int64),
                dtype=torch.long,
                device=device,
            )
            if history_boards is not None:
                if history_actions is None:
                    history_actions = []
                if len(history_boards) != len(history_actions) + 1:
                    raise ValueError("history_boards must contain exactly one more item than history_actions.")
                history_t = torch.as_tensor(np.stack(history_boards), dtype=torch.long, device=device)[None]
                prefix_emb = model.encode_sequence(history_t)
                if history_actions:
                    prefix_actions_t = torch.as_tensor(
                        np.asarray([[action_to_array(action) for action in history_actions]], dtype=np.int64),
                        dtype=torch.long,
                        device=device,
                    )
                else:
                    prefix_actions_t = torch.zeros((1, 0, 3), dtype=torch.long, device=device)
                start_emb = prefix_emb[:, -1]
                leaf_emb = model.rollout_latent(
                    start_emb,
                    action_t,
                    prefix_embeddings=prefix_emb,
                    prefix_actions=prefix_actions_t,
                )[:, -1]
            else:
                start_t = torch.as_tensor(board[None], dtype=torch.long, device=device)
                start_emb = model.encode_board(start_t)
                leaf_emb = model.rollout_latent(start_emb, action_t, position_offset=position_offset)[:, -1]
        if score_mode == "oracle_goal_distance":
            cost = torch.linalg.vector_norm(leaf_emb - goal_emb, dim=-1).item()
        elif score_mode == "predicted_goal_distance":
            cost = model.score_value(leaf_emb).item()
        else:
            raise ValueError(f"Unknown score_mode {score_mode!r}.")
    return SequenceScore(float(cost), leaf, terminal)


def apply_action_sequence(board: np.ndarray, actions: list[WorldAction]) -> tuple[np.ndarray, bool]:
    current = np.asarray(board, dtype=np.int64).copy()
    for action in actions:
        try:
            current = apply_fill_action(current, action, allow_conflicts=True)
        except ValueError:
            return current, False
    return current, True


def greedy_plan_once(
    model: LeWMSudokuModel | None,
    board: np.ndarray,
    goal: np.ndarray,
    *,
    transition_mode: TransitionMode,
    score_mode: ScoreMode,
    device: torch.device,
    position_offset: int = 0,
    history_boards: list[np.ndarray] | None = None,
    history_actions: list[WorldAction] | None = None,
) -> WorldAction | None:
    actions = legal_fill_actions(board, allow_conflicts=True)
    if not actions:
        return None
    scored = [
        (
            score_action_sequence(
                model,
                board,
                goal,
                [action],
                transition_mode=transition_mode,
                score_mode=score_mode,
                device=device,
                position_offset=position_offset,
                history_boards=history_boards,
                history_actions=history_actions,
            ).cost,
            action,
        )
        for action in actions
    ]
    return min(scored, key=lambda item: item[0])[1]


def beam_plan_once(
    model: LeWMSudokuModel | None,
    board: np.ndarray,
    goal: np.ndarray,
    *,
    horizon: int,
    beam_width: int,
    branch_size: int,
    transition_mode: TransitionMode,
    score_mode: ScoreMode,
    device: torch.device,
    position_offset: int = 0,
    history_boards: list[np.ndarray] | None = None,
    history_actions: list[WorldAction] | None = None,
) -> WorldAction | None:
    beam: list[tuple[float, np.ndarray, list[WorldAction]]] = [(0.0, board.copy(), [])]
    best: tuple[float, list[WorldAction]] | None = None
    for _ in range(max(1, horizon)):
        candidates: list[tuple[float, np.ndarray, list[WorldAction]]] = []
        for _, node_board, seq in beam:
            actions = legal_fill_actions(node_board, allow_conflicts=True)
            if branch_size > 0:
                actions = _rank_immediate_actions(
                    model,
                    node_board,
                    goal,
                    actions,
                    branch_size,
                    transition_mode=transition_mode,
                    score_mode=score_mode,
                    device=device,
                    position_offset=position_offset + len(seq),
                    history_boards=_history_boards_for_sequence(history_boards, board, seq),
                    history_actions=_history_actions_for_sequence(history_actions, seq),
                )
            for action in actions:
                score = score_action_sequence(
                    model,
                    board,
                    goal,
                    [*seq, action],
                    transition_mode=transition_mode,
                    score_mode=score_mode,
                    device=device,
                    position_offset=position_offset,
                    history_boards=history_boards,
                    history_actions=history_actions,
                )
                if not math.isfinite(score.cost):
                    continue
                next_seq = [*seq, action]
                candidates.append((score.cost, score.leaf_board, next_seq))
                if best is None or score.cost < best[0]:
                    best = (score.cost, next_seq)
        if not candidates:
            break
        candidates.sort(key=lambda item: item[0])
        beam = candidates[: max(1, beam_width)]
        if beam[0][0] <= 0.0:
            break
    if best is None or not best[1]:
        return None
    return best[1][0]


def best_first_plan_once(
    model: LeWMSudokuModel | None,
    board: np.ndarray,
    goal: np.ndarray,
    *,
    horizon: int,
    max_expansions: int,
    branch_size: int,
    heuristic_weight: float,
    transition_mode: TransitionMode,
    score_mode: ScoreMode,
    device: torch.device,
    position_offset: int = 0,
    history_boards: list[np.ndarray] | None = None,
    history_actions: list[WorldAction] | None = None,
) -> WorldAction | None:
    counter = 0
    heap: list[tuple[float, int, np.ndarray, list[WorldAction]]] = [(0.0, counter, board.copy(), [])]
    best_seq: list[WorldAction] | None = None
    best_cost = float("inf")
    visited: set[bytes] = set()
    for _ in range(max(1, max_expansions)):
        if not heap:
            break
        _, _, node_board, seq = heapq.heappop(heap)
        key = node_board.tobytes()
        if key in visited:
            continue
        visited.add(key)
        node_score = score_action_sequence(
            model,
            board,
            goal,
            seq,
            transition_mode=transition_mode,
            score_mode=score_mode,
            device=device,
            position_offset=position_offset,
            history_boards=history_boards,
            history_actions=history_actions,
        )
        if node_score.cost < best_cost and seq:
            best_cost = node_score.cost
            best_seq = seq
        if len(seq) >= horizon or node_score.terminal:
            continue
        actions = legal_fill_actions(node_board, allow_conflicts=True)
        if branch_size > 0:
            actions = _rank_immediate_actions(
                model,
                node_board,
                goal,
                actions,
                branch_size,
                transition_mode=transition_mode,
                score_mode=score_mode,
                device=device,
                position_offset=position_offset + len(seq),
                history_boards=_history_boards_for_sequence(history_boards, board, seq),
                history_actions=_history_actions_for_sequence(history_actions, seq),
            )
        for action in actions:
            next_board, valid = apply_action_sequence(node_board, [action])
            if not valid:
                continue
            next_seq = [*seq, action]
            heuristic = score_action_sequence(
                model,
                board,
                goal,
                next_seq,
                transition_mode=transition_mode,
                score_mode=score_mode,
                device=device,
                position_offset=position_offset,
                history_boards=history_boards,
                history_actions=history_actions,
            ).cost
            counter += 1
            priority = len(next_seq) + heuristic_weight * heuristic
            heapq.heappush(heap, (priority, counter, next_board, next_seq))
    return None if not best_seq else best_seq[0]


def categorical_cem_plan_once(
    model: LeWMSudokuModel | None,
    board: np.ndarray,
    goal: np.ndarray,
    *,
    horizon: int,
    candidates: int,
    elites: int,
    iterations: int,
    smoothing: float,
    transition_mode: TransitionMode,
    score_mode: ScoreMode,
    rng: np.random.Generator,
    device: torch.device,
    position_offset: int = 0,
    history_boards: list[np.ndarray] | None = None,
    history_actions: list[WorldAction] | None = None,
) -> WorldAction | None:
    horizon = max(1, horizon)
    probs = np.full((horizon, 729), 1.0 / 729.0, dtype=np.float64)
    best_seq: list[WorldAction] | None = None
    best_cost = float("inf")
    for _ in range(max(1, iterations)):
        scored: list[tuple[float, list[WorldAction]]] = []
        for _ in range(max(1, candidates)):
            seq = _sample_categorical_sequence(board, probs, rng)
            score = score_action_sequence(
                model,
                board,
                goal,
                seq,
                transition_mode=transition_mode,
                score_mode=score_mode,
                device=device,
                position_offset=position_offset,
                history_boards=history_boards,
                history_actions=history_actions,
            )
            scored.append((score.cost, seq))
            if score.cost < best_cost and seq:
                best_cost = score.cost
                best_seq = seq
        scored.sort(key=lambda item: item[0])
        elite_set = scored[: max(1, min(elites, len(scored)))]
        new_probs = np.full_like(probs, 1.0e-6)
        for _, seq in elite_set:
            for step, action in enumerate(seq[:horizon]):
                new_probs[step, action_id(action)] += 1.0
        new_probs /= new_probs.sum(axis=1, keepdims=True)
        probs = smoothing * probs + (1.0 - smoothing) * new_probs
        probs /= probs.sum(axis=1, keepdims=True)
    return None if not best_seq else best_seq[0]


def local_search_plan_once(
    model: LeWMSudokuModel | None,
    board: np.ndarray,
    goal: np.ndarray,
    *,
    horizon: int,
    candidates: int,
    iterations: int,
    temperature: float,
    transition_mode: TransitionMode,
    score_mode: ScoreMode,
    rng: np.random.Generator,
    device: torch.device,
    position_offset: int = 0,
    history_boards: list[np.ndarray] | None = None,
    history_actions: list[WorldAction] | None = None,
) -> WorldAction | None:
    pool = [_sample_random_sequence(board, horizon, rng) for _ in range(max(1, candidates))]
    scored = [
        (
            score_action_sequence(
                model,
                board,
                goal,
                seq,
                transition_mode=transition_mode,
                score_mode=score_mode,
                device=device,
                position_offset=position_offset,
                history_boards=history_boards,
                history_actions=history_actions,
            ).cost,
            seq,
        )
        for seq in pool
        if seq
    ]
    if not scored:
        return None
    scored.sort(key=lambda item: item[0])
    best_cost, best_seq = scored[0]
    for index in range(max(1, iterations)):
        base_idx = int(rng.integers(0, len(scored)))
        _, base_seq = scored[base_idx]
        if not base_seq:
            continue
        cut = int(rng.integers(0, len(base_seq)))
        prefix = base_seq[:cut]
        prefix_board, valid = apply_action_sequence(board, prefix)
        if not valid:
            prefix = []
            prefix_board = board.copy()
        proposal = [*prefix, *_sample_random_sequence(prefix_board, horizon - len(prefix), rng)]
        proposal_score = score_action_sequence(
            model,
            board,
            goal,
            proposal,
            transition_mode=transition_mode,
            score_mode=score_mode,
            device=device,
            position_offset=position_offset,
            history_boards=history_boards,
            history_actions=history_actions,
        ).cost
        old_score, _ = scored[base_idx]
        accept_worse = temperature > 0.0 and rng.random() < math.exp(-(proposal_score - old_score) / max(temperature, 1.0e-6))
        if proposal_score <= old_score or accept_worse:
            scored[base_idx] = (proposal_score, proposal)
        if proposal_score < best_cost and proposal:
            best_cost, best_seq = proposal_score, proposal
        if index % 8 == 0:
            scored.sort(key=lambda item: item[0])
    return None if not best_seq else best_seq[0]


@dataclass
class MCTSNode:
    board: np.ndarray
    seq: list[WorldAction]
    parent: "MCTSNode | None" = None
    parent_action: WorldAction | None = None
    visits: int = 0
    total_reward: float = 0.0
    children: dict[int, "MCTSNode"] = field(default_factory=dict)
    untried_actions: list[WorldAction] = field(default_factory=list)
    action_count: int = 0

    @property
    def value(self) -> float:
        return self.total_reward / max(1, self.visits)


def mcts_plan_once(
    model: LeWMSudokuModel | None,
    board: np.ndarray,
    goal: np.ndarray,
    *,
    horizon: int,
    simulations: int,
    exploration: float,
    transition_mode: TransitionMode,
    score_mode: ScoreMode,
    rng: np.random.Generator,
    device: torch.device,
    progressive_c: float = 2.0,
    progressive_alpha: float = 0.5,
    expansion_branch_size: int = 32,
    position_offset: int = 0,
    history_boards: list[np.ndarray] | None = None,
    history_actions: list[WorldAction] | None = None,
) -> WorldAction | None:
    root = _make_mcts_node(
        model,
        board.copy(),
        goal,
        root_board=board.copy(),
        seq=[],
        parent=None,
        parent_action=None,
        transition_mode=transition_mode,
        score_mode=score_mode,
        expansion_branch_size=expansion_branch_size,
        position_offset=position_offset,
        history_boards=history_boards,
        history_actions=history_actions,
        device=device,
    )
    if root.action_count == 0:
        return None
    for _ in range(max(1, simulations)):
        node = root
        while node.children and len(node.seq) < horizon and not _can_expand(node, progressive_c, progressive_alpha):
            node = max(node.children.values(), key=lambda child: _uct_score(child, exploration))
        if len(node.seq) < horizon and node.untried_actions and _can_expand(node, progressive_c, progressive_alpha):
            action = node.untried_actions.pop(0)
            next_board, valid = apply_action_sequence(node.board, [action])
            if valid:
                child = _make_mcts_node(
                    model,
                    next_board,
                    goal,
                    root_board=board,
                    seq=[*node.seq, action],
                    parent=node,
                    parent_action=action,
                    transition_mode=transition_mode,
                    score_mode=score_mode,
                    expansion_branch_size=expansion_branch_size,
                    position_offset=position_offset + len(node.seq) + 1,
                    history_boards=history_boards,
                    history_actions=history_actions,
                    device=device,
                )
                node.children[action_id(action)] = child
                node = child
        rollout_seq = [*node.seq]
        rollout_board = node.board.copy()
        while len(rollout_seq) < horizon:
            actions = legal_fill_actions(rollout_board, allow_conflicts=True)
            if not actions:
                break
            action = actions[int(rng.integers(0, len(actions)))]
            rollout_board = apply_fill_action(rollout_board, action, allow_conflicts=True)
            rollout_seq.append(action)
        cost = score_action_sequence(
            model,
            board,
            goal,
            rollout_seq,
            transition_mode=transition_mode,
            score_mode=score_mode,
            device=device,
            position_offset=position_offset,
            history_boards=history_boards,
            history_actions=history_actions,
        ).cost
        reward = -cost
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent
    if not root.children:
        return root.untried_actions[0] if root.untried_actions else None
    return max(root.children.values(), key=lambda child: (child.visits, child.value)).parent_action


def _make_mcts_node(
    model: LeWMSudokuModel | None,
    board: np.ndarray,
    goal: np.ndarray,
    *,
    root_board: np.ndarray,
    seq: list[WorldAction],
    parent: MCTSNode | None,
    parent_action: WorldAction | None,
    transition_mode: TransitionMode,
    score_mode: ScoreMode,
    expansion_branch_size: int,
    position_offset: int,
    history_boards: list[np.ndarray] | None,
    history_actions: list[WorldAction] | None,
    device: torch.device,
) -> MCTSNode:
    actions = legal_fill_actions(board, allow_conflicts=True)
    action_count = len(actions)
    if actions and expansion_branch_size > 0:
        actions = _rank_immediate_actions(
            model,
            board,
            goal,
            actions,
            expansion_branch_size,
            transition_mode=transition_mode,
            score_mode=score_mode,
            position_offset=position_offset,
            history_boards=_history_boards_for_sequence(history_boards, root_board, seq),
            history_actions=_history_actions_for_sequence(history_actions, seq),
            device=device,
        )
    return MCTSNode(
        board=board,
        seq=seq,
        parent=parent,
        parent_action=parent_action,
        untried_actions=actions,
        action_count=action_count,
    )


def _can_expand(node: MCTSNode, progressive_c: float, progressive_alpha: float) -> bool:
    if not node.untried_actions:
        return False
    allowed = int(math.ceil(max(1.0, progressive_c) * max(1, node.visits) ** progressive_alpha))
    allowed = min(max(1, allowed), node.action_count)
    return len(node.children) < allowed


def _uct_score(node: MCTSNode, exploration: float) -> float:
    if node.visits == 0:
        return float("inf")
    parent_visits = max(1, node.parent.visits if node.parent is not None else node.visits)
    return node.value + exploration * math.sqrt(math.log(parent_visits + 1) / node.visits)


def run_mpc(
    model: LeWMSudokuModel | None,
    board: np.ndarray,
    goal: np.ndarray,
    *,
    planner: PlannerName,
    horizon: int,
    transition_mode: TransitionMode = "symbolic_reencode",
    score_mode: ScoreMode = "oracle_goal_distance",
    max_steps: int = 81,
    beam_width: int = 8,
    branch_size: int = 24,
    best_first_expansions: int = 256,
    best_first_weight: float = 1.0,
    cem_candidates: int = 128,
    cem_elites: int = 16,
    cem_iterations: int = 4,
    cem_smoothing: float = 0.2,
    local_candidates: int = 64,
    local_iterations: int = 128,
    local_temperature: float = 0.0,
    mcts_simulations: int = 256,
    mcts_exploration: float = 1.4,
    mcts_progressive_c: float = 2.0,
    mcts_progressive_alpha: float = 0.5,
    mcts_branch_size: int = 32,
    rng: np.random.Generator | None = None,
    device: torch.device | None = None,
) -> MPCResult:
    rng = rng or np.random.default_rng()
    device = device or torch.device("cpu")
    world = SudokuWorld()
    current = world.validate_state(board).copy()
    target = world.validate_state(goal)
    start_hamming = hamming_distance(current, target)
    actions_taken: list[WorldAction] = []
    history_boards: list[np.ndarray] = [current.copy()]
    if planner == "exact":
        solved = solve_sudoku_exact(current)
        if solved is not None:
            current = solved
        return MPCResult(
            solved=bool(solved is not None and np.array_equal(current, target)),
            steps=0,
            start_hamming=start_hamming,
            remaining_hamming=hamming_distance(current, target),
            actions=[],
            final_board=current,
            planner=planner,
            transition_mode="symbolic_reencode",
            score_mode="true_hamming_oracle",
            horizon=0,
        )

    for _ in range(max_steps):
        if np.array_equal(current, target):
            break
        if not legal_fill_actions(current, allow_conflicts=True):
            break
        action = _plan_once(
            model,
            current,
            target,
            planner=planner,
            horizon=horizon,
            transition_mode=transition_mode,
            score_mode=score_mode,
            beam_width=beam_width,
            branch_size=branch_size,
            best_first_expansions=best_first_expansions,
            best_first_weight=best_first_weight,
            cem_candidates=cem_candidates,
            cem_elites=cem_elites,
            cem_iterations=cem_iterations,
            cem_smoothing=cem_smoothing,
            local_candidates=local_candidates,
            local_iterations=local_iterations,
            local_temperature=local_temperature,
            mcts_simulations=mcts_simulations,
            mcts_exploration=mcts_exploration,
            mcts_progressive_c=mcts_progressive_c,
            mcts_progressive_alpha=mcts_progressive_alpha,
            mcts_branch_size=mcts_branch_size,
            rng=rng,
            device=device,
            position_offset=len(actions_taken),
            history_boards=history_boards,
            history_actions=actions_taken,
        )
        if action is None:
            break
        current = apply_fill_action(current, action, allow_conflicts=True)
        actions_taken.append(action)
        history_boards.append(current.copy())
    return MPCResult(
        solved=bool(np.array_equal(current, target)),
        steps=len(actions_taken),
        start_hamming=start_hamming,
        remaining_hamming=hamming_distance(current, target),
        actions=actions_taken,
        final_board=current,
        planner=_planner_result_name(planner, mcts_branch_size),
        transition_mode=transition_mode,
        score_mode=score_mode,
        horizon=horizon,
    )


def _plan_once(
    model: LeWMSudokuModel | None,
    board: np.ndarray,
    goal: np.ndarray,
    *,
    planner: PlannerName,
    horizon: int,
    transition_mode: TransitionMode,
    score_mode: ScoreMode,
    beam_width: int,
    branch_size: int,
    best_first_expansions: int,
    best_first_weight: float,
    cem_candidates: int,
    cem_elites: int,
    cem_iterations: int,
    cem_smoothing: float,
    local_candidates: int,
    local_iterations: int,
    local_temperature: float,
    mcts_simulations: int,
    mcts_exploration: float,
    mcts_progressive_c: float,
    mcts_progressive_alpha: float,
    mcts_branch_size: int,
    rng: np.random.Generator,
    device: torch.device,
    position_offset: int = 0,
    history_boards: list[np.ndarray] | None = None,
    history_actions: list[WorldAction] | None = None,
) -> WorldAction | None:
    if planner == "greedy":
        return greedy_plan_once(
            model,
            board,
            goal,
            transition_mode=transition_mode,
            score_mode=score_mode,
            device=device,
            position_offset=position_offset,
            history_boards=history_boards,
            history_actions=history_actions,
        )
    if planner == "beam":
        return beam_plan_once(
            model,
            board,
            goal,
            horizon=horizon,
            beam_width=beam_width,
            branch_size=branch_size,
            transition_mode=transition_mode,
            score_mode=score_mode,
            device=device,
            position_offset=position_offset,
            history_boards=history_boards,
            history_actions=history_actions,
        )
    if planner == "best_first":
        return best_first_plan_once(
            model,
            board,
            goal,
            horizon=horizon,
            max_expansions=best_first_expansions,
            branch_size=branch_size,
            heuristic_weight=best_first_weight,
            transition_mode=transition_mode,
            score_mode=score_mode,
            device=device,
            position_offset=position_offset,
            history_boards=history_boards,
            history_actions=history_actions,
        )
    if planner == "categorical_cem":
        return categorical_cem_plan_once(
            model,
            board,
            goal,
            horizon=horizon,
            candidates=cem_candidates,
            elites=cem_elites,
            iterations=cem_iterations,
            smoothing=cem_smoothing,
            transition_mode=transition_mode,
            score_mode=score_mode,
            rng=rng,
            device=device,
            position_offset=position_offset,
            history_boards=history_boards,
            history_actions=history_actions,
        )
    if planner == "local_search":
        return local_search_plan_once(
            model,
            board,
            goal,
            horizon=horizon,
            candidates=local_candidates,
            iterations=local_iterations,
            temperature=local_temperature,
            transition_mode=transition_mode,
            score_mode=score_mode,
            rng=rng,
            device=device,
            position_offset=position_offset,
            history_boards=history_boards,
            history_actions=history_actions,
        )
    if planner == "mcts":
        return mcts_plan_once(
            model,
            board,
            goal,
            horizon=horizon,
            simulations=mcts_simulations,
            exploration=mcts_exploration,
            progressive_c=mcts_progressive_c,
            progressive_alpha=mcts_progressive_alpha,
            expansion_branch_size=mcts_branch_size,
            transition_mode=transition_mode,
            score_mode=score_mode,
            rng=rng,
            device=device,
            position_offset=position_offset,
            history_boards=history_boards,
            history_actions=history_actions,
        )
    raise ValueError(f"Unknown planner {planner!r}.")


def _planner_result_name(planner: PlannerName, mcts_branch_size: int) -> str:
    if planner != "mcts":
        return planner
    return "score_pruned_progressive_uct" if mcts_branch_size > 0 else "progressive_uct"


def _history_boards_for_sequence(
    history_boards: list[np.ndarray] | None,
    root_board: np.ndarray,
    sequence: list[WorldAction],
) -> list[np.ndarray] | None:
    if history_boards is None:
        return None
    extended = [np.asarray(board, dtype=np.int64).copy() for board in history_boards]
    current = np.asarray(root_board, dtype=np.int64).copy()
    for action in sequence:
        current = apply_fill_action(current, action, allow_conflicts=True)
        extended.append(current.copy())
    return extended


def _history_actions_for_sequence(
    history_actions: list[WorldAction] | None,
    sequence: list[WorldAction],
) -> list[WorldAction] | None:
    if history_actions is None:
        return None
    return [*history_actions, *sequence]


def _rank_immediate_actions(
    model: LeWMSudokuModel | None,
    board: np.ndarray,
    goal: np.ndarray,
    actions: list[WorldAction],
    limit: int,
    *,
    transition_mode: TransitionMode,
    score_mode: ScoreMode,
    position_offset: int = 0,
    history_boards: list[np.ndarray] | None = None,
    history_actions: list[WorldAction] | None = None,
    device: torch.device,
) -> list[WorldAction]:
    scored = [
        (
            score_action_sequence(
                model,
                board,
                goal,
                [action],
                transition_mode=transition_mode,
                score_mode=score_mode,
                device=device,
                position_offset=position_offset,
                history_boards=history_boards,
                history_actions=history_actions,
            ).cost,
            action,
        )
        for action in actions
    ]
    scored.sort(key=lambda item: item[0])
    return [action for _, action in scored[:limit]]


def _sample_categorical_sequence(board: np.ndarray, probs: np.ndarray, rng: np.random.Generator) -> list[WorldAction]:
    current = board.copy()
    seq: list[WorldAction] = []
    for step in range(probs.shape[0]):
        legal = legal_fill_actions(current, allow_conflicts=True)
        if not legal:
            break
        ids = np.asarray([action_id(action) for action in legal], dtype=np.int64)
        weights = probs[step, ids].astype(np.float64)
        weights = weights / weights.sum() if weights.sum() > 0 else np.full(len(ids), 1.0 / len(ids))
        action = action_from_id(int(rng.choice(ids, p=weights)))
        current = apply_fill_action(current, action, allow_conflicts=True)
        seq.append(action)
    return seq


def _sample_random_sequence(board: np.ndarray, horizon: int, rng: np.random.Generator) -> list[WorldAction]:
    current = board.copy()
    seq: list[WorldAction] = []
    for _ in range(max(0, horizon)):
        legal = legal_fill_actions(current, allow_conflicts=True)
        if not legal:
            break
        action = legal[int(rng.integers(0, len(legal)))]
        current = apply_fill_action(current, action, allow_conflicts=True)
        seq.append(action)
    return seq


def solve_sudoku_exact(board: np.ndarray) -> np.ndarray | None:
    world = SudokuWorld()
    arr = world.validate_state(board).copy()

    def candidates(row: int, col: int) -> list[int]:
        return [value for value in range(1, 10) if world.is_value_allowed_after_write(arr, row, col, value)]

    def search() -> bool:
        empties = np.argwhere(arr == 0)
        if len(empties) == 0:
            return world.is_valid_solution(arr)
        best_cell = None
        best_candidates: list[int] | None = None
        for row, col in empties:
            row_i, col_i = int(row), int(col)
            opts = candidates(row_i, col_i)
            if not opts:
                return False
            if best_candidates is None or len(opts) < len(best_candidates):
                best_cell = (row_i, col_i)
                best_candidates = opts
        assert best_cell is not None and best_candidates is not None
        row_i, col_i = best_cell
        for value in best_candidates:
            arr[row_i, col_i] = value
            if search():
                return True
            arr[row_i, col_i] = 0
        return False

    return arr.copy() if search() else None
