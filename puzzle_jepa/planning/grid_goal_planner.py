from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from puzzle_jepa.data.grid_goal_sudoku import apply_sudoku_action, legal_fill_actions, legal_sudoku_actions
from puzzle_jepa.data.worlds import WorldAction
from puzzle_jepa.models.grid_goal_jepa import GridTokenGoalJEPA, _affected_token_weights


ScoreMode = Literal[
    "oracle_goal_distance",
    "predicted_goal_distance",
    "oracle_goal_raw_euclidean_distance",
    "predicted_goal_raw_euclidean_distance",
    "oracle_goal_raw_squared_euclidean_distance",
    "predicted_goal_raw_squared_euclidean_distance",
    "oracle_goal_raw_mse_distance",
    "predicted_goal_raw_mse_distance",
    "oracle_goal_raw_cosine_distance",
    "predicted_goal_raw_cosine_distance",
    "oracle_goal_raw_hybrid_distance",
    "predicted_goal_raw_hybrid_distance",
    "oracle_goal_raw_euclidean_progress",
    "predicted_goal_raw_euclidean_progress",
    "oracle_goal_changed_cell_raw_euclidean_distance",
    "predicted_goal_changed_cell_raw_euclidean_distance",
    "oracle_goal_affected_context_raw_euclidean_distance",
    "predicted_goal_affected_context_raw_euclidean_distance",
    "oracle_goal_delta_top1_raw_euclidean_distance",
    "predicted_goal_delta_top1_raw_euclidean_distance",
    "oracle_goal_delta_top3_raw_euclidean_distance",
    "predicted_goal_delta_top3_raw_euclidean_distance",
    "oracle_goal_delta_top5_raw_euclidean_distance",
    "predicted_goal_delta_top5_raw_euclidean_distance",
    "oracle_goal_projected_euclidean_distance",
    "predicted_goal_projected_euclidean_distance",
    "predicted_goal_waypoint_goal_raw_euclidean_distance",
    "success_metric_distance",
    "terminal_value",
    "compatibility_energy",
    "remaining_edit_count",
    "verifier_energy",
    "oracle_waypoint_raw_euclidean_distance",
    "predicted_waypoint_raw_euclidean_distance",
    "predicted_waypoint_goal_raw_euclidean_distance",
]
TransitionMode = Literal["symbolic_reencode", "latent_rollout"]
PlannerMode = Literal[
    "mpc_beam",
    "categorical_cem",
    "hierarchical_cem",
    "hierarchical_beam",
    "waypoint_beam",
    "waypoint_hierarchical_beam",
    "waypoint_hierarchical_cem",
]


@dataclass(frozen=True, slots=True)
class BeamMPCResult:
    solved: bool
    steps: int
    remaining_hamming: int
    actions: list[WorldAction]
    final_board: np.ndarray
    score_mode: str
    transition_mode: str
    beam_width: int
    beam_depth: int
    action_evals: int
    elapsed_seconds: float


ACTION_VOCAB = tuple(WorldAction(row=row, col=col, value=value) for row in range(9) for col in range(9) for value in range(1, 10))


def hamming_distance(board: np.ndarray, goal: np.ndarray) -> int:
    return int(np.not_equal(board, goal).sum())


def remaining_blanks(board: np.ndarray) -> int:
    return int(np.count_nonzero(np.asarray(board) == 0))


def capped_horizon(requested: int, board: np.ndarray) -> int:
    blanks = remaining_blanks(board)
    if blanks <= 0:
        return 0
    return min(max(1, int(requested)), blanks)


def planning_horizon(requested: int, board: np.ndarray, *, allow_overwrite: bool) -> int:
    if allow_overwrite:
        return max(1, int(requested))
    return capped_horizon(requested, board)


@torch.no_grad()
def run_beam_mpc(
    model: GridTokenGoalJEPA,
    puzzle: np.ndarray,
    goal: np.ndarray,
    *,
    score_mode: ScoreMode,
    transition_mode: TransitionMode,
    beam_width: int,
    beam_depth: int,
    max_steps: int = 81,
    device: torch.device,
    allow_overwrite: bool = False,
) -> BeamMPCResult:
    start = time.time()
    model.eval()
    current = np.asarray(puzzle, dtype=np.int64).copy()
    actions_taken: list[WorldAction] = []
    clue_mask = np.asarray(puzzle, dtype=np.int64) != 0
    editable_mask = ~clue_mask
    active_mask = np.ones((9, 9), dtype=bool)
    context_latents, predicted_goal, oracle_goal, initial_latents = _prepare_goal_latents(
        model,
        current,
        goal,
        clue_mask,
        editable_mask,
        active_mask,
        device=device,
        score_mode=score_mode,
    )
    action_evals = 0
    for _ in range(max_steps):
        if np.array_equal(current, goal) or ((not allow_overwrite) and not np.any(current == 0)):
            break
        depth = planning_horizon(beam_depth, current, allow_overwrite=allow_overwrite)
        if depth <= 0:
            break
        predicted_goal = _predict_goal_for_board(
            model,
            current,
            context_latents,
            initial_latents,
            clue_mask,
            editable_mask,
            active_mask,
            device=device,
        )
        first, evals = beam_plan_once(
            model,
            current,
            goal,
            context_latents,
            predicted_goal,
            oracle_goal,
            clue_mask,
            editable_mask,
            active_mask,
            score_mode=score_mode,
            transition_mode=transition_mode,
            beam_width=beam_width,
            beam_depth=depth,
            device=device,
            allow_overwrite=allow_overwrite,
        )
        action_evals += evals
        if first is None:
            break
        try:
            current = apply_sudoku_action(
                current,
                first,
                clue_mask=clue_mask,
                allow_conflicts=True,
                allow_overwrite=allow_overwrite,
            )
        except ValueError:
            break
        actions_taken.append(first)
    return BeamMPCResult(
        solved=bool(np.array_equal(current, goal)),
        steps=len(actions_taken),
        remaining_hamming=hamming_distance(current, goal),
        actions=actions_taken,
        final_board=current,
        score_mode=score_mode,
        transition_mode=transition_mode,
        beam_width=beam_width,
        beam_depth=beam_depth,
        action_evals=action_evals,
        elapsed_seconds=time.time() - start,
    )


@torch.no_grad()
def run_categorical_cem_mpc(
    model: GridTokenGoalJEPA,
    puzzle: np.ndarray,
    goal: np.ndarray,
    *,
    score_mode: ScoreMode,
    transition_mode: TransitionMode,
    beam_width: int,
    beam_depth: int,
    max_steps: int = 81,
    device: torch.device,
    cem_samples: int = 128,
    cem_iters: int = 4,
    cem_elites: int = 16,
    cem_momentum: float = 0.7,
    seed: int = 0,
    allow_overwrite: bool = False,
) -> BeamMPCResult:
    del beam_width
    start = time.time()
    rng = np.random.default_rng(seed)
    model.eval()
    current = np.asarray(puzzle, dtype=np.int64).copy()
    actions_taken: list[WorldAction] = []
    clue_mask = np.asarray(puzzle, dtype=np.int64) != 0
    editable_mask = ~clue_mask
    active_mask = np.ones((9, 9), dtype=bool)
    context_latents, predicted_goal, oracle_goal, initial_latents = _prepare_goal_latents(
        model, current, goal, clue_mask, editable_mask, active_mask, device=device
    )
    action_evals = 0
    for _ in range(max_steps):
        if np.array_equal(current, goal) or ((not allow_overwrite) and not np.any(current == 0)):
            break
        depth = planning_horizon(beam_depth, current, allow_overwrite=allow_overwrite)
        if depth <= 0:
            break
        predicted_goal = _predict_goal_for_board(
            model,
            current,
            context_latents,
            initial_latents,
            clue_mask,
            editable_mask,
            active_mask,
            device=device,
            allow_overwrite=allow_overwrite,
        )
        first, evals = categorical_cem_plan_once(
            model,
            current,
            goal,
            context_latents,
            predicted_goal,
            oracle_goal,
            clue_mask,
            editable_mask,
            active_mask,
            score_mode=score_mode,
            transition_mode=transition_mode,
            horizon=depth,
            samples=cem_samples,
            iterations=cem_iters,
            elites=cem_elites,
            momentum=cem_momentum,
            rng=rng,
            device=device,
            allow_overwrite=allow_overwrite,
        )
        action_evals += evals
        if first is None:
            break
        try:
            current = apply_sudoku_action(
                current,
                first,
                clue_mask=clue_mask,
                allow_conflicts=True,
                allow_overwrite=allow_overwrite,
            )
        except ValueError:
            break
        actions_taken.append(first)
    return BeamMPCResult(
        solved=bool(np.array_equal(current, goal)),
        steps=len(actions_taken),
        remaining_hamming=hamming_distance(current, goal),
        actions=actions_taken,
        final_board=current,
        score_mode=score_mode,
        transition_mode=transition_mode,
        beam_width=cem_samples,
        beam_depth=beam_depth,
        action_evals=action_evals,
        elapsed_seconds=time.time() - start,
    )


@torch.no_grad()
def run_hierarchical_cem_mpc(
    model: GridTokenGoalJEPA,
    puzzle: np.ndarray,
    goal: np.ndarray,
    *,
    score_mode: ScoreMode,
    transition_mode: TransitionMode,
    beam_width: int,
    beam_depth: int,
    max_steps: int = 81,
    device: torch.device,
    cem_samples: int = 128,
    cem_iters: int = 4,
    cem_elites: int = 16,
    cem_momentum: float = 0.7,
    high_cem_samples: int = 128,
    high_cem_iters: int = 4,
    high_cem_elites: int = 16,
    high_cem_momentum: float = 0.7,
    high_cem_std: float = 1.0,
    high_cem_optimizer: str = "cem",
    high_cem_temperature: float = 1.0,
    high_cem_codebook: str = "none",
    high_cem_codebook_size: int = 0,
    seed: int = 0,
    allow_overwrite: bool = False,
) -> BeamMPCResult:
    del beam_width
    if not model.hierarchy_levels:
        raise ValueError("hierarchical_cem requires a checkpoint trained with hierarchy_levels.")
    start = time.time()
    rng = np.random.default_rng(seed)
    model.eval()
    current = np.asarray(puzzle, dtype=np.int64).copy()
    actions_taken: list[WorldAction] = []
    clue_mask = np.asarray(puzzle, dtype=np.int64) != 0
    editable_mask = ~clue_mask
    active_mask = np.ones((9, 9), dtype=bool)
    context_latents, predicted_goal, oracle_goal, initial_latents = _prepare_goal_latents(
        model, current, goal, clue_mask, editable_mask, active_mask, device=device
    )
    action_evals = 0
    for _ in range(max_steps):
        if np.array_equal(current, goal) or ((not allow_overwrite) and not np.any(current == 0)):
            break
        depth = planning_horizon(beam_depth, current, allow_overwrite=allow_overwrite)
        if depth <= 0:
            break
        predicted_goal = _predict_goal_for_board(
            model,
            current,
            context_latents,
            initial_latents,
            clue_mask,
            editable_mask,
            active_mask,
            device=device,
            allow_overwrite=allow_overwrite,
        )
        _, current_latent = score_board(
            model,
            current,
            context_latents,
            predicted_goal,
            oracle_goal,
            clue_mask,
            editable_mask,
            active_mask,
            score_mode=score_mode,
            device=device,
        )
        target_goal = _target_goal_latents(score_mode, predicted_goal, oracle_goal)
        subgoal = target_goal
        high_levels = tuple(level for level in sorted(model.hierarchy_levels, reverse=True) if level <= depth)
        for level in high_levels:
            macro_horizon = max(1, int(np.ceil(depth / level)))
            subgoal, evals = hierarchical_subgoal_cem(
                model,
                current_latent,
                subgoal,
                context_latents,
                active_mask,
                board=current,
                clue_mask=clue_mask,
                score_mode=score_mode,
                level=level,
                macro_horizon=macro_horizon,
                samples=high_cem_samples,
                iterations=high_cem_iters,
                elites=high_cem_elites,
                momentum=high_cem_momentum,
                init_std=high_cem_std,
                optimizer=high_cem_optimizer,
                temperature=high_cem_temperature,
                codebook=high_cem_codebook,
                codebook_size=high_cem_codebook_size,
                rng=rng,
                device=device,
            )
            action_evals += evals
        first, evals = categorical_cem_plan_once(
            model,
            current,
            goal,
            context_latents,
            subgoal,
            subgoal,
            clue_mask,
            editable_mask,
            active_mask,
            score_mode=_subgoal_score_mode(score_mode),  # type: ignore[arg-type]
            transition_mode=transition_mode,
            horizon=min(depth, min(high_levels) if high_levels else depth),
            samples=cem_samples,
            iterations=cem_iters,
            elites=cem_elites,
            momentum=cem_momentum,
            rng=rng,
            device=device,
        )
        action_evals += evals
        if first is None:
            break
        try:
            current = apply_sudoku_action(
                current,
                first,
                clue_mask=clue_mask,
                allow_conflicts=True,
                allow_overwrite=allow_overwrite,
            )
        except ValueError:
            break
        actions_taken.append(first)
    return BeamMPCResult(
        solved=bool(np.array_equal(current, goal)),
        steps=len(actions_taken),
        remaining_hamming=hamming_distance(current, goal),
        actions=actions_taken,
        final_board=current,
        score_mode=score_mode,
        transition_mode=transition_mode,
        beam_width=cem_samples,
        beam_depth=beam_depth,
        action_evals=action_evals,
        elapsed_seconds=time.time() - start,
    )


@torch.no_grad()
def run_hierarchical_beam_mpc(
    model: GridTokenGoalJEPA,
    puzzle: np.ndarray,
    goal: np.ndarray,
    *,
    score_mode: ScoreMode,
    transition_mode: TransitionMode,
    beam_width: int,
    beam_depth: int,
    max_steps: int = 81,
    device: torch.device,
    allow_overwrite: bool = False,
) -> BeamMPCResult:
    if not model.hierarchy_levels:
        raise ValueError("hierarchical_beam requires a checkpoint trained with hierarchy_levels.")
    start = time.time()
    model.eval()
    current = np.asarray(puzzle, dtype=np.int64).copy()
    actions_taken: list[WorldAction] = []
    clue_mask = np.asarray(puzzle, dtype=np.int64) != 0
    editable_mask = ~clue_mask
    active_mask = np.ones((9, 9), dtype=bool)
    context_latents, predicted_goal, oracle_goal, initial_latents = _prepare_goal_latents(
        model, current, goal, clue_mask, editable_mask, active_mask, device=device
    )
    action_evals = 0
    for _ in range(max_steps):
        if np.array_equal(current, goal) or ((not allow_overwrite) and not np.any(current == 0)):
            break
        depth = planning_horizon(beam_depth, current, allow_overwrite=allow_overwrite)
        if depth <= 0:
            break
        predicted_goal = _predict_goal_for_board(
            model,
            current,
            context_latents,
            initial_latents,
            clue_mask,
            editable_mask,
            active_mask,
            device=device,
            allow_overwrite=allow_overwrite,
        )
        _, current_latent = score_board(
            model,
            current,
            context_latents,
            predicted_goal,
            oracle_goal,
            clue_mask,
            editable_mask,
            active_mask,
            score_mode=score_mode,
            device=device,
        )
        target_goal = _target_goal_latents(score_mode, predicted_goal, oracle_goal)
        subgoal = target_goal
        high_levels = tuple(level for level in sorted(model.hierarchy_levels, reverse=True) if level <= depth)
        for level in high_levels:
            subgoal, evals = hierarchical_subgoal_beam(
                model,
                current,
                current_latent,
                subgoal,
                context_latents,
                clue_mask,
                editable_mask,
                active_mask,
                score_mode=score_mode,
                level=level,
                beam_width=beam_width,
                device=device,
                allow_overwrite=allow_overwrite,
            )
            action_evals += evals
        first, evals = beam_plan_once(
            model,
            current,
            goal,
            context_latents,
            subgoal,
            subgoal,
            clue_mask,
            editable_mask,
            active_mask,
            score_mode=_subgoal_score_mode(score_mode),  # type: ignore[arg-type]
            transition_mode=transition_mode,
            beam_width=beam_width,
            beam_depth=min(depth, min(high_levels) if high_levels else depth),
            device=device,
            allow_overwrite=allow_overwrite,
        )
        action_evals += evals
        if first is None:
            break
        try:
            current = apply_sudoku_action(
                current,
                first,
                clue_mask=clue_mask,
                allow_conflicts=True,
                allow_overwrite=allow_overwrite,
            )
        except ValueError:
            break
        actions_taken.append(first)
    return BeamMPCResult(
        solved=bool(np.array_equal(current, goal)),
        steps=len(actions_taken),
        remaining_hamming=hamming_distance(current, goal),
        actions=actions_taken,
        final_board=current,
        score_mode=score_mode,
        transition_mode=transition_mode,
        beam_width=beam_width,
        beam_depth=beam_depth,
        action_evals=action_evals,
        elapsed_seconds=time.time() - start,
    )


@torch.no_grad()
def run_waypoint_beam_mpc(
    model: GridTokenGoalJEPA,
    puzzle: np.ndarray,
    goal: np.ndarray,
    *,
    score_mode: ScoreMode,
    transition_mode: TransitionMode,
    beam_width: int,
    beam_depth: int,
    max_steps: int = 81,
    device: torch.device,
    allow_overwrite: bool = False,
    waypoint_horizon: int = 8,
    hierarchical: bool = False,
) -> BeamMPCResult:
    if hierarchical and not model.hierarchy_levels:
        raise ValueError("waypoint_hierarchical_beam requires a checkpoint trained with hierarchy_levels.")
    start = time.time()
    model.eval()
    current = np.asarray(puzzle, dtype=np.int64).copy()
    actions_taken: list[WorldAction] = []
    clue_mask = np.asarray(puzzle, dtype=np.int64) != 0
    editable_mask = ~clue_mask
    active_mask = np.ones((9, 9), dtype=bool)
    context_latents, predicted_goal, oracle_goal, initial_latents = _prepare_goal_latents(
        model, current, goal, clue_mask, editable_mask, active_mask, device=device
    )
    del oracle_goal
    action_evals = 0
    for _ in range(max_steps):
        if np.array_equal(current, goal) or ((not allow_overwrite) and not np.any(current == 0)):
            break
        depth = planning_horizon(beam_depth, current, allow_overwrite=allow_overwrite)
        if depth <= 0:
            break
        if str(score_mode).startswith("oracle_waypoint_"):
            waypoint = _oracle_future_waypoint_latents(
                model,
                current,
                goal,
                context_latents,
                clue_mask,
                editable_mask,
                active_mask,
                horizon=waypoint_horizon,
                device=device,
            )
        else:
            waypoint = _predict_waypoint_for_board(
                model,
                current,
                context_latents,
                initial_latents,
                clue_mask,
                editable_mask,
                active_mask,
                horizon=waypoint_horizon,
                device=device,
            )
        terminal_goal = waypoint
        inner_score_mode = "oracle_goal_raw_euclidean_distance"
        if str(score_mode) == "predicted_waypoint_goal_raw_euclidean_distance":
            predicted_goal = _predict_goal_for_board(
                model,
                current,
                context_latents,
                initial_latents,
                clue_mask,
                editable_mask,
                active_mask,
                device=device,
                allow_overwrite=allow_overwrite,
            )
            terminal_goal = predicted_goal
            inner_score_mode = "predicted_goal_waypoint_goal_raw_euclidean_distance"
        if hierarchical:
            first, evals = _waypoint_hierarchical_beam_once(
                model,
                current,
                goal,
                context_latents,
                waypoint,
                clue_mask,
                editable_mask,
                active_mask,
                transition_mode=transition_mode,
                beam_width=beam_width,
                beam_depth=depth,
                device=device,
                allow_overwrite=allow_overwrite,
            )
        else:
            first, evals = beam_plan_once(
                model,
                current,
                goal,
                context_latents,
                waypoint,
                terminal_goal,
                clue_mask,
                editable_mask,
                active_mask,
                score_mode=inner_score_mode,
                transition_mode=transition_mode,
                beam_width=beam_width,
                beam_depth=depth,
                device=device,
                allow_overwrite=allow_overwrite,
            )
        action_evals += evals
        if first is None:
            break
        try:
            current = apply_sudoku_action(
                current,
                first,
                clue_mask=clue_mask,
                allow_conflicts=True,
                allow_overwrite=allow_overwrite,
            )
        except ValueError:
            break
        actions_taken.append(first)
    return BeamMPCResult(
        solved=bool(np.array_equal(current, goal)),
        steps=len(actions_taken),
        remaining_hamming=hamming_distance(current, goal),
        actions=actions_taken,
        final_board=current,
        score_mode=score_mode,
        transition_mode=transition_mode,
        beam_width=beam_width,
        beam_depth=beam_depth,
        action_evals=action_evals,
        elapsed_seconds=time.time() - start,
    )


@torch.no_grad()
def run_waypoint_hierarchical_cem_mpc(
    model: GridTokenGoalJEPA,
    puzzle: np.ndarray,
    goal: np.ndarray,
    *,
    score_mode: ScoreMode,
    transition_mode: TransitionMode,
    beam_width: int,
    beam_depth: int,
    max_steps: int = 81,
    device: torch.device,
    cem_samples: int = 128,
    cem_iters: int = 4,
    cem_elites: int = 16,
    cem_momentum: float = 0.7,
    high_cem_samples: int = 128,
    high_cem_iters: int = 4,
    high_cem_elites: int = 16,
    high_cem_momentum: float = 0.7,
    high_cem_std: float = 1.0,
    high_cem_optimizer: str = "cem",
    high_cem_temperature: float = 1.0,
    high_cem_codebook: str = "none",
    high_cem_codebook_size: int = 0,
    seed: int = 0,
    allow_overwrite: bool = False,
    waypoint_horizon: int = 8,
) -> BeamMPCResult:
    del beam_width
    if not model.hierarchy_levels:
        raise ValueError("waypoint_hierarchical_cem requires a checkpoint trained with hierarchy_levels.")
    start = time.time()
    rng = np.random.default_rng(seed)
    model.eval()
    current = np.asarray(puzzle, dtype=np.int64).copy()
    actions_taken: list[WorldAction] = []
    clue_mask = np.asarray(puzzle, dtype=np.int64) != 0
    editable_mask = ~clue_mask
    active_mask = np.ones((9, 9), dtype=bool)
    context_latents, predicted_goal, oracle_goal, initial_latents = _prepare_goal_latents(
        model, current, goal, clue_mask, editable_mask, active_mask, device=device
    )
    del predicted_goal, oracle_goal
    action_evals = 0
    for _ in range(max_steps):
        if np.array_equal(current, goal) or ((not allow_overwrite) and not np.any(current == 0)):
            break
        depth = planning_horizon(beam_depth, current, allow_overwrite=allow_overwrite)
        if depth <= 0:
            break
        if str(score_mode).startswith("oracle_waypoint_"):
            waypoint = _oracle_future_waypoint_latents(
                model,
                current,
                goal,
                context_latents,
                clue_mask,
                editable_mask,
                active_mask,
                horizon=waypoint_horizon,
                device=device,
            )
        else:
            waypoint = _predict_waypoint_for_board(
                model,
                current,
                context_latents,
                initial_latents,
                clue_mask,
                editable_mask,
                active_mask,
                horizon=waypoint_horizon,
                device=device,
            )
        first, evals = _waypoint_hierarchical_cem_once(
            model,
            current,
            goal,
            context_latents,
            waypoint,
            clue_mask,
            editable_mask,
            active_mask,
            transition_mode=transition_mode,
            beam_depth=depth,
            device=device,
            cem_samples=cem_samples,
            cem_iters=cem_iters,
            cem_elites=cem_elites,
            cem_momentum=cem_momentum,
            high_cem_samples=high_cem_samples,
            high_cem_iters=high_cem_iters,
            high_cem_elites=high_cem_elites,
            high_cem_momentum=high_cem_momentum,
            high_cem_std=high_cem_std,
            high_cem_optimizer=high_cem_optimizer,
            high_cem_temperature=high_cem_temperature,
            high_cem_codebook=high_cem_codebook,
            high_cem_codebook_size=high_cem_codebook_size,
            rng=rng,
            allow_overwrite=allow_overwrite,
        )
        action_evals += evals
        if first is None:
            break
        try:
            current = apply_sudoku_action(
                current,
                first,
                clue_mask=clue_mask,
                allow_conflicts=True,
                allow_overwrite=allow_overwrite,
            )
        except ValueError:
            break
        actions_taken.append(first)
    return BeamMPCResult(
        solved=bool(np.array_equal(current, goal)),
        steps=len(actions_taken),
        remaining_hamming=hamming_distance(current, goal),
        actions=actions_taken,
        final_board=current,
        score_mode=score_mode,
        transition_mode=transition_mode,
        beam_width=cem_samples,
        beam_depth=beam_depth,
        action_evals=action_evals,
        elapsed_seconds=time.time() - start,
    )


@torch.no_grad()
def _waypoint_hierarchical_beam_once(
    model: GridTokenGoalJEPA,
    current: np.ndarray,
    goal: np.ndarray,
    context_latents: torch.Tensor,
    waypoint: torch.Tensor,
    clue_mask: np.ndarray,
    editable_mask: np.ndarray,
    active_mask: np.ndarray,
    *,
    transition_mode: TransitionMode,
    beam_width: int,
    beam_depth: int,
    device: torch.device,
    allow_overwrite: bool,
) -> tuple[WorldAction | None, int]:
    _, current_latent = score_board(
        model,
        current,
        context_latents,
        waypoint,
        waypoint,
        clue_mask,
        editable_mask,
        active_mask,
        score_mode="oracle_goal_raw_euclidean_distance",
        device=device,
    )
    subgoal = waypoint
    action_evals = 0
    high_levels = tuple(level for level in sorted(model.hierarchy_levels, reverse=True) if level <= beam_depth)
    for level in high_levels:
        subgoal, evals = hierarchical_subgoal_beam(
            model,
            current,
            current_latent,
            subgoal,
            context_latents,
            clue_mask,
            editable_mask,
            active_mask,
            score_mode="oracle_goal_raw_euclidean_distance",
            level=level,
            beam_width=beam_width,
            device=device,
            allow_overwrite=allow_overwrite,
        )
        action_evals += evals
    first, evals = beam_plan_once(
        model,
        current,
        goal,
        context_latents,
        subgoal,
        subgoal,
        clue_mask,
        editable_mask,
        active_mask,
        score_mode="oracle_goal_raw_euclidean_distance",
        transition_mode=transition_mode,
        beam_width=beam_width,
        beam_depth=min(beam_depth, min(high_levels) if high_levels else beam_depth),
        device=device,
        allow_overwrite=allow_overwrite,
    )
    return first, action_evals + evals


@torch.no_grad()
def _waypoint_hierarchical_cem_once(
    model: GridTokenGoalJEPA,
    current: np.ndarray,
    goal: np.ndarray,
    context_latents: torch.Tensor,
    waypoint: torch.Tensor,
    clue_mask: np.ndarray,
    editable_mask: np.ndarray,
    active_mask: np.ndarray,
    *,
    transition_mode: TransitionMode,
    beam_depth: int,
    device: torch.device,
    cem_samples: int,
    cem_iters: int,
    cem_elites: int,
    cem_momentum: float,
    high_cem_samples: int,
    high_cem_iters: int,
    high_cem_elites: int,
    high_cem_momentum: float,
    high_cem_std: float,
    high_cem_optimizer: str,
    high_cem_temperature: float,
    high_cem_codebook: str,
    high_cem_codebook_size: int,
    rng: np.random.Generator,
    allow_overwrite: bool,
) -> tuple[WorldAction | None, int]:
    _, current_latent = score_board(
        model,
        current,
        context_latents,
        waypoint,
        waypoint,
        clue_mask,
        editable_mask,
        active_mask,
        score_mode="oracle_goal_raw_euclidean_distance",
        device=device,
    )
    subgoal = waypoint
    action_evals = 0
    high_levels = tuple(level for level in sorted(model.hierarchy_levels, reverse=True) if level <= beam_depth)
    for level in high_levels:
        macro_horizon = max(1, int(np.ceil(beam_depth / level)))
        subgoal, evals = hierarchical_subgoal_cem(
            model,
            current_latent,
            subgoal,
            context_latents,
                active_mask,
                board=current,
                clue_mask=clue_mask,
                score_mode="oracle_goal_raw_euclidean_distance",
            level=level,
            macro_horizon=macro_horizon,
            samples=high_cem_samples,
            iterations=high_cem_iters,
            elites=high_cem_elites,
            momentum=high_cem_momentum,
            init_std=high_cem_std,
            optimizer=high_cem_optimizer,
            temperature=high_cem_temperature,
            codebook=high_cem_codebook,
            codebook_size=high_cem_codebook_size,
            rng=rng,
            device=device,
            allow_overwrite=allow_overwrite,
        )
        action_evals += evals
    first, evals = categorical_cem_plan_once(
        model,
        current,
        goal,
        context_latents,
        subgoal,
        subgoal,
        clue_mask,
        editable_mask,
        active_mask,
        score_mode="oracle_goal_raw_euclidean_distance",
        transition_mode=transition_mode,
        horizon=min(beam_depth, min(high_levels) if high_levels else beam_depth),
        samples=cem_samples,
        iterations=cem_iters,
        elites=cem_elites,
        momentum=cem_momentum,
        rng=rng,
        device=device,
        allow_overwrite=allow_overwrite,
    )
    return first, action_evals + evals


@torch.no_grad()
def hierarchical_subgoal_beam(
    model: GridTokenGoalJEPA,
    board: np.ndarray,
    start_latent: torch.Tensor,
    goal_latent: torch.Tensor,
    context_latents: torch.Tensor,
    clue_mask: np.ndarray,
    editable_mask: np.ndarray,
    active_mask: np.ndarray,
    *,
    score_mode: ScoreMode,
    level: int,
    beam_width: int,
    device: torch.device,
    allow_overwrite: bool = False,
) -> tuple[torch.Tensor, int]:
    candidates, evals = _latent_beam_candidates(
        model,
        board,
        start_latent,
        context_latents,
        goal_latent,
        clue_mask,
        active_mask,
        score_mode=_subgoal_score_mode(score_mode),  # type: ignore[arg-type]
        beam_width=beam_width,
        beam_depth=level,
        device=device,
        allow_overwrite=allow_overwrite,
    )
    if not candidates:
        return goal_latent, evals
    sequences = [seq for _, seq, _ in candidates if len(seq) >= level]
    if not sequences:
        return goal_latent, evals
    action_t = torch.as_tensor(
        [[[action.row, action.col, action.value] for action in seq[:level]] for seq in sequences],
        dtype=torch.long,
        device=device,
    )
    start = start_latent.expand(action_t.shape[0], -1, -1)
    context = context_latents.expand(action_t.shape[0], -1, -1)
    predicted_waypoints = model.predict_high_level(start, action_t, context, level=level)
    mask = torch.as_tensor(active_mask[None], dtype=torch.bool, device=device).expand(action_t.shape[0], -1, -1)
    target = _expand_tokens_like(goal_latent, predicted_waypoints)
    scores = latent_distance(model, predicted_waypoints, target, target, mask, _subgoal_score_mode(score_mode))
    if _policy_prior_planning_weight(model) > 0.0:
        macro_priors = model.score_macro_action_prior(start, target, context, mask, action_t, level=level)
        scores = scores - _policy_prior_planning_weight(model) * macro_priors.to(dtype=scores.dtype)
    best = int(torch.argmin(scores).item())
    return predicted_waypoints[best : best + 1].detach(), evals + action_t.shape[0]


@torch.no_grad()
def beam_plan_once(
    model: GridTokenGoalJEPA,
    board: np.ndarray,
    goal: np.ndarray,
    context_latents: torch.Tensor,
    predicted_goal: torch.Tensor,
    oracle_goal: torch.Tensor,
    clue_mask: np.ndarray,
    editable_mask: np.ndarray,
    active_mask: np.ndarray,
    *,
    score_mode: ScoreMode,
    transition_mode: TransitionMode,
    beam_width: int,
    beam_depth: int,
    device: torch.device,
    allow_overwrite: bool = False,
) -> tuple[WorldAction | None, int]:
    depth = planning_horizon(beam_depth, board, allow_overwrite=allow_overwrite)
    if depth <= 0:
        return None, 0
    if transition_mode == "symbolic_reencode":
        return _beam_plan_once_symbolic(
            model,
            board,
            context_latents,
            predicted_goal,
            oracle_goal,
            clue_mask,
            editable_mask,
            active_mask,
            score_mode=score_mode,
            beam_width=beam_width,
            beam_depth=depth,
            device=device,
            allow_overwrite=allow_overwrite,
        )
    return _beam_plan_once_latent(
        model,
        board,
        context_latents,
        predicted_goal,
        oracle_goal,
        clue_mask,
        editable_mask,
        active_mask,
        score_mode=score_mode,
        beam_width=beam_width,
        beam_depth=depth,
        device=device,
        allow_overwrite=allow_overwrite,
    )


@torch.no_grad()
def categorical_cem_plan_once(
    model: GridTokenGoalJEPA,
    board: np.ndarray,
    goal: np.ndarray,
    context_latents: torch.Tensor,
    predicted_goal: torch.Tensor,
    oracle_goal: torch.Tensor,
    clue_mask: np.ndarray,
    editable_mask: np.ndarray,
    active_mask: np.ndarray,
    *,
    score_mode: ScoreMode,
    transition_mode: TransitionMode,
    horizon: int,
    samples: int,
    iterations: int,
    elites: int,
    momentum: float,
    rng: np.random.Generator,
    device: torch.device,
    allow_overwrite: bool = False,
) -> tuple[WorldAction | None, int]:
    del goal
    horizon = planning_horizon(horizon, board, allow_overwrite=allow_overwrite)
    if horizon <= 0:
        return None, 0
    samples = max(1, int(samples))
    iterations = max(1, int(iterations))
    elites = max(1, min(int(elites), samples))
    momentum = float(np.clip(momentum, 0.0, 0.999))
    probs = np.full((horizon, len(ACTION_VOCAB)), 1.0 / len(ACTION_VOCAB), dtype=np.float64)
    best_score = float("inf")
    best_first: WorldAction | None = None
    action_evals = 0
    for _ in range(iterations):
        seq_ids, final_boards = _sample_categorical_action_sequences(
            board,
            probs,
            clue_mask=clue_mask,
            allow_overwrite=allow_overwrite,
            samples=samples,
            rng=rng,
        )
        scores = _score_cem_sequences(
            model,
            board,
            final_boards,
            seq_ids,
            context_latents,
            predicted_goal,
            oracle_goal,
            clue_mask,
            editable_mask,
            active_mask,
            score_mode=score_mode,
            transition_mode=transition_mode,
            device=device,
        )
        action_evals += samples * horizon
        order = np.argsort(scores)
        if float(scores[order[0]]) < best_score:
            best_score = float(scores[order[0]])
            best_first = ACTION_VOCAB[int(seq_ids[order[0], 0])]
        elite_ids = seq_ids[order[:elites]]
        elite_probs = np.full_like(probs, 1.0e-6)
        for step in range(horizon):
            counts = np.bincount(elite_ids[:, step], minlength=len(ACTION_VOCAB)).astype(np.float64)
            elite_probs[step] += counts / counts.sum().clip(min=1.0)
            elite_probs[step] /= elite_probs[step].sum()
        probs = momentum * probs + (1.0 - momentum) * elite_probs
        probs /= probs.sum(axis=1, keepdims=True)
    return best_first, action_evals


@torch.no_grad()
def hierarchical_subgoal_cem(
    model: GridTokenGoalJEPA,
    start_latent: torch.Tensor,
    goal_latent: torch.Tensor,
    context_latents: torch.Tensor,
    active_mask: np.ndarray,
    *,
    board: np.ndarray | None = None,
    clue_mask: np.ndarray | None = None,
    score_mode: ScoreMode,
    level: int,
    macro_horizon: int,
    samples: int,
    iterations: int,
    elites: int,
    momentum: float,
    init_std: float,
    optimizer: str = "cem",
    temperature: float = 1.0,
    codebook: str = "none",
    codebook_size: int = 0,
    rng: np.random.Generator,
    device: torch.device,
    allow_overwrite: bool = False,
) -> tuple[torch.Tensor, int]:
    level = int(level)
    macro_horizon = max(1, int(macro_horizon))
    samples = max(1, int(samples))
    iterations = max(1, int(iterations))
    elites = max(1, min(int(elites), samples))
    momentum = float(np.clip(momentum, 0.0, 0.999))
    optimizer = str(optimizer)
    if optimizer not in {"cem", "mppi"}:
        raise ValueError("hierarchical_subgoal_cem optimizer must be 'cem' or 'mppi'.")
    codebook = str(codebook)
    if codebook not in {"none", "init"}:
        raise ValueError("hierarchical_subgoal_cem codebook must be 'none' or 'init'.")
    macro_dim = int(getattr(model, "macro_action_dim", model.d_model))
    mean = torch.zeros((macro_horizon, macro_dim), dtype=start_latent.dtype, device=device)
    std = torch.full_like(mean, float(init_std))
    if codebook == "init" and board is not None:
        prior = _sample_macro_action_codebook(
            model,
            board,
            level=level,
            samples=max(samples, int(codebook_size)),
            rng=rng,
            device=device,
            dtype=start_latent.dtype,
            clue_mask=clue_mask,
            allow_overwrite=allow_overwrite,
        )
        if prior is not None:
            mean = prior.mean(dim=0, keepdim=True).expand(macro_horizon, -1).clone()
            prior_std = prior.std(dim=0, unbiased=False).clamp_min(1.0e-3)
            std = (float(init_std) * prior_std).unsqueeze(0).expand_as(mean).clone()
    mask = torch.as_tensor(active_mask[None], dtype=torch.bool, device=device).expand(samples, -1, -1)
    best_score = float("inf")
    best_subgoal = goal_latent.detach()
    action_evals = 0
    for _ in range(iterations):
        noise = torch.as_tensor(
            rng.standard_normal((samples, macro_horizon, macro_dim)),
            dtype=start_latent.dtype,
            device=device,
        )
        macro_actions = mean.unsqueeze(0) + std.unsqueeze(0) * noise
        rollout = start_latent.expand(samples, -1, -1)
        context = context_latents.expand(samples, -1, -1)
        first_subgoals: torch.Tensor | None = None
        for step in range(macro_horizon):
            rollout = model.predict_high_level_from_macro(rollout, macro_actions[:, step], context, level=level)
            if step == 0:
                first_subgoals = rollout.detach()
        target = _expand_tokens_like(goal_latent, rollout)
        score_values = latent_distance(model, rollout, target, target, mask, _subgoal_score_mode(score_mode))
        scores = score_values.detach().float()
        action_evals += samples * macro_horizon
        order = torch.argsort(scores)
        best_index = int(order[0].item())
        if float(scores[best_index].item()) < best_score and first_subgoals is not None:
            best_score = float(scores[best_index].item())
            best_subgoal = first_subgoals[best_index : best_index + 1].detach()
        if optimizer == "mppi":
            weights = torch.softmax(-(scores - scores.min()) / max(float(temperature), 1.0e-6), dim=0)
            view = weights.view(-1, 1, 1)
            elite_mean = (macro_actions * view).sum(dim=0)
            elite_var = ((macro_actions - elite_mean.unsqueeze(0)).square() * view).sum(dim=0)
            elite_std = elite_var.clamp_min(1.0e-6).sqrt()
        else:
            elite = macro_actions[order[:elites]]
            elite_mean = elite.mean(dim=0)
            elite_std = elite.std(dim=0, unbiased=False).clamp_min(1.0e-3)
        mean = momentum * mean + (1.0 - momentum) * elite_mean
        std = momentum * std + (1.0 - momentum) * elite_std
    return best_subgoal, action_evals


@torch.no_grad()
def _sample_macro_action_codebook(
    model: GridTokenGoalJEPA,
    board: np.ndarray,
    *,
    level: int,
    samples: int,
    rng: np.random.Generator,
    device: torch.device,
    dtype: torch.dtype,
    clue_mask: np.ndarray | None = None,
    allow_overwrite: bool = False,
) -> torch.Tensor | None:
    sequences = _sample_macro_action_sequences(
        board,
        level=level,
        samples=samples,
        rng=rng,
        clue_mask=clue_mask,
        allow_overwrite=allow_overwrite,
    )
    if len(sequences) == 0:
        return None
    action_t = torch.as_tensor(sequences, dtype=torch.long, device=device)
    latents: list[torch.Tensor] = []
    for chunk in action_t.split(512):
        latents.append(model.encode_macro_action(chunk).detach().to(dtype=dtype))
    return torch.cat(latents, dim=0)


def _sample_macro_action_sequences(
    board: np.ndarray,
    *,
    level: int,
    samples: int,
    rng: np.random.Generator,
    clue_mask: np.ndarray | None = None,
    allow_overwrite: bool = False,
) -> np.ndarray:
    level = max(1, int(level))
    samples = max(1, int(samples))
    sequences: list[list[tuple[int, int, int]]] = []
    attempts = 0
    max_attempts = max(samples * 4, 16)
    while len(sequences) < samples and attempts < max_attempts:
        attempts += 1
        current = np.asarray(board, dtype=np.int64).copy()
        clue = None if clue_mask is None else np.asarray(clue_mask, dtype=bool)
        sequence: list[tuple[int, int, int]] = []
        for _ in range(level):
            valid = np.flatnonzero(_valid_action_vocab_mask(current, clue_mask=clue, allow_overwrite=allow_overwrite))
            if len(valid) == 0:
                break
            action = ACTION_VOCAB[int(rng.choice(valid))]
            sequence.append((action.row, action.col, action.value))
            try:
                current = apply_sudoku_action(
                    current,
                    action,
                    clue_mask=clue,
                    allow_conflicts=True,
                    allow_overwrite=allow_overwrite,
                )
            except ValueError:
                break
        if len(sequence) == level:
            sequences.append(sequence)
    if not sequences:
        return np.empty((0, level, 3), dtype=np.int64)
    return np.asarray(sequences, dtype=np.int64)


@torch.no_grad()
def _beam_plan_once_latent(
    model: GridTokenGoalJEPA,
    board: np.ndarray,
    context_latents: torch.Tensor,
    predicted_goal: torch.Tensor,
    oracle_goal: torch.Tensor,
    clue_mask: np.ndarray,
    editable_mask: np.ndarray,
    active_mask: np.ndarray,
    *,
    score_mode: ScoreMode,
    beam_width: int,
    beam_depth: int,
    device: torch.device,
    allow_overwrite: bool = False,
    chunk_size: int = 2048,
) -> tuple[WorldAction | None, int]:
    _, start_latent = score_board(
        model,
        board,
        context_latents,
        predicted_goal,
        oracle_goal,
        clue_mask,
        editable_mask,
        active_mask,
        score_mode=score_mode,
        device=device,
    )
    use_history = _uses_single_state_history(model, start_latent)
    start_history = start_latent.unsqueeze(1) if use_history else None
    beam: list[tuple[float, np.ndarray, list[WorldAction], torch.Tensor | None, torch.Tensor | None]] = [
        (0.0, board.copy(), [], start_latent, start_history)
    ]
    best: tuple[float, list[WorldAction]] | None = None
    action_evals = 0
    base_progress_mode = _progress_base_score_mode(score_mode)
    target_goal = _target_goal_latents(score_mode, predicted_goal, oracle_goal)
    for _ in range(max(1, int(beam_depth))):
        parent_latents: list[torch.Tensor] = []
        parent_histories: list[torch.Tensor] = []
        leaves: list[np.ndarray] = []
        seqs: list[list[WorldAction]] = []
        actions: list[WorldAction] = []
        for _, node_board, seq, latent, history in beam:
            if latent is None:
                raise RuntimeError("Latent rollout beam node is missing its latent state.")
            for action in legal_sudoku_actions(
                node_board,
                clue_mask=clue_mask,
                allow_conflicts=True,
                allow_overwrite=allow_overwrite,
            ):
                try:
                    leaf = apply_sudoku_action(
                        node_board,
                        action,
                        clue_mask=clue_mask,
                        allow_conflicts=True,
                        allow_overwrite=allow_overwrite,
                    )
                except ValueError:
                    continue
                parent_latents.append(latent)
                if use_history:
                    if history is None:
                        raise RuntimeError("Single-state latent rollout beam node is missing its history.")
                    parent_histories.append(history)
                leaves.append(leaf)
                seqs.append([*seq, action])
                actions.append(action)
        if not leaves:
            break
        candidates: list[tuple[float, np.ndarray, list[WorldAction], torch.Tensor | None, torch.Tensor | None]] = []
        mask_t = torch.as_tensor(active_mask[None], dtype=torch.bool, device=device)
        for start in range(0, len(leaves), chunk_size):
            end = min(len(leaves), start + chunk_size)
            parent_batch = torch.cat(parent_latents[start:end], dim=0)
            action_t = torch.as_tensor(
                [[action.row, action.col, action.value] for action in actions[start:end]],
                dtype=torch.long,
                device=device,
            )
            context_t = context_latents.expand(parent_batch.shape[0], -1, -1)
            if use_history:
                parent_history_batch = torch.cat(parent_histories[start:end], dim=0)
                action_history_t = _actions_to_tensor(seqs[start:end], device=device)
                next_latents = model.predict_next_sequence(parent_history_batch, action_history_t, context_t)[:, -1]
            else:
                parent_history_batch = None
                next_latents = model.predict_next(parent_batch, action_t, context_t)
            chunk_mask = mask_t.expand(parent_batch.shape[0], -1, -1)
            if base_progress_mode is not None:
                next_score = raw_tokenwise_euclidean_distance(next_latents, _expand_tokens_like(target_goal, next_latents), chunk_mask)
                node_score = raw_tokenwise_euclidean_distance(parent_batch, _expand_tokens_like(target_goal, parent_batch), chunk_mask)
                score_values = next_score - node_score
            elif _is_delta_topk_score(score_mode):
                score_values = delta_topk_raw_euclidean_distances(
                    next_latents,
                    parent_batch,
                    _expand_tokens_like(target_goal, next_latents),
                    chunk_mask,
                    top_k=_delta_topk_value(score_mode),
                )
            elif _is_changed_cell_score(score_mode):
                score_values = changed_cell_raw_euclidean_distances(next_latents, _expand_tokens_like(target_goal, next_latents), actions[start:end])
            elif _is_affected_context_score(score_mode):
                score_values = affected_context_raw_euclidean_distances(
                    next_latents,
                    _expand_tokens_like(target_goal, next_latents),
                    actions[start:end],
                )
            else:
                score_values = latent_distance(model, next_latents, predicted_goal, oracle_goal, chunk_mask, score_mode)
            score_values = _apply_policy_prior_bias(
                model,
                score_values,
                parent_batch,
                target_goal,
                context_t,
                chunk_mask,
                action_t,
                score_mode=score_mode,
            )
            scores = [float(x) for x in score_values.detach().cpu().tolist()]
            for offset, score in enumerate(scores):
                idx = start + offset
                latent = next_latents[offset : offset + 1].detach()
                if use_history:
                    if parent_history_batch is None:
                        raise RuntimeError("Single-state parent history was not built.")
                    history = torch.cat([parent_history_batch[offset : offset + 1], latent.unsqueeze(1)], dim=1).detach()
                else:
                    history = None
                candidates.append((score, leaves[idx], seqs[idx], latent, history))
                if best is None or score < best[0]:
                    best = (score, seqs[idx])
            candidates.sort(key=lambda item: item[0])
            candidates = candidates[: max(1, int(beam_width))]
            action_evals += end - start
        beam = candidates
        if _can_early_stop(score_mode) and beam[0][0] <= 1.0e-8:
            break
    if best is None or not best[1]:
        return None, action_evals
    return best[1][0], action_evals


@torch.no_grad()
def _latent_beam_candidates(
    model: GridTokenGoalJEPA,
    board: np.ndarray,
    start_latent: torch.Tensor,
    context_latents: torch.Tensor,
    target_goal: torch.Tensor,
    clue_mask: np.ndarray,
    active_mask: np.ndarray,
    *,
    score_mode: ScoreMode,
    beam_width: int,
    beam_depth: int,
    device: torch.device,
    allow_overwrite: bool = False,
    chunk_size: int = 2048,
) -> tuple[list[tuple[float, list[WorldAction], torch.Tensor]], int]:
    use_history = _uses_single_state_history(model, start_latent)
    start_history = start_latent.unsqueeze(1) if use_history else None
    beam: list[tuple[float, np.ndarray, list[WorldAction], torch.Tensor, torch.Tensor | None]] = [
        (0.0, board.copy(), [], start_latent, start_history)
    ]
    action_evals = 0
    kept: list[tuple[float, list[WorldAction], torch.Tensor]] = []
    for _ in range(max(1, int(beam_depth))):
        parent_latents: list[torch.Tensor] = []
        parent_histories: list[torch.Tensor] = []
        leaves: list[np.ndarray] = []
        seqs: list[list[WorldAction]] = []
        actions: list[WorldAction] = []
        for _, node_board, seq, latent, history in beam:
            for action in legal_sudoku_actions(
                node_board,
                clue_mask=clue_mask,
                allow_conflicts=True,
                allow_overwrite=allow_overwrite,
            ):
                try:
                    leaf = apply_sudoku_action(
                        node_board,
                        action,
                        clue_mask=clue_mask,
                        allow_conflicts=True,
                        allow_overwrite=allow_overwrite,
                    )
                except ValueError:
                    continue
                parent_latents.append(latent)
                if use_history:
                    if history is None:
                        raise RuntimeError("Single-state latent rollout beam node is missing its history.")
                    parent_histories.append(history)
                leaves.append(leaf)
                seqs.append([*seq, action])
                actions.append(action)
        if not leaves:
            break
        candidates: list[tuple[float, np.ndarray, list[WorldAction], torch.Tensor, torch.Tensor | None]] = []
        mask_t = torch.as_tensor(active_mask[None], dtype=torch.bool, device=device)
        for start in range(0, len(leaves), chunk_size):
            end = min(len(leaves), start + chunk_size)
            parent_batch = torch.cat(parent_latents[start:end], dim=0)
            action_t = torch.as_tensor(
                [[action.row, action.col, action.value] for action in actions[start:end]],
                dtype=torch.long,
                device=device,
            )
            context_t = context_latents.expand(parent_batch.shape[0], -1, -1)
            if use_history:
                parent_history_batch = torch.cat(parent_histories[start:end], dim=0)
                action_history_t = _actions_to_tensor(seqs[start:end], device=device)
                next_latents = model.predict_next_sequence(parent_history_batch, action_history_t, context_t)[:, -1]
            else:
                parent_history_batch = None
                next_latents = model.predict_next(parent_batch, action_t, context_t)
            chunk_mask = mask_t.expand(parent_batch.shape[0], -1, -1)
            if _is_delta_topk_score(score_mode):
                score_values = delta_topk_raw_euclidean_distances(
                    next_latents,
                    parent_batch,
                    _expand_tokens_like(target_goal, next_latents),
                    chunk_mask,
                    top_k=_delta_topk_value(score_mode),
                )
            elif _is_changed_cell_score(score_mode):
                score_values = changed_cell_raw_euclidean_distances(
                    next_latents,
                    _expand_tokens_like(target_goal, next_latents),
                    actions[start:end],
                )
            elif _is_affected_context_score(score_mode):
                score_values = affected_context_raw_euclidean_distances(
                    next_latents,
                    _expand_tokens_like(target_goal, next_latents),
                    actions[start:end],
                )
            else:
                score_values = latent_distance(model, next_latents, target_goal, target_goal, chunk_mask, score_mode)
            score_values = _apply_policy_prior_bias(
                model,
                score_values,
                parent_batch,
                target_goal,
                context_t,
                chunk_mask,
                action_t,
                score_mode=score_mode,
            )
            scores = [float(x) for x in score_values.detach().cpu().tolist()]
            for offset, score in enumerate(scores):
                idx = start + offset
                latent = next_latents[offset : offset + 1].detach()
                if use_history:
                    if parent_history_batch is None:
                        raise RuntimeError("Single-state parent history was not built.")
                    history = torch.cat([parent_history_batch[offset : offset + 1], latent.unsqueeze(1)], dim=1).detach()
                else:
                    history = None
                candidates.append((score, leaves[idx], seqs[idx], latent, history))
            action_evals += end - start
        candidates.sort(key=lambda item: item[0])
        beam = candidates[: max(1, int(beam_width))]
        kept = [(score, seq, latent) for score, _, seq, latent, _ in beam]
    return kept, action_evals


@torch.no_grad()
def _beam_plan_once_symbolic(
    model: GridTokenGoalJEPA,
    board: np.ndarray,
    context_latents: torch.Tensor,
    predicted_goal: torch.Tensor,
    oracle_goal: torch.Tensor,
    clue_mask: np.ndarray,
    editable_mask: np.ndarray,
    active_mask: np.ndarray,
    *,
    score_mode: ScoreMode,
    beam_width: int,
    beam_depth: int,
    device: torch.device,
    allow_overwrite: bool = False,
) -> tuple[WorldAction | None, int]:
    beam: list[tuple[float, np.ndarray, list[WorldAction], torch.Tensor | None]] = [(0.0, board.copy(), [], None)]
    best: tuple[float, list[WorldAction]] | None = None
    action_evals = 0
    base_progress_mode = _progress_base_score_mode(score_mode)
    target_goal = _target_goal_latents(score_mode, predicted_goal, oracle_goal)
    for _ in range(max(1, int(beam_depth))):
        parent_boards: list[np.ndarray] = []
        leaves: list[np.ndarray] = []
        seqs: list[list[WorldAction]] = []
        actions: list[WorldAction] = []
        node_scores: list[float] = []
        for _, node_board, seq, _ in beam:
            node_score = 0.0
            if base_progress_mode is not None:
                node_score, _ = score_board(
                    model,
                    node_board,
                    context_latents,
                    predicted_goal,
                    oracle_goal,
                    clue_mask,
                    editable_mask,
                    active_mask,
                    score_mode=base_progress_mode,  # type: ignore[arg-type]
                    device=device,
                )
            for action in legal_sudoku_actions(
                node_board,
                clue_mask=clue_mask,
                allow_conflicts=True,
                allow_overwrite=allow_overwrite,
            ):
                try:
                    leaf = apply_sudoku_action(
                        node_board,
                        action,
                        clue_mask=clue_mask,
                        allow_conflicts=True,
                        allow_overwrite=allow_overwrite,
                    )
                except ValueError:
                    continue
                parent_boards.append(node_board)
                leaves.append(leaf)
                seqs.append([*seq, action])
                actions.append(action)
                node_scores.append(node_score)
        if not leaves:
            break
        leaf_latents = encode_boards(
            model,
            leaves,
            context_latents,
            clue_mask,
            editable_mask,
            active_mask,
            device=device,
        )
        mask_t = torch.as_tensor(active_mask[None], dtype=torch.bool, device=device).expand(len(leaves), -1, -1)
        if base_progress_mode is not None:
            absolute = latent_distance(model, leaf_latents, predicted_goal, oracle_goal, mask_t, base_progress_mode)
            score_values = absolute - torch.as_tensor(node_scores, dtype=absolute.dtype, device=device)
        elif _is_delta_topk_score(score_mode):
            parent_latents = encode_boards(
                model,
                parent_boards,
                context_latents,
                clue_mask,
                editable_mask,
                active_mask,
                device=device,
            )
            score_values = delta_topk_raw_euclidean_distances(
                leaf_latents,
                parent_latents,
                _expand_tokens_like(target_goal, leaf_latents),
                mask_t,
                top_k=_delta_topk_value(score_mode),
            )
        elif _is_changed_cell_score(score_mode):
            score_values = torch.stack(
                [changed_cell_raw_euclidean_distance(leaf_latents[i : i + 1], target_goal, action) for i, action in enumerate(actions)]
            ).reshape(-1)
        elif _is_affected_context_score(score_mode):
            score_values = affected_context_raw_euclidean_distances(leaf_latents, _expand_tokens_like(target_goal, leaf_latents), actions)
        else:
            score_values = latent_distance(model, leaf_latents, predicted_goal, oracle_goal, mask_t, score_mode)
        if _policy_prior_planning_weight(model) > 0.0:
            parent_latents = encode_boards(
                model,
                parent_boards,
                context_latents,
                clue_mask,
                editable_mask,
                active_mask,
                device=device,
            )
            action_t = torch.as_tensor(
                [[action.row, action.col, action.value] for action in actions],
                dtype=torch.long,
                device=device,
            )
            score_values = _apply_policy_prior_bias(
                model,
                score_values,
                parent_latents,
                target_goal,
                context_latents.expand(parent_latents.shape[0], -1, -1),
                mask_t,
                action_t,
                score_mode=score_mode,
            )
        scores = [float(x) for x in score_values.detach().cpu().tolist()]
        action_evals += len(leaves)
        candidates: list[tuple[float, np.ndarray, list[WorldAction], torch.Tensor | None]] = []
        for score, leaf, seq in zip(scores, leaves, seqs, strict=True):
            candidates.append((score, leaf, seq, None))
            if best is None or score < best[0]:
                best = (score, seq)
        candidates.sort(key=lambda item: item[0])
        beam = candidates[: max(1, int(beam_width))]
        if _can_early_stop(score_mode) and beam[0][0] <= 1.0e-8:
            break
    if best is None or not best[1]:
        return None, action_evals
    return best[1][0], action_evals


@torch.no_grad()
def score_board(
    model: GridTokenGoalJEPA,
    board: np.ndarray,
    context_latents: torch.Tensor,
    predicted_goal: torch.Tensor,
    oracle_goal: torch.Tensor,
    clue_mask: np.ndarray,
    editable_mask: np.ndarray,
    active_mask: np.ndarray,
    *,
    score_mode: ScoreMode,
    device: torch.device,
) -> tuple[float, torch.Tensor]:
    board_t = torch.as_tensor(board[None], dtype=torch.long, device=device)
    clue_t = torch.as_tensor(clue_mask[None], dtype=torch.bool, device=device)
    edit_t = torch.as_tensor(editable_mask[None], dtype=torch.bool, device=device)
    active_t = torch.as_tensor(active_mask[None], dtype=torch.bool, device=device)
    latent = model.encode_state(board_t, context_latents, clue_t, edit_t, active_t)
    return float(latent_distance(model, latent, predicted_goal, oracle_goal, active_t, score_mode).item()), latent


@torch.no_grad()
def encode_boards(
    model: GridTokenGoalJEPA,
    boards: list[np.ndarray],
    context_latents: torch.Tensor,
    clue_mask: np.ndarray,
    editable_mask: np.ndarray,
    active_mask: np.ndarray,
    *,
    device: torch.device,
    chunk_size: int = 512,
) -> torch.Tensor:
    board_t = torch.as_tensor(np.stack(boards), dtype=torch.long, device=device)
    count = board_t.shape[0]
    clue_t = torch.as_tensor(clue_mask[None], dtype=torch.bool, device=device).expand(count, -1, -1)
    edit_t = torch.as_tensor(editable_mask[None], dtype=torch.bool, device=device).expand(count, -1, -1)
    active_t = torch.as_tensor(active_mask[None], dtype=torch.bool, device=device).expand(count, -1, -1)
    context_t = context_latents.expand(count, -1, -1)
    latents = []
    for start in range(0, count, chunk_size):
        end = min(count, start + chunk_size)
        latents.append(model.encode_state(board_t[start:end], context_t[start:end], clue_t[start:end], edit_t[start:end], active_t[start:end]))
    return torch.cat(latents, dim=0)


def latent_distance(
    model: GridTokenGoalJEPA,
    latent: torch.Tensor,
    predicted_goal: torch.Tensor,
    oracle_goal: torch.Tensor,
    mask: torch.Tensor,
    score_mode: str,
) -> torch.Tensor:
    if mask.ndim == 3:
        mask = mask.reshape(mask.shape[0], -1)
    predicted_goal = _expand_tokens_like(predicted_goal, latent)
    oracle_goal = _expand_tokens_like(oracle_goal, latent)
    if score_mode == "predicted_goal_distance":
        return model.distance(latent, predicted_goal, mask)
    if score_mode == "oracle_goal_distance":
        return model.distance(latent, oracle_goal, mask)
    if score_mode == "predicted_goal_waypoint_goal_raw_euclidean_distance":
        waypoint_weight = float(getattr(model, "waypoint_planning_weight", 1.0) or 1.0)
        goal_weight = float(getattr(model, "goal_planning_weight", 0.1) or 0.1)
        return waypoint_weight * raw_tokenwise_euclidean_distance(
            latent,
            predicted_goal,
            mask,
        ) + goal_weight * raw_tokenwise_euclidean_distance(latent, oracle_goal, mask)
    target_goal = _target_goal_latents(score_mode, predicted_goal, oracle_goal)
    metric = _score_metric_name(score_mode)
    if metric in {"raw_euclidean_distance", "raw_euclidean_progress"}:
        return raw_tokenwise_euclidean_distance(latent, target_goal, mask)
    if metric == "raw_squared_euclidean_distance":
        return raw_tokenwise_squared_euclidean_distance(latent, target_goal, mask)
    if metric == "raw_mse_distance":
        return raw_full_board_mse_distance(latent, target_goal, mask)
    if metric == "raw_cosine_distance":
        return raw_tokenwise_cosine_distance(latent, target_goal, mask)
    if metric == "raw_hybrid_distance":
        return raw_tokenwise_euclidean_distance(latent, target_goal, mask) + raw_tokenwise_cosine_distance(latent, target_goal, mask)
    if metric == "projected_euclidean_distance":
        scores = model.metric_distance(latent, target_goal, mask)
        bad_weight = float(getattr(model, "bad_state_planning_weight", 0.0) or 0.0)
        if bad_weight > 0.0:
            scores = scores + bad_weight * torch.sigmoid(model.bad_state_logits(latent, mask)).to(dtype=scores.dtype)
        return scores
    if metric == "success_metric_distance":
        scores = model.success_distance(latent, mask)
        bad_weight = float(getattr(model, "bad_state_planning_weight", 0.0) or 0.0)
        if bad_weight > 0.0:
            scores = scores + bad_weight * torch.sigmoid(model.bad_state_logits(latent, mask)).to(dtype=scores.dtype)
        return scores
    if metric == "terminal_value":
        scores = model.terminal_value(latent, mask)
        bad_weight = float(getattr(model, "bad_state_planning_weight", 0.0) or 0.0)
        if bad_weight > 0.0:
            scores = scores + bad_weight * torch.sigmoid(model.bad_state_logits(latent, mask)).to(dtype=scores.dtype)
        return scores
    if metric == "compatibility_energy":
        return model.compatibility_energy(latent, mask)
    if metric == "remaining_edit_count":
        return model.remaining_edit_count(latent, mask)
    if metric == "verifier_energy":
        return model.verifier_score(latent, mask)
    if metric in {"changed_cell_raw_euclidean_distance", "affected_context_raw_euclidean_distance"}:
        return raw_tokenwise_euclidean_distance(latent, target_goal, mask)
    if _is_delta_topk_score(score_mode):
        return raw_tokenwise_euclidean_distance(latent, target_goal, mask)
    raise ValueError(f"Unknown score_mode: {score_mode}")


def _expand_tokens_like(tokens: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if tokens.shape == reference.shape:
        return tokens
    if tokens.shape[0] == 1 and tokens.shape[1:] == reference.shape[1:]:
        return tokens.expand(reference.shape[0], -1, -1)
    return tokens


def _sample_categorical_action_sequences(
    board: np.ndarray,
    probs: np.ndarray,
    *,
    clue_mask: np.ndarray | None = None,
    allow_overwrite: bool = False,
    samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[np.ndarray]]:
    horizon = probs.shape[0]
    seq_ids = np.zeros((samples, horizon), dtype=np.int64)
    final_boards: list[np.ndarray] = []
    for sample in range(samples):
        current = np.asarray(board, dtype=np.int64).copy()
        for step in range(horizon):
            valid = _valid_action_vocab_mask(current, clue_mask=clue_mask, allow_overwrite=allow_overwrite)
            if not bool(valid.any()):
                seq_ids[sample, step:] = seq_ids[sample, max(0, step - 1)]
                break
            step_probs = probs[step] * valid
            if step_probs.sum() <= 0:
                step_probs = valid.astype(np.float64)
            step_probs = step_probs / step_probs.sum()
            action_id = int(rng.choice(len(ACTION_VOCAB), p=step_probs))
            seq_ids[sample, step] = action_id
            try:
                current = apply_sudoku_action(
                    current,
                    ACTION_VOCAB[action_id],
                    clue_mask=clue_mask,
                    allow_conflicts=True,
                    allow_overwrite=allow_overwrite,
                )
            except ValueError:
                pass
        final_boards.append(current)
    return seq_ids, final_boards


def _valid_action_vocab_mask(
    board: np.ndarray,
    *,
    clue_mask: np.ndarray | None = None,
    allow_overwrite: bool = False,
) -> np.ndarray:
    mask = np.zeros((len(ACTION_VOCAB),), dtype=bool)
    for index, action in enumerate(ACTION_VOCAB):
        if clue_mask is not None and bool(clue_mask[action.row, action.col]):
            mask[index] = False
        elif allow_overwrite:
            mask[index] = int(board[action.row, action.col]) != int(action.value)
        else:
            mask[index] = bool(board[action.row, action.col] == 0)
    return mask


@torch.no_grad()
def _score_cem_sequences(
    model: GridTokenGoalJEPA,
    start_board: np.ndarray,
    final_boards: list[np.ndarray],
    seq_ids: np.ndarray,
    context_latents: torch.Tensor,
    predicted_goal: torch.Tensor,
    oracle_goal: torch.Tensor,
    clue_mask: np.ndarray,
    editable_mask: np.ndarray,
    active_mask: np.ndarray,
    *,
    score_mode: ScoreMode,
    transition_mode: TransitionMode,
    device: torch.device,
) -> np.ndarray:
    if transition_mode == "symbolic_reencode":
        latents = encode_boards(
            model,
            final_boards,
            context_latents,
            clue_mask,
            editable_mask,
            active_mask,
            device=device,
        )
    else:
        _, start_latent = score_board(
            model,
            start_board,
            context_latents,
            predicted_goal,
            oracle_goal,
            clue_mask,
            editable_mask,
            active_mask,
            score_mode=score_mode,
            device=device,
        )
        latents = start_latent.expand(seq_ids.shape[0], -1, -1)
        context = context_latents.expand(seq_ids.shape[0], -1, -1)
        if _uses_single_state_history(model, latents):
            state_history = latents.unsqueeze(1)
            action_history = []
        for step in range(seq_ids.shape[1]):
            actions = torch.as_tensor(
                [[ACTION_VOCAB[int(action_id)].row, ACTION_VOCAB[int(action_id)].col, ACTION_VOCAB[int(action_id)].value] for action_id in seq_ids[:, step]],
                dtype=torch.long,
                device=device,
            )
            if _uses_single_state_history(model, latents):
                action_history.append(actions)
                action_history_t = torch.stack(action_history, dim=1)
                latents = model.predict_next_sequence(state_history, action_history_t, context)[:, -1]
                state_history = torch.cat([state_history, latents.unsqueeze(1)], dim=1)
            else:
                latents = model.predict_next(latents, actions, context)
    mask = torch.as_tensor(active_mask[None], dtype=torch.bool, device=device).expand(latents.shape[0], -1, -1)
    target_goal = _target_goal_latents(score_mode, predicted_goal, oracle_goal)
    if _is_changed_cell_score(score_mode):
        final_actions = [ACTION_VOCAB[int(action_id)] for action_id in seq_ids[:, -1]]
        scores = changed_cell_raw_euclidean_distances(latents, _expand_tokens_like(target_goal, latents), final_actions)
    elif _is_affected_context_score(score_mode):
        final_actions = [ACTION_VOCAB[int(action_id)] for action_id in seq_ids[:, -1]]
        scores = affected_context_raw_euclidean_distances(latents, _expand_tokens_like(target_goal, latents), final_actions)
    else:
        scores = latent_distance(model, latents, predicted_goal, oracle_goal, mask, score_mode)
    return scores.detach().cpu().numpy()


def _target_goal_latents(score_mode: str, predicted_goal: torch.Tensor, oracle_goal: torch.Tensor) -> torch.Tensor:
    if score_mode.startswith("predicted_goal_"):
        return predicted_goal
    return oracle_goal


def _score_metric_name(score_mode: str) -> str:
    for prefix in ("oracle_goal_", "predicted_goal_"):
        if score_mode.startswith(prefix):
            return score_mode.removeprefix(prefix)
    return score_mode


def _is_progress_score(score_mode: str) -> bool:
    return _score_metric_name(score_mode) == "raw_euclidean_progress"


def _is_changed_cell_score(score_mode: str) -> bool:
    return _score_metric_name(score_mode) == "changed_cell_raw_euclidean_distance"


def _is_affected_context_score(score_mode: str) -> bool:
    return _score_metric_name(score_mode) == "affected_context_raw_euclidean_distance"


def _is_delta_topk_score(score_mode: str) -> bool:
    metric = _score_metric_name(score_mode)
    return metric.startswith("delta_top") and metric.endswith("_raw_euclidean_distance")


def _is_verifier_score(score_mode: str) -> bool:
    return _score_metric_name(score_mode) in {"compatibility_energy", "remaining_edit_count", "verifier_energy"}


def _delta_topk_value(score_mode: str) -> int:
    metric = _score_metric_name(score_mode)
    prefix = "delta_top"
    suffix = "_raw_euclidean_distance"
    if not metric.startswith(prefix) or not metric.endswith(suffix):
        raise ValueError(f"Score mode {score_mode!r} is not a delta top-k score.")
    return int(metric[len(prefix) : -len(suffix)])


def _subgoal_score_mode(score_mode: str) -> str:
    metric = _score_metric_name(score_mode)
    if metric == "raw_euclidean_progress":
        metric = "raw_euclidean_distance"
    return f"oracle_goal_{metric}"


def _progress_base_score_mode(score_mode: str) -> str | None:
    if not _is_progress_score(score_mode):
        return None
    if score_mode.startswith("predicted_goal_"):
        return "predicted_goal_raw_euclidean_distance"
    return "oracle_goal_raw_euclidean_distance"


def _can_early_stop(score_mode: str) -> bool:
    return (not _is_progress_score(score_mode)) and (not _is_verifier_score(score_mode))


def _policy_prior_planning_weight(model: GridTokenGoalJEPA) -> float:
    return float(getattr(model, "policy_prior_planning_weight", 0.0))


def _apply_policy_prior_bias(
    model: GridTokenGoalJEPA,
    scores: torch.Tensor,
    parent_latents: torch.Tensor,
    target_goal: torch.Tensor,
    context_latents: torch.Tensor,
    active_mask: torch.Tensor,
    actions: torch.Tensor,
    *,
    score_mode: str,
) -> torch.Tensor:
    weight = _policy_prior_planning_weight(model)
    if weight <= 0.0:
        return scores
    if _score_metric_name(score_mode) in {"success_metric_distance", "terminal_value"} or _is_verifier_score(score_mode):
        target_goal = model.success_policy_goal_like(parent_latents)
    priors = model.score_action_prior(parent_latents, target_goal, context_latents, active_mask, actions)
    return scores - weight * priors.to(dtype=scores.dtype)


def _uses_single_state_history(model: GridTokenGoalJEPA, latents: torch.Tensor) -> bool:
    return bool(getattr(model, "latent_representation", "grid") == "single" and latents.shape[-2] == 1)


def _actions_to_tensor(sequences: list[list[WorldAction]], *, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(
        [[[action.row, action.col, action.value] for action in sequence] for sequence in sequences],
        dtype=torch.long,
        device=device,
    )


def _prepare_token_mask(mask: torch.Tensor, *, token_count: int) -> torch.Tensor:
    if mask.ndim == 3:
        mask = mask.reshape(mask.shape[0], -1)
    if mask.shape[-1] == token_count:
        return mask
    if int(token_count) == 1:
        return torch.ones((*mask.shape[:-1], 1), dtype=torch.bool, device=mask.device)
    if int(token_count) > int(mask.shape[-1]):
        extra = torch.ones(
            (*mask.shape[:-1], int(token_count) - int(mask.shape[-1])),
            dtype=torch.bool,
            device=mask.device,
        )
        return torch.cat([mask, extra], dim=-1)
    raise ValueError(f"Active mask with {mask.shape[-1]} tokens cannot mask latent sequence with {token_count} tokens.")


def raw_tokenwise_euclidean_distance(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if a.shape != b.shape:
        raise ValueError(f"Raw distance inputs must have matching shapes, got {tuple(a.shape)} and {tuple(b.shape)}.")
    mask = _prepare_token_mask(mask, token_count=a.shape[-2])
    if mask.shape != a.shape[:-1]:
        raise ValueError(f"Raw distance mask must have shape {tuple(a.shape[:-1])}, got {tuple(mask.shape)}.")
    per_token = (a.float() - b.float()).square().sum(dim=-1).sqrt()
    weights = mask.float()
    return (per_token * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)


def raw_tokenwise_squared_euclidean_distance(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if a.shape != b.shape:
        raise ValueError(f"Raw squared distance inputs must have matching shapes, got {tuple(a.shape)} and {tuple(b.shape)}.")
    mask = _prepare_token_mask(mask, token_count=a.shape[-2])
    if mask.shape != a.shape[:-1]:
        raise ValueError(f"Raw squared distance mask must have shape {tuple(a.shape[:-1])}, got {tuple(mask.shape)}.")
    per_token = (a.float() - b.float()).square().sum(dim=-1)
    weights = mask.float()
    return (per_token * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)


def raw_full_board_mse_distance(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if a.shape != b.shape:
        raise ValueError(f"Raw MSE distance inputs must have matching shapes, got {tuple(a.shape)} and {tuple(b.shape)}.")
    mask = _prepare_token_mask(mask, token_count=a.shape[-2])
    if mask.shape != a.shape[:-1]:
        raise ValueError(f"Raw MSE distance mask must have shape {tuple(a.shape[:-1])}, got {tuple(mask.shape)}.")
    per_token = (a.float() - b.float()).square().mean(dim=-1)
    weights = mask.float()
    return (per_token * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)


def raw_tokenwise_cosine_distance(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if a.shape != b.shape:
        raise ValueError(f"Raw cosine distance inputs must have matching shapes, got {tuple(a.shape)} and {tuple(b.shape)}.")
    mask = _prepare_token_mask(mask, token_count=a.shape[-2])
    if mask.shape != a.shape[:-1]:
        raise ValueError(f"Raw cosine distance mask must have shape {tuple(a.shape[:-1])}, got {tuple(mask.shape)}.")
    per_token = 1.0 - torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1, eps=1.0e-6)
    weights = mask.float()
    return (per_token * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)


def projected_tokenwise_euclidean_distance(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor, projector: torch.nn.Module) -> torch.Tensor:
    if a.shape != b.shape:
        raise ValueError(f"Projected distance inputs must have matching shapes, got {tuple(a.shape)} and {tuple(b.shape)}.")
    mask = _prepare_token_mask(mask, token_count=a.shape[-2])
    if mask.shape != a.shape[:-1]:
        raise ValueError(f"Projected distance mask must have shape {tuple(a.shape[:-1])}, got {tuple(mask.shape)}.")
    per_token = (projector(a.float()) - projector(b.float())).square().sum(dim=-1).sqrt()
    weights = mask.float()
    return (per_token * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)


def changed_cell_raw_euclidean_distance(a: torch.Tensor, b: torch.Tensor, action: WorldAction) -> torch.Tensor:
    if a.shape != b.shape:
        raise ValueError(f"Changed-cell distance inputs must have matching shapes, got {tuple(a.shape)} and {tuple(b.shape)}.")
    if a.shape[-2] == 1:
        return (a[..., 0, :].float() - b[..., 0, :].float()).square().sum(dim=-1).sqrt()
    idx = int(action.row) * 9 + int(action.col)
    if idx < 0 or idx >= a.shape[-2]:
        raise ValueError(f"Action cell index {idx} is outside token count {a.shape[-2]}.")
    return (a[..., idx, :].float() - b[..., idx, :].float()).square().sum(dim=-1).sqrt()


def changed_cell_raw_euclidean_distances(a: torch.Tensor, b: torch.Tensor, actions: list[WorldAction]) -> torch.Tensor:
    if a.shape != b.shape:
        raise ValueError(f"Changed-cell distance inputs must have matching shapes, got {tuple(a.shape)} and {tuple(b.shape)}.")
    if len(actions) != a.shape[0]:
        raise ValueError(f"Expected one action per batch item, got {len(actions)} actions for batch size {a.shape[0]}.")
    if a.shape[-2] == 1:
        return (a[:, 0, :].float() - b[:, 0, :].float()).square().sum(dim=-1).sqrt()
    indices = torch.as_tensor([int(action.row) * 9 + int(action.col) for action in actions], dtype=torch.long, device=a.device)
    if bool(((indices < 0) | (indices >= a.shape[-2])).any().item()):
        raise ValueError(f"Action cell indices must be inside token count {a.shape[-2]}.")
    batch = torch.arange(a.shape[0], device=a.device)
    return (a[batch, indices, :].float() - b[batch, indices, :].float()).square().sum(dim=-1).sqrt()


def affected_context_raw_euclidean_distances(
    a: torch.Tensor,
    b: torch.Tensor,
    actions: list[WorldAction],
    *,
    rows: int = 9,
    cols: int = 9,
    affected_weight: float = 8.0,
    context_weight: float = 2.0,
) -> torch.Tensor:
    if a.shape != b.shape:
        raise ValueError(f"Affected-context distance inputs must have matching shapes, got {tuple(a.shape)} and {tuple(b.shape)}.")
    if len(actions) != a.shape[0]:
        raise ValueError(f"Expected one action per batch item, got {len(actions)} actions for batch size {a.shape[0]}.")
    if a.shape[-2] == 1:
        return (a[:, 0, :].float() - b[:, 0, :].float()).square().sum(dim=-1).sqrt()
    if a.shape[-2] != rows * cols:
        raise ValueError(f"Expected {rows * cols} board tokens for a {rows}x{cols} grid, got {a.shape[-2]}.")
    action_t = torch.as_tensor(
        [[int(action.row), int(action.col), int(action.value)] for action in actions],
        dtype=torch.long,
        device=a.device,
    )
    weights = _affected_token_weights(
        action_t,
        token_count=rows * cols,
        rows=rows,
        cols=cols,
        affected_weight=affected_weight,
        context_weight=context_weight,
        horizon=1,
    ).to(dtype=a.dtype, device=a.device)
    per_token = (a.float() - b.float()).square().sum(dim=-1).sqrt()
    weights = weights.to(dtype=per_token.dtype)
    return (per_token * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)


def delta_topk_raw_euclidean_distances(
    next_latents: torch.Tensor,
    previous_latents: torch.Tensor,
    goal_latents: torch.Tensor,
    mask: torch.Tensor,
    *,
    top_k: int,
) -> torch.Tensor:
    if next_latents.shape != previous_latents.shape:
        raise ValueError(
            f"Delta top-k inputs must have matching next/previous shapes, got "
            f"{tuple(next_latents.shape)} and {tuple(previous_latents.shape)}."
        )
    goal_latents = _expand_tokens_like(goal_latents, next_latents)
    if goal_latents.shape != next_latents.shape:
        raise ValueError(f"Goal latents must expand to {tuple(next_latents.shape)}, got {tuple(goal_latents.shape)}.")
    mask = _prepare_token_mask(mask, token_count=next_latents.shape[-2])
    if mask.shape != next_latents.shape[:-1]:
        raise ValueError(f"Delta top-k mask must have shape {tuple(next_latents.shape[:-1])}, got {tuple(mask.shape)}.")
    delta = (next_latents.float() - previous_latents.float()).square().sum(dim=-1)
    delta = delta.masked_fill(~mask, float("-inf"))
    k = min(max(1, int(top_k)), next_latents.shape[-2])
    selected_delta, indices = delta.topk(k=k, dim=-1)
    per_token = (next_latents.float() - goal_latents.float()).square().sum(dim=-1).sqrt()
    selected = per_token.gather(dim=-1, index=indices)
    selected_mask = torch.isfinite(selected_delta)
    weights = selected_mask.to(dtype=selected.dtype)
    return (selected * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)


@torch.no_grad()
def _prepare_goal_latents(
    model: GridTokenGoalJEPA,
    puzzle: np.ndarray,
    goal: np.ndarray,
    clue_mask: np.ndarray,
    editable_mask: np.ndarray,
    active_mask: np.ndarray,
    *,
    device: torch.device,
    score_mode: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    context_t = torch.as_tensor(puzzle[None], dtype=torch.long, device=device)
    clue_t = torch.as_tensor(clue_mask[None], dtype=torch.bool, device=device)
    edit_t = torch.as_tensor(editable_mask[None], dtype=torch.bool, device=device)
    active_t = torch.as_tensor(active_mask[None], dtype=torch.bool, device=device)
    context_latents = model.encode_context(context_t, clue_t, edit_t, active_t)
    initial_latents = model.encode_state(context_t, context_latents, clue_t, edit_t, active_t)
    if score_mode is not None and _is_verifier_score(str(score_mode)):
        # Verifier scores are intentionally goal-free at inference. Return a
        # same-shape placeholder for legacy planner APIs without encoding the
        # solved board into an oracle latent.
        return context_latents, initial_latents, initial_latents, initial_latents
    goal_t = torch.as_tensor(goal[None], dtype=torch.long, device=device)
    predicted_goal = model.predict_goal(
        context_latents,
        active_t,
        initial_latents=initial_latents if model.goal_conditioning == "initial_current" else None,
        current_latents=initial_latents if model.goal_conditioning in {"initial_current", "context_current"} else None,
    )
    oracle_goal = model.encode_state(goal_t, context_latents, clue_t, edit_t, active_t)
    return context_latents, predicted_goal, oracle_goal, initial_latents


@torch.no_grad()
def _predict_goal_for_board(
    model: GridTokenGoalJEPA,
    board: np.ndarray,
    context_latents: torch.Tensor,
    initial_latents: torch.Tensor,
    clue_mask: np.ndarray,
    editable_mask: np.ndarray,
    active_mask: np.ndarray,
    *,
    device: torch.device,
    allow_overwrite: bool = False,
) -> torch.Tensor:
    del allow_overwrite
    active_t = torch.as_tensor(active_mask[None], dtype=torch.bool, device=device)
    if model.goal_conditioning not in {"initial_current", "context_current"}:
        return model.predict_goal(context_latents, active_t)
    board_t = torch.as_tensor(board[None], dtype=torch.long, device=device)
    clue_t = torch.as_tensor(clue_mask[None], dtype=torch.bool, device=device)
    edit_t = torch.as_tensor(editable_mask[None], dtype=torch.bool, device=device)
    current_latents = model.encode_state(board_t, context_latents, clue_t, edit_t, active_t)
    return model.predict_goal(
        context_latents,
        active_t,
        initial_latents=initial_latents if model.goal_conditioning == "initial_current" else None,
        current_latents=current_latents,
    )


@torch.no_grad()
def _predict_waypoint_for_board(
    model: GridTokenGoalJEPA,
    board: np.ndarray,
    context_latents: torch.Tensor,
    initial_latents: torch.Tensor | None,
    clue_mask: np.ndarray,
    editable_mask: np.ndarray,
    active_mask: np.ndarray,
    *,
    horizon: int | None = None,
    device: torch.device,
) -> torch.Tensor:
    board_t = torch.as_tensor(board[None], dtype=torch.long, device=device)
    clue_t = torch.as_tensor(clue_mask[None], dtype=torch.bool, device=device)
    edit_t = torch.as_tensor(editable_mask[None], dtype=torch.bool, device=device)
    active_t = torch.as_tensor(active_mask[None], dtype=torch.bool, device=device)
    current_latents = model.encode_state(board_t, context_latents, clue_t, edit_t, active_t)
    goal_latents = None
    if getattr(model, "waypoint_conditioning", "current") == "current_goal":
        if initial_latents is None:
            raise ValueError("waypoint_conditioning='current_goal' requires initial_latents during planning.")
        goal_latents = model.predict_goal(
            context_latents,
            active_t,
            initial_latents=initial_latents if model.goal_conditioning == "initial_current" else None,
            current_latents=current_latents if model.goal_conditioning in {"initial_current", "context_current"} else None,
        )
    return model.predict_waypoint(context_latents, active_t, current_latents, horizon=horizon, goal_latents=goal_latents)


@torch.no_grad()
def _oracle_future_waypoint_latents(
    model: GridTokenGoalJEPA,
    board: np.ndarray,
    goal: np.ndarray,
    context_latents: torch.Tensor,
    clue_mask: np.ndarray,
    editable_mask: np.ndarray,
    active_mask: np.ndarray,
    *,
    horizon: int,
    device: torch.device,
) -> torch.Tensor:
    waypoint = np.asarray(board, dtype=np.int64).copy()
    target = np.asarray(goal, dtype=np.int64)
    editable = ~np.asarray(clue_mask, dtype=bool)
    differing = np.argwhere(editable & (waypoint != target))
    for row, col in differing[: max(0, int(horizon))]:
        waypoint[int(row), int(col)] = int(target[int(row), int(col)])
    return encode_boards(
        model,
        [waypoint],
        context_latents,
        clue_mask,
        editable_mask,
        active_mask,
        device=device,
    )
