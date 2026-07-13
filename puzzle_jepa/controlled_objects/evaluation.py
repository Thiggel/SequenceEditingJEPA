from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from puzzle_jepa.controlled_objects.batching import ControlledTrajectoryDataset
from puzzle_jepa.controlled_objects.domain import RigidAction
from puzzle_jepa.controlled_objects.generator import ControlledObjectGenerator
from puzzle_jepa.controlled_objects.model import ControlledObjectJEPA


@torch.no_grad()
def evaluate_controlled_model(
    model: ControlledObjectJEPA,
    dataset: ControlledTrajectoryDataset,
    generator: ControlledObjectGenerator,
    *,
    seed: int,
    batch_size: int,
    device: torch.device,
    planning_episodes: int = 0,
    planning_candidates: int = 16,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    rng = np.random.default_rng(seed)
    batch = dataset.sample_batch(
        rng,
        batch_size=batch_size,
        horizon=model.required_horizon,
        device=device,
    )
    output = model(batch)
    current = model.encode(batch.states[:, 0])
    metrics: dict[str, Any] = {
        "eval_loss": float(output.loss.cpu()),
        "eval_prediction_loss": float(output.prediction_loss.cpu()),
        "eval_vicreg_loss": float(output.vicreg_loss.cpu()),
        "eval_ldad_loss": float(output.ldad_loss.cpu()),
        "eval_action_valid_fraction": float(batch.action_validity.float().mean().cpu()),
        "eval_latent_std": float(_pooled(current).std(dim=0, unbiased=False).mean().cpu()),
        "eval_latent_effective_rank": _effective_rank(_pooled(current)),
    }
    for level, (predicted, targets) in enumerate(
        zip(output.predictions, output.targets, strict=True)
    ):
        reduce_dims = tuple(range(2, predicted.ndim))
        per_step = (predicted - targets).square().mean(dim=reduce_dims).mean(dim=0)
        identity = current.unsqueeze(1).expand_as(targets)
        identity_per_step = (identity - targets).square().mean(dim=reduce_dims).mean(dim=0)
        for step, (mse, identity_mse) in enumerate(
            zip(per_step, identity_per_step, strict=True), start=1
        ):
            metrics[f"eval_level{level}_rollout{step}_mse"] = float(mse.cpu())
            metrics[f"eval_level{level}_rollout{step}_identity_mse"] = float(
                identity_mse.cpu()
            )
            metrics[f"eval_level{level}_rollout{step}_gain"] = float(
                (identity_mse - mse).cpu()
            )

    if model.hierarchy_depth > 1:
        metrics.update(_hierarchy_diagnostics(model, batch))
    if planning_episodes > 0:
        metrics.update(
            _planning_diagnostics(
                model,
                generator,
                rng,
                episodes=planning_episodes,
                candidates=planning_candidates,
                device=device,
            )
        )
    model.train(was_training)
    return metrics


@torch.no_grad()
def _hierarchy_diagnostics(
    model: ControlledObjectJEPA, batch
) -> dict[str, float]:
    current = model.encode(batch.states[:, 0])
    diagnostics: dict[str, float] = {}
    primitive_state = current
    primitive_predictions = []
    for index in range(max(model.level_spans)):
        primitive_state = model.predict_chunk(
            0, primitive_state, batch.actions[:, index : index + 1]
        )
        primitive_predictions.append(primitive_state)
    for level, span in enumerate(model.level_spans[1:], start=1):
        high_prediction = model.predict_chunk(
            level, current, batch.actions[:, :span]
        )
        target = model.encode(batch.states[:, span], target=True)
        low_prediction = primitive_predictions[span - 1]
        diagnostics[f"eval_level{level}_one_step_mse"] = float(
            F.mse_loss(high_prediction, target).cpu()
        )
        diagnostics[f"eval_level{level}_primitive_rollout_mse"] = float(
            F.mse_loss(low_prediction, target).cpu()
        )
        macros = model.encode_action_chunk(level, batch.actions[:, :span])
        diagnostics[f"eval_level{level}_macro_std"] = float(
            macros.std(dim=0, unbiased=False).mean().cpu()
        )
        diagnostics[f"eval_level{level}_macro_effective_rank"] = _effective_rank(macros)
        if len(macros) > 2:
            on_distance = torch.cdist(macros, macros)
            on_distance.fill_diagonal_(torch.inf)
            on_nearest = on_distance.min(dim=1).values
            feature_std = macros.std(dim=0, unbiased=False).clamp_min(1.0e-3)
            off = macros.mean(dim=0) + 3.0 * feature_std * torch.randn_like(macros)
            off_distance = torch.cdist(off, macros)
            off_nearest, nearest_index = off_distance.min(dim=1)
            off_prediction = model.predict_from_macro(level, current, off)
            on_reachability = _latent_distance(high_prediction, low_prediction)
            off_reachability = _latent_distance(
                off_prediction, low_prediction[nearest_index]
            )
            diagnostics[f"eval_level{level}_support_margin"] = float(
                (off_nearest.mean() - on_nearest.mean()).cpu()
            )
            diagnostics[f"eval_level{level}_support_energy_auroc"] = _binary_auroc(
                on_nearest, off_nearest
            )
            diagnostics[f"eval_level{level}_reachability_energy_margin"] = float(
                (off_reachability.mean() - on_reachability.mean()).cpu()
            )
            diagnostics[f"eval_level{level}_reachability_energy_auroc"] = _binary_auroc(
                on_reachability, off_reachability
            )
    return diagnostics


@torch.no_grad()
def _planning_diagnostics(
    model: ControlledObjectJEPA,
    generator: ControlledObjectGenerator,
    rng: np.random.Generator,
    *,
    episodes: int,
    candidates: int,
    device: torch.device,
) -> dict[str, float]:
    planning_horizon = model.required_horizon
    exact_successes = 0
    exact_receding_successes = 0
    learned_successes = 0
    oracle_macro_successes = 0
    for _ in range(episodes):
        trajectory = generator.sample_trajectory(rng, horizon=planning_horizon)
        initial = trajectory.states[0]
        goal = trajectory.states[-1]
        replayed = generator.replay(
            initial,
            (RigidAction(*(int(value) for value in action)) for action in trajectory.actions),
        )
        exact_successes += int(np.array_equal(replayed, goal))
        exact_receding = _exact_receding_plan(
            generator,
            initial,
            goal,
            trajectory.actions,
            rng,
            candidates=candidates,
        )
        exact_receding_successes += int(np.array_equal(exact_receding, goal))
        learned = _receding_plan(
            model,
            generator,
            initial,
            goal,
            rng,
            max_steps=2 * planning_horizon,
            candidates=candidates,
            device=device,
            oracle_actions=None,
        )
        learned_successes += int(np.array_equal(learned, goal))
        oracle_macro = _receding_plan(
            model,
            generator,
            initial,
            goal,
            rng,
            max_steps=2 * planning_horizon,
            candidates=candidates,
            device=device,
            oracle_actions=trajectory.actions,
        )
        oracle_macro_successes += int(np.array_equal(oracle_macro, goal))
    return {
        "eval_exact_replay_success_rate": exact_successes / episodes,
        "eval_exact_receding_success_rate": exact_receding_successes / episodes,
        "eval_learned_receding_success_rate": learned_successes / episodes,
        "eval_oracle_macro_learned_low_success_rate": oracle_macro_successes / episodes,
        "eval_planning_horizon": float(planning_horizon),
    }


def _exact_receding_plan(
    generator: ControlledObjectGenerator,
    initial: np.ndarray,
    goal: np.ndarray,
    oracle_actions: np.ndarray,
    rng: np.random.Generator,
    *,
    candidates: int,
) -> np.ndarray:
    state = initial.copy()
    horizon = len(oracle_actions)
    for executed in range(horizon):
        remaining = horizon - executed
        sequences = _candidate_action_sequences(
            generator,
            state,
            rng,
            horizon=remaining,
            count=candidates,
        )
        sequences[0] = oracle_actions[executed:]
        costs = []
        for sequence in sequences:
            outcome = generator.replay(
                state,
                (RigidAction(*(int(value) for value in action)) for action in sequence),
            )
            costs.append(np.count_nonzero(outcome != goal))
        chosen = int(np.argmin(costs))
        action = RigidAction(*(int(value) for value in sequences[chosen, 0]))
        state, _ = generator.apply_action(state, action)
        if np.array_equal(state, goal):
            break
    return state


def _receding_plan(
    model: ControlledObjectJEPA,
    generator: ControlledObjectGenerator,
    initial: np.ndarray,
    goal: np.ndarray,
    rng: np.random.Generator,
    *,
    max_steps: int,
    candidates: int,
    device: torch.device,
    oracle_actions: np.ndarray | None,
) -> np.ndarray:
    state = initial.copy()
    goal_tensor = torch.as_tensor(goal[None], dtype=torch.long, device=device)
    goal_latent = model.encode(goal_tensor, target=True)
    top_level = model.hierarchy_depth - 1
    span = model.level_spans[top_level]
    for executed in range(max_steps):
        if np.array_equal(state, goal):
            break
        state_tensor = torch.as_tensor(state[None], dtype=torch.long, device=device)
        state_latent = model.encode(state_tensor)
        if top_level == 0:
            action = _best_flat_plan_action(
                model,
                generator,
                state,
                state_latent,
                goal_latent,
                rng,
                horizon=model.required_horizon,
                candidates=candidates,
                device=device,
            )
        else:
            macro_steps = model.level_rollout_steps(top_level)
            if oracle_actions is not None:
                needed = span * macro_steps
                suffix = oracle_actions[executed : executed + needed]
                if len(suffix) < needed:
                    suffix = np.concatenate(
                        [suffix, np.zeros((needed - len(suffix), 3), dtype=np.int64)]
                    )
                chunk_sequences = suffix.reshape(1, macro_steps, span, 3)
            else:
                chunk_sequences = _candidate_chunk_sequences(
                    generator,
                    state,
                    rng,
                    span=span,
                    macro_steps=macro_steps,
                    count=candidates,
                )
            chunk_tensor = torch.as_tensor(
                chunk_sequences, dtype=torch.long, device=device
            )
            rollout_state = state_latent.expand(
                len(chunk_sequences), *state_latent.shape[1:]
            )
            first_subgoals = None
            for macro_index in range(macro_steps):
                macro = model.encode_action_chunk(
                    top_level, chunk_tensor[:, macro_index]
                )
                rollout_state = model.predict_from_macro(
                    top_level, rollout_state, macro
                )
                if first_subgoals is None:
                    first_subgoals = rollout_state
            scores = _latent_distance(rollout_state, goal_latent.expand_as(rollout_state))
            chosen = int(scores.argmin())
            assert first_subgoals is not None
            chosen_subgoal = first_subgoals[chosen : chosen + 1]
            action = _best_primitive_action(
                model,
                generator,
                state,
                state_latent,
                chosen_subgoal,
                device=device,
            )
        state, _ = generator.apply_action(state, action)
    return state


def _best_flat_plan_action(
    model: ControlledObjectJEPA,
    generator: ControlledObjectGenerator,
    state: np.ndarray,
    state_latent: torch.Tensor,
    target_latent: torch.Tensor,
    rng: np.random.Generator,
    *,
    horizon: int,
    candidates: int,
    device: torch.device,
) -> RigidAction:
    sequences = _candidate_action_sequences(
        generator,
        state,
        rng,
        horizon=horizon,
        count=candidates,
    )
    action_tensor = torch.as_tensor(sequences, dtype=torch.long, device=device)
    rollout_state = state_latent.expand(len(sequences), *state_latent.shape[1:])
    for step in range(horizon):
        rollout_state = model.predict_chunk(
            0, rollout_state, action_tensor[:, step : step + 1]
        )
    scores = _latent_distance(rollout_state, target_latent.expand_as(rollout_state))
    chosen = int(scores.argmin())
    return RigidAction(*(int(value) for value in sequences[chosen, 0]))


def _best_primitive_action(
    model: ControlledObjectJEPA,
    generator: ControlledObjectGenerator,
    state: np.ndarray,
    state_latent: torch.Tensor,
    target_latent: torch.Tensor,
    *,
    device: torch.device,
) -> RigidAction:
    actions = generator.candidate_actions(state)
    action_tensor = torch.as_tensor(
        np.stack([action.as_array() for action in actions])[:, None],
        dtype=torch.long,
        device=device,
    )
    repeated_state = state_latent.expand(len(actions), *state_latent.shape[1:])
    predictions = model.predict_chunk(0, repeated_state, action_tensor)
    scores = _latent_distance(predictions, target_latent.expand_as(predictions))
    return actions[int(scores.argmin())]


def _candidate_chunk_sequences(
    generator: ControlledObjectGenerator,
    state: np.ndarray,
    rng: np.random.Generator,
    *,
    span: int,
    macro_steps: int,
    count: int,
) -> np.ndarray:
    actions = _candidate_action_sequences(
        generator,
        state,
        rng,
        horizon=span * macro_steps,
        count=count,
    )
    return actions.reshape(count, macro_steps, span, 3)


def _candidate_action_sequences(
    generator: ControlledObjectGenerator,
    state: np.ndarray,
    rng: np.random.Generator,
    *,
    horizon: int,
    count: int,
) -> np.ndarray:
    sequences = []
    first_actions = generator.candidate_actions(state)
    for candidate_index in range(count):
        rollout_state = state.copy()
        actions = []
        for step in range(horizon):
            if step == 0 and candidate_index < len(first_actions):
                action = first_actions[candidate_index]
            else:
                action = generator.sample_action(rollout_state, rng)
            actions.append(action.as_array())
            rollout_state, _ = generator.apply_action(rollout_state, action)
        sequences.append(np.stack(actions))
    return np.stack(sequences)


def _latent_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left - right).square().flatten(1).mean(dim=1)


def _pooled(latent: torch.Tensor) -> torch.Tensor:
    return latent if latent.ndim == 2 else latent.mean(dim=1)


def _effective_rank(latent: torch.Tensor) -> float:
    centered = latent - latent.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered.float())
    probabilities = singular.square()
    probabilities = probabilities / probabilities.sum().clamp_min(1.0e-12)
    entropy = -(probabilities * probabilities.clamp_min(1.0e-12).log()).sum()
    return float(entropy.exp().cpu())


def _binary_auroc(negative_scores: torch.Tensor, positive_scores: torch.Tensor) -> float:
    comparisons = positive_scores[:, None] - negative_scores[None, :]
    return float(
        ((comparisons > 0).float() + 0.5 * (comparisons == 0).float()).mean().cpu()
    )
