from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from puzzle_jepa.controlled_objects.batching import ControlledTrajectoryDataset
from puzzle_jepa.controlled_objects.domain import RigidTransform
from puzzle_jepa.controlled_objects.generator import ControlledObjectGenerator
from puzzle_jepa.controlled_objects.model import ControlledObjectJEPA


@dataclass(frozen=True, slots=True)
class MacroSupport:
    state_bank: torch.Tensor
    bank: torch.Tensor
    action_bank: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    state_mean: torch.Tensor
    state_std: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor


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
    changed_cells = (batch.states[:, 1:] != batch.states[:, :-1]).flatten(2).sum(dim=2)
    object_areas = torch.stack(
        [(batch.states == color).flatten(2).sum(dim=2) for color in range(1, model.num_colors)],
        dim=-1,
    )
    area_preserved = object_areas[:, 1:] == object_areas[:, :-1]
    metrics: dict[str, Any] = {
        "eval_loss": float(output.loss.cpu()),
        "eval_prediction_loss": float(output.prediction_loss.cpu()),
        "eval_teacher_forcing_loss": float(output.teacher_forcing_loss.cpu()),
        "eval_rollout_loss": float(output.rollout_loss.cpu()),
        "eval_vicreg_loss": float(output.vicreg_loss.cpu()),
        "eval_vicreg_variance_loss": float(output.vicreg_variance_loss.cpu()),
        "eval_vicreg_covariance_loss": float(output.vicreg_covariance_loss.cpu()),
        "eval_ldad_loss": float(output.ldad_loss.cpu()),
        "eval_action_valid_fraction": float(batch.action_validity.float().mean().cpu()),
        "eval_changed_cells_per_action_mean": float(changed_cells.float().mean().cpu()),
        "eval_changed_cells_per_action_min": float(changed_cells.min().cpu()),
        "eval_changed_cells_per_action_max": float(changed_cells.max().cpu()),
        "eval_object_area_preserved_fraction": float(area_preserved.float().mean().cpu()),
        "eval_latent_std": float(_pooled(current).std(dim=0, unbiased=False).mean().cpu()),
        "eval_latent_effective_rank": _effective_rank(_pooled(current)),
    }
    metrics.update(
        _action_ranking_diagnostics(
            model,
            batch,
            generator,
            device=device,
            max_samples=min(16, batch_size),
        )
    )
    if output.ldad_logits is not None:
        horizon = model.ldad_horizon
        changed = batch.states[:, 1 : horizon + 1] != batch.states[:, :horizon]
        changed = changed.flatten(2).any(dim=2).all(dim=1)
        effective = batch.action_validity[:, :horizon].all(dim=1) & changed
        metrics["eval_ldad_horizon"] = float(horizon)
        metrics["eval_ldad_effective_fraction"] = float(effective.float().mean().cpu())
        if bool(effective.any()):
            predictions = torch.stack(
                [logits.argmax(dim=-1) for logits in output.ldad_logits], dim=-1
            )
            targets = batch.actions[effective, :horizon]
            if horizon == 1:
                targets = targets[:, 0]
            correct = predictions[effective] == targets
            metrics["eval_ldad_row_accuracy"] = float(correct[..., 0].float().mean().cpu())
            metrics["eval_ldad_col_accuracy"] = float(correct[..., 1].float().mean().cpu())
            metrics["eval_ldad_transform_accuracy"] = float(
                correct[..., 2].float().mean().cpu()
            )
            per_action = correct.all(dim=-1)
            metrics["eval_ldad_per_action_exact_accuracy"] = float(
                per_action.float().mean().cpu()
            )
            metrics["eval_ldad_exact_accuracy"] = float(
                per_action.flatten(1).all(dim=1).float().mean().cpu()
                if horizon > 1
                else per_action.float().mean().cpu()
            )
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
            endpoint = step * model.level_spans[level]
            changed = (batch.states[:, endpoint] != batch.states[:, 0]).flatten(1).any(dim=1)
            if bool(changed.any()):
                changed_mse = (predicted[changed, step - 1] - targets[changed, step - 1]).square()
                changed_identity = (
                    identity[changed, step - 1] - targets[changed, step - 1]
                ).square()
                metrics[f"eval_level{level}_rollout{step}_changed_gain"] = float(
                    (changed_identity.mean() - changed_mse.mean()).cpu()
                )

    if model.hierarchy_depth > 1:
        metrics.update(_hierarchy_diagnostics(model, batch))
    if planning_episodes > 0:
        metrics.update(
            _planning_diagnostics(
                model,
                dataset,
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
def _action_ranking_diagnostics(
    model: ControlledObjectJEPA,
    batch,
    generator: ControlledObjectGenerator,
    *,
    device: torch.device,
    max_samples: int,
) -> dict[str, float]:
    learned_correct = 0
    oracle_correct = 0
    margins = []
    candidate_counts = []
    evaluated = 0
    for sample in range(min(max_samples, len(batch.states))):
        state = batch.states[sample, 0].detach().cpu().numpy()
        actual_values = tuple(int(value) for value in batch.actions[sample, 0])
        actions = _candidate_subset(
            generator.candidate_actions(state, state_changing_only=True),
            limit=256,
            required=RigidTransform(*actual_values),
        )
        action_values = [tuple(int(value) for value in action.as_array()) for action in actions]
        if actual_values not in action_values:
            continue
        actual_index = action_values.index(actual_values)
        action_tensor = torch.as_tensor(
            np.stack([action.as_array() for action in actions])[:, None],
            dtype=torch.long,
            device=device,
        )
        state_tensor = batch.states[sample : sample + 1, 0]
        state_latent = model.encode(state_tensor)
        predictions = model.predict_chunk(
            0,
            state_latent.expand(len(actions), *state_latent.shape[1:]),
            action_tensor,
        )
        target = model.encode(batch.states[sample : sample + 1, 1], target=True)
        scores = _latent_distance(predictions, target.expand_as(predictions))
        learned_correct += int(int(scores.argmin()) == actual_index)
        wrong = torch.cat((scores[:actual_index], scores[actual_index + 1 :]))
        if len(wrong):
            margins.append(float((wrong.min() - scores[actual_index]).cpu()))

        successor_states = []
        for action in actions:
            successor, _ = generator.apply_action(state, action)
            successor_states.append(successor)
        encoded_successors = model.encode(
            torch.as_tensor(np.stack(successor_states), dtype=torch.long, device=device),
            target=True,
        )
        oracle_scores = _latent_distance(
            encoded_successors, target.expand_as(encoded_successors)
        )
        oracle_correct += int(int(oracle_scores.argmin()) == actual_index)
        candidate_counts.append(len(actions))
        evaluated += 1
    denominator = max(1, evaluated)
    return {
        "eval_action_ranking_samples": float(evaluated),
        "eval_action_candidate_count_mean": (
            float(np.mean(candidate_counts)) if candidate_counts else 0.0
        ),
        "eval_action_top1_accuracy": learned_correct / denominator,
        "eval_oracle_geometry_action_top1_accuracy": oracle_correct / denominator,
        "eval_action_margin_mean": float(np.mean(margins)) if margins else 0.0,
    }


@torch.no_grad()
def _hierarchy_diagnostics(
    model: ControlledObjectJEPA, batch
) -> dict[str, float]:
    current = model.encode(batch.states[:, 0])
    diagnostics: dict[str, float] = {}
    primitive_predictions = model.rollout_level(
        0, current, batch.actions[:, : max(model.level_spans)]
    ).unbind(dim=1)
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
        valid_distances = torch.cdist(high_prediction, target).square() / target.shape[-1]
        diagnostics[f"eval_level{level}_predicted_valid_nn_mse"] = float(
            valid_distances.min(dim=1).values.mean().cpu()
        )
        if len(macros) > 2:
            pooled_current = _pooled(current)
            state_mean = pooled_current.mean(dim=0)
            state_std = pooled_current.std(dim=0, unbiased=False).clamp_min(1.0e-3)
            macro_mean = macros.mean(dim=0)
            feature_std = macros.std(dim=0, unbiased=False).clamp_min(1.0e-3)
            on_joint = torch.cat(
                (
                    (pooled_current - state_mean) / state_std,
                    (macros - macro_mean) / feature_std,
                ),
                dim=-1,
            )
            on_distance = torch.cdist(on_joint, on_joint)
            on_distance.fill_diagonal_(torch.inf)
            on_nearest = on_distance.min(dim=1).values
            off = macro_mean + 3.0 * feature_std * torch.randn_like(macros)
            off_joint = torch.cat(
                (
                    (pooled_current - state_mean) / state_std,
                    (off - macro_mean) / feature_std,
                ),
                dim=-1,
            )
            off_distance = torch.cdist(off_joint, on_joint)
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
    dataset: ControlledTrajectoryDataset,
    generator: ControlledObjectGenerator,
    rng: np.random.Generator,
    *,
    episodes: int,
    candidates: int,
    device: torch.device,
) -> dict[str, float]:
    planning_horizon = min(max(model.level_spans), dataset.horizon)
    replay_successes = 0
    symbolic_successes = 0
    learned_successes = 0
    oracle_candidate_successes = 0
    bounded_cem_successes = 0
    support_cem_successes = 0
    mppi_successes = 0
    manual_subgoal_successes = 0
    final_errors: dict[str, list[float]] = {
        "retrieval": [],
        "oracle_candidate": [],
        "bounded_cem": [],
        "support_cem": [],
        "mppi": [],
    }
    symbolic_horizon = min(4, planning_horizon)
    supports = {
        level: _estimate_macro_support(
            model,
            dataset,
            level=level,
            seed=int(rng.integers(0, 2**31 - 1)),
            sample_count=max(1024, candidates),
            device=device,
        )
        for level in range(model.hierarchy_depth)
    }
    torch_rng = torch.Generator(device=device)
    torch_rng.manual_seed(int(rng.integers(0, 2**31 - 1)))
    for _ in range(episodes):
        trajectory = generator.sample_trajectory(rng, horizon=planning_horizon)
        initial = trajectory.states[0]
        goal = trajectory.states[-1]
        manual_horizon = min(10, planning_horizon)
        manual_subgoal = _manual_low_level_plan(
            model,
            generator,
            initial,
            trajectory.states[manual_horizon],
            max_steps=manual_horizon,
            beam_width=64,
            device=device,
        )
        manual_subgoal_successes += int(
            np.array_equal(manual_subgoal, trajectory.states[manual_horizon])
        )
        replayed = generator.replay(
            initial,
            (
                RigidTransform(*(int(value) for value in action))
                for action in trajectory.actions
            ),
        )
        replay_successes += int(np.array_equal(replayed, goal))
        short_trajectory = generator.sample_trajectory(rng, horizon=symbolic_horizon)
        symbolic = _symbolic_receding_plan(
            generator,
            short_trajectory.states[0],
            short_trajectory.states[-1],
            max_depth=symbolic_horizon,
            beam_width=max(256, candidates),
        )
        symbolic_successes += int(np.array_equal(symbolic, short_trajectory.states[-1]))
        learned = _receding_on_support_plan(
            model,
            generator,
            initial,
            goal,
            rng,
            max_steps=planning_horizon,
            candidates=min(256, candidates),
            device=device,
            oracle_actions=None,
            supports=supports,
        )
        learned_successes += int(np.array_equal(learned, goal))
        final_errors["retrieval"].append(float(np.mean(learned != goal)))
        oracle_candidate = _receding_on_support_plan(
            model,
            generator,
            initial,
            goal,
            rng,
            max_steps=planning_horizon,
            candidates=min(256, candidates),
            device=device,
            oracle_actions=trajectory.actions,
            supports=supports,
        )
        oracle_candidate_successes += int(np.array_equal(oracle_candidate, goal))
        final_errors["oracle_candidate"].append(
            float(np.mean(oracle_candidate != goal))
        )
        if model.hierarchy_depth > 1:
            bounded_cem = _receding_cem_plan(
                model,
                generator,
                initial,
                goal,
                rng,
                torch_rng,
                supports=supports,
                max_steps=planning_horizon,
                candidates=max(512, candidates),
                support_weight=0.0,
                planner="cem",
                device=device,
            )
            bounded_cem_successes += int(np.array_equal(bounded_cem, goal))
            final_errors["bounded_cem"].append(float(np.mean(bounded_cem != goal)))
            support_cem = _receding_cem_plan(
                model,
                generator,
                initial,
                goal,
                rng,
                torch_rng,
                supports=supports,
                max_steps=planning_horizon,
                candidates=max(512, candidates),
                support_weight=0.1,
                planner="cem",
                device=device,
            )
            support_cem_successes += int(np.array_equal(support_cem, goal))
            final_errors["support_cem"].append(float(np.mean(support_cem != goal)))
            mppi = _receding_cem_plan(
                model,
                generator,
                initial,
                goal,
                rng,
                torch_rng,
                supports=supports,
                max_steps=planning_horizon,
                candidates=max(1000, candidates),
                support_weight=0.1,
                planner="mppi",
                device=device,
            )
            mppi_successes += int(np.array_equal(mppi, goal))
            final_errors["mppi"].append(float(np.mean(mppi != goal)))
    return {
        "eval_oracle_replay_success_rate": replay_successes / episodes,
        "eval_manual_low_level_subgoal_success_rate": (
            manual_subgoal_successes / episodes
        ),
        "eval_symbolic_receding_success_rate": symbolic_successes / episodes,
        "eval_symbolic_planning_horizon": float(symbolic_horizon),
        "eval_learned_receding_success_rate": learned_successes / episodes,
        "eval_oracle_candidate_receding_success_rate": (
            oracle_candidate_successes / episodes
        ),
        "eval_bounded_cem_receding_success_rate": (
            bounded_cem_successes / episodes if model.hierarchy_depth > 1 else learned_successes / episodes
        ),
        "eval_support_cem_receding_success_rate": (
            support_cem_successes / episodes if model.hierarchy_depth > 1 else learned_successes / episodes
        ),
        "eval_mppi_receding_success_rate": (
            mppi_successes / episodes if model.hierarchy_depth > 1 else learned_successes / episodes
        ),
        **{
            f"eval_{name}_final_pixel_error": float(np.mean(values))
            for name, values in final_errors.items()
            if values
        },
        "eval_planning_horizon": float(planning_horizon),
    }


def _manual_low_level_plan(
    model: ControlledObjectJEPA,
    generator: ControlledObjectGenerator,
    initial: np.ndarray,
    goal: np.ndarray,
    *,
    max_steps: int,
    beam_width: int,
    device: torch.device,
) -> np.ndarray:
    state = initial.copy()
    target = model.encode(
        torch.as_tensor(goal[None], dtype=torch.long, device=device), target=True
    )
    for executed in range(max_steps):
        if np.array_equal(state, goal):
            break
        state_latent = model.encode(
            torch.as_tensor(state[None], dtype=torch.long, device=device)
        )
        action = _best_flat_plan_action(
            model,
            generator,
            state,
            state_latent,
            target,
            horizon=max_steps - executed,
            beam_width=beam_width,
            device=device,
        )
        state, _ = generator.apply_action(state, action)
    return state


def _estimate_macro_support(
    model: ControlledObjectJEPA,
    dataset: ControlledTrajectoryDataset,
    *,
    level: int,
    seed: int,
    sample_count: int,
    device: torch.device,
) -> MacroSupport:
    span = model.level_spans[level]
    batch = dataset.sample_batch(
        np.random.default_rng(seed),
        batch_size=sample_count,
        horizon=span,
        device=device,
    )
    state_bank = _pooled(model.encode(batch.states[:, 0]))
    bank = model.encode_action_chunk(level, batch.actions[:, :span])
    lower = torch.quantile(bank, 0.02, dim=0)
    upper = torch.quantile(bank, 0.98, dim=0)
    return MacroSupport(
        state_bank=state_bank,
        bank=bank,
        action_bank=batch.actions[:, :span],
        lower=lower,
        upper=upper,
        state_mean=state_bank.mean(dim=0),
        state_std=state_bank.std(dim=0, unbiased=False).clamp_min(1.0e-3),
        mean=bank.mean(dim=0),
        std=bank.std(dim=0, unbiased=False).clamp_min(1.0e-3),
    )


def _symbolic_receding_plan(
    generator: ControlledObjectGenerator,
    initial: np.ndarray,
    goal: np.ndarray,
    *,
    max_depth: int,
    beam_width: int,
) -> np.ndarray:
    state = initial.copy()
    for executed in range(max_depth):
        if np.array_equal(state, goal):
            break
        path = _symbolic_beam_search(
            generator,
            state,
            goal,
            max_depth=max_depth - executed,
            beam_width=beam_width,
        )
        if not path:
            break
        state, _ = generator.apply_action(state, path[0])
    return state


def _symbolic_beam_search(
    generator: ControlledObjectGenerator,
    initial: np.ndarray,
    goal: np.ndarray,
    *,
    max_depth: int,
    beam_width: int,
) -> tuple[RigidTransform, ...]:
    if np.array_equal(initial, goal):
        return ()
    beam: list[tuple[np.ndarray, tuple[RigidTransform, ...]]] = [(initial.copy(), ())]
    seen = {initial.tobytes()}
    for _ in range(max_depth):
        proposals: list[tuple[int, np.ndarray, tuple[RigidTransform, ...]]] = []
        for state, path in beam:
            for action in generator.candidate_actions(state, state_changing_only=True):
                successor, valid = generator.apply_action(state, action)
                if not valid:
                    continue
                key = successor.tobytes()
                if key in seen:
                    continue
                next_path = (*path, action)
                if np.array_equal(successor, goal):
                    return next_path
                seen.add(key)
                proposals.append((int(np.count_nonzero(successor != goal)), successor, next_path))
        proposals.sort(key=lambda item: item[0])
        beam = [(state, path) for _, state, path in proposals[:beam_width]]
        if not beam:
            break
    return ()


def _receding_on_support_plan(
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
    supports: dict[int, MacroSupport],
) -> np.ndarray:
    state = initial.copy()
    goal_tensor = torch.as_tensor(goal[None], dtype=torch.long, device=device)
    goal_latent = model.encode(goal_tensor, target=True)
    for executed in range(max_steps):
        if np.array_equal(state, goal):
            break
        oracle_suffix = None
        if oracle_actions is not None:
            oracle_suffix = oracle_actions[executed:]
        action = _recursive_on_support_action(
            model,
            generator,
            state,
            goal_latent,
            rng,
            candidates=candidates,
            device=device,
            oracle_actions=oracle_suffix,
            supports=supports,
            max_primitive_horizon=max_steps - executed,
        )
        state, _ = generator.apply_action(state, action)
    return state


def _receding_cem_plan(
    model: ControlledObjectJEPA,
    generator: ControlledObjectGenerator,
    initial: np.ndarray,
    goal: np.ndarray,
    rng: np.random.Generator,
    torch_rng: torch.Generator,
    *,
    supports: dict[int, MacroSupport],
    max_steps: int,
    candidates: int,
    support_weight: float,
    planner: str,
    device: torch.device,
) -> np.ndarray:
    state = initial.copy()
    goal_tensor = torch.as_tensor(goal[None], dtype=torch.long, device=device)
    goal_latent = model.encode(goal_tensor, target=True)
    for executed in range(max_steps):
        if np.array_equal(state, goal):
            break
        action = _recursive_cem_action(
            model,
            generator,
            state,
            goal_latent,
            rng,
            torch_rng,
            supports=supports,
            candidates=candidates,
            support_weight=support_weight,
            planner=planner,
            device=device,
            max_primitive_horizon=max_steps - executed,
        )
        state, _ = generator.apply_action(state, action)
    return state


def _recursive_cem_action(
    model: ControlledObjectJEPA,
    generator: ControlledObjectGenerator,
    state: np.ndarray,
    target_latent: torch.Tensor,
    rng: np.random.Generator,
    torch_rng: torch.Generator,
    *,
    supports: dict[int, MacroSupport],
    candidates: int,
    support_weight: float,
    planner: str,
    device: torch.device,
    max_primitive_horizon: int,
) -> RigidTransform:
    state_tensor = torch.as_tensor(state[None], dtype=torch.long, device=device)
    state_latent = model.encode(state_tensor)
    top_level = model.hierarchy_depth - 1
    while (
        top_level > 0
        and model.level_spans[top_level] > max_primitive_horizon
    ):
        top_level -= 1
    return _plan_cem_level(
        model,
        generator,
        state,
        state_latent,
        target_latent,
        rng,
        torch_rng,
        level=top_level,
        transition_count=_planning_transition_count(
            model, top_level, max_primitive_horizon=max_primitive_horizon
        ),
        supports=supports,
        candidates=candidates,
        support_weight=support_weight,
        planner=planner,
        device=device,
    )


def _plan_cem_level(
    model: ControlledObjectJEPA,
    generator: ControlledObjectGenerator,
    state: np.ndarray,
    state_latent: torch.Tensor,
    target_latent: torch.Tensor,
    rng: np.random.Generator,
    torch_rng: torch.Generator,
    *,
    level: int,
    transition_count: int,
    supports: dict[int, MacroSupport],
    candidates: int,
    support_weight: float,
    planner: str,
    device: torch.device,
) -> RigidTransform:
    if level == 0:
        return _best_flat_plan_action(
            model,
            generator,
            state,
            state_latent,
            target_latent,
            horizon=transition_count,
            beam_width=min(64, candidates),
            device=device,
        )
    sequence_planner = _cem_macro_sequence if planner == "cem" else _mppi_macro_sequence
    macros, first_subgoal = sequence_planner(
        model,
        state_latent,
        target_latent,
        level=level,
        transition_count=transition_count,
        support=supports[level],
        candidates=candidates,
        iterations=10,
        support_weight=support_weight,
        torch_rng=torch_rng,
    )
    del macros
    span = model.level_spans[level]
    lower_span = model.level_spans[level - 1]
    return _plan_cem_level(
        model,
        generator,
        state,
        state_latent,
        first_subgoal,
        rng,
        torch_rng,
        level=level - 1,
        transition_count=span // lower_span,
        supports=supports,
        candidates=candidates,
        support_weight=support_weight,
        planner=planner,
        device=device,
    )


def _cem_macro_sequence(
    model: ControlledObjectJEPA,
    state_latent: torch.Tensor,
    target_latent: torch.Tensor,
    *,
    level: int,
    transition_count: int,
    support: MacroSupport,
    candidates: int,
    iterations: int,
    support_weight: float,
    torch_rng: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    variance_ema = 0.7
    mean = support.mean.expand(transition_count, -1).clone()
    std = support.std.expand(transition_count, -1).clone()
    elite_count = max(2, candidates // 8)
    best_macros = mean
    best_first = model.predict_from_macro(level, state_latent, mean[:1])
    for _ in range(iterations):
        noise = torch.randn(
            candidates,
            transition_count,
            support.mean.numel(),
            device=state_latent.device,
            generator=torch_rng,
        )
        macro_candidates = mean.unsqueeze(0) + std.unsqueeze(0) * noise
        macro_candidates = torch.maximum(
            torch.minimum(macro_candidates, support.upper), support.lower
        )
        initial = state_latent.expand(candidates, -1)
        rollout = model.dynamics[level].rollout(initial, macro_candidates)
        support_states = torch.cat((initial[:, None], rollout[:, :-1]), dim=1)
        costs = _latent_distance(rollout[:, -1], target_latent.expand(candidates, -1))
        if support_weight > 0.0:
            costs = costs + support_weight * _macro_support_energy(
                macro_candidates,
                support_states,
                support,
            )
        elite_ids = costs.topk(elite_count, largest=False).indices
        elites = macro_candidates[elite_ids]
        mean = elites.mean(dim=0)
        elite_variance = elites.var(dim=0, unbiased=False)
        variance = variance_ema * std.square() + (1.0 - variance_ema) * elite_variance
        std = torch.maximum(
            variance.clamp_min(1.0e-6).sqrt(), 0.05 * support.std
        )
        chosen = int(costs.argmin())
        best_macros = macro_candidates[chosen]
        best_first = rollout[chosen : chosen + 1, 0]
    return best_macros, best_first


def _mppi_macro_sequence(
    model: ControlledObjectJEPA,
    state_latent: torch.Tensor,
    target_latent: torch.Tensor,
    *,
    level: int,
    transition_count: int,
    support: MacroSupport,
    candidates: int,
    iterations: int,
    support_weight: float,
    torch_rng: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = support.mean.expand(transition_count, -1).clone()
    std = support.std.expand(transition_count, -1).clone()
    for _ in range(iterations):
        noise = torch.randn(
            candidates,
            transition_count,
            support.mean.numel(),
            device=state_latent.device,
            generator=torch_rng,
        )
        candidates_macro = mean.unsqueeze(0) + std.unsqueeze(0) * noise
        candidates_macro = torch.maximum(
            torch.minimum(candidates_macro, support.upper), support.lower
        )
        initial = state_latent.expand(candidates, -1)
        rollout = model.dynamics[level].rollout(initial, candidates_macro)
        support_states = torch.cat((initial[:, None], rollout[:, :-1]), dim=1)
        costs = _latent_distance(
            rollout[:, -1], target_latent.expand(candidates, -1)
        )
        if support_weight > 0.0:
            costs = costs + support_weight * _macro_support_energy(
                candidates_macro, support_states, support
            )
        temperature = costs.std(unbiased=False).clamp_min(1.0e-4)
        weights = torch.softmax(-(costs - costs.min()) / temperature, dim=0)
        mean = torch.einsum("b,btd->td", weights, candidates_macro)
        centered = candidates_macro - mean
        std = torch.sqrt(
            torch.einsum("b,btd->td", weights, centered.square()) + 1.0e-6
        )
        std = torch.maximum(std, 0.05 * support.std)
    selected = mean
    selected_rollout = model.dynamics[level].rollout(state_latent, selected[None])
    return selected, selected_rollout[:, 0]


def _macro_support_energy(
    macros: torch.Tensor,
    states: torch.Tensor,
    support: MacroSupport,
) -> torch.Tensor:
    if states.shape[:2] != macros.shape[:2]:
        raise ValueError("Macro support states and actions must share batch/time axes.")
    normalized = (macros - support.mean) / support.std
    normalized_states = (states - support.state_mean) / support.state_std
    queries = torch.cat((normalized_states, normalized), dim=-1)
    bank = torch.cat(
        (
            (support.state_bank - support.state_mean) / support.state_std,
            (support.bank - support.mean) / support.std,
        ),
        dim=-1,
    )
    distances = torch.cdist(queries.flatten(0, 1), bank)
    nearest = distances.min(dim=1).values.square()
    return nearest.view(macros.shape[:2]).mean(dim=1)


def _recursive_on_support_action(
    model: ControlledObjectJEPA,
    generator: ControlledObjectGenerator,
    state: np.ndarray,
    target_latent: torch.Tensor,
    rng: np.random.Generator,
    *,
    candidates: int,
    device: torch.device,
    oracle_actions: np.ndarray | None = None,
    supports: dict[int, MacroSupport] | None = None,
    max_primitive_horizon: int | None = None,
) -> RigidTransform:
    state_tensor = torch.as_tensor(state[None], dtype=torch.long, device=device)
    state_latent = model.encode(state_tensor)
    top_level = model.hierarchy_depth - 1
    available_horizon = (
        max(model.level_spans)
        if max_primitive_horizon is None
        else max_primitive_horizon
    )
    while top_level > 0 and model.level_spans[top_level] > available_horizon:
        top_level -= 1
    if oracle_actions is not None:
        while top_level > 0 and model.level_spans[top_level] > len(oracle_actions):
            top_level -= 1
    transition_count = _planning_transition_count(
        model, top_level, max_primitive_horizon=available_horizon
    )
    if oracle_actions is not None:
        transition_count = min(
            transition_count,
            max(1, len(oracle_actions) // model.level_spans[top_level]),
        )
    return _plan_on_support_level(
        model,
        generator,
        state,
        state_latent,
        target_latent,
        rng,
        level=top_level,
        transition_count=transition_count,
        candidates=candidates,
        device=device,
        oracle_actions=oracle_actions,
        supports=supports or {},
    )


def _plan_on_support_level(
    model: ControlledObjectJEPA,
    generator: ControlledObjectGenerator,
    state: np.ndarray,
    state_latent: torch.Tensor,
    target_latent: torch.Tensor,
    rng: np.random.Generator,
    *,
    level: int,
    transition_count: int,
    candidates: int,
    device: torch.device,
    oracle_actions: np.ndarray | None,
    supports: dict[int, MacroSupport],
) -> RigidTransform:
    if level == 0 and oracle_actions is None:
        return _best_flat_plan_action(
            model,
            generator,
            state,
            state_latent,
            target_latent,
            horizon=transition_count,
            beam_width=candidates,
            device=device,
        )
    span = model.level_spans[level]
    primitive_horizon = span * transition_count
    sequences = _candidate_action_sequences(
        generator,
        state,
        rng,
        horizon=primitive_horizon,
        count=candidates,
        action_bank=(supports[level].action_bank if level in supports else None),
    )
    if oracle_actions is not None and len(oracle_actions) >= primitive_horizon:
        sequences[0] = oracle_actions[:primitive_horizon]
    action_tensor = torch.as_tensor(sequences, dtype=torch.long, device=device)
    rollouts = model.rollout_level(
        level,
        state_latent.expand(len(sequences), -1),
        action_tensor,
    )
    scores = _latent_distance(
        rollouts[:, -1], target_latent.expand(len(sequences), -1)
    )
    chosen = int(scores.argmin())
    if level == 0:
        return RigidTransform(*(int(value) for value in sequences[chosen, 0]))
    chosen_subgoal = rollouts[chosen : chosen + 1, 0]
    lower_span = model.level_spans[level - 1]
    return _plan_on_support_level(
        model,
        generator,
        state,
        state_latent,
        chosen_subgoal,
        rng,
        level=level - 1,
        transition_count=span // lower_span,
        candidates=candidates,
        device=device,
        oracle_actions=None,
        supports=supports,
    )


@torch.no_grad()
def _best_flat_plan_action(
    model: ControlledObjectJEPA,
    generator: ControlledObjectGenerator,
    state: np.ndarray,
    state_latent: torch.Tensor,
    target_latent: torch.Tensor,
    *,
    horizon: int,
    beam_width: int,
    device: torch.device,
) -> RigidTransform:
    if horizon < 1 or beam_width < 1:
        raise ValueError("Latent beam planning requires positive horizon and width.")
    beam: list[tuple[np.ndarray, tuple[RigidTransform, ...]]] = [(state.copy(), ())]
    best_score = float("inf")
    best_action = None
    for _ in range(horizon):
        proposal_states = []
        proposal_sequences = []
        first_actions = []
        for symbolic_state, sequence in beam:
            actions = _candidate_subset(
                generator.candidate_actions(symbolic_state, state_changing_only=True),
                limit=max(32, 2 * beam_width),
            )
            for action in actions:
                successor, _ = generator.apply_action(symbolic_state, action)
                proposal_states.append(successor)
                proposal_sequences.append((*sequence, action))
                first_actions.append(sequence[0] if sequence else action)
        if not proposal_sequences:
            break
        action_batch = torch.as_tensor(
            np.stack(
                [np.stack([action.as_array() for action in sequence]) for sequence in proposal_sequences]
            ),
            dtype=torch.long,
            device=device,
        )
        predictions = model.rollout_level(
            0,
            state_latent.expand(len(proposal_sequences), -1),
            action_batch,
        )[:, -1]
        scores = _latent_distance(predictions, target_latent.expand_as(predictions))
        step_best = int(scores.argmin())
        step_score = float(scores[step_best].cpu())
        if step_score < best_score:
            best_score = step_score
            best_action = first_actions[step_best]
        keep = scores.topk(min(beam_width, len(scores)), largest=False).indices.tolist()
        beam = [
            (
                proposal_states[index],
                proposal_sequences[index],
            )
            for index in keep
        ]
    if best_action is None:
        raise RuntimeError("No valid primitive action is available for latent beam planning.")
    return best_action


def _best_primitive_action(
    model: ControlledObjectJEPA,
    generator: ControlledObjectGenerator,
    state: np.ndarray,
    state_latent: torch.Tensor,
    target_latent: torch.Tensor,
    *,
    device: torch.device,
) -> RigidTransform:
    actions = _candidate_subset(
        generator.candidate_actions(state, state_changing_only=True), limit=256
    )
    action_tensor = torch.as_tensor(
        np.stack([action.as_array() for action in actions])[:, None],
        dtype=torch.long,
        device=device,
    )
    repeated_state = state_latent.expand(len(actions), *state_latent.shape[1:])
    predictions = model.predict_chunk(0, repeated_state, action_tensor)
    scores = _latent_distance(predictions, target_latent.expand_as(predictions))
    return actions[int(scores.argmin())]


def _candidate_action_sequences(
    generator: ControlledObjectGenerator,
    state: np.ndarray,
    rng: np.random.Generator,
    *,
    horizon: int,
    count: int,
    action_bank: torch.Tensor | None = None,
) -> np.ndarray:
    if action_bank is not None:
        bank = action_bank.detach().cpu().numpy()
        if horizon % bank.shape[1]:
            raise ValueError("Planning horizon must contain complete support chunks.")
        chunk_count = horizon // bank.shape[1]
        ids = rng.integers(0, len(bank), size=(count, chunk_count))
        return bank[ids].reshape(count, horizon, 3)
    sequences = []
    first_actions = _candidate_subset(
        generator.candidate_actions(state, state_changing_only=True), limit=count
    )
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


def _candidate_subset(
    actions: tuple[RigidTransform, ...],
    *,
    limit: int,
    required: RigidTransform | None = None,
) -> tuple[RigidTransform, ...]:
    if len(actions) <= limit:
        return actions
    indices = np.linspace(0, len(actions) - 1, num=limit, dtype=np.int64)
    selected = [actions[int(index)] for index in indices]
    if required is not None and required not in selected:
        selected[-1] = required
    return tuple(selected)


def _planning_transition_count(
    model: ControlledObjectJEPA,
    level: int,
    *,
    max_primitive_horizon: int | None = None,
) -> int:
    horizon = (
        max(model.level_spans)
        if max_primitive_horizon is None
        else max_primitive_horizon
    )
    return max(1, min(8, horizon // model.level_spans[level]))


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
