from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW

from puzzle_jepa.data.arc import ARCGrid, grid_distance, grid_exact, load_arc_tasks, make_initial_arc_candidates
from puzzle_jepa.data.arc_actions import ARCAction, apply_arc_action, episode_candidate_shapes, episode_palette, generate_arc_actions
from puzzle_jepa.data.arc_proposals import build_arc_sources, extract_arc_proposals
from puzzle_jepa.data.arc_training import (
    ARCCandidateRecord,
    arc_action_features,
    collate_arc_records,
    episodes_from_tasks,
    sample_arc_candidate_record,
)
from puzzle_jepa.models.arc_models import ARCCandidateScorer


VARIANTS: dict[str, dict[str, Any]] = {
    "raw_grid_energy": {"use_action_features": False, "use_jepa": False, "dynamics_weight": 0.0},
    "proposal_energy": {"use_action_features": True, "use_jepa": False, "dynamics_weight": 0.0},
    "jepa_energy": {"use_action_features": True, "use_jepa": True, "dynamics_weight": 0.25},
}


def run_arc_training(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", 0))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))

    variant = str(config["variant"])
    if variant not in VARIANTS:
        raise ValueError(f"Unknown ARC variant {variant!r}; choices are {sorted(VARIANTS)}.")
    data_cfg = dict(config["data"])
    tasks = load_arc_tasks(data_cfg["data_root"], split=str(data_cfg.get("split", "training")), limit=data_cfg.get("task_limit"))
    eval_task_count = int(data_cfg.get("eval_task_count", 40))
    eval_tasks = tasks[-eval_task_count:] if eval_task_count > 0 else []
    train_tasks = tasks[: max(0, len(tasks) - eval_task_count)] if eval_tasks else tasks
    train_episodes = episodes_from_tasks(train_tasks)
    eval_episodes = episodes_from_tasks(eval_tasks)
    if not train_episodes:
        raise ValueError("ARC training requires at least one train episode.")
    if not eval_episodes:
        eval_episodes = train_episodes[: max(1, min(32, len(train_episodes)))]

    model_cfg = dict(config["model"])
    model = ARCCandidateScorer(**{**model_cfg, **VARIANTS[variant]}).to(device)
    train_cfg = dict(config["training"])
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 3.0e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1.0e-4)),
    )
    max_steps = int(train_cfg.get("max_steps", 1000))
    batch_size = int(train_cfg.get("batch_size", 16))
    eval_every = int(train_cfg.get("eval_every_steps", 200))
    save_every = int(train_cfg.get("save_every_steps", 1000))
    use_amp = bool(train_cfg.get("bf16", True)) and device.type == "cuda"
    metrics_path = output_dir / "metrics.jsonl"
    latest: dict[str, Any] = {}
    sampler_cfg = dict(config["sampler"])

    for step in range(1, max_steps + 1):
        model.train()
        records = [
            sample_arc_candidate_record(
                train_episodes,
                rng,
                oracle_shape=bool(sampler_cfg.get("oracle_shape", False)),
                include_cell_actions=bool(sampler_cfg.get("include_cell_actions", True)),
                max_actions=int(sampler_cfg.get("max_actions", 800)),
                positive_probability=float(sampler_cfg.get("positive_probability", 0.25)),
                best_action_probability=float(sampler_cfg.get("best_action_probability", 0.5)),
            )
            for _ in range(batch_size)
        ]
        batch = collate_arc_records(records, max_context=int(data_cfg.get("max_context", 4)), device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            output = model(batch)
        if not torch.isfinite(output.loss.detach()):
            raise FloatingPointError(f"Non-finite ARC loss at step {step}: {float(output.loss.detach().cpu())}")
        output.loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 1.0)))
        optimizer.step()

        if step == 1 or step % eval_every == 0 or step == max_steps:
            eval_metrics = evaluate_arc_model(
                model,
                eval_episodes[: int(config["eval"].get("episodes", 64))],
                device=device,
                max_context=int(data_cfg.get("max_context", 4)),
                oracle_shape=bool(config["eval"].get("oracle_shape", False)),
                include_cell_actions=bool(config["eval"].get("include_cell_actions", True)),
                max_actions=int(config["eval"].get("max_actions", 800)),
                beam_width=int(config["eval"].get("beam_width", 1)),
            )
            latest = {
                "step": step,
                "variant": variant,
                "train_loss": float(output.loss.detach().cpu()),
                "train_energy_loss": float(output.energy_loss.detach().cpu()),
                "train_dynamics_loss": float(output.dynamics_loss.detach().cpu()),
                "grad_norm_pre_clip": float(grad_norm.detach().cpu()),
                "batch_positive_rate": float(batch.labels.mean().detach().cpu()),
                "device": device.type,
                "num_train_tasks": len(train_tasks),
                "num_eval_tasks": len(eval_tasks),
                "num_train_episodes": len(train_episodes),
                "num_eval_episodes": len(eval_episodes),
                **eval_metrics,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(latest, sort_keys=True) + "\n")
            print(json.dumps(latest, sort_keys=True), flush=True)
        if step % save_every == 0 or step == max_steps:
            _save_checkpoint(output_dir / "checkpoint.pt", model, optimizer, step, latest, config)

    (output_dir / "metrics.json").write_text(json.dumps(latest, indent=2, sort_keys=True))
    return latest


def evaluate_arc_model(
    model: ARCCandidateScorer,
    episodes,
    *,
    device: torch.device,
    max_context: int,
    oracle_shape: bool,
    include_cell_actions: bool,
    max_actions: int,
    beam_width: int,
) -> dict[str, Any]:
    del beam_width
    model.eval()
    rows = []
    with torch.no_grad():
        for episode in episodes:
            records = _candidate_records_for_eval(
                episode,
                oracle_shape=oracle_shape,
                include_cell_actions=include_cell_actions,
                max_actions=max_actions,
            )
            batch = collate_arc_records(records, max_context=max_context, device=device)
            logits = model(batch).logits.detach().float().cpu().numpy()
            order = np.argsort(-logits)
            best_idx = int(order[0])
            oracle_idx = min(range(len(records)), key=lambda index: records[index].candidate_distance)
            rows.append(
                {
                    "task_id": episode.task_id,
                    "query_index": episode.query_index,
                    "pass1": grid_exact(records[best_idx].candidate, episode.target_output),
                    "oracle_reachable": grid_exact(records[oracle_idx].candidate, episode.target_output),
                    "pred_distance": records[best_idx].candidate_distance,
                    "oracle_distance": records[oracle_idx].candidate_distance,
                    "num_candidates": len(records),
                }
            )
    if not rows:
        return {"eval_pass1": 0.0, "eval_oracle_reachable": 0.0, "eval_mean_pred_distance": 0.0, "eval_mean_oracle_distance": 0.0}
    return {
        "eval_pass1": sum(bool(row["pass1"]) for row in rows) / len(rows),
        "eval_oracle_reachable": sum(bool(row["oracle_reachable"]) for row in rows) / len(rows),
        "eval_mean_pred_distance": sum(int(row["pred_distance"]) for row in rows) / len(rows),
        "eval_mean_oracle_distance": sum(int(row["oracle_distance"]) for row in rows) / len(rows),
        "eval_mean_candidates": sum(int(row["num_candidates"]) for row in rows) / len(rows),
    }


def _candidate_records_for_eval(
    episode,
    *,
    oracle_shape: bool,
    include_cell_actions: bool,
    max_actions: int,
) -> list[ARCCandidateRecord]:
    records: list[ARCCandidateRecord] = []
    for current in make_initial_arc_candidates(episode, oracle_shape=oracle_shape):
        current_distance = grid_distance(current, episode.target_output)
        records.append(
            ARCCandidateRecord(
                episode=episode,
                current=current,
                candidate=current,
                label=1.0 if grid_exact(current, episode.target_output) else 0.0,
                action=ARCAction(op="initial", params={}, label="initial"),
                action_features=arc_action_features(ARCAction(op="initial", params={}, label="initial"), current=current, candidate=current, proposals={}),
                current_distance=current_distance,
                candidate_distance=current_distance,
            )
        )
        proposals = extract_arc_proposals(episode.context, episode.query_input, current)
        sources = build_arc_sources(episode.context, episode.query_input, current)
        actions = generate_arc_actions(
            episode.context,
            episode.query_input,
            current,
            proposals=proposals,
            candidate_shapes=episode_candidate_shapes(
                episode.context,
                episode.query_input,
                oracle_shape=episode.target_output.shape if oracle_shape else None,
            ),
            palette=episode_palette(episode.context, episode.query_input, current),
            include_cell_actions=include_cell_actions,
            max_actions=max_actions,
        )
        seen = {(current.shape, current.values.tobytes())}
        for action in actions:
            try:
                candidate = apply_arc_action(current, action, proposals=proposals, sources=sources)
            except ValueError:
                continue
            key = (candidate.shape, candidate.values.tobytes())
            if key in seen:
                continue
            seen.add(key)
            records.append(
                ARCCandidateRecord(
                    episode=episode,
                    current=current,
                    candidate=candidate,
                    label=1.0 if grid_exact(candidate, episode.target_output) else 0.0,
                    action=action,
                    action_features=arc_action_features(action, current=current, candidate=candidate, proposals=proposals),
                    current_distance=current_distance,
                    candidate_distance=grid_distance(candidate, episode.target_output),
                )
            )
    return records[: max(1, min(len(records), max_actions))]


def _save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, metrics: dict[str, Any], config: dict[str, Any]) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "metrics": metrics,
            "config": config,
        },
        path,
    )


@hydra.main(config_path="../../configs/puzzle", config_name="arc_jepa", version_base=None)
def main(cfg: DictConfig) -> None:
    run_arc_training(OmegaConf.to_container(cfg, resolve=True))


if __name__ == "__main__":
    main()
