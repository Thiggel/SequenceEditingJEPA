from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


CORE_METRICS = (
    "latent_std_mean",
    "latent_effective_rank",
    "probe_object_count_acc",
    "probe_current_object_acc",
    "probe_delta_action_object_acc",
    "probe_object_map_foreground_miou",
    "probe_grid_foreground_miou",
    "rollout_error_invalid_auroc",
)

ENDPOINT_FIELDS = (
    "train_loss",
    "latent_std_ratio",
    "delta_latent_effective_rank",
    "delta_probe_object_count_acc",
    "delta_probe_current_object_acc",
    "delta_probe_delta_action_object_acc",
    "delta_probe_object_map_foreground_miou",
    "delta_probe_grid_foreground_miou",
    "delta_rollout_error_invalid_auroc",
)

BALANCED_FIELDS = (
    "delta_probe_object_count_acc",
    "delta_probe_object_count_balanced_acc",
    "delta_probe_current_object_acc",
    "delta_probe_current_object_balanced_acc",
    "delta_probe_delta_action_object_acc",
    "delta_probe_object_map_foreground_miou",
    "delta_probe_grid_foreground_miou",
    "delta_rollout_error_invalid_auroc",
    "delta_probe_mlp_object_count_acc",
    "delta_probe_mlp_current_object_balanced_acc",
    "delta_probe_rollout_object_count_acc",
    "delta_probe_rollout_object_count_balanced_acc",
    "delta_probe_delta_action_process_acc",
    "delta_probe_rollout_bbox_mse",
    "delta_probe_rollout_completion_mse",
    "delta_probe_object_part_count_acc",
    "delta_probe_relation_inside_acc",
    "delta_probe_relation_inside_balanced_acc",
    "delta_probe_chunk_correction_acc",
    "delta_probe_attention_current_object_iou",
    "delta_probe_attention_current_object_iou_ge4",
    "delta_probe_attention_current_object_future_iou",
    "delta_probe_attention_incomplete_object_iou",
    "delta_probe_hierarchy_endpoint_mse",
    "delta_probe_hierarchy_macro_retrieval_acc",
    "delta_probe_hierarchy_low_level_retrieval_acc",
    "delta_probe_hierarchy_optimized_goal_l1",
    "delta_probe_hierarchy_subgoal_reachability_l1",
    "delta_probe_hierarchy_cem_subgoal_l1",
    "delta_probe_hierarchy_cem_goal_l1",
    "delta_probe_hierarchy_retrieval_goal_hamming",
    "delta_probe_hierarchy_retrieval_goal_success",
    "delta_probe_hierarchy_cem_executed_goal_hamming",
    "delta_probe_hierarchy_cem_executed_goal_success",
    "delta_probe_hierarchy_cem_model_bias_l1",
    "probe_delta_action_process_acc",
    "probe_delta_action_process_balanced_acc",
    "raw_probe_action_process_provenance_acc",
    "raw_probe_action_process_provenance_balanced_acc",
    "probe_action_process_provenance_majority_acc",
    "probe_action_process_provenance_majority_balanced_acc",
    "latent_nn_current_shape_acc",
    "pixel_nn_current_shape_acc",
    "latent_nn_current_color_acc",
    "pixel_nn_current_color_acc",
    "latent_nn_current_completion_mae",
    "pixel_nn_current_completion_mae",
    "probe_hierarchy_endpoint_mse",
    "probe_hierarchy_level_agreement",
    "probe_hierarchy_macro_retrieval_acc",
    "probe_hierarchy_low_level_retrieval_acc",
    "probe_hierarchy_retrieval_goal_success",
    "probe_hierarchy_retrieval_goal_hamming",
    "probe_hierarchy_cem_executed_goal_success",
    "probe_hierarchy_cem_executed_goal_hamming",
    "probe_hierarchy_subgoal_reachability_l1",
    "probe_hierarchy_cem_model_bias_l1",
)


def summarize_object_dynamics_runs(root: Path) -> dict[str, Any]:
    runs = []
    checkpoints = []
    balanced_reprobes = []
    for metrics_path in sorted(root.glob("*/metrics.jsonl")):
        balanced_reprobes.extend(_load_balanced_reprobes(metrics_path.parent))
        run = _load_run(metrics_path.parent)
        if run is None:
            continue
        runs.append(run[0])
        checkpoints.extend(run[1])
    endpoints = [row for row in checkpoints if row["is_endpoint"]]
    return {
        "root": str(root),
        "run_count": len(runs),
        "complete_run_count": sum(bool(run["complete"]) for run in runs),
        "runs": runs,
        "checkpoints": checkpoints,
        "endpoint_aggregates": _aggregate_endpoints(endpoints),
        "balanced_reprobe_count": len(balanced_reprobes),
        "balanced_reprobes": balanced_reprobes,
        "balanced_reprobe_aggregates": _aggregate_balanced_reprobes(balanced_reprobes),
    }


def write_summary(summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "object_dynamics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "object_dynamics_summary.md").write_text(_render_markdown(summary), encoding="utf-8")


def _load_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.jsonl"
    if not config_path.exists() or not metrics_path.exists():
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        return None
    records.sort(key=lambda record: int(record["step"]))
    baseline = next((record for record in records if int(record["step"]) == 0), None)
    max_steps = int(config["training"]["max_steps"])
    endpoint_step = int(records[-1]["step"])
    complete = endpoint_step >= max_steps and (run_dir / "checkpoint.pt").exists()
    identity = {
        "run_name": run_dir.name,
        "run_family": _run_family(run_dir.name),
        "data": str(config["data"]["name"]),
        "model": str(config["model"]["name"]),
        "objective": str(config["objective"]["name"]),
        "seed": int(config["seed"]),
        "learning_rate": float(config["training"]["learning_rate"]),
        "max_steps": max_steps,
        "probe_trajectory_kind": _probe_trajectory_kind(config, baseline or {}),
    }
    run_summary = {
        **identity,
        "endpoint_step": endpoint_step,
        "complete": complete,
        "checkpoint_count": max(0, len(records) - int(baseline is not None)),
        "run_dir": str(run_dir),
    }
    if baseline is None:
        return run_summary, []
    rows = []
    for record in records:
        if int(record["step"]) == 0:
            continue
        row: dict[str, Any] = {
            **identity,
            "step": int(record["step"]),
            "is_endpoint": int(record["step"]) == endpoint_step,
            "complete": complete,
            "probe_fit_version": int(record.get("probe_fit_version", 1)),
            "train_loss": _number(record.get("train_loss")),
        }
        metric_names = sorted(
            set(CORE_METRICS)
            | {name for name in baseline if _is_probe_metric(name)}
            | {name for name in record if _is_probe_metric(name)}
        )
        for metric in metric_names:
            value = _number(record.get(metric))
            initial = _number(baseline.get(metric))
            row[metric] = value
            row[f"initial_{metric}"] = initial
            row[f"delta_{metric}"] = value - initial if value is not None and initial is not None else None
        initial_std = row["initial_latent_std_mean"]
        row["latent_std_ratio"] = (
            row["latent_std_mean"] / initial_std
            if row["latent_std_mean"] is not None and initial_std not in {None, 0.0}
            else None
        )
        rows.append(row)
    run_summary["checkpoint_count"] = len(rows)
    return run_summary, rows


def _aggregate_endpoints(endpoints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    keys = (
        "probe_fit_version",
        "probe_trajectory_kind",
        "run_family",
        "data",
        "model",
        "objective",
        "learning_rate",
        "max_steps",
    )
    for row in endpoints:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    aggregate_fields = sorted(
        set(ENDPOINT_FIELDS)
        | {
            key
            for row in endpoints
            for key in row
            if key.startswith("delta_") and _number(row.get(key)) is not None
        }
    )
    aggregates = []
    for group_key, rows in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        aggregate: dict[str, Any] = dict(zip(keys, group_key, strict=True))
        aggregate["seeds"] = sorted(int(row["seed"]) for row in rows)
        aggregate["n"] = len(rows)
        aggregate["complete_n"] = sum(bool(row["complete"]) for row in rows)
        aggregate["endpoint_steps"] = sorted({int(row["step"]) for row in rows})
        for field in aggregate_fields:
            values = [float(row[field]) for row in rows if _number(row.get(field)) is not None]
            aggregate[f"{field}_mean"] = statistics.fmean(values) if values else None
            aggregate[f"{field}_std"] = statistics.pstdev(values) if len(values) > 1 else (0.0 if values else None)
        aggregates.append(aggregate)
    return aggregates


def _load_balanced_reprobes(run_dir: Path) -> list[dict[str, Any]]:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return []
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for version in (4, 3, 2):
        rows_by_trajectory: dict[str, dict[str, Any]] = {}
        paths = sorted(
            run_dir.glob(f"probe_eval_*v{version}.json"),
            key=lambda path: (path.name == f"probe_eval_balanced_v{version}.json", path.name),
        )
        for result_path in paths:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if (
                int(result.get("probe_fit_version", -1)) != version
                or "initial_probe_current_object_acc" not in result
            ):
                continue
            max_steps = int(config["training"]["max_steps"])
            trajectory_kind = str(
                result.get("probe_trajectory_kind", _probe_trajectory_kind(config, result))
            )
            row: dict[str, Any] = {
                "run_name": run_dir.name,
                "run_family": _run_family(run_dir.name),
                "probe_file": result_path.name,
                "data": str(config["data"]["name"]),
                "model": str(config["model"]["name"]),
                "objective": str(config["objective"]["name"]),
                "seed": int(config["seed"]),
                "learning_rate": float(config["training"]["learning_rate"]),
                "max_steps": max_steps,
                "probe_fit_version": version,
                "probe_trajectory_kind": trajectory_kind,
                "checkpoint_step": int(result["checkpoint_step"]),
                "complete": int(result["checkpoint_step"]) >= max_steps,
            }
            for field in sorted(set(BALANCED_FIELDS) | {key for key in result if key.startswith("delta_")}):
                row[field] = _number(result.get(field))
            initial_std = _number(result.get("initial_latent_std_mean"))
            trained_std = _number(result.get("latent_std_mean"))
            row["latent_std_ratio"] = (
                trained_std / initial_std if trained_std is not None and initial_std else None
            )
            rows_by_trajectory[trajectory_kind] = row
        if rows_by_trajectory:
            return [rows_by_trajectory[key] for key in sorted(rows_by_trajectory)]
    return []


def _aggregate_balanced_reprobes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    keys = (
        "probe_fit_version",
        "probe_trajectory_kind",
        "run_family",
        "data",
        "model",
        "objective",
        "learning_rate",
        "max_steps",
    )
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    aggregate_fields = sorted(
        {"latent_std_ratio", *BALANCED_FIELDS}
        | {
            key
            for row in rows
            for key in row
            if key.startswith("delta_") and _number(row.get(key)) is not None
        }
    )
    aggregates = []
    for group_key, group_rows in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        aggregate: dict[str, Any] = dict(zip(keys, group_key, strict=True))
        aggregate["seeds"] = sorted(int(row["seed"]) for row in group_rows)
        aggregate["n"] = len(group_rows)
        aggregate["complete_n"] = sum(bool(row["complete"]) for row in group_rows)
        for field in aggregate_fields:
            values = [float(row[field]) for row in group_rows if _number(row.get(field)) is not None]
            aggregate[f"{field}_mean"] = statistics.fmean(values) if values else None
            aggregate[f"{field}_std"] = statistics.pstdev(values) if len(values) > 1 else (0.0 if values else None)
        aggregates.append(aggregate)
    return aggregates


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Object Dynamics Result Summary",
        "",
        f"Runs: {summary['run_count']} ({summary['complete_run_count']} complete)",
        "",
        "| Probe | Distribution | Family | Model | Objective | LR | Max steps | Seeds | Complete | Loss | Std ratio | dCurrent | dObject map | dGrid | dInvalid AUROC |",
        "|---:|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["endpoint_aggregates"]:
        lines.append(
            "| v{version} | {distribution} | {family} | {model} | {objective} | {lr:.1e} | {steps} | {seeds} | {complete}/{n} | {loss} | {std} | {current} | {object_map} | {grid} | {invalid} |".format(
                version=row["probe_fit_version"],
                distribution=row["probe_trajectory_kind"],
                family=row["run_family"],
                model=row["model"],
                objective=row["objective"],
                lr=float(row["learning_rate"]),
                steps=row["max_steps"],
                seeds=",".join(str(seed) for seed in row["seeds"]),
                complete=row["complete_n"],
                n=row["n"],
                loss=_format(row["train_loss_mean"]),
                std=_format(row["latent_std_ratio_mean"]),
                current=_format(row["delta_probe_current_object_acc_mean"]),
                object_map=_format(row["delta_probe_object_map_foreground_miou_mean"]),
                grid=_format(row["delta_probe_grid_foreground_miou_mean"]),
                invalid=_format(row["delta_rollout_error_invalid_auroc_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## Class-Balanced Re-Probes",
            "",
            "| Probe | Distribution | Family | Model | Objective | LR | Max steps | Seeds | Complete | Std ratio | dCount | dCurrent | dCurrent balanced | dAction object | dObject map | dGrid | dInvalid AUROC |",
            "|---:|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["balanced_reprobe_aggregates"]:
        lines.append(
            "| v{version} | {distribution} | {family} | {model} | {objective} | {lr:.1e} | {steps} | {seeds} | {complete}/{n} | {std} | {count} | {current} | {balanced} | {action} | {object_map} | {grid} | {invalid} |".format(
                version=row["probe_fit_version"],
                distribution=row["probe_trajectory_kind"],
                family=row["run_family"],
                model=row["model"],
                objective=row["objective"],
                lr=float(row["learning_rate"]),
                steps=row["max_steps"],
                seeds=",".join(str(seed) for seed in row["seeds"]),
                complete=row["complete_n"],
                n=row["n"],
                std=_format(row["latent_std_ratio_mean"]),
                count=_format(row["delta_probe_object_count_acc_mean"]),
                current=_format(row["delta_probe_current_object_acc_mean"]),
                balanced=_format(row["delta_probe_current_object_balanced_acc_mean"]),
                action=_format(row["delta_probe_delta_action_object_acc_mean"]),
                object_map=_format(row["delta_probe_object_map_foreground_miou_mean"]),
                grid=_format(row["delta_probe_grid_foreground_miou_mean"]),
                invalid=_format(row["delta_rollout_error_invalid_auroc_mean"]),
            )
        )
    return "\n".join(lines) + "\n"


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _probe_trajectory_kind(config: dict[str, Any], metrics: dict[str, Any]) -> str:
    if "probe_trajectory_kind" in metrics:
        return str(metrics["probe_trajectory_kind"])
    eval_config = dict(config.get("eval", {}))
    data_config = dict(config["data"])
    return str(eval_config.get("probe_trajectory_kind", data_config.get("trajectory_kind", data_config["name"])))


def _run_family(run_name: str) -> str:
    return re.sub(r"_seed\d+$", "", run_name)


def _is_probe_metric(name: str) -> bool:
    return name.startswith(
        (
            "latent_",
            "pixel_nn_",
            "probe_",
            "raw_probe_",
            "rollout_error_",
        )
    ) and name not in {
        "probe_fit_version",
        "probe_seed",
        "probe_class_balanced_objectives",
        "probe_trajectory_kind",
    }


def _format(value: Any) -> str:
    number = _number(value)
    return "" if number is None else f"{number:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize object-dynamics checkpoint probe deltas.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize_object_dynamics_runs(args.root)
    write_summary(summary, args.output_dir)
    print(json.dumps({"runs": summary["run_count"], "complete": summary["complete_run_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
