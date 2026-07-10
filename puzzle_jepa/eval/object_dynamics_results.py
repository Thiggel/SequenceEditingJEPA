from __future__ import annotations

import argparse
import json
import math
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
)


def summarize_object_dynamics_runs(root: Path) -> dict[str, Any]:
    runs = []
    checkpoints = []
    balanced_reprobes = []
    for metrics_path in sorted(root.glob("*/metrics.jsonl")):
        run = _load_run(metrics_path.parent)
        if run is None:
            continue
        runs.append(run[0])
        checkpoints.extend(run[1])
        balanced = _load_balanced_reprobe(metrics_path.parent)
        if balanced is not None:
            balanced_reprobes.append(balanced)
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
    if baseline is None:
        return None
    max_steps = int(config["training"]["max_steps"])
    endpoint_step = int(records[-1]["step"])
    complete = endpoint_step >= max_steps and (run_dir / "checkpoint.pt").exists()
    identity = {
        "run_name": run_dir.name,
        "data": str(config["data"]["name"]),
        "model": str(config["model"]["name"]),
        "objective": str(config["objective"]["name"]),
        "seed": int(config["seed"]),
        "learning_rate": float(config["training"]["learning_rate"]),
        "max_steps": max_steps,
    }
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
        for metric in CORE_METRICS:
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
    run_summary = {
        **identity,
        "endpoint_step": endpoint_step,
        "complete": complete,
        "checkpoint_count": len(rows),
        "run_dir": str(run_dir),
    }
    return run_summary, rows


def _aggregate_endpoints(endpoints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    keys = ("probe_fit_version", "data", "model", "objective", "learning_rate", "max_steps")
    for row in endpoints:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    aggregates = []
    for group_key, rows in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        aggregate: dict[str, Any] = dict(zip(keys, group_key, strict=True))
        aggregate["seeds"] = sorted(int(row["seed"]) for row in rows)
        aggregate["n"] = len(rows)
        aggregate["complete_n"] = sum(bool(row["complete"]) for row in rows)
        aggregate["endpoint_steps"] = sorted({int(row["step"]) for row in rows})
        for field in ENDPOINT_FIELDS:
            values = [float(row[field]) for row in rows if _number(row.get(field)) is not None]
            aggregate[f"{field}_mean"] = statistics.fmean(values) if values else None
            aggregate[f"{field}_std"] = statistics.pstdev(values) if len(values) > 1 else (0.0 if values else None)
        aggregates.append(aggregate)
    return aggregates


def _load_balanced_reprobe(run_dir: Path) -> dict[str, Any] | None:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return None
    result = None
    probe_fit_version = None
    for version in (3, 2):
        result_path = run_dir / f"probe_eval_balanced_v{version}.json"
        if not result_path.exists():
            continue
        candidate = json.loads(result_path.read_text(encoding="utf-8"))
        if int(candidate.get("probe_fit_version", -1)) != version:
            continue
        result = candidate
        probe_fit_version = version
        break
    if result is None or probe_fit_version is None or "initial_probe_current_object_acc" not in result:
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    max_steps = int(config["training"]["max_steps"])
    row: dict[str, Any] = {
        "run_name": run_dir.name,
        "data": str(config["data"]["name"]),
        "model": str(config["model"]["name"]),
        "objective": str(config["objective"]["name"]),
        "seed": int(config["seed"]),
        "learning_rate": float(config["training"]["learning_rate"]),
        "max_steps": max_steps,
        "probe_fit_version": probe_fit_version,
        "checkpoint_step": int(result["checkpoint_step"]),
        "complete": int(result["checkpoint_step"]) >= max_steps,
    }
    for field in BALANCED_FIELDS:
        row[field] = _number(result.get(field))
    initial_std = _number(result.get("initial_latent_std_mean"))
    trained_std = _number(result.get("latent_std_mean"))
    row["latent_std_ratio"] = trained_std / initial_std if trained_std is not None and initial_std else None
    return row


def _aggregate_balanced_reprobes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    keys = ("probe_fit_version", "data", "model", "objective", "learning_rate", "max_steps")
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    aggregates = []
    for group_key, group_rows in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        aggregate: dict[str, Any] = dict(zip(keys, group_key, strict=True))
        aggregate["seeds"] = sorted(int(row["seed"]) for row in group_rows)
        aggregate["n"] = len(group_rows)
        aggregate["complete_n"] = sum(bool(row["complete"]) for row in group_rows)
        for field in ("latent_std_ratio", *BALANCED_FIELDS):
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
        "| Probe | Model | Objective | LR | Max steps | Seeds | Complete | Loss | Std ratio | dCurrent | dObject map | dGrid | dInvalid AUROC |",
        "|---:|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["endpoint_aggregates"]:
        lines.append(
            "| v{version} | {model} | {objective} | {lr:.1e} | {steps} | {seeds} | {complete}/{n} | {loss} | {std} | {current} | {object_map} | {grid} | {invalid} |".format(
                version=row["probe_fit_version"],
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
            "| Probe | Model | Objective | LR | Seeds | Complete | Std ratio | dCount | dCurrent | dCurrent balanced | dAction object | dObject map | dGrid | dInvalid AUROC |",
            "|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["balanced_reprobe_aggregates"]:
        lines.append(
            "| v{version} | {model} | {objective} | {lr:.1e} | {seeds} | {complete}/{n} | {std} | {count} | {current} | {balanced} | {action} | {object_map} | {grid} | {invalid} |".format(
                version=row["probe_fit_version"],
                model=row["model"],
                objective=row["objective"],
                lr=float(row["learning_rate"]),
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
    result = float(value)
    return result if math.isfinite(result) else None


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
