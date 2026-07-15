from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


PROBE_METRICS = (
    ("final", "probe_latent_effective_rank"),
    ("final", "probe_latent_std_mean"),
    ("final", "probe_object_presence_balanced_acc"),
    ("final", "probe_shape_balanced_acc"),
    ("final", "probe_position_r2"),
    ("final", "probe_area_r2"),
    ("final", "probe_relation_r2"),
    ("final", "probe_pixel_decoder_foreground_iou"),
    ("final_mixed_load", "probe_object_count_balanced_acc"),
    ("evaluation", "eval_prediction_loss"),
    ("evaluation", "eval_rollout_loss"),
    ("evaluation", "eval_cross_level_consistency_loss"),
    ("evaluation", "eval_action_top1_accuracy"),
    ("evaluation", "eval_level1_one_step_mse"),
    ("evaluation", "eval_level1_primitive_rollout_mse"),
    ("evaluation", "eval_level2_one_step_mse"),
    ("evaluation", "eval_level2_primitive_rollout_mse"),
)


def _read_tasks(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _probe_run_name(task: dict[str, str], gate: str) -> str:
    if gate == "objective":
        return f"{task['variant']}_s{task['seed']}_joint_h110100"
    return f"{task['objective']}_{task['setting']}_s{task['seed']}_dense_h110100"


def analyze_probes(
    tasks: list[dict[str, str]], output_root: Path, gate: str
) -> dict[str, Any]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    missing = []
    group_fields = ("variant",) if gate == "objective" else ("objective", "setting")
    for task in tasks:
        run_name = _probe_run_name(task, gate)
        path = output_root / run_name / "probe_eval_v5.json"
        if not path.is_file():
            missing.append(run_name)
            continue
        grouped[tuple(task[field] for field in group_fields)].append(
            json.loads(path.read_text(encoding="utf-8"))
        )

    groups = []
    for key, rows in sorted(grouped.items()):
        summary: dict[str, Any] = dict(zip(group_fields, key, strict=True))
        summary["seeds"] = sorted(int(row["seed"]) for row in rows)
        for section, metric in PROBE_METRICS:
            values = [
                float(row[section][metric])
                for row in rows
                if metric in row.get(section, {})
            ]
            if values:
                summary[f"{metric}_mean"] = mean(values)
            if section == "final":
                deltas = [
                    float(row["delta"][metric])
                    for row in rows
                    if metric in row.get("delta", {})
                ]
                if deltas:
                    summary[f"{metric}_delta_mean"] = mean(deltas)
                    summary[f"{metric}_delta_min"] = min(deltas)
        groups.append(summary)
    return {
        "schema": f"controlled_objects_{gate}_gate_v1",
        "expected_cells": len(tasks),
        "completed_cells": sum(len(rows) for rows in grouped.values()),
        "missing_runs": missing,
        "groups": groups,
    }


def analyze_planner(
    tasks: list[dict[str, str]], output_root: Path
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    missing = []
    for task in tasks:
        path = output_root / f"{task['objective']}_s{task['seed']}" / f"{task['mode']}.json"
        if not path.is_file():
            missing.append(str(path.relative_to(output_root)))
            continue
        grouped[(task["objective"], task["mode"])].append(
            json.loads(path.read_text(encoding="utf-8"))
        )
    groups = []
    for (objective, mode), rows in sorted(grouped.items()):
        groups.append(
            {
                "objective": objective,
                "mode": mode,
                "seeds": sorted(int(row["seed"]) for row in rows),
                "planning_success_rate_mean": mean(
                    float(row["planning_success_rate"]) for row in rows
                ),
                "planning_final_pixel_error_mean": mean(
                    float(row["planning_final_pixel_error"]) for row in rows
                ),
            }
        )
    return {
        "schema": "controlled_objects_planner_interface_gate_v1",
        "expected_cells": len(tasks),
        "completed_cells": sum(len(rows) for rows in grouped.values()),
        "missing_runs": missing,
        "groups": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gate", choices=("objective", "dense", "planner"))
    parser.add_argument("task_manifest", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks = _read_tasks(args.task_manifest)
    result = (
        analyze_planner(tasks, args.output_root)
        if args.gate == "planner"
        else analyze_probes(tasks, args.output_root, args.gate)
    )
    result["task_manifest"] = str(args.task_manifest)
    result["output_root"] = str(args.output_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "completed_cells": result["completed_cells"],
                "expected_cells": result["expected_cells"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
