from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


METRICS = (
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
    ("evaluation", "eval_action_top1_accuracy"),
    ("evaluation", "eval_level1_one_step_mse"),
    ("evaluation", "eval_level1_primitive_rollout_mse"),
    ("evaluation", "eval_level1_support_energy_auroc"),
    ("evaluation", "eval_level1_reachability_energy_auroc"),
    ("evaluation", "eval_level2_one_step_mse"),
    ("evaluation", "eval_level2_primitive_rollout_mse"),
    ("evaluation", "eval_level2_support_energy_auroc"),
    ("evaluation", "eval_level2_reachability_energy_auroc"),
    ("evaluation", "eval_ldad_per_action_exact_accuracy"),
)


def analyze(task_manifest: Path, output_root: Path) -> dict[str, Any]:
    with task_manifest.open(newline="", encoding="utf-8") as handle:
        tasks = list(csv.DictReader(handle, delimiter="\t"))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing = []
    for task in tasks:
        run_name = f"{task['objective']}_s{task['seed']}_joint_h110100"
        path = output_root / run_name / "probe_eval_v5.json"
        if not path.is_file():
            missing.append(run_name)
            continue
        grouped[task["objective"]].append(
            json.loads(path.read_text(encoding="utf-8"))
        )

    groups = []
    for objective in sorted(grouped):
        rows = grouped[objective]
        summary: dict[str, Any] = {
            "objective": objective,
            "seeds": sorted(int(row["seed"]) for row in rows),
        }
        for section, metric in METRICS:
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
        "schema": "controlled_objects_joint_hwm_objectives_v1",
        "task_manifest": str(task_manifest),
        "output_root": str(output_root),
        "expected_cells": len(tasks),
        "completed_cells": sum(len(rows) for rows in grouped.values()),
        "missing_runs": missing,
        "groups": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task_manifest", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.task_manifest, args.output_root)
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
