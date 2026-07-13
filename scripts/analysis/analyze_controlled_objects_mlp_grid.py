from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


SCHEDULES = ("h1", "h14", "h1416", "h124")
GROUP_KEYS = ("architecture", "schedule", "rollout_steps", "weighting", "object_count")
EVAL_METRICS = (
    "eval_prediction_loss",
    "eval_teacher_forcing_loss",
    "eval_rollout_loss",
    "eval_latent_effective_rank",
    "eval_learned_receding_success_rate",
    "eval_bounded_cem_receding_success_rate",
    "eval_support_cem_receding_success_rate",
    "eval_mppi_receding_success_rate",
    "eval_retrieval_final_pixel_error",
    "eval_bounded_cem_final_pixel_error",
    "eval_support_cem_final_pixel_error",
    "eval_mppi_final_pixel_error",
)
PROBE_METRICS = (
    "probe_object_count_balanced_acc",
    "probe_object_presence_balanced_acc",
    "probe_shape_balanced_acc",
    "probe_motion_policy_balanced_acc",
    "probe_position_r2",
    "probe_area_r2",
    "probe_relation_r2",
    "probe_delta_action_row_balanced_acc",
    "probe_delta_action_col_balanced_acc",
    "probe_delta_action_color_balanced_acc",
    "probe_pixel_decoder_acc",
    "probe_pixel_decoder_foreground_iou",
    "probe_latent_effective_rank",
)


def analyze(task_manifest: Path, output_root: Path) -> dict[str, Any]:
    with task_manifest.open(newline="", encoding="utf-8") as handle:
        tasks = list(csv.DictReader(handle, delimiter="\t"))
    records = []
    missing = []
    for task in tasks:
        common = (
            f"a{task['architecture']}_r{task['rollout_steps']}_"
            f"w{task['weighting']}_n{task['object_count']}_s{task['seed']}"
        )
        for schedule in SCHEDULES:
            run_name = f"{common}_{schedule}"
            run_dir = output_root / run_name
            metrics_path = run_dir / "metrics.json"
            probe_path = run_dir / "probe_eval_v3.json"
            if not metrics_path.is_file() or not probe_path.is_file():
                missing.append(run_name)
                continue
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            probes = json.loads(probe_path.read_text(encoding="utf-8"))["final"]
            records.append(
                {
                    **task,
                    "schedule": schedule,
                    "metrics": metrics,
                    "probes": probes,
                }
            )
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[tuple(record[key] for key in GROUP_KEYS)].append(record)
    summaries = []
    for key, rows in sorted(groups.items()):
        summary: dict[str, Any] = dict(zip(GROUP_KEYS, key, strict=True))
        summary["seeds"] = sorted(int(row["seed"]) for row in rows)
        for name in EVAL_METRICS:
            values = [float(row["metrics"][name]) for row in rows if name in row["metrics"]]
            if values:
                summary[f"{name}_mean"] = mean(values)
                summary[f"{name}_min"] = min(values)
        for name in PROBE_METRICS:
            values = [float(row["probes"][name]) for row in rows if name in row["probes"]]
            if values:
                summary[f"{name}_mean"] = mean(values)
        summaries.append(summary)
    return {
        "schema": "controlled_objects_mlp_grid_v1",
        "task_manifest": str(task_manifest),
        "output_root": str(output_root),
        "expected_cells": len(tasks) * len(SCHEDULES),
        "completed_cells": len(records),
        "missing_runs": missing,
        "groups": summaries,
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
