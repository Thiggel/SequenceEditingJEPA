from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


PROBE_METRICS = (
    "probe_latent_effective_rank",
    "probe_latent_std_mean",
    "probe_object_presence_balanced_acc",
    "probe_shape_balanced_acc",
    "probe_position_r2",
    "probe_area_r2",
    "probe_relation_r2",
    "probe_pixel_decoder_acc",
    "probe_pixel_decoder_foreground_iou",
)
EVAL_METRICS = (
    "eval_prediction_loss",
    "eval_rollout_loss",
    "eval_latent_effective_rank",
    "eval_manual_low_level_subgoal_success_rate",
    "eval_learned_receding_success_rate",
    "eval_oracle_candidate_receding_success_rate",
    "eval_bounded_cem_receding_success_rate",
    "eval_support_cem_receding_success_rate",
    "eval_mppi_receding_success_rate",
    "eval_retrieval_final_pixel_error",
    "eval_bounded_cem_final_pixel_error",
    "eval_support_cem_final_pixel_error",
    "eval_mppi_final_pixel_error",
)


def analyze(task_manifest: Path, output_root: Path) -> dict[str, Any]:
    with task_manifest.open(newline="", encoding="utf-8") as handle:
        tasks = list(csv.DictReader(handle, delimiter="\t"))
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    missing = []
    for task in tasks:
        variance_tag = task["variance_weight"].replace(".", "p")
        covariance_tag = task["covariance_weight"].replace(".", "p")
        run_name = f"v{variance_tag}_c{covariance_tag}_s{task['seed']}_h110100"
        path = output_root / run_name / "probe_eval_v4.json"
        if not path.is_file():
            missing.append(run_name)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        grouped[(task["variance_weight"], task["covariance_weight"])].append(payload)

    groups = []
    for (variance, covariance), rows in sorted(grouped.items(), key=_numeric_key):
        summary: dict[str, Any] = {
            "variance_weight": float(variance),
            "covariance_weight": float(covariance),
            "seeds": sorted(int(row["seed"]) for row in rows),
        }
        for metric in PROBE_METRICS:
            values = [float(row["final"][metric]) for row in rows if metric in row["final"]]
            if values:
                summary[f"{metric}_mean"] = mean(values)
                initial = [
                    float(row["initial"][metric])
                    for row in rows
                    if metric in row["initial"]
                ]
                if initial:
                    summary[f"{metric}_delta_mean"] = mean(values) - mean(initial)
        for metric in EVAL_METRICS:
            values = [
                float(row["evaluation"][metric])
                for row in rows
                if metric in row["evaluation"]
            ]
            if values:
                summary[f"{metric}_mean"] = mean(values)
        mixed_count = [
            float(row["final_mixed_load"]["probe_object_count_balanced_acc"])
            for row in rows
        ]
        summary["probe_mixed_load_object_count_balanced_acc_mean"] = mean(mixed_count)
        groups.append(summary)
    return {
        "schema": "controlled_objects_valid_hwm_vicreg_v1",
        "task_manifest": str(task_manifest),
        "output_root": str(output_root),
        "expected_cells": len(tasks),
        "completed_cells": sum(len(rows) for rows in grouped.values()),
        "missing_runs": missing,
        "groups": groups,
    }


def _numeric_key(item: tuple[tuple[str, str], list[dict[str, Any]]]) -> tuple[float, float]:
    return float(item[0][0]), float(item[0][1])


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
