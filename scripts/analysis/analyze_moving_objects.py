from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


KEYS = (
    "probe_object_count_balanced_acc",
    "probe_shape_count_mae",
    "probe_shape_count_r2",
    "probe_color_count_mae",
    "probe_velocity_count_mae",
    "probe_velocity_count_r2",
    "probe_relations_mae",
    "probe_grid_foreground_iou",
    "probe_latent_effective_rank",
)


def analyze(root: Path) -> dict[str, Any]:
    runs = []
    for metrics_path in sorted(root.glob("motion_*/metrics.jsonl")):
        rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
        if len(rows) < 2:
            continue
        initial, final = rows[0], rows[-1]
        if int(final.get("step", 0)) <= 0:
            continue
        runs.append(
            {
                "run": metrics_path.parent.name,
                "latent_dim": int(final["latent_dim"]),
                "max_objects": int(final["max_objects"]),
                "seed": int(final["seed"]),
                "step": int(final["step"]),
                "absolute": {key: final.get(key) for key in KEYS},
                "delta": {key: _delta(final.get(key), initial.get(key)) for key in KEYS},
            }
        )
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        groups[(run["latent_dim"], run["max_objects"])].append(run)
    aggregates = []
    for (latent_dim, max_objects), members in sorted(groups.items()):
        aggregate: dict[str, Any] = {
            "latent_dim": latent_dim,
            "max_objects": max_objects,
            "n": len(members),
            "seeds": sorted(member["seed"] for member in members),
        }
        for mode in ("absolute", "delta"):
            aggregate[mode] = {}
            for key in KEYS:
                values = [member[mode][key] for member in members if member[mode][key] is not None]
                aggregate[mode][key] = {
                    "mean": mean(values) if values else None,
                    "std": pstdev(values) if values else None,
                }
        aggregates.append(aggregate)
    return {"schema": "moving_objects_summary_v1", "runs": runs, "aggregates": aggregates}


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Moving-Object Bottleneck Summary",
        "",
        "Trained-minus-initial metrics; lower is better for MAE columns.",
        "",
        "| z | max objects | n | dCount bal | dShape R2 | dVelocity R2 | dRelation MAE | dGrid fg IoU | rank |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["aggregates"]:
        delta = row["delta"]
        absolute = row["absolute"]
        lines.append(
            "| {z} | {objects} | {n} | {count} | {shape} | {velocity} | {relation} | {grid} | {rank} |".format(
                z=row["latent_dim"], objects=row["max_objects"], n=row["n"],
                count=_format(delta["probe_object_count_balanced_acc"]),
                shape=_format(delta["probe_shape_count_r2"]),
                velocity=_format(delta["probe_velocity_count_r2"]),
                relation=_format(delta["probe_relations_mae"]),
                grid=_format(delta["probe_grid_foreground_iou"]),
                rank=_format(absolute["probe_latent_effective_rank"]),
            )
        )
    return "\n".join(lines) + "\n"


def _delta(final: Any, initial: Any) -> float | None:
    if final is None or initial is None:
        return None
    return float(final) - float(initial)


def _format(value: dict[str, float | None]) -> str:
    if value["mean"] is None:
        return ""
    return f"{value['mean']:+.4f} +/- {value['std']:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    summary = analyze(args.root)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True))
    args.markdown_output.write_text(render_markdown(summary))
    print(json.dumps({"complete_runs": len(summary["runs"]), "groups": len(summary["aggregates"])}))


if __name__ == "__main__":
    main()
