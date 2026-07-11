from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


KEYS = (
    "probe_object_count_balanced_acc",
    "raw_probe_object_count_balanced_acc",
    "probe_visible_object_count_balanced_acc",
    "raw_probe_visible_object_count_balanced_acc",
    "probe_scene_object_count_balanced_acc",
    "raw_probe_scene_object_count_balanced_acc",
    "probe_rollout_object_count_balanced_acc",
    "probe_shape_count_mae",
    "probe_shape_count_r2",
    "raw_probe_shape_count_r2",
    "probe_rollout_shape_count_r2",
    "probe_color_count_r2",
    "raw_probe_color_count_r2",
    "probe_rollout_color_count_r2",
    "probe_color_count_mae",
    "probe_velocity_count_mae",
    "probe_velocity_count_r2",
    "raw_probe_velocity_count_r2",
    "probe_rollout_velocity_count_r2",
    "probe_angular_velocity_count_r2",
    "raw_probe_angular_velocity_count_r2",
    "probe_rollout_angular_velocity_count_r2",
    "probe_relations_mae",
    "probe_relations_r2",
    "raw_probe_relations_r2",
    "probe_rollout_relations_r2",
    "probe_completion_mae",
    "probe_completion_r2",
    "raw_probe_completion_r2",
    "probe_rollout_completion_r2",
    "probe_bound_shape_acc",
    "raw_probe_bound_shape_acc",
    "probe_rollout_bound_shape_acc",
    "probe_bound_shape_r2",
    "raw_probe_bound_shape_r2",
    "probe_rollout_bound_shape_r2",
    "probe_bound_velocity_r2",
    "raw_probe_bound_velocity_r2",
    "probe_rollout_bound_velocity_r2",
    "probe_bound_angular_velocity_r2",
    "raw_probe_bound_angular_velocity_r2",
    "probe_rollout_bound_angular_velocity_r2",
    "probe_bound_position_r2",
    "raw_probe_bound_position_r2",
    "probe_rollout_bound_position_r2",
    "probe_bound_completion_r2",
    "raw_probe_bound_completion_r2",
    "probe_rollout_bound_completion_r2",
    "probe_grid_foreground_iou",
    "probe_latent_std_mean",
    "probe_latent_effective_rank",
    "train_prediction_loss",
)
DYNAMICS_KEYS = (
    "dynamics_pixel_change_rate",
    "dynamics_prediction_squared_error",
    "dynamics_identity_squared_error",
    "dynamics_prediction_gain",
    "dynamics_predictor_win",
    "dynamics_transition_to_variance_ratio",
)


def analyze(root: Path, run_names: set[str] | None = None) -> dict[str, Any]:
    runs = []
    for metrics_path in sorted(root.glob("motion_*/metrics.jsonl")):
        if run_names is not None and metrics_path.parent.name not in run_names:
            continue
        rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
        if len(rows) < 2:
            continue
        initial, final = rows[0], rows[-1]
        if int(final.get("step", 0)) <= 0:
            continue
        probe_path = metrics_path.parent / "probe_eval_v4.json"
        probe_payload = json.loads(probe_path.read_text()) if probe_path.exists() else None
        probe_initial = probe_payload["initial"] if probe_payload is not None else initial
        probe_final = probe_payload["final"] if probe_payload is not None else final
        run = {
            "run": metrics_path.parent.name,
            "data": str(final.get("data", "unknown")),
            "objective": str(final.get("objective", "unknown")),
            "latent_dim": int(final["latent_dim"]),
            "max_objects": int(final["max_objects"]),
            "seed": int(final["seed"]),
            "step": int(final["step"]),
            "probe_source": "probe_eval_v4.json" if probe_payload is not None else "metrics.jsonl",
            "absolute": {
                key: probe_final.get(key, final.get(key)) if key != "train_prediction_loss" else final.get(key)
                for key in KEYS
            },
            "delta": {
                key: (
                    _delta(probe_final.get(key), probe_initial.get(key))
                    if key != "train_prediction_loss" and key in probe_final
                    else _delta(final.get(key), initial.get(key))
                )
                for key in KEYS
            },
        }
        dynamics_path = metrics_path.parent / "dynamics_eval_v1.json"
        if dynamics_path.exists():
            dynamics = json.loads(dynamics_path.read_text())
            final_dynamics = dict(dynamics["final"])
            final_dynamics["dynamics_predictor_win"] = float(
                final_dynamics["dynamics_prediction_squared_error"]
                < final_dynamics["dynamics_identity_squared_error"]
            )
            run["dynamics_final"] = {key: final_dynamics.get(key) for key in DYNAMICS_KEYS}
            run["dynamics_delta"] = {key: dynamics["delta"].get(key) for key in DYNAMICS_KEYS}
        runs.append(run)
    groups: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        groups[(run["data"], run["objective"], run["latent_dim"], run["max_objects"])].append(run)
    aggregates = []
    for (data, objective, latent_dim, max_objects), members in sorted(groups.items()):
        aggregate: dict[str, Any] = {
            "data": data,
            "objective": objective,
            "latent_dim": latent_dim,
            "max_objects": max_objects,
            "n": len(members),
            "probe_v4_n": sum(member["probe_source"] == "probe_eval_v4.json" for member in members),
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
        for mode in ("dynamics_final", "dynamics_delta"):
            aggregate[mode] = {}
            for key in DYNAMICS_KEYS:
                values = [member[mode][key] for member in members if mode in member and member[mode][key] is not None]
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
        "| data | objective | z | max objects | n | dCount bal | dShape R2 | dVelocity R2 | dRelation MAE | dGrid fg IoU | rank |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["aggregates"]:
        delta = row["delta"]
        absolute = row["absolute"]
        lines.append(
            "| {data} | {objective} | {z} | {objects} | {n} | {count} | {shape} | {velocity} | {relation} | {grid} | {rank} |".format(
                data=row["data"], objective=row["objective"],
                z=row["latent_dim"], objects=row["max_objects"], n=row["n"],
                count=_format(delta["probe_object_count_balanced_acc"]),
                shape=_format(delta["probe_shape_count_r2"]),
                velocity=_format(delta["probe_velocity_count_r2"]),
                relation=_format(delta["probe_relations_mae"]),
                grid=_format(delta["probe_grid_foreground_iou"]),
                rank=_format(absolute["probe_latent_effective_rank"]),
            )
        )
    lines.extend(
        [
            "",
            "Final color-indexed object binding, learned/raw/one-step-rollout.",
            "",
            "| data | objective | z | max objects | v4 n | Shape acc | Shape R2 | Velocity R2 | Angular R2 | Position R2 | Completion R2 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["aggregates"]:
        absolute = row["absolute"]
        lines.append(
            "| {data} | {objective} | {z} | {objects} | {probe_v4_n} | {shape_acc} | {shape} | {velocity} | {angular} | {position} | {completion} |".format(
                data=row["data"], objective=row["objective"],
                z=row["latent_dim"], objects=row["max_objects"],
                probe_v4_n=row["probe_v4_n"],
                shape_acc=_triple_suffix(absolute, "bound_shape", "acc"),
                shape=_triple(absolute, "bound_shape"),
                velocity=_triple(absolute, "bound_velocity"),
                angular=_triple(absolute, "bound_angular_velocity"),
                position=_triple(absolute, "bound_position"),
                completion=_triple(absolute, "bound_completion"),
            )
        )
    lines.extend(
        [
            "",
            "Final absolute learned/raw/one-step-rollout R2; count is learned/raw balanced accuracy.",
            "",
            "| data | objective | z | max objects | Visible count | Scene count | Shape R2 | Color R2 | Velocity R2 | Angular R2 | Relation R2 | Completion R2 | fg IoU | rank |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["aggregates"]:
        absolute = row["absolute"]
        lines.append(
            "| {data} | {objective} | {z} | {objects} | {count} | {scene} | {shape} | {color} | {velocity} | {angular} | {relation} | {completion} | {grid} | {rank} |".format(
                data=row["data"], objective=row["objective"],
                z=row["latent_dim"], objects=row["max_objects"],
                count=_pair(absolute, "probe_object_count_balanced_acc", "raw_probe_object_count_balanced_acc"),
                scene=_pair(
                    absolute,
                    "probe_scene_object_count_balanced_acc",
                    "raw_probe_scene_object_count_balanced_acc",
                ),
                shape=_triple(absolute, "shape_count"),
                color=_triple(absolute, "color_count"),
                velocity=_triple(absolute, "velocity_count"),
                angular=_triple(absolute, "angular_velocity_count"),
                relation=_triple(absolute, "relations"),
                completion=_triple(absolute, "completion"),
                grid=_mean(absolute["probe_grid_foreground_iou"]),
                rank=_mean(absolute["probe_latent_effective_rank"]),
            )
        )
    lines.extend(
        [
            "",
            "Final latent dynamics. Positive gain means the predictor beats identity persistence.",
            "",
            "| data | objective | z | max objects | pixel change | predictor MSE | identity MSE | identity-predictor | wins | transition/variance |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["aggregates"]:
        dynamics = row["dynamics_final"]
        lines.append(
            "| {data} | {objective} | {z} | {objects} | {pixel} | {prediction} | {identity} | {gain} | {wins} | {ratio} |".format(
                data=row["data"], objective=row["objective"],
                z=row["latent_dim"], objects=row["max_objects"],
                pixel=_mean(dynamics["dynamics_pixel_change_rate"]),
                prediction=_scientific(dynamics["dynamics_prediction_squared_error"]),
                identity=_scientific(dynamics["dynamics_identity_squared_error"]),
                gain=_scientific(dynamics["dynamics_prediction_gain"]),
                wins=_mean(dynamics["dynamics_predictor_win"]),
                ratio=_mean(dynamics["dynamics_transition_to_variance_ratio"]),
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


def _mean(value: dict[str, float | None]) -> str:
    return "" if value["mean"] is None else f"{value['mean']:.3f}"


def _scientific(value: dict[str, float | None]) -> str:
    return "" if value["mean"] is None else f"{value['mean']:.2e}"


def _pair(values: dict[str, dict[str, float | None]], learned: str, raw: str) -> str:
    return f"{_mean(values[learned])}/{_mean(values[raw])}"


def _triple(values: dict[str, dict[str, float | None]], stem: str) -> str:
    return "/".join(
        _mean(values[key])
        for key in (f"probe_{stem}_r2", f"raw_probe_{stem}_r2", f"probe_rollout_{stem}_r2")
    )


def _triple_suffix(
    values: dict[str, dict[str, float | None]], stem: str, suffix: str
) -> str:
    return "/".join(
        _mean(values[key])
        for key in (
            f"probe_{stem}_{suffix}",
            f"raw_probe_{stem}_{suffix}",
            f"probe_rollout_{stem}_{suffix}",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    run_names = None
    if args.manifest is not None:
        with args.manifest.open(newline="") as handle:
            run_names = {row["run_name"] for row in csv.DictReader(handle, delimiter="\t")}
    summary = analyze(args.root, run_names)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True))
    args.markdown_output.write_text(render_markdown(summary))
    print(json.dumps({"complete_runs": len(summary["runs"]), "groups": len(summary["aggregates"])}))


if __name__ == "__main__":
    main()
