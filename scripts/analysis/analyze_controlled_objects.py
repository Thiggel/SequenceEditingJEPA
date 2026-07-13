from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


GROUP_KEYS = (
    "block",
    "depth",
    "stride",
    "rollout_steps",
    "rollout_all_levels",
    "lambda",
    "representation",
    "latent_dim",
    "ldad_horizon",
    "ldad_weight",
    "objective",
)


def analyze_manifest(manifest_path: Path) -> dict[str, Any]:
    rows = _read_manifest(manifest_path)
    completed = []
    missing = []
    for row in rows:
        metrics_path = Path(row["output_dir"]) / "metrics.json"
        if not metrics_path.is_file():
            missing.append(row["run_name"])
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        probes = _read_current_probes(Path(row["output_dir"]))
        objective = row["objective"]
        default_ldad_horizon = (
            row["rollout_steps"]
            if "controlled_hwm_v1" in manifest_path.name
            and objective.startswith("ldad")
            else "1"
        )
        completed.append(
            {
                **row,
                "latent_dim": row.get("latent_dim") or str(metrics.get("latent_dim", 32)),
                "ldad_horizon": row.get("ldad_horizon")
                or str(metrics.get("ldad_horizon", default_ldad_horizon)),
                "ldad_weight": row.get("ldad_weight")
                or str(
                    metrics.get(
                        "ldad_weight", 1.0 if objective.startswith("ldad") else 0.0
                    )
                ),
                "metrics": metrics,
                "probes": probes,
            }
        )
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in completed:
        groups[tuple(row[key] for key in GROUP_KEYS)].append(row)
    summaries = [
        _summarize_group(key, group_rows)
        for key, group_rows in sorted(groups.items())
    ]
    return {
        "manifest": str(manifest_path),
        "expected_runs": len(rows),
        "completed_runs": len(completed),
        "missing_runs": missing,
        "complete": len(completed) == len(rows),
        "groups": summaries,
    }


def _summarize_group(
    key: tuple[str, ...], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    metrics = [row["metrics"] for row in rows]
    gains = [
        float(value)
        for item in metrics
        for name, value in item.items()
        if name.startswith("eval_level") and "_rollout" in name and name.endswith("_gain")
    ]
    learned = [float(item["eval_learned_receding_success_rate"]) for item in metrics]
    oracle_macro = [
        _first_metric(
            item,
            "eval_oracle_candidate_receding_success_rate",
            "eval_oracle_macro_learned_low_success_rate",
        )
        for item in metrics
    ]
    exact = [
        _first_metric(
            item,
            "eval_symbolic_receding_success_rate",
            "eval_exact_receding_success_rate",
        )
        for item in metrics
    ]
    bounded_cem = _metric_values(metrics, "bounded_cem_receding_success_rate")
    support_cem = _metric_values(metrics, "support_cem_receding_success_rate")
    ldad_accuracy = _metric_values(metrics, "ldad_exact_accuracy")
    action_top1 = [
        float(item["eval_action_top1_accuracy"])
        for item in metrics
        if "eval_action_top1_accuracy" in item
    ]
    support = _metric_values(metrics, "support_energy_auroc")
    reachability = _metric_values(metrics, "reachability_energy_auroc")
    output: dict[str, Any] = dict(zip(GROUP_KEYS, key, strict=True))
    output.update(
        {
            "seeds": sorted(int(row["seed"]) for row in rows),
            "seed_count": len(rows),
            "prediction_loss_mean": mean(
                float(item["eval_prediction_loss"]) for item in metrics
            ),
            "all_horizon_gain_mean": mean(gains),
            "all_horizon_gain_min": min(gains),
            "learned_receding_mean": mean(learned),
            "learned_receding_min": min(learned),
            "oracle_macro_learned_low_mean": mean(oracle_macro),
            "oracle_macro_learned_low_min": min(oracle_macro),
            "exact_receding_min": min(exact),
            "ldad_loss_mean": mean(float(item["eval_ldad_loss"]) for item in metrics),
            "latent_effective_rank_mean": mean(
                float(item["eval_latent_effective_rank"]) for item in metrics
            ),
            "support_energy_auroc_min": min(support) if support else None,
            "reachability_energy_auroc_min": (
                min(reachability) if reachability else None
            ),
            "bounded_cem_min": min(bounded_cem) if bounded_cem else None,
            "support_cem_min": min(support_cem) if support_cem else None,
            "ldad_exact_accuracy_min": min(ldad_accuracy) if ldad_accuracy else None,
            "action_top1_min": min(action_top1) if action_top1 else None,
        }
    )
    probe_rows = [row["probes"] for row in rows if row.get("probes") is not None]
    output["probe_seed_count"] = len(probe_rows)
    output["probe_metrics"] = _summarize_probes(probe_rows)
    output["exact_gate"] = len(rows) == 3 and output["exact_receding_min"] == 1.0
    output["prediction_gate"] = len(rows) == 3 and output["all_horizon_gain_min"] > 0.0
    output["planning_gate"] = len(rows) == 3 and output["learned_receding_min"] >= 0.95
    output["ldad_gate"] = (
        len(rows) == 3
        and output["ldad_exact_accuracy_min"] is not None
        and output["ldad_exact_accuracy_min"] >= 0.5
    )
    output["action_gate"] = (
        len(rows) == 3
        and output["action_top1_min"] is not None
        and output["action_top1_min"] >= 0.95
    )
    return output


def _summarize_probes(probes: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    if not probes:
        return {}
    names = sorted(
        set.intersection(
            *(
                (
                    set(probe["initial"])
                    & set(probe["final"])
                    & set(probe["delta"])
                )
                - {"probe_schema"}
                for probe in probes
            )
        )
    )
    output = {}
    for name in names:
        initial = [float(probe["initial"][name]) for probe in probes]
        final = [float(probe["final"][name]) for probe in probes]
        delta = [float(probe["delta"][name]) for probe in probes]
        output[name] = {
            "initial_mean": mean(initial),
            "final_mean": mean(final),
            "final_min": min(final),
            "delta_mean": mean(delta),
        }
    return output


def _first_metric(item: dict[str, Any], *names: str) -> float:
    for name in names:
        if name in item:
            return float(item[name])
    raise KeyError(f"None of the metric names are present: {names}")


def _metric_values(metrics: list[dict[str, Any]], suffix: str) -> list[float]:
    return [
        float(value)
        for item in metrics
        for name, value in item.items()
        if name.endswith(suffix)
    ]


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _read_current_probes(output_dir: Path) -> dict[str, Any] | None:
    for name in ("probe_eval_v2.json", "probe_eval_v1.json"):
        path = output_dir / name
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("probe_schema") == "controlled_objects_checkpoint_v2":
            return payload
    return None


def _markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# Controlled HWM Sweep",
        "",
        f"Completed: {analysis['completed_runs']}/{analysis['expected_runs']}",
        "",
        "| block | depth | stride | rollout | all levels | lambda | representation | z | LDAD h | LDAD w | objective | seeds | gain min | action top-1 | learned min | oracle-candidate min | symbolic | bounded CEM | support CEM | LDAD exact | pred gate | plan gate |",
        "|---|---:|---:|---:|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for group in analysis["groups"]:
        lines.append(
            "| {block} | {depth} | {stride} | {rollout_steps} | "
            "{rollout_all_levels} | {lambda} | {representation} | {latent_dim} | "
            "{ldad_horizon} | {ldad_weight} | {objective} | {seed_count} | "
            "{all_horizon_gain_min:.4f} | {action_top1} | "
            "{learned_receding_min:.2f} | "
            "{oracle_macro_learned_low_min:.2f} | {exact_receding_min:.2f} | "
            "{bounded} | {support_cem} | {ldad_accuracy} | "
            "{prediction_gate} | {planning_gate} |".format(
                **group,
                action_top1=_format_optional(group["action_top1_min"]),
                bounded=_format_optional(group["bounded_cem_min"]),
                support_cem=_format_optional(group["support_cem_min"]),
                ldad_accuracy=_format_optional(group["ldad_exact_accuracy_min"]),
            )
        )
    probe_groups = [group for group in analysis["groups"] if group["probe_seed_count"]]
    if probe_groups:
        lines.extend(
            [
                "",
                "## Frozen semantic probes",
                "",
                "All values are final means. Parenthesized representation changes are from matched initialization; unbounded endpoint R2 changes remain in JSON only.",
                "",
                "| block | depth | rollout | z | probes | presence | shape | position R2 | relation R2 | endpoint position R2 | endpoint relation R2 | endpoint grid IoU | predicted action | rank |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for group in probe_groups:
            probes = group["probe_metrics"]
            lines.append(
                "| {block} | {depth} | {rollout_steps} | {latent_dim} | "
                "{probe_seed_count} | {presence} | {shape} | {position} | "
                "{relation} | {rollout_position} | {rollout_relation} | {grid} | "
                "{action} | {rank} |".format(
                    **group,
                    presence=_format_probe(probes, "probe_object_presence_balanced_acc"),
                    shape=_format_probe(probes, "probe_shape_balanced_acc"),
                    position=_format_probe(probes, "probe_position_r2"),
                    relation=_format_probe(probes, "probe_relation_r2"),
                    rollout_position=_format_furthest_probe(
                        group, "position_r2"
                    ),
                    rollout_relation=_format_furthest_probe(
                        group, "relation_r2"
                    ),
                    grid=_format_furthest_probe(
                        group, "grid_foreground_iou"
                    ),
                    action=_format_probe(
                        probes, "probe_predicted_delta_transform_balanced_acc"
                    ),
                    rank=_format_probe(probes, "probe_latent_effective_rank"),
                )
            )
    return "\n".join(lines) + "\n"


def _format_optional(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _format_probe(metrics: dict[str, dict[str, float]], name: str) -> str:
    if name not in metrics:
        return "-"
    value = metrics[name]
    return f"{value['final_mean']:.3f} ({value['delta_mean']:+.3f})"


def _format_furthest_probe(group: dict[str, Any], suffix: str) -> str:
    pattern = re.compile(rf"probe_level(\d+)_rollout(\d+)_{re.escape(suffix)}")
    candidates = []
    for name in group["probe_metrics"]:
        match = pattern.fullmatch(name)
        if match:
            level, rollout = (int(value) for value in match.groups())
            horizon = int(group["stride"]) ** level * rollout
            candidates.append((horizon, level, rollout, name))
    if not candidates:
        return "-"
    value = group["probe_metrics"][max(candidates)[-1]]
    return f"{value['final_mean']:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    analysis = analyze_manifest(args.manifest)
    output = args.output or args.manifest.with_suffix(".summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(analysis, indent=2, sort_keys=True), encoding="utf-8")
    output.with_suffix(".md").write_text(_markdown(analysis), encoding="utf-8")
    print(
        json.dumps(
            {
                "completed_runs": analysis["completed_runs"],
                "expected_runs": analysis["expected_runs"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
