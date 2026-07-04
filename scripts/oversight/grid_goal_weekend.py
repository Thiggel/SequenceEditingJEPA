from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Six-hour oversight for the counterfactual/editable Grid Goal weekend wave.")
    parser.add_argument("--manifest", type=Path, default=Path("scripts/experiments/grid_goal_weekend_manifest.json"))
    parser.add_argument("--work-root", type=Path, default=Path(os.environ.get("PUZZLE_JEPA_WORK_ROOT", ".")))
    parser.add_argument("--report-root", type=Path, default=Path("../sequence-editing-report"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--run-suffix",
        default=os.environ.get("GRID_GOAL_WEEKEND_RUN_SUFFIX", os.environ.get("RUN_SUFFIX", "")),
        help="Suffix appended to grid_goal_weekend run directories, for replacement waves.",
    )
    parser.add_argument("--repair-evals", action="store_true", default=os.environ.get("GRID_GOAL_WEEKEND_REPAIR_EVALS", "1") == "1")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    validate_delta_pairing(manifest)
    run_root = args.work_root / "runs" / str(manifest["run_family"])
    variants = all_variants(manifest)
    summary = summarize_variants(run_root, variants, args.run_suffix)
    submissions = repair_incomplete_evals(summary, variants, args.run_suffix) if args.repair_evals else []
    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "manifest": str(args.manifest),
        "run_root": str(run_root),
        "run_suffix": args.run_suffix,
        "oversight_cadence_hours": manifest.get("oversight", {}).get("cadence_hours"),
        "required_diagnostics": manifest.get("oversight", {}).get("required_diagnostics", []),
        "repair_evals": args.repair_evals,
        "research_questions": manifest.get("research_questions", []),
        "summary": summary,
        "slurm": slurm_snapshot(),
        "submissions": submissions,
        "insights": derive_insights(summary),
    }
    print(json.dumps(record, indent=2, sort_keys=True))
    update_report(args.report_root, args.repo_root, manifest, record)


def validate_delta_pairing(manifest: dict[str, Any]) -> None:
    for stage in manifest.get("stages", []):
        if not stage.get("requires_paired_latents"):
            continue
        variants = set(stage.get("variants", []))
        for base in stage.get("base_variants", []):
            missing = [name for name in (f"{base}_grid", f"{base}_single") if name not in variants]
            if missing:
                raise ValueError(f"Delta stage {stage.get('id')} is missing paired variants: {missing}")


def all_variants(manifest: dict[str, Any]) -> list[str]:
    variants: list[str] = []
    for stage in manifest.get("stages", []):
        for variant in stage.get("variants", []):
            if variant not in variants:
                variants.append(str(variant))
    return variants


def summarize_variants(run_root: Path, variants: list[str], run_suffix: str = "") -> dict[str, Any]:
    by_variant = {}
    for variant in variants:
        root = run_root / f"grid_goal_weekend_{variant}{run_suffix}"
        rows = []
        for matrix in root.glob("planner_eval_*/planner_matrix.jsonl"):
            rows.extend(read_jsonl(matrix))
        metrics = {}
        metrics_path = root / "metrics.json"
        if metrics_path.is_file():
            try:
                metrics = json.loads(metrics_path.read_text())
            except json.JSONDecodeError:
                metrics = {}
        by_variant[variant] = {
            "checkpoint": (root / "checkpoint.pt").is_file(),
            "rows": len(rows),
            "best": best_row(rows),
            "metrics": metrics,
        }
    return {
        "variants": by_variant,
        "best_overall": best_row([row for item in by_variant.values() for row in rows_from_best(item)]),
        "completed_checkpoints": sum(1 for item in by_variant.values() if item["checkpoint"]),
        "total_rows": sum(int(item["rows"]) for item in by_variant.values()),
    }


def rows_from_best(item: dict[str, Any]) -> list[dict[str, Any]]:
    best = item.get("best")
    return [] if best is None else [best]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            row["path"] = str(path)
            rows.append(row)
    return rows


def best_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return min(
        rows,
        key=lambda row: (
            -float(row.get("solve_rate", 0.0)),
            float(row.get("remaining_hamming_mean", 9999.0)),
            int(row.get("beam_depth", 9999)),
        ),
    )


def derive_insights(summary: dict[str, Any]) -> list[str]:
    insights = []
    variants = summary.get("variants", {})
    best = summary.get("best_overall")
    if best:
        insights.append(
            "Best current row: "
            f"{best.get('planner')} {best.get('transition_mode')} {best.get('score_mode')} "
            f"solve={best.get('solved')}/{best.get('examples')} h={best.get('remaining_hamming_mean')}"
        )
    delta_rows = [item for name, item in variants.items() if str(name).startswith("D")]
    if delta_rows and any(item.get("rows", 0) for item in delta_rows):
        insights.append("Delta branch has eval rows; compare every grid variant against its single-CLS paired variant before promoting.")
    waypoint_rows = [item for name, item in variants.items() if "waypoint" in str(name)]
    if waypoint_rows and any(item.get("rows", 0) for item in waypoint_rows):
        insights.append("Waypoint rows are present; prioritize predicted-waypoint versus oracle-waypoint gap before terminal predicted-goal variants.")
        macro_waypoint = [
            item
            for item in variants.values()
            if (best := item.get("best")) and best.get("planner") == "waypoint_hierarchical_cem"
        ]
        if macro_waypoint:
            insights.append("Waypoint hierarchy has macro CEM/MPPI rows; use those, not flat waypoint_beam rows, to judge hierarchical planning.")
        else:
            insights.append("Current waypoint rows may be flat-only; do not treat waypoint_beam solves as evidence that hierarchical MPC works.")
        insights.append(
            "If predicted-waypoint solve rate is still zero, inspect waypoint quality directly: latent alignment to oracle future waypoints, Hamming progress after one tracked chunk, and trackability distance."
        )
    return insights


def repair_incomplete_evals(summary: dict[str, Any], variants: list[str], run_suffix: str = "") -> list[dict[str, Any]]:
    submissions = []
    for index, variant in enumerate(variants):
        item = summary["variants"].get(variant, {})
        if not item.get("checkpoint") or int(item.get("rows", 0)) > 0:
            continue
        job = subprocess.check_output(
            [
                "sbatch",
                "--parsable",
                f"--export=ALL,VARIANT_INDEX={index},RUN_SUFFIX={run_suffix},GRID_GOAL_WEEKEND_RUN_SUFFIX={run_suffix}",
                "scripts/slurm/run_grid_goal_weekend_eval.slurm",
            ],
            text=True,
        ).strip()
        submissions.append({"variant": variant, "eval_job": job, "reason": "checkpoint exists but no eval rows"})
    return submissions


def slurm_snapshot() -> str:
    try:
        return subprocess.check_output(
            ["squeue", "-u", os.environ.get("USER", ""), "-o", "%.18i %.9P %.32j %.8T %.10M %.10l %R"],
            text=True,
        )
    except Exception as exc:  # pragma: no cover - cluster dependent
        return f"slurm snapshot unavailable: {exc}"


def update_report(report_root: Path, repo_root: Path, manifest: dict[str, Any], record: dict[str, Any]) -> None:
    report_root.mkdir(parents=True, exist_ok=True)
    now = record["time"]
    lines = [
        "# Current Experiments",
        "",
        f"Last updated: {now}",
        "",
        "## Active: Counterfactual Editable Weekend Wave",
        "",
        "Research questions:",
        *[f"- {question}" for question in manifest.get("research_questions", [])],
        "",
        "Current summary:",
        f"- run suffix: `{record.get('run_suffix', '')}`",
        f"- oversight cadence: `{record.get('oversight_cadence_hours')}` hours",
        f"- repair evals enabled: `{record.get('repair_evals')}`",
        f"- completed checkpoints: {record['summary']['completed_checkpoints']}",
        f"- total planner rows: {record['summary']['total_rows']}",
        "",
        "Insights:",
        *[f"- {item}" for item in record.get("insights", [])],
        "",
        "Required oversight diagnostics:",
        *[f"- {item}" for item in record.get("required_diagnostics", [])],
        "",
        "Variant table:",
        "",
        "| Variant | Checkpoint | Rows | Best result |",
        "|---|---:|---:|---|",
    ]
    for variant, item in record["summary"]["variants"].items():
        best = item.get("best")
        if best:
            best_text = (
                f"{best.get('planner')} / {best.get('transition_mode')} / {best.get('score_mode')} "
                f"= {best.get('solved')}/{best.get('examples')}, h {best.get('remaining_hamming_mean')}"
            )
        else:
            best_text = ""
        lines.append(f"| `{variant}` | `{bool(item.get('checkpoint'))}` | `{item.get('rows', 0)}` | {best_text} |")
    text = "\n".join(lines) + "\n"
    (report_root / "CURRENT_EXPERIMENTS.md").write_text(text)
    (report_root / "STATUS.md").write_text(text)
    log_path = report_root / "LOG.md"
    with log_path.open("a") as handle:
        handle.write(f"\n## {now} Weekend Oversight\n\n")
        for insight in record.get("insights", []):
            handle.write(f"- {insight}\n")
        if record.get("submissions"):
            handle.write(f"- Submitted repairs: `{record['submissions']}`\n")
    docs_current = repo_root / "docs" / "CURRENT_EXPERIMENTS.md"
    if docs_current.parent.is_dir():
        docs_current.write_text("# Current Experiments\n\nSource of truth: `../sequence-editing-report/CURRENT_EXPERIMENTS.md`.\n\n" + text)


if __name__ == "__main__":
    main()
