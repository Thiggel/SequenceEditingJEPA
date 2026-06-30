from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


VARIANTS = (
    "anchor_h1",
    "action_token",
    "action_local_feature",
    "action_old_local_value",
    "action_old_local_concat",
    "dynamics_affected",
    "dynamics_affected_context",
    "no_temporal",
    "no_progress",
    "no_action_rank",
    "no_terminal_corrupt",
    "no_vicreg",
    "minimal_aux",
    "hier_none",
    "hier_l4",
    "hier_l16",
    "hier_l4_l16_l32",
)

ACTION_CONDITIONING_BY_VARIANT = {
    "anchor_h1": "affected_marker",
    "action_token": "action_token",
    "action_local_feature": "local_action_feature",
    "action_old_local_value": "old_local_value",
    "action_old_local_concat": "old_local_concat",
}

DYNAMICS_BY_VARIANT = {
    "anchor_h1": "uniform",
    "dynamics_affected": "affected",
    "dynamics_affected_context": "affected_context",
}

LOCAL_ORACLE_SCORES = {
    "oracle_goal_changed_cell_raw_euclidean_distance",
    "oracle_goal_affected_context_raw_euclidean_distance",
}


@dataclass(frozen=True)
class Submission:
    kind: str
    train: str | None = None
    eval: str | None = None
    details: dict[str, Any] | None = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Oversight for the H1 recipe sweep.")
    parser.add_argument("--work-root", type=Path, default=Path(os.environ.get("PUZZLE_JEPA_WORK_ROOT", ".")))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--report-root", type=Path, default=Path("../sequence-editing-report"))
    parser.add_argument("--submit-next", action="store_true", default=os.environ.get("H1_RECIPE_SUBMIT_NEXT", "0") == "1")
    parser.add_argument("--cleanup", action="store_true", default=os.environ.get("H1_RECIPE_CLEANUP", "0") == "1")
    args = parser.parse_args()

    run_root = args.work_root / "runs" / "grid_goal_h1_recipe"
    summary = summarize_h1_recipe(run_root)
    submissions: list[Submission] = []
    if args.submit_next:
        submissions.extend(repair_missing_evals(summary))
        if ready_for_second_wave(summary):
            submissions.extend(submit_second_wave(summary))

    cleanup = cleanup_disposable(args.repo_root) if args.cleanup else []
    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "run_root": str(run_root),
        "summary": summary,
        "submissions": [submission.__dict__ for submission in submissions],
        "cleanup": cleanup,
        "slurm": slurm_snapshot(),
    }
    print(json.dumps(record, indent=2, sort_keys=True))
    update_report(args.report_root, record)


def summarize_h1_recipe(run_root: Path) -> dict[str, Any]:
    rows_by_variant: dict[str, list[dict[str, Any]]] = {variant: [] for variant in VARIANTS}
    metrics_by_variant: dict[str, dict[str, Any]] = {}
    checkpoint_by_variant: dict[str, bool] = {}
    for variant in VARIANTS:
        root = run_root / f"grid_goal_h1_recipe_{variant}"
        checkpoint_by_variant[variant] = (root / "checkpoint.pt").is_file()
        metrics_path = root / "metrics.json"
        if metrics_path.is_file():
            metrics_by_variant[variant] = json.loads(metrics_path.read_text())
        for matrix in root.glob("planner_eval_*/planner_matrix.jsonl"):
            for row in read_jsonl(matrix):
                row["variant"] = variant
                row["path"] = str(matrix)
                rows_by_variant[variant].append(row)

    all_rows = [row for rows in rows_by_variant.values() for row in rows]
    best_overall = best_row(all_rows)
    best_local = best_row([row for row in all_rows if row.get("score_mode") in LOCAL_ORACLE_SCORES])
    best_action = best_variant_from_candidates(rows_by_variant, ACTION_CONDITIONING_BY_VARIANT)
    best_dynamics = best_variant_from_candidates(rows_by_variant, DYNAMICS_BY_VARIANT)
    expected_rows = {variant: expected_eval_rows(variant) for variant in VARIANTS}
    row_counts = {variant: len(rows) for variant, rows in rows_by_variant.items()}
    missing_eval_variants = [
        variant
        for variant in VARIANTS
        if checkpoint_by_variant.get(variant)
        and row_counts.get(variant, 0) < expected_rows[variant]
    ]
    incomplete_train_variants = [variant for variant in VARIANTS if not checkpoint_by_variant.get(variant)]
    return {
        "variants": list(VARIANTS),
        "row_counts": row_counts,
        "expected_rows": expected_rows,
        "metrics": metrics_by_variant,
        "checkpoints": checkpoint_by_variant,
        "best_overall": best_overall,
        "best_local_oracle": best_local,
        "best_action_variant": best_action,
        "best_action_conditioning": ACTION_CONDITIONING_BY_VARIANT.get(best_action or "", "affected_marker"),
        "best_dynamics_variant": best_dynamics,
        "best_dynamics_weighting": DYNAMICS_BY_VARIANT.get(best_dynamics or "", "uniform"),
        "missing_eval_variants": missing_eval_variants,
        "incomplete_train_variants": incomplete_train_variants,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError:
                continue
    return rows


def expected_eval_rows(variant: str) -> int:
    planners = 1 if variant == "hier_none" else 2
    transitions = 2
    scores = 10
    depths = 4
    return planners * transitions * scores * depths


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


def best_variant_from_candidates(rows_by_variant: dict[str, list[dict[str, Any]]], candidates: dict[str, str]) -> str | None:
    scored = []
    for variant in candidates:
        rows = [
            row
            for row in rows_by_variant.get(variant, [])
            if row.get("planner") == "mpc_beam"
            and row.get("transition_mode") == "latent_rollout"
            and row.get("score_mode") in LOCAL_ORACLE_SCORES
        ]
        row = best_row(rows) or best_row([r for r in rows_by_variant.get(variant, []) if str(r.get("score_mode", "")).startswith("oracle_goal_")])
        if row is not None:
            scored.append((variant, row))
    if not scored:
        return None
    return min(
        scored,
        key=lambda item: (
            -float(item[1].get("solve_rate", 0.0)),
            float(item[1].get("remaining_hamming_mean", 9999.0)),
        ),
    )[0]


def ready_for_second_wave(summary: dict[str, Any]) -> bool:
    row_counts = summary["row_counts"]
    expected = summary["expected_rows"]
    complete_variants = sum(1 for variant, count in row_counts.items() if count >= expected[variant])
    if complete_variants < max(8, len(VARIANTS) // 2):
        return False
    marker = Path(summary_path_root()) / "h1_recipe_wave2_submitted.json"
    return not marker.exists()


def repair_missing_evals(summary: dict[str, Any]) -> list[Submission]:
    variants = summary.get("missing_eval_variants", [])
    if not variants:
        return []
    indices = ",".join(str(VARIANTS.index(variant)) for variant in variants)
    eval_job = subprocess.check_output(
        [
            "sbatch",
            "--parsable",
            f"--array={indices}%8",
            "--export=ALL,EVAL_OUTPUT_DIR_NAME=planner_eval_h1_recipe_repair",
            "scripts/slurm/run_grid_goal_h1_recipe_eval.slurm",
        ],
        text=True,
    ).strip()
    return [Submission(kind="repair_eval", eval=eval_job, details={"variants": variants})]


def submit_second_wave(summary: dict[str, Any]) -> list[Submission]:
    action = str(summary.get("best_action_conditioning") or "affected_marker")
    dynamics = str(summary.get("best_dynamics_weighting") or "uniform")
    marker_root = Path(summary_path_root())
    marker_root.mkdir(parents=True, exist_ok=True)
    marker = marker_root / "h1_recipe_wave2_submitted.json"
    if marker.exists():
        return []
    export = (
        "ALL,"
        f"H1_WAVE2_ACTION_CONDITIONING={action},"
        f"H1_WAVE2_DYNAMICS_WEIGHTING={dynamics},"
        "TRAIN_MAX_STEPS=20000,"
        "SAVE_EVERY_STEPS=5000,"
        "EVAL_EVERY_STEPS=1000"
    )
    train = subprocess.check_output(
        [
            "sbatch",
            "--parsable",
            "--array=0-7%8",
            f"--export={export}",
            "scripts/slurm/run_grid_goal_h1_wave2_train.slurm",
        ],
        text=True,
    ).strip()
    eval_job = subprocess.check_output(
        [
            "sbatch",
            "--parsable",
            "--array=0-7%8",
            f"--dependency=aftercorr:{train}",
            f"--export=ALL,H1_WAVE2_ACTION_CONDITIONING={action},H1_WAVE2_DYNAMICS_WEIGHTING={dynamics}",
            "scripts/slurm/run_grid_goal_h1_wave2_eval.slurm",
        ],
        text=True,
    ).strip()
    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "action_conditioning": action,
        "dynamics_weighting": dynamics,
        "train": train,
        "eval": eval_job,
        "basis": {
            "best_action_variant": summary.get("best_action_variant"),
            "best_dynamics_variant": summary.get("best_dynamics_variant"),
            "best_local_oracle": summary.get("best_local_oracle"),
        },
    }
    marker.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return [Submission(kind="second_wave", train=train, eval=eval_job, details=payload)]


def summary_path_root() -> str:
    return os.environ.get("PUZZLE_JEPA_WORK_ROOT", ".")


def cleanup_disposable(repo_root: Path) -> list[str]:
    removed = []
    for path in (repo_root / "logs").glob("grid_goal_h1r_train_*_4.err"):
        if path.stat().st_size == 0:
            path.unlink(missing_ok=True)
            removed.append(str(path))
    return removed


def slurm_snapshot() -> str:
    try:
        return subprocess.check_output(
            ["squeue", "-u", os.environ.get("USER", ""), "-o", "%.18i %.12P %.28j %.2t %.12M %.22R"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except Exception as exc:
        return f"squeue unavailable: {exc}"


def update_report(report_root: Path, record: dict[str, Any]) -> None:
    report_root.mkdir(parents=True, exist_ok=True)
    log_path = report_root / "LOG.md"
    with log_path.open("a") as handle:
        handle.write(f"\n## {record['time']} H1 Recipe Oversight\n\n")
        summary = record["summary"]
        best = summary.get("best_local_oracle") or summary.get("best_overall")
        handle.write(f"- Run root: `{record['run_root']}`\n")
        handle.write(f"- Best action conditioning: `{summary.get('best_action_conditioning')}` from `{summary.get('best_action_variant')}`\n")
        handle.write(f"- Best dynamics weighting: `{summary.get('best_dynamics_weighting')}` from `{summary.get('best_dynamics_variant')}`\n")
        if best:
            handle.write(
                f"- Best row: `{best.get('variant')}` / `{best.get('planner')}` / "
                f"`{best.get('transition_mode')}` / `{best.get('score_mode')}`, depth "
                f"`{best.get('beam_depth')}`, solve rate `{best.get('solve_rate')}`, "
                f"remaining Hamming `{best.get('remaining_hamming_mean')}`\n"
            )
        for submission in record.get("submissions", []):
            handle.write(f"- Submitted `{submission['kind']}`: `{submission}`\n")


if __name__ == "__main__":
    main()
