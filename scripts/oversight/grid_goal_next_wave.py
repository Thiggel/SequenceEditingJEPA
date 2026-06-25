from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


STAGES = (
    "goal_conditioning",
    "dense_horizon",
    "hierarchy_levels",
    "predictor_delta_topk",
    "ranking_losses",
    "hierarchical_planning",
    "policy_prior",
)

STAGE_VARIANTS = {
    "goal_conditioning": ("G0_context", "G1_initial_current", "G2_initial_current_oracle_progress"),
    "dense_horizon": ("DK2", "DK4", "DK8", "DK16", "DK32"),
    "hierarchy_levels": ("H_empty", "H2", "H2_4", "H2_4_8", "H2_4_8_16", "H2_4_8_16_32"),
    "predictor_delta_topk": ("P0_separate_delta", "P1_shared_delta", "P2_shared_no_delta", "P3_shared_delta_topk_train"),
    "ranking_losses": (
        "R_pred_progress",
        "R_oracle_progress",
        "R_both_progress",
        "R_no_progress",
        "R_pairwise",
        "R_listwise",
        "R_no_action_rank",
    ),
    "hierarchical_planning": ("BASE",),
    "policy_prior": ("PP0_no_prior", "PP1_pairwise_prior", "PP2_listwise_prior", "PP3_listwise_prior_strong"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid-Goal JEPA staged experiment oversight.")
    parser.add_argument("--work-root", type=Path, default=Path(os.environ.get("PUZZLE_JEPA_WORK_ROOT", ".")))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--report-root", type=Path, default=Path("../sequence-editing-report"))
    parser.add_argument("--stage", choices=STAGES, default=os.environ.get("GRID_GOAL_STAGE", "goal_conditioning"))
    parser.add_argument("--submit-next", action="store_true", default=os.environ.get("OVERSIGHT_SUBMIT_NEXT", "0") == "1")
    parser.add_argument("--cleanup", action="store_true", default=os.environ.get("OVERSIGHT_CLEANUP", "0") == "1")
    parser.add_argument("--delete-checkpoints", action="store_true", default=os.environ.get("OVERSIGHT_DELETE_CHECKPOINTS", "0") == "1")
    args = parser.parse_args()

    run_root = args.work_root / "runs" / "grid_goal_next_wave"
    summary = summarize_runs(run_root)
    slurm = slurm_snapshot()
    cleanup = cleanup_disposable(args.repo_root, args.work_root, delete_checkpoints=args.delete_checkpoints) if args.cleanup else []
    maybe_submit = None
    if args.submit_next and stage_complete(summary, args.stage):
        stage = next_stage(args.stage)
        if stage is not None:
            maybe_submit = submit_stage(stage)

    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "stage": args.stage,
        "run_root": str(run_root),
        "summary": summary,
        "slurm": slurm,
        "cleanup": cleanup,
        "submitted": maybe_submit,
    }
    print(json.dumps(record, indent=2, sort_keys=True))
    update_report(args.report_root, record)


def summarize_runs(run_root: Path) -> dict[str, Any]:
    rows = []
    failed = []
    completed = []
    if not run_root.exists():
        return {"exists": False, "runs": 0, "planner_rows": 0, "best": None, "failed": []}
    for metrics in run_root.glob("*/metrics.json"):
        completed.append(metrics.parent.name)
    for err in Path("logs").glob("grid_goal_next_*_*.err"):
        if err.stat().st_size > 0 and "OutOfMemoryError" in err.read_text(errors="ignore"):
            failed.append(str(err))
    for matrix in run_root.glob("*/planner_eval_*/planner_matrix.jsonl"):
        with matrix.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                item["path"] = str(matrix)
                rows.append(item)
    best = None
    if rows:
        best = min(rows, key=lambda row: row.get("remaining_hamming_mean", 9999.0))
    return {
        "exists": True,
        "runs": len(list(run_root.glob("*"))),
        "completed_train_runs": sorted(completed),
        "planner_rows": len(rows),
        "best": best,
        "failed": failed,
        "row_counts": dict(row_count_histogram(run_root)),
        "rows_by_run": rows_by_run(run_root),
    }


def row_count_histogram(run_root: Path) -> dict[int, int]:
    hist: dict[int, int] = defaultdict(int)
    for matrix in run_root.glob("*/planner_eval_*/planner_matrix.jsonl"):
        count = sum(1 for line in matrix.open() if line.strip())
        hist[count] += 1
    return dict(sorted(hist.items()))


def rows_by_run(run_root: Path) -> dict[str, int]:
    rows = {}
    for matrix in run_root.glob("*/planner_eval_*/planner_matrix.jsonl"):
        rows[matrix.parent.parent.name] = rows.get(matrix.parent.parent.name, 0) + sum(1 for line in matrix.open() if line.strip())
    return dict(sorted(rows.items()))


def slurm_snapshot() -> str:
    try:
        return subprocess.check_output(
            ["squeue", "-u", os.environ.get("USER", ""), "-o", "%.18i %.12P %.28j %.2t %.12M %.22R"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except Exception as exc:  # pragma: no cover - cluster command availability varies.
        return f"squeue unavailable: {exc}"


def stage_complete(summary: dict[str, Any], stage: str) -> bool:
    if not summary.get("exists"):
        return False
    expected = {f"grid_goal_next_{stage}_{variant}" for variant in STAGE_VARIANTS[stage]}
    completed = set(summary.get("completed_train_runs", ()))
    rows = summary.get("rows_by_run", {})
    return expected <= completed and all(int(rows.get(run, 0)) > 0 for run in expected)


def next_stage(stage: str) -> str | None:
    index = STAGES.index(stage)
    if index + 1 >= len(STAGES):
        return None
    return STAGES[index + 1]


def submit_stage(stage: str) -> dict[str, str]:
    last = len(STAGE_VARIANTS[stage]) - 1
    train = subprocess.check_output(
        [
            "sbatch",
            "--parsable",
            f"--array=0-{last}%4",
            f"--export=ALL,GRID_GOAL_STAGE={stage}",
            "scripts/slurm/run_grid_goal_next_train.slurm",
        ],
        text=True,
    ).strip()
    eval_job = subprocess.check_output(
        [
            "sbatch",
            "--parsable",
            f"--array=0-{last}%4",
            f"--dependency=aftercorr:{train}",
            f"--export=ALL,GRID_GOAL_STAGE={stage}",
            "scripts/slurm/run_grid_goal_next_eval.slurm",
        ],
        text=True,
    ).strip()
    return {"train": train, "eval": eval_job}


def cleanup_disposable(repo_root: Path, work_root: Path, *, delete_checkpoints: bool) -> list[str]:
    removed = []
    pytest_cache = repo_root / ".pytest_cache"
    if pytest_cache.exists():
        shutil.rmtree(pytest_cache, ignore_errors=True)
        removed.append(str(pytest_cache))
    for path in repo_root.rglob("__pycache__"):
        shutil.rmtree(path, ignore_errors=True)
        removed.append(str(path))
    for path in (work_root / "runs" / "grid_goal_followups").glob("*_failed_*"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            removed.append(str(path))
    if delete_checkpoints:
        for checkpoint in (work_root / "runs").glob("grid_goal_*/**/checkpoint-*.pt"):
            checkpoint.unlink(missing_ok=True)
            removed.append(str(checkpoint))
    return removed


def update_report(report_root: Path, record: dict[str, Any]) -> None:
    report_root.mkdir(parents=True, exist_ok=True)
    log_path = report_root / "LOG.md"
    with log_path.open("a") as handle:
        handle.write(f"\n## {record['time']} Grid-Goal Oversight\n\n")
        handle.write(f"- Stage: `{record['stage']}`\n")
        handle.write(f"- Planner rows: `{record['summary'].get('planner_rows', 0)}`\n")
        best = record["summary"].get("best")
        if best:
            handle.write(
                f"- Best row: `{best.get('score_mode')}` / `{best.get('planner')}` / "
                f"depth `{best.get('beam_depth')}`, remaining Hamming "
                f"`{best.get('remaining_hamming_mean')}`, solve rate `{best.get('solve_rate')}`\n"
            )
        if record.get("submitted"):
            handle.write(f"- Submitted: `{record['submitted']}`\n")
        if record.get("cleanup"):
            handle.write(f"- Cleanup removed `{len(record['cleanup'])}` disposable paths.\n")


if __name__ == "__main__":
    main()
