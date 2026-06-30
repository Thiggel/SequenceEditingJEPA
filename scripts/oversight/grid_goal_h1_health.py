from __future__ import annotations

import json
import os
import subprocess
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

SAFE_NODES = "a[0531-0537,0631-0633,0731,0831-0833,0931-0934,2041,2043,2141-2143,2841,2843,2941]"
TRAIN_JOBS = tuple(
    job_id
    for job_id in os.environ.get("H1_HEALTH_TRAIN_JOBS", "3799696,3799777,3800228").split(",")
    if job_id
)


def main() -> None:
    repo = Path.cwd()
    work_root = Path(os.environ["PUZZLE_JEPA_WORK_ROOT"])
    marker_path = work_root / "h1_recipe_health_resubmissions.json"
    marker = load_marker(marker_path)
    failures = find_train_failures(repo)
    submissions = []
    for failure in failures:
        if not failure["oom"]:
            continue
        index = int(failure["index"])
        variant = VARIANTS[index]
        if checkpoint_exists(work_root, variant):
            continue
        key = f"{variant}:oom_batch4_accum2"
        if key in marker:
            continue
        train = subprocess.check_output(
            [
                "sbatch",
                "--parsable",
                "--partition=rtxpro6k,a100",
                f"--nodelist={SAFE_NODES}",
                f"--array={index}",
                "--export=ALL,BATCH_SIZE=4,GRADIENT_ACCUMULATION_STEPS=2",
                "scripts/slurm/run_grid_goal_h1_recipe_train.slurm",
            ],
            text=True,
        ).strip()
        eval_job = subprocess.check_output(
            [
                "sbatch",
                "--parsable",
                "--partition=rtxpro6k,a100",
                f"--nodelist={SAFE_NODES}",
                f"--array={index}",
                f"--dependency=afterok:{train}",
                "--export=ALL,EVAL_OUTPUT_DIR_NAME=planner_eval_h1_recipe_health_repair",
                "scripts/slurm/run_grid_goal_h1_recipe_eval.slurm",
            ],
            text=True,
        ).strip()
        marker[key] = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "index": index,
            "variant": variant,
            "failed_job": failure["job_id"],
            "train": train,
            "eval": eval_job,
        }
        submissions.append(marker[key])
    marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True))
    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "failures": failures,
        "submissions": submissions,
        "squeue": squeue_snapshot(),
    }
    print(json.dumps(record, indent=2, sort_keys=True))
    update_report(Path("../sequence-editing-report"), record)


def load_marker(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def checkpoint_exists(work_root: Path, variant: str) -> bool:
    return (work_root / "runs" / "grid_goal_h1_recipe" / f"grid_goal_h1_recipe_{variant}" / "checkpoint.pt").is_file()


def find_train_failures(repo: Path) -> list[dict[str, Any]]:
    failures = []
    for job_id in TRAIN_JOBS:
        output = subprocess.check_output(
            ["sacct", "-j", job_id, "--format=JobID,State,ExitCode", "-P"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        for line in output.splitlines()[1:]:
            parts = line.split("|")
            if len(parts) < 3:
                continue
            slurm_id, state, exit_code = parts[:3]
            if "_" not in slurm_id or "." in slurm_id or not state.startswith(("FAILED", "OUT_OF_MEMORY", "TIMEOUT")):
                continue
            index_text = slurm_id.rsplit("_", 1)[-1]
            if not index_text.isdigit():
                continue
            index = int(index_text)
            if index < 0 or index >= len(VARIANTS):
                continue
            log_text = read_logs(repo, job_id, index)
            oom = state.startswith("OUT_OF_MEMORY") or "OutOfMemoryError" in log_text or "CUDA out of memory" in log_text
            failures.append(
                {
                    "job_id": slurm_id,
                    "array_job": job_id,
                    "index": index,
                    "variant": VARIANTS[index],
                    "state": state,
                    "exit_code": exit_code,
                    "oom": oom,
                }
            )
    return failures


def read_logs(repo: Path, job_id: str, index: int) -> str:
    texts = []
    for suffix in ("err", "out"):
        path = repo / "logs" / f"grid_goal_h1r_train_{job_id}_{index}.{suffix}"
        if path.is_file():
            texts.append(path.read_text(errors="ignore"))
    return "\n".join(texts)


def squeue_snapshot() -> str:
    return subprocess.check_output(
        ["squeue", "-u", os.environ.get("USER", ""), "-o", "%.18i %.12P %.28j %.2t %.12M %.22R"],
        text=True,
        stderr=subprocess.STDOUT,
    )


def update_report(report_root: Path, record: dict[str, Any]) -> None:
    if not record["submissions"]:
        return
    report_root.mkdir(parents=True, exist_ok=True)
    with (report_root / "LOG.md").open("a") as handle:
        handle.write(f"\n## {record['time']} H1 Recipe Health Repair\n\n")
        for item in record["submissions"]:
            handle.write(
                f"- Resubmitted `{item['variant']}` after OOM-like failure "
                f"`{item['failed_job']}` with batch `4` and grad accumulation `2`: "
                f"train `{item['train']}`, eval `{item['eval']}`.\n"
            )


if __name__ == "__main__":
    main()
