import os
import subprocess
from pathlib import Path


STAGE_VARIANTS = {
    "hierarchy_levels": ("H_empty", "H2", "H2_4", "H2_4_8", "H2_4_8_16", "H2_4_8_16_32"),
    "ranking_losses": (
        "R_pred_progress",
        "R_oracle_progress",
        "R_both_progress",
        "R_no_progress",
        "R_pairwise",
        "R_listwise",
        "R_no_action_rank",
    ),
}


def _default_eval_planners(tmp_path: Path, *, stage: str, array_index: int) -> str:
    repo_root = Path(__file__).resolve().parents[1]
    work = tmp_path / "work"
    venv_bin = work / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    capture_file = tmp_path / "python_args.txt"
    fake_python = venv_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$@\" > \"${CAPTURE_FILE:?}\"\n"
    )
    fake_python.chmod(0o755)

    variant = STAGE_VARIANTS[stage][array_index]
    run_root = work / "runs" / "grid_goal_next_wave" / f"grid_goal_next_{stage}_{variant}"
    run_root.mkdir(parents=True)
    (run_root / "checkpoint.pt").write_bytes(b"placeholder")

    env = os.environ.copy()
    env.pop("PLANNERS", None)
    env.update(
        {
            "WORK": str(work),
            "SCRATCH": str(work / "scratch"),
            "VIRTUAL_ENV": str(work / ".venv"),
            "PUZZLE_JEPA_WORK_ROOT": str(work),
            "SLURM_ARRAY_TASK_ID": str(array_index),
            "GRID_GOAL_STAGE": stage,
            "CAPTURE_FILE": str(capture_file),
        }
    )

    subprocess.run(
        ["bash", "scripts/slurm/run_grid_goal_next_eval.slurm"],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    args = capture_file.read_text().splitlines()
    return args[args.index("--planners") + 1]


def test_next_wave_eval_skips_hierarchical_planner_for_empty_hierarchy_variant(tmp_path):
    planners = _default_eval_planners(tmp_path, stage="hierarchy_levels", array_index=0)

    assert planners == "mpc_beam"


def test_next_wave_eval_includes_hierarchical_planner_for_hierarchy_variant(tmp_path):
    planners = _default_eval_planners(tmp_path, stage="hierarchy_levels", array_index=1)

    assert planners == "mpc_beam,hierarchical_beam"


def test_next_wave_eval_skips_hierarchical_planner_for_ranking_loss_variants(tmp_path):
    planners = _default_eval_planners(tmp_path, stage="ranking_losses", array_index=0)

    assert planners == "mpc_beam"
