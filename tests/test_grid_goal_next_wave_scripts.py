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

OLDLOCAL_VARIANTS = (
    "dense_k1",
    "dense_k4",
    "dense_k8",
    "dense_k16",
    "dense_k32",
    "hier_l4",
    "hier_l4_l16",
    "hier_l4_l16_l32",
    "hier_l4_l16_shared",
    "hier_l4_l16_hier_dense",
    "rank_oracle_progress",
    "rank_both_progress",
    "rank_no_progress",
    "rank_pairwise_oracle_action",
    "rank_pairwise_both_action",
    "rank_listwise_pred_action",
    "rank_listwise_both_action",
    "rank_no_action",
)


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


def _capture_python_args(tmp_path: Path, *, script: str, array_index: int, checkpoint: bool = False) -> list[str]:
    repo_root = Path(__file__).resolve().parents[1]
    work = tmp_path / "work"
    venv_bin = work / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    capture_file = tmp_path / f"{Path(script).stem}_{array_index}_args.txt"
    fake_python = venv_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$@\" > \"${CAPTURE_FILE:?}\"\n"
    )
    fake_python.chmod(0o755)

    if checkpoint:
        variant = OLDLOCAL_VARIANTS[array_index]
        run_root = work / "runs" / "grid_goal_oldlocal_fast" / f"grid_goal_oldlocal_fast_{variant}"
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
            "CAPTURE_FILE": str(capture_file),
        }
    )
    subprocess.run(
        ["bash", script],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    return capture_file.read_text().splitlines()


def test_oldlocal_fast_train_uses_faithful_local_value_conditioning_and_conditional_goal(tmp_path):
    args = _capture_python_args(tmp_path, script="scripts/slurm/run_grid_goal_oldlocal_fast_train.slurm", array_index=0)

    assert "model.action_conditioning=old_local_value" in args
    assert "model.goal_conditioning=initial_current" in args
    assert "model.dense_rollout_all_steps=true" in args
    assert "model.regularizer=vicreg" in args
    assert "model.use_ema_target_encoder=true" in args
    assert "training.max_steps=5000" in args


def test_oldlocal_fast_dense_k32_supervises_all_steps_to_32_without_hierarchy(tmp_path):
    args = _capture_python_args(tmp_path, script="scripts/slurm/run_grid_goal_oldlocal_fast_train.slurm", array_index=4)

    assert "model.multi_step_horizons=[32]" in args
    assert "model.hierarchy_levels=[]" in args
    assert "model.hierarchy_loss_weight=0.0" in args


def test_oldlocal_fast_hierarchy_variant_uses_l4_l16_l32(tmp_path):
    args = _capture_python_args(tmp_path, script="scripts/slurm/run_grid_goal_oldlocal_fast_train.slurm", array_index=7)

    assert "model.hierarchy_levels=[4,16,32]" in args
    assert "model.hierarchy_loss_weight=1.0" in args


def test_oldlocal_fast_eval_skips_hierarchy_for_dense_variants(tmp_path):
    args = _capture_python_args(
        tmp_path,
        script="scripts/slurm/run_grid_goal_oldlocal_fast_eval.slurm",
        array_index=0,
        checkpoint=True,
    )

    assert args[args.index("--planners") + 1] == "mpc_beam"


def test_oldlocal_fast_eval_includes_hierarchy_for_hierarchy_variants(tmp_path):
    args = _capture_python_args(
        tmp_path,
        script="scripts/slurm/run_grid_goal_oldlocal_fast_eval.slurm",
        array_index=6,
        checkpoint=True,
    )

    assert args[args.index("--planners") + 1] == "mpc_beam,hierarchical_beam"


def test_oldlocal_fast_eval_records_old_full_board_raw_mse_for_oracle_and_predicted_goals(tmp_path):
    args = _capture_python_args(
        tmp_path,
        script="scripts/slurm/run_grid_goal_oldlocal_fast_eval.slurm",
        array_index=0,
        checkpoint=True,
    )
    scores = args[args.index("--scores") + 1].split(",")

    assert "oracle_goal_raw_mse_distance" in scores
    assert "predicted_goal_raw_mse_distance" in scores
    assert args[args.index("--beam-depths") + 1] == "1,4,16,32"
    assert args[args.index("--transitions") + 1] == "symbolic_reencode,latent_rollout"
