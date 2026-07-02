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

H1_RECIPE_VARIANTS = (
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

MINAUX_VARIANTS = (
    "base",
    "reg_vicreg",
    "reg_sigreg",
    "reg_no_ema",
    "reg_vicreg_no_ema",
    "reg_sigreg_no_ema",
    "reg_vicreg_sigreg",
    "reg_vicreg_sigreg_no_ema",
    "rank_pairwise_pred_action",
    "rank_listwise_pred_action",
    "rank_pairwise_oracle_action",
    "rank_listwise_oracle_action",
    "geom_temporal",
    "geom_pred_progress",
    "geom_oracle_progress",
    "dense_k1",
    "dense_k2",
    "dense_k4",
    "dense_k8",
    "dense_k16",
    "hier_none",
    "hier_l4",
    "hier_l16",
    "hier_l4_l16",
    "hier_l4_l16_l32",
    "goal_initial_current",
    "goal_no_stopgrad",
    "goal_initial_current_no_stopgrad",
    "goal_distance_field_distill",
)

DENSE_EXACT_VARIANTS = (
    "dense_exact_k8_uniform",
    "dense_exact_k8_inv_sqrt",
    "dense_exact_k8_gamma",
)

CLEAN17_VARIANTS = (
    "W_uniform_H0_G_none",
    "W_uniform_H4_G_none",
    "W_uniform_H4_16_G_none",
    "W_uniform_H4_16_32_G_none",
    "W_inv_sqrt_H0_G_none",
    "W_inv_sqrt_H4_G_none",
    "W_inv_sqrt_H4_16_G_none",
    "W_inv_sqrt_H4_16_32_G_none",
    "W_gamma_H0_G_none",
    "W_gamma_H4_G_none",
    "W_gamma_H4_16_G_none",
    "W_gamma_H4_16_32_G_none",
    "G_ic_detached",
    "G_ic_non_detached",
    "G_ic_online_no_stopgrad",
    "G_ic_field_plus_mse",
    "G_ic_field_only",
)

MACRO_HWM_VARIANTS = (
    "D4_H4_16",
    "D8_H4_16",
    "D16_H4_16",
    "D8_H4_16_32",
)

MINAUX_FACTOR_VARIANTS = (
    "A_anchor_repro",
    "A_refactor_equiv_14816",
    "A_refactor_equiv_14816_dropout_off",
    "A_smooth_14816_like",
    "A_uniform_k16",
    "A_inv_sqrt_k16",
    "A_gamma_k16",
    "A_inv_sqrt_k8",
    "A_old_path_h16_only",
    "A_old_path_h8_only",
    "A_no_goal_mse",
    "A_initial_current_goal",
    "A_no_hierarchy",
    "A_no_predict_delta",
)


def _default_eval_planners(tmp_path: Path, *, stage: str, array_index: int) -> str:
    repo_root = Path(__file__).resolve().parents[1]
    work = tmp_path / "work"
    venv_bin = work / ".venv" / "bin"
    venv_bin.mkdir(parents=True, exist_ok=True)
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
    venv_bin.mkdir(parents=True, exist_ok=True)
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


def _capture_h1_recipe_args(tmp_path: Path, *, script: str, array_index: int, checkpoint: bool = False) -> list[str]:
    repo_root = Path(__file__).resolve().parents[1]
    work = tmp_path / "work"
    venv_bin = work / ".venv" / "bin"
    venv_bin.mkdir(parents=True, exist_ok=True)
    capture_file = tmp_path / f"{Path(script).stem}_{array_index}_args.txt"
    fake_python = venv_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$@\" > \"${CAPTURE_FILE:?}\"\n"
    )
    fake_python.chmod(0o755)

    if checkpoint:
        variant = H1_RECIPE_VARIANTS[array_index]
        run_root = work / "runs" / "grid_goal_h1_recipe" / f"grid_goal_h1_recipe_{variant}"
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


def _capture_minaux_args(tmp_path: Path, *, script: str, array_index: int, checkpoint: bool = False) -> list[str]:
    repo_root = Path(__file__).resolve().parents[1]
    work = tmp_path / "work"
    venv_bin = work / ".venv" / "bin"
    venv_bin.mkdir(parents=True, exist_ok=True)
    capture_file = tmp_path / f"{Path(script).stem}_{array_index}_args.txt"
    fake_python = venv_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$@\" > \"${CAPTURE_FILE:?}\"\n"
    )
    fake_python.chmod(0o755)

    if checkpoint:
        variant = MINAUX_VARIANTS[array_index]
        run_root = work / "runs" / "grid_goal_minaux_wave" / f"grid_goal_minaux_wave_{variant}"
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


def _capture_dense_exact_args(tmp_path: Path, *, script: str, array_index: int, checkpoint: bool = False) -> list[str]:
    repo_root = Path(__file__).resolve().parents[1]
    work = tmp_path / "work"
    venv_bin = work / ".venv" / "bin"
    venv_bin.mkdir(parents=True, exist_ok=True)
    capture_file = tmp_path / f"{Path(script).stem}_{array_index}_args.txt"
    fake_python = venv_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$@\" > \"${CAPTURE_FILE:?}\"\n"
    )
    fake_python.chmod(0o755)

    if checkpoint:
        variant = DENSE_EXACT_VARIANTS[array_index]
        run_root = work / "runs" / "grid_goal_dense_exact" / f"grid_goal_dense_exact_{variant}"
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


def _capture_clean17_args(tmp_path: Path, *, script: str, array_index: int, checkpoint: bool = False) -> list[str]:
    repo_root = Path(__file__).resolve().parents[1]
    work = tmp_path / "work"
    venv_bin = work / ".venv" / "bin"
    venv_bin.mkdir(parents=True, exist_ok=True)
    capture_file = tmp_path / f"{Path(script).stem}_{array_index}_args.txt"
    fake_python = venv_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$@\" > \"${CAPTURE_FILE:?}\"\n"
    )
    fake_python.chmod(0o755)

    if checkpoint:
        variant = CLEAN17_VARIANTS[array_index]
        run_root = work / "runs" / "grid_goal_clean17" / f"grid_goal_clean17_{variant}"
        run_root.mkdir(parents=True)
        (run_root / "checkpoint.pt").write_bytes(b"placeholder")

    env = os.environ.copy()
    env.pop("PLANNERS", None)
    env.pop("SCORES", None)
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


def _capture_macro_hwm_args(
    tmp_path: Path,
    *,
    script: str,
    array_index: int,
    checkpoint: bool = False,
    eval_mode: str | None = None,
) -> list[str]:
    repo_root = Path(__file__).resolve().parents[1]
    work = tmp_path / "work"
    venv_bin = work / ".venv" / "bin"
    venv_bin.mkdir(parents=True, exist_ok=True)
    capture_file = tmp_path / f"{Path(script).stem}_{array_index}_{eval_mode or 'train'}_args.txt"
    fake_python = venv_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$@\" > \"${CAPTURE_FILE:?}\"\n"
    )
    fake_python.chmod(0o755)

    if checkpoint:
        variant = MACRO_HWM_VARIANTS[array_index]
        run_root = work / "runs" / "grid_goal_macro_hwm" / f"grid_goal_macro_hwm_{variant}"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "checkpoint.pt").write_bytes(b"placeholder")

    env = os.environ.copy()
    env.pop("PLANNERS", None)
    env.pop("SCORES", None)
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
    if eval_mode is not None:
        env["EVAL_MODE"] = eval_mode
    subprocess.run(
        ["bash", script],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    return capture_file.read_text().splitlines()


def _capture_minaux_factor_args(tmp_path: Path, *, script: str, array_index: int, checkpoint: bool = False) -> list[str]:
    repo_root = Path(__file__).resolve().parents[1]
    work = tmp_path / "work"
    venv_bin = work / ".venv" / "bin"
    venv_bin.mkdir(parents=True, exist_ok=True)
    capture_file = tmp_path / f"{Path(script).stem}_{array_index}_args.txt"
    fake_python = venv_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$@\" > \"${CAPTURE_FILE:?}\"\n"
    )
    fake_python.chmod(0o755)

    if checkpoint:
        variant = MINAUX_FACTOR_VARIANTS[array_index]
        run_root = work / "runs" / "grid_goal_minaux_factor" / f"grid_goal_minaux_factor_{variant}"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "checkpoint.pt").write_bytes(b"placeholder")

    env = os.environ.copy()
    env.pop("PLANNERS", None)
    env.pop("SCORES", None)
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


def test_h1_recipe_anchor_uses_h1_compatible_basis(tmp_path):
    args = _capture_h1_recipe_args(tmp_path, script="scripts/slurm/run_grid_goal_h1_recipe_train.slurm", array_index=0)

    assert "model.action_conditioning=affected_marker" in args
    assert "model.predict_delta=true" in args
    assert "model.dense_future_weight=1.0" in args
    assert "model.multi_step_horizons=[1,4,8,16]" in args
    assert "model.hierarchy_levels=[4,16]" in args
    assert "model.goal_conditioning=context" in args
    assert "model.goal_nce_weight=0.0" in args
    assert "training.max_steps=45000" in args


def test_h1_recipe_action_variants_are_single_factor_overrides(tmp_path):
    value_args = _capture_h1_recipe_args(tmp_path, script="scripts/slurm/run_grid_goal_h1_recipe_train.slurm", array_index=3)
    concat_args = _capture_h1_recipe_args(tmp_path, script="scripts/slurm/run_grid_goal_h1_recipe_train.slurm", array_index=4)

    assert "model.action_conditioning=old_local_value" in value_args
    assert "model.action_conditioning=old_local_concat" in concat_args
    assert "model.hierarchy_levels=[4,16]" in value_args
    assert "model.hierarchy_levels=[4,16]" in concat_args


def test_h1_recipe_affected_context_variant_enables_local_context_weighting(tmp_path):
    args = _capture_h1_recipe_args(tmp_path, script="scripts/slurm/run_grid_goal_h1_recipe_train.slurm", array_index=6)

    assert "model.dynamics_weighting=affected_context" in args
    assert "model.affected_dynamics_weight=8.0" in args
    assert "model.context_dynamics_weight=2.0" in args


def test_h1_recipe_auxiliary_ablation_removes_all_auxiliary_geometry_losses(tmp_path):
    args = _capture_h1_recipe_args(tmp_path, script="scripts/slurm/run_grid_goal_h1_recipe_train.slurm", array_index=12)

    assert "model.temporal_straightening_weight=0.0" in args
    assert "model.progress_rank_target=none" in args
    assert "model.progress_rank_weight=0.0" in args
    assert "model.action_rank_mode=none" in args
    assert "model.action_rank_weight=0.0" in args
    assert "model.terminal_corrupt_weight=0.0" in args
    assert "model.regularizer=none" in args


def test_h1_recipe_eval_includes_local_context_scores_and_both_transition_modes(tmp_path):
    args = _capture_h1_recipe_args(
        tmp_path,
        script="scripts/slurm/run_grid_goal_h1_recipe_eval.slurm",
        array_index=0,
        checkpoint=True,
    )
    scores = args[args.index("--scores") + 1].split(",")

    assert "oracle_goal_affected_context_raw_euclidean_distance" in scores
    assert "predicted_goal_affected_context_raw_euclidean_distance" in scores
    assert "oracle_goal_raw_mse_distance" in scores
    assert args[args.index("--transitions") + 1] == "symbolic_reencode,latent_rollout"
    assert args[args.index("--beam-depths") + 1] == "4,16,32,64"
    assert args[args.index("--planners") + 1] == "mpc_beam,hierarchical_beam"


def test_h1_recipe_eval_skips_hierarchical_beam_without_hierarchy(tmp_path):
    args = _capture_h1_recipe_args(
        tmp_path,
        script="scripts/slurm/run_grid_goal_h1_recipe_eval.slurm",
        array_index=13,
        checkpoint=True,
    )

    assert args[args.index("--planners") + 1] == "mpc_beam"


def test_minaux_wave_base_is_5k_minimal_aux_with_hierarchy(tmp_path):
    args = _capture_minaux_args(tmp_path, script="scripts/slurm/run_grid_goal_minaux_wave_train.slurm", array_index=0)

    assert "training.max_steps=5000" in args
    assert "model.action_conditioning=affected_marker" in args
    assert "model.predict_delta=true" in args
    assert "model.regularizer=none" in args
    assert "model.use_ema_target_encoder=true" in args
    assert "model.temporal_straightening_weight=0.0" in args
    assert "model.progress_rank_target=none" in args
    assert "model.action_rank_mode=none" in args
    assert "model.terminal_corrupt_weight=0.0" in args
    assert "model.hierarchy_levels=[4,16]" in args
    assert "model.goal_conditioning=context" in args


def test_minaux_wave_regularizer_and_no_ema_variants_are_single_factor_args(tmp_path):
    both_args = _capture_minaux_args(tmp_path, script="scripts/slurm/run_grid_goal_minaux_wave_train.slurm", array_index=6)
    no_ema_args = _capture_minaux_args(tmp_path, script="scripts/slurm/run_grid_goal_minaux_wave_train.slurm", array_index=7)

    assert "model.regularizer=both" in both_args
    assert "model.use_ema_target_encoder=true" in both_args
    assert "model.regularizer=both" in no_ema_args
    assert "model.use_ema_target_encoder=false" in no_ema_args


def test_minaux_wave_goal_no_stopgrad_uses_online_goal_target_mode(tmp_path):
    args = _capture_minaux_args(tmp_path, script="scripts/slurm/run_grid_goal_minaux_wave_train.slurm", array_index=26)

    assert "model.goal_target_mode=online_no_stopgrad" in args
    assert "model.use_ema_target_encoder=true" in args
    assert "model.goal_conditioning=context" in args


def test_minaux_wave_initial_current_no_stopgrad_combines_both_goal_changes(tmp_path):
    args = _capture_minaux_args(tmp_path, script="scripts/slurm/run_grid_goal_minaux_wave_train.slurm", array_index=27)

    assert "model.goal_conditioning=initial_current" in args
    assert "model.goal_target_mode=online_no_stopgrad" in args


def test_minaux_wave_distance_field_distillation_enables_goal_field_loss(tmp_path):
    args = _capture_minaux_args(tmp_path, script="scripts/slurm/run_grid_goal_minaux_wave_train.slurm", array_index=28)

    assert "model.goal_distance_field_weight=1.0" in args


def test_minaux_wave_fast_eval_uses_latent_rollout_depths_4_16_and_global_scores(tmp_path):
    args = _capture_minaux_args(
        tmp_path,
        script="scripts/slurm/run_grid_goal_minaux_wave_eval.slurm",
        array_index=0,
        checkpoint=True,
    )
    scores = args[args.index("--scores") + 1].split(",")

    assert args[args.index("--transitions") + 1] == "latent_rollout"
    assert args[args.index("--beam-widths") + 1] == "16"
    assert args[args.index("--beam-depths") + 1] == "4,16"
    assert args[args.index("--examples") + 1] == "8"
    assert scores == [
        "oracle_goal_distance",
        "predicted_goal_distance",
        "oracle_goal_raw_euclidean_distance",
        "predicted_goal_raw_euclidean_distance",
    ]
    assert args[args.index("--planners") + 1] == "mpc_beam,hierarchical_beam"


def test_minaux_wave_fast_eval_skips_hierarchical_beam_for_no_hierarchy_variant(tmp_path):
    args = _capture_minaux_args(
        tmp_path,
        script="scripts/slurm/run_grid_goal_minaux_wave_eval.slurm",
        array_index=20,
        checkpoint=True,
    )

    assert args[args.index("--planners") + 1] == "mpc_beam"


def test_dense_exact_train_uses_variable_start_k8_minimal_aux_base(tmp_path):
    args = _capture_dense_exact_args(
        tmp_path,
        script="scripts/slurm/run_grid_goal_dense_exact_train.slurm",
        array_index=0,
    )

    assert "training.max_steps=5000" in args
    assert "model.action_conditioning=affected_marker" in args
    assert "model.predict_delta=true" in args
    assert "model.hierarchy_levels=[4,16]" in args
    assert "model.goal_conditioning=context" in args
    assert "model.dense_rollout_all_steps=false" in args
    assert "model.dense_rollout_variable_starts=true" in args
    assert "model.multi_step_horizons=[8]" in args
    assert "model.dense_rollout_weighting=uniform" in args


def test_dense_exact_train_weighting_variants_are_single_factor(tmp_path):
    inv_args = _capture_dense_exact_args(
        tmp_path,
        script="scripts/slurm/run_grid_goal_dense_exact_train.slurm",
        array_index=1,
    )
    gamma_args = _capture_dense_exact_args(
        tmp_path,
        script="scripts/slurm/run_grid_goal_dense_exact_train.slurm",
        array_index=2,
    )

    assert "model.dense_rollout_weighting=inverse_sqrt" in inv_args
    assert "model.dense_rollout_variable_starts=true" in inv_args
    assert "model.multi_step_horizons=[8]" in inv_args
    assert "model.dense_rollout_weighting=geometric" in gamma_args
    assert "model.dense_rollout_gamma=0.8" in gamma_args
    assert "model.dense_rollout_variable_starts=true" in gamma_args
    assert "model.multi_step_horizons=[8]" in gamma_args


def test_dense_exact_eval_matches_fast_latent_global_matrix(tmp_path):
    args = _capture_dense_exact_args(
        tmp_path,
        script="scripts/slurm/run_grid_goal_dense_exact_eval.slurm",
        array_index=0,
        checkpoint=True,
    )
    scores = args[args.index("--scores") + 1].split(",")

    assert args[args.index("--transitions") + 1] == "latent_rollout"
    assert args[args.index("--beam-widths") + 1] == "16"
    assert args[args.index("--beam-depths") + 1] == "4,16"
    assert args[args.index("--examples") + 1] == "8"
    assert args[args.index("--planners") + 1] == "mpc_beam,hierarchical_beam"
    assert scores == [
        "oracle_goal_distance",
        "predicted_goal_distance",
        "oracle_goal_raw_euclidean_distance",
        "predicted_goal_raw_euclidean_distance",
    ]


def test_clean17_has_expected_unique_variant_count():
    assert len(CLEAN17_VARIANTS) == 17
    assert len(set(CLEAN17_VARIANTS)) == 17


def test_clean17_anchor_is_deduped_inv_sqrt_h4_l16_g_none(tmp_path):
    args = _capture_clean17_args(
        tmp_path,
        script="scripts/slurm/run_grid_goal_clean17_train.slurm",
        array_index=6,
    )

    assert "ablation=M0_full" in args
    assert "+experiment_variant=W_inv_sqrt_H4_16_G_none" in args
    assert "training.max_steps=5000" in args
    assert "model.dense_rollout_variable_starts=true" in args
    assert "model.multi_step_horizons=[8]" in args
    assert "model.dense_rollout_weighting=inverse_sqrt" in args
    assert "model.hierarchy_levels=[4,16]" in args
    assert "model.hierarchy_loss_weight=1.0" in args
    assert "model.goal_mse_weight=0.0" in args
    assert "model.goal_conditioning=context" in args
    assert "model.progress_rank_target=none" in args
    assert "model.action_rank_mode=none" in args


def test_clean17_rollout_hierarchy_grid_only_changes_weight_and_hierarchy(tmp_path):
    no_hier = _capture_clean17_args(
        tmp_path,
        script="scripts/slurm/run_grid_goal_clean17_train.slurm",
        array_index=0,
    )
    deep_hier = _capture_clean17_args(
        tmp_path,
        script="scripts/slurm/run_grid_goal_clean17_train.slurm",
        array_index=11,
    )

    assert "model.dense_rollout_weighting=uniform" in no_hier
    assert "model.hierarchy_levels=[]" in no_hier
    assert "model.hierarchy_loss_weight=0.0" in no_hier
    assert "model.dense_rollout_weighting=geometric" in deep_hier
    assert "model.dense_rollout_gamma=0.8" in deep_hier
    assert "model.hierarchy_levels=[4,16,32]" in deep_hier
    assert "model.hierarchy_loss_weight=1.0" in deep_hier
    assert "model.goal_mse_weight=0.0" in no_hier
    assert "model.goal_mse_weight=0.0" in deep_hier


def test_clean17_goal_variants_are_separate_goal_objectives(tmp_path):
    detached = _capture_clean17_args(
        tmp_path,
        script="scripts/slurm/run_grid_goal_clean17_train.slurm",
        array_index=12,
    )
    non_detached = _capture_clean17_args(
        tmp_path,
        script="scripts/slurm/run_grid_goal_clean17_train.slurm",
        array_index=13,
    )
    online = _capture_clean17_args(
        tmp_path,
        script="scripts/slurm/run_grid_goal_clean17_train.slurm",
        array_index=14,
    )
    field_plus_mse = _capture_clean17_args(
        tmp_path,
        script="scripts/slurm/run_grid_goal_clean17_train.slurm",
        array_index=15,
    )
    field_only = _capture_clean17_args(
        tmp_path,
        script="scripts/slurm/run_grid_goal_clean17_train.slurm",
        array_index=16,
    )

    assert "model.goal_conditioning=initial_current" in detached
    assert "model.goal_conditioning_detach_state=true" in detached
    assert "model.goal_mse_weight=1.0" in detached
    assert "model.goal_distance_field_weight=0.0" in detached
    assert "model.goal_conditioning_detach_state=false" in non_detached
    assert "model.goal_target_mode=online_no_stopgrad" in online
    assert "model.goal_conditioning_detach_state=false" in online
    assert "model.goal_mse_weight=1.0" in field_plus_mse
    assert "model.goal_distance_field_weight=1.0" in field_plus_mse
    assert "model.goal_mse_weight=0.0" in field_only
    assert "model.goal_distance_field_weight=1.0" in field_only
    for args in (detached, non_detached, online, field_plus_mse, field_only):
        assert "model.dense_rollout_weighting=inverse_sqrt" in args
        assert "model.hierarchy_levels=[4,16]" in args


def test_clean17_eval_uses_raw_oracle_only_for_g_none_and_adds_predicted_for_goal_jobs(tmp_path):
    no_hier = _capture_clean17_args(
        tmp_path,
        script="scripts/slurm/run_grid_goal_clean17_eval.slurm",
        array_index=0,
        checkpoint=True,
    )
    hier_anchor = _capture_clean17_args(
        tmp_path,
        script="scripts/slurm/run_grid_goal_clean17_eval.slurm",
        array_index=6,
        checkpoint=True,
    )
    goal_job = _capture_clean17_args(
        tmp_path,
        script="scripts/slurm/run_grid_goal_clean17_eval.slurm",
        array_index=12,
        checkpoint=True,
    )

    assert no_hier[no_hier.index("--planners") + 1] == "mpc_beam"
    assert no_hier[no_hier.index("--scores") + 1] == "oracle_goal_raw_euclidean_distance"
    assert hier_anchor[hier_anchor.index("--planners") + 1] == "mpc_beam,hierarchical_beam"
    assert hier_anchor[hier_anchor.index("--scores") + 1] == "oracle_goal_raw_euclidean_distance"
    assert goal_job[goal_job.index("--planners") + 1] == "mpc_beam,hierarchical_beam"
    assert goal_job[goal_job.index("--scores") + 1] == (
        "oracle_goal_raw_euclidean_distance,predicted_goal_raw_euclidean_distance"
    )
    assert goal_job[goal_job.index("--transitions") + 1] == "latent_rollout"
    assert goal_job[goal_job.index("--beam-depths") + 1] == "4,16"


def test_minaux_factor_has_expected_unique_variant_count():
    assert len(MINAUX_FACTOR_VARIANTS) == 14
    assert len(set(MINAUX_FACTOR_VARIANTS)) == 14


def test_minaux_factor_anchor_reproduces_minimal_aux_basis(tmp_path):
    args = _capture_minaux_factor_args(
        tmp_path,
        script="scripts/slurm/run_grid_goal_minaux_factor_train.slurm",
        array_index=0,
    )

    assert "training.max_steps=5000" in args
    assert "training.batch_size=8" in args
    assert "training.learning_rate=1.0e-4" in args
    assert "model.action_conditioning=affected_marker" in args
    assert "model.predict_delta=true" in args
    assert "model.goal_mse_weight=1.0" in args
    assert "model.goal_nce_weight=0.0" in args
    assert "model.goal_conditioning=context" in args
    assert "model.dense_future_weight=1.0" in args
    assert "model.dense_rollout_all_steps=false" in args
    assert "model.dense_rollout_variable_starts=false" in args
    assert "model.dense_rollout_refactor_mode=none" in args
    assert "model.multi_step_horizons=[1,4,8,16]" in args
    assert "model.dense_rollout_weighting=inverse_sqrt" in args
    assert "model.hierarchy_levels=[4,16]" in args
    assert "model.hierarchy_loss_weight=1.0" in args
    assert "model.regularizer=none" in args
    assert "model.use_ema_target_encoder=true" in args
    assert "model.progress_rank_target=none" in args
    assert "model.action_rank_mode=none" in args
    assert "model.temporal_straightening_weight=0.0" in args


def test_minaux_factor_refactor_variants_select_equivalent_and_smooth_count_modes(tmp_path):
    exact = _capture_minaux_factor_args(
        tmp_path,
        script="scripts/slurm/run_grid_goal_minaux_factor_train.slurm",
        array_index=1,
    )
    exact_no_dropout = _capture_minaux_factor_args(
        tmp_path,
        script="scripts/slurm/run_grid_goal_minaux_factor_train.slurm",
        array_index=2,
    )
    smooth = _capture_minaux_factor_args(
        tmp_path,
        script="scripts/slurm/run_grid_goal_minaux_factor_train.slurm",
        array_index=3,
    )

    assert "model.dense_rollout_refactor_mode=legacy_equivalent" in exact
    assert "model.multi_step_horizons=[1,4,8,16]" in exact
    assert "model.dense_rollout_refactor_mode=legacy_equivalent" in exact_no_dropout
    assert "model.dropout=0.0" in exact_no_dropout
    assert "model.dense_rollout_refactor_mode=legacy_count" in smooth
    assert "model.multi_step_horizons=[1,4,8,16]" in smooth


def test_minaux_factor_single_rollout_weighting_variants_are_clean_controls(tmp_path):
    uniform = _capture_minaux_factor_args(
        tmp_path,
        script="scripts/slurm/run_grid_goal_minaux_factor_train.slurm",
        array_index=4,
    )
    inv = _capture_minaux_factor_args(
        tmp_path,
        script="scripts/slurm/run_grid_goal_minaux_factor_train.slurm",
        array_index=5,
    )
    gamma = _capture_minaux_factor_args(
        tmp_path,
        script="scripts/slurm/run_grid_goal_minaux_factor_train.slurm",
        array_index=6,
    )
    inv8 = _capture_minaux_factor_args(
        tmp_path,
        script="scripts/slurm/run_grid_goal_minaux_factor_train.slurm",
        array_index=7,
    )

    assert "model.dense_rollout_all_steps=true" in uniform
    assert "model.multi_step_horizons=[16]" in uniform
    assert "model.dense_rollout_weighting=uniform" in uniform
    assert "model.dense_rollout_all_steps=true" in inv
    assert "model.multi_step_horizons=[16]" in inv
    assert "model.dense_rollout_weighting=inverse_sqrt" in inv
    assert "model.dense_rollout_all_steps=true" in gamma
    assert "model.multi_step_horizons=[16]" in gamma
    assert "model.dense_rollout_weighting=geometric" in gamma
    assert "model.dense_rollout_gamma=0.8" in gamma
    assert "model.dense_rollout_all_steps=true" in inv8
    assert "model.multi_step_horizons=[8]" in inv8


def test_minaux_factor_old_path_single_horizon_controls_keep_legacy_loop(tmp_path):
    h16 = _capture_minaux_factor_args(
        tmp_path,
        script="scripts/slurm/run_grid_goal_minaux_factor_train.slurm",
        array_index=8,
    )
    h8 = _capture_minaux_factor_args(
        tmp_path,
        script="scripts/slurm/run_grid_goal_minaux_factor_train.slurm",
        array_index=9,
    )

    assert "model.dense_rollout_all_steps=false" in h16
    assert "model.dense_rollout_variable_starts=false" in h16
    assert "model.dense_rollout_refactor_mode=none" in h16
    assert "model.multi_step_horizons=[16]" in h16
    assert "model.dense_rollout_all_steps=false" in h8
    assert "model.dense_rollout_variable_starts=false" in h8
    assert "model.dense_rollout_refactor_mode=none" in h8
    assert "model.multi_step_horizons=[8]" in h8


def test_minaux_factor_one_factor_ablation_variants_only_change_named_factor(tmp_path):
    no_goal = _capture_minaux_factor_args(
        tmp_path,
        script="scripts/slurm/run_grid_goal_minaux_factor_train.slurm",
        array_index=10,
    )
    ic_goal = _capture_minaux_factor_args(
        tmp_path,
        script="scripts/slurm/run_grid_goal_minaux_factor_train.slurm",
        array_index=11,
    )
    no_hier = _capture_minaux_factor_args(
        tmp_path,
        script="scripts/slurm/run_grid_goal_minaux_factor_train.slurm",
        array_index=12,
    )
    no_delta = _capture_minaux_factor_args(
        tmp_path,
        script="scripts/slurm/run_grid_goal_minaux_factor_train.slurm",
        array_index=13,
    )

    assert "model.goal_mse_weight=0.0" in no_goal
    assert "model.multi_step_horizons=[1,4,8,16]" in no_goal
    assert "model.goal_conditioning=initial_current" in ic_goal
    assert "model.goal_conditioning_detach_state=false" in ic_goal
    assert "model.hierarchy_levels=[]" in no_hier
    assert "model.hierarchy_loss_weight=0.0" in no_hier
    assert "model.predict_delta=false" in no_delta
    assert "model.hierarchy_levels=[4,16]" in no_delta


def test_minaux_factor_eval_is_independent_fast_latent_matrix(tmp_path):
    args = _capture_minaux_factor_args(
        tmp_path,
        script="scripts/slurm/run_grid_goal_minaux_factor_eval.slurm",
        array_index=0,
        checkpoint=True,
    )
    scores = args[args.index("--scores") + 1].split(",")

    assert args[args.index("--transitions") + 1] == "latent_rollout"
    assert args[args.index("--beam-widths") + 1] == "16"
    assert args[args.index("--beam-depths") + 1] == "4,16"
    assert args[args.index("--examples") + 1] == "8"
    assert args[args.index("--planners") + 1] == "mpc_beam,hierarchical_beam"
    assert "--skip-diagnostics" in args
    assert scores == [
        "oracle_goal_raw_euclidean_distance",
        "predicted_goal_raw_euclidean_distance",
    ]


def test_minaux_factor_eval_skips_hierarchical_beam_without_hierarchy(tmp_path):
    args = _capture_minaux_factor_args(
        tmp_path,
        script="scripts/slurm/run_grid_goal_minaux_factor_eval.slurm",
        array_index=12,
        checkpoint=True,
    )

    assert args[args.index("--planners") + 1] == "mpc_beam"


def test_macro_hwm_train_uses_low_dimensional_macro_bottleneck_on_clean17_anchor(tmp_path):
    args = _capture_macro_hwm_args(
        tmp_path,
        script="scripts/slurm/run_grid_goal_macro_hwm_train.slurm",
        array_index=1,
    )

    assert "training.max_steps=5000" in args
    assert "model.dense_rollout_variable_starts=true" in args
    assert "model.multi_step_horizons=[8]" in args
    assert "model.dense_rollout_weighting=inverse_sqrt" in args
    assert "model.hierarchy_levels=[4,16]" in args
    assert "model.hierarchy_loss_weight=1.0" in args
    assert "model.macro_action_dim=8" in args
    assert "model.goal_mse_weight=0.0" in args
    assert "model.progress_rank_target=none" in args
    assert "model.action_rank_mode=none" in args


def test_macro_hwm_train_has_three_level_bottleneck_variant(tmp_path):
    args = _capture_macro_hwm_args(
        tmp_path,
        script="scripts/slurm/run_grid_goal_macro_hwm_train.slurm",
        array_index=3,
    )

    assert "model.hierarchy_levels=[4,16,32]" in args
    assert "model.macro_action_dim=8" in args


def test_macro_hwm_eval_baseline_is_flat_mpc_only(tmp_path):
    args = _capture_macro_hwm_args(
        tmp_path,
        script="scripts/slurm/run_grid_goal_macro_hwm_eval.slurm",
        array_index=1,
        checkpoint=True,
        eval_mode="baseline",
    )

    assert args[args.index("--planners") + 1] == "mpc_beam"
    assert args[args.index("--scores") + 1] == "oracle_goal_raw_euclidean_distance"
    assert args[args.index("--transitions") + 1] == "latent_rollout"
    assert args[args.index("--beam-depths") + 1] == "4,16"
    assert "--skip-diagnostics" in args


def test_macro_hwm_eval_ablate_cem_mppi_and_codebook(tmp_path):
    cem_codebook = _capture_macro_hwm_args(
        tmp_path,
        script="scripts/slurm/run_grid_goal_macro_hwm_eval.slurm",
        array_index=1,
        checkpoint=True,
        eval_mode="cem_codebook",
    )
    mppi_none = _capture_macro_hwm_args(
        tmp_path,
        script="scripts/slurm/run_grid_goal_macro_hwm_eval.slurm",
        array_index=1,
        checkpoint=True,
        eval_mode="mppi_none",
    )

    assert cem_codebook[cem_codebook.index("--planners") + 1] == "hierarchical_cem"
    assert cem_codebook[cem_codebook.index("--high-cem-optimizer") + 1] == "cem"
    assert cem_codebook[cem_codebook.index("--high-cem-codebook") + 1] == "init"
    assert cem_codebook[cem_codebook.index("--high-cem-codebook-size") + 1] == "64"
    assert mppi_none[mppi_none.index("--planners") + 1] == "hierarchical_cem"
    assert mppi_none[mppi_none.index("--high-cem-optimizer") + 1] == "mppi"
    assert mppi_none[mppi_none.index("--high-cem-codebook") + 1] == "none"
