# Runbook

Last updated: 2026-06-16 15:05 CEST

Long-form handoff source of truth: `../sequence-editing-report`.

## Active Surface

This repo has been reset to one active experiment path: **Grid-Token
Goal-JEPA** for Sudoku.

- Config: `configs/puzzle/grid_goal_sudoku.yaml`
- Model: `puzzle_jepa/models/grid_goal_jepa.py`
- Data sampler: `puzzle_jepa/data/grid_goal_sudoku.py`
- Trainer: `puzzle_jepa/train/grid_goal_sudoku.py`
- Diagnostics: `puzzle_jepa/eval/grid_goal_diagnostics.py`
- Planner matrix: `puzzle_jepa/eval/grid_goal_planner_matrix.py`
- Planner: `puzzle_jepa/planning/grid_goal_planner.py`
- Training Slurm array: `scripts/slurm/run_grid_goal_sudoku_ablation.slurm`
- Dependency-ready planner eval array:
  `scripts/slurm/run_grid_goal_sudoku_planner_eval.slurm`

All previous LeWM/CLS/value-head jobs were cancelled or completed before this
reset. No new Grid-Token jobs have been submitted yet.

## Verify

```bash
source scripts/env.sh
pytest -q
python -m compileall -q puzzle_jepa configs
bash -n scripts/slurm/run_grid_goal_sudoku_ablation.slurm
bash -n scripts/slurm/run_grid_goal_sudoku_planner_eval.slurm
```

Current verification after the refactor:

- `pytest -q`: `8 passed`
- `python -m compileall -q puzzle_jepa configs`: passed
- Import check confirms 12 ablations and beam widths/depths:
  widths `1,4,16,64`; depths `8,16,32,64`

## Submit When Asked

Do not submit until the user says `go`.

Recommended submission:

```bash
train_job=$(sbatch --parsable scripts/slurm/run_grid_goal_sudoku_ablation.slurm)
sbatch --dependency=afterok:${train_job} scripts/slurm/run_grid_goal_sudoku_planner_eval.slurm
```

Each training task writes to:

```text
$PUZZLE_JEPA_WORK_ROOT/runs/grid_goal_sudoku_<ablation>
```

Each dependency eval task writes:

```text
$PUZZLE_JEPA_WORK_ROOT/runs/grid_goal_sudoku_<ablation>/planner_eval/
```

## Active Ablations

`M0_full`, `R1_no_context_masks`, `R2_mean_pooled_distance`, `R3_k1_only`,
`R3_k4`, `R3_k8`, `R3_k16`, `R4_no_goal_nce`,
`R5_no_progress_rank`, `R6_no_action_rank`,
`R7_no_terminal_corrupt`, `R8_no_sigreg`.

Planner eval axes:

- Planner: MPC with beam search inner optimizer
- Beam width: `1,4,16,64`
- Beam depth: `8,16,32,64`
- Score: `oracle_goal_distance`, `predicted_goal_distance`
- Transition: `symbolic_reencode`, `latent_rollout`
