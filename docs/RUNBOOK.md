# Runbook

Last updated: 2026-06-16 17:27 CEST

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
reset. Review-time `squeue -u "$USER"` showed no active Slurm jobs. No new
Grid-Token jobs have been submitted yet.

## Verify

```bash
source scripts/env.sh
pytest -q
python -m compileall -q puzzle_jepa configs
bash -n scripts/slurm/run_grid_goal_sudoku_ablation.slurm
bash -n scripts/slurm/run_grid_goal_sudoku_planner_eval.slurm
```

Current verification after fixing final action-rank state sampling:

- `source scripts/env.sh && pytest -q`: `26 passed`
- `python -m compileall -q puzzle_jepa configs`: passed
- Slurm launcher syntax checks: passed
- Import check confirms 13 ablations and beam widths/depths:
  widths `1,4,16,64`; depths `8,16,32,64`

Regression tests in `tests/test_grid_goal_plan_regressions.py` now pass. They
cover:

- action-rank positives are target-consistent solution actions
- `R1_no_context_masks` removes context value conditioning
- model `forward` accepts non-9x9 active grid-token tensors
- legacy CLS/value/causal paths are removed from the active source tree

Second-pass regression tests in `tests/test_grid_goal_plan_regressions.py` now
pass. They cover:

- progress ranking ignores random non-solution trajectories unless
  `oracle_mask` marks them successful
- action ranking uses encoded symbolic successor boards, not predictor latents
- `puzzle_jepa/models/recursive.py` and `puzzle_jepa/models/layers.py` remain
  intentionally as future HRM/TRM baselines
- rollout and goal-alignment diagnostics are present

Final-review regression test now passes:

- training samples action-rank boards from valid trajectory states, not only
  `batch.boards[:, 0]`.

Operational risk:

- the largest planner matrix settings (`beam_width=64`, `beam_depth=64`) expand
  many unbatched successor scores and are likely to be very slow without
  batching or a branch policy.

## Submit When Asked

Do not submit until the user says `go`. The largest planner settings remain a
runtime risk unless batched/pruned or explicitly accepted.

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
`R7_no_terminal_corrupt`, `R8_no_sigreg`,
`R9_no_temporal_straightening`.

Training defaults:

- optimizer steps: `20000`
- microbatch size: `64`
- gradient accumulation: `4`
- effective batch size: `256` full trajectories per optimizer step
- peak LR: `1e-4`
- warmup: `1000` steps
- schedule: linear warmup then cosine decay
- final LR: `1e-5`
- temporal straightening weight: `0.1`

Planner eval axes:

- Planner: MPC with beam search inner optimizer
- Beam width: `1,4,16,64`
- Beam depth: `8,16,32,64`
- Score: `oracle_goal_distance`, `predicted_goal_distance`
- Transition: `symbolic_reencode`, `latent_rollout`
