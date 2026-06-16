# Runbook

Last updated: 2026-06-16 18:22 CEST

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

## Slurm Snapshot

RTX Pro 6000 batch probes for `M0_full` were submitted and all failed quickly
with CUDA OOM:

- `3748744`: batch 64, `logs/grid_goal_bs64_3748744.out/.err`
- `3748745`: batch 128, `logs/grid_goal_bs128_3748745.out/.err`
- `3748746`: batch 256, `logs/grid_goal_bs256_3748746.out/.err`
- `3748747`: batch 512, `logs/grid_goal_bs512_3748747.out/.err`

Each probe requested one `rtxpro6k` GPU and 24h, and sampled GPU utilization
with `nvidia-smi`. None of the requested microbatch sizes fit; batch 64 already
used roughly the full 96 GB VRAM before failing.

Smaller full-trajectory probes:

- `3748774`: batch 4, canceled after confirming it fit
- `3748775`: batch 8, canceled after confirming it fit and submitting the full suite
- `3748776`: batch 10, canceled after confirming it fit but was near the
  VRAM ceiling
- `3748777`: batch 12, failed CUDA OOM after `00:00:23`
- `3748778`: batch 16, failed CUDA OOM after `00:00:23`

Wrong trajectories have the same frame count as oracle trajectories:
`#editable cells + 1`; they differ only in using random fill values.
In a 512-example train sample, trajectory lengths were min 47, median 57, mean
56.94, max 65 frames. Batch 8 logged roughly 100 optimizer steps/minute early
in training, implying about 144k steps in 24h at that rate; the configured
20k-step run would finish in roughly 3.3 hours plus final diagnostics.

Full suite submission:

- Training array: `3748789`, `rtxpro6k`, array `0-12%13`, running
- Planner eval array: `3748790`, `rtxpro6k`, array `0-12%13`, pending on
  `afterok:3748789_*`
- Training overrides: `TRAIN_MAX_STEPS=60000`, `BATCH_SIZE=8`,
  `GRADIENT_ACCUMULATION_STEPS=1`, `LEARNING_RATE=1e-4`
- Logs: `logs/grid_goal_train_3748789_<task>.out/.err` and
  `logs/grid_goal_plan_3748790_<task>.out/.err`

## Verify

```bash
source scripts/env.sh
pytest -q
python -m compileall -q puzzle_jepa configs
bash -n scripts/slurm/run_grid_goal_sudoku_ablation.slurm
bash -n scripts/slurm/run_grid_goal_sudoku_planner_eval.slurm
```

Current verification after fixing temporal straightening:

- `source scripts/env.sh && pytest -q tests/test_grid_goal_jepa.py`:
  `13 passed`
- `source scripts/env.sh && pytest -q`: `31 passed`
- `source scripts/env.sh && python -m compileall -q puzzle_jepa configs`:
  passed
- Slurm launcher syntax checks: passed

Previous verification after fixing final action-rank state sampling:

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

Temporal-straightening regression tests now pass. They cover:

- a two-frame sequence must have zero curvature loss
- a masked sequence with no fully valid three-frame triplet must have zero
  curvature loss
- changing only the goal must not change the curvature loss of a fixed
  encoded trajectory
- full active grid-token latents are used rather than only mean summaries

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
