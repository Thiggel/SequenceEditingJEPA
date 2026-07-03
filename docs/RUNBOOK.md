# Runbook

Last updated: 2026-07-03 10:13 CEST

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
- Follow-up train array:
  `scripts/slurm/run_grid_goal_followup_train.slurm`
- Follow-up planner eval array:
  `scripts/slurm/run_grid_goal_followup_eval.slurm`
- Current-best checkpoint-time eval array:
  `scripts/slurm/run_grid_goal_best_checkpoint_eval.slurm`
- Next-wave staged train array:
  `scripts/slurm/run_grid_goal_next_train.slurm`
- Next-wave staged eval array:
  `scripts/slurm/run_grid_goal_next_eval.slurm`
- Next-wave submit wrapper:
  `scripts/experiments/submit_grid_goal_next_wave.sh`
- Next-wave oversight:
  `scripts/slurm/run_grid_goal_oversight.slurm`,
  `scripts/oversight/submit_grid_goal_oversight.sh`, and
  `scripts/oversight/grid_goal_next_wave.py`
- H1 debug/extra train and eval:
  `scripts/slurm/run_grid_goal_h1_debug_train.slurm`,
  `scripts/slurm/run_grid_goal_h1_debug_eval.slurm`,
  `scripts/slurm/run_grid_goal_h1_debug_hier_eval.slurm`,
  `scripts/slurm/run_grid_goal_h1_extra_train.slurm`, and
  `scripts/slurm/run_grid_goal_h1_extra_eval.slurm`
- Active clean17 train/eval:
  `scripts/slurm/run_grid_goal_clean17_train.slurm`,
  `scripts/slurm/run_grid_goal_clean17_eval.slurm`, and
  `scripts/experiments/submit_grid_goal_clean17.sh`
- Prepared Delta-JEPA train/eval, implemented and not submitted:
  `scripts/slurm/run_grid_goal_delta_jepa_train.slurm` and
  `scripts/slurm/run_grid_goal_delta_jepa_eval.slurm`

Current active/prepared jobs are documented in `docs/CURRENT_EXPERIMENTS.md` and
`../sequence-editing-report/CURRENT_EXPERIMENTS.md`. All previous
LeWM/CLS/value-head jobs were cancelled or completed before this reset.

## Next-Wave Weekend State

The 12-hour oversight chain advanced from `goal_conditioning` to
`dense_horizon`, but then stopped on malformed partial eval JSONL produced by
timed-out eval jobs.

| Stage | Train jobs | Eval jobs | Outcome |
| --- | --- | --- | --- |
| `goal_conditioning` | `3780027`, tasks `0-2`, completed | `3780028`, tasks `0-2`, completed | 120 valid planner rows. Best row solved `1/10` with oracle changed-cell local scoring; predicted-goal rows solved `0/10`. |
| `dense_horizon` | `3782967`, tasks `0-4`, completed | `3782968`, tasks `0-4`, timed out | Submitted by oversight at 2026-06-25 23:55. Partial rows were written. |
| `dense_horizon` duplicate | `3784073`, tasks `0-4`, completed | `3784074`, tasks `0-4`, timed out | Submitted again by oversight at 2026-06-26 11:55; it wrote the same run dirs. Current dense-horizon files contain 65-66 rows each, with malformed trailing JSON in two files. |
| later stages | none | none | Not submitted. Oversight jobs `3780036`-`3780040` failed with `JSONDecodeError`; stale pending oversight jobs `3780041`-`3780042` were canceled on 2026-06-29 after this audit. |

Current next-wave run root:
`$PUZZLE_JEPA_WORK_ROOT/runs/grid_goal_next_wave`.

Best valid weekend rows:

| Run | Valid rows | Best oracle row | Best predicted row |
| --- | ---: | --- | --- |
| `G0_context` | 40 | changed-cell raw L2, `mpc_beam`, depth 32: `1/10`, rem Hamming `8.7` | delta-top3 raw L2, depth 32: `0/10`, rem Hamming `42.2` |
| `G1_initial_current` | 40 | delta-top1 raw L2, depth 4: `0/10`, rem Hamming `42.9` | delta-top3 raw L2, depth 4: `0/10`, rem Hamming `48.1` |
| `G2_initial_current_oracle_progress` | 40 | changed-cell raw L2, depth 4: `0/10`, rem Hamming `31.2` | delta-top5 raw L2, depth 64: `0/10`, rem Hamming `48.9` |
| `DK2` | 65 | changed-cell raw L2, depth 32: `0/10`, rem Hamming `36.6` | delta-top3 raw L2, depth 16: `0/10`, rem Hamming `48.7` |
| `DK4` | 65 | delta-top5 raw L2, depth 32: `0/10`, rem Hamming `45.1` | normalized predicted, hierarchical beam, depth 16: `0/10`, rem Hamming `48.8` |
| `DK8` | 65 | changed-cell raw L2, depth 32: `0/10`, rem Hamming `38.8` | delta-top5 raw L2, depth 32: `0/10`, rem Hamming `47.6` |
| `DK16` | 65 valid, 1 malformed | changed-cell raw L2, depth 64: `0/10`, rem Hamming `48.3` | changed-cell raw L2, depth 16: `0/10`, rem Hamming `48.0` |
| `DK32` | 65 valid, 1 malformed | changed-cell raw L2, depth 64: `0/10`, rem Hamming `48.1` | normalized predicted, depth 4: `0/10`, rem Hamming `48.9` |

Interpretation: the weekend next-wave did not improve the predicted-goal
planner. Conditional predicted goals and the dense-horizon stage regressed
relative to the previous `H1_hierarchy_dense_l4_l16` oracle-local result.
The only exact solve was a single oracle-local `G0_context` board, not a
predicted-goal success.

## H1 Debug Sweep

The controlled H1 debug sweeps are complete; H1-extra evals are still running.

Completed:

- H1 delta train/eval: `3795127`/`3795128`, six variants, batch `8`, 45k
  steps, hierarchy `[4,16]`, context-only goal predictor, `affected_marker`,
  delta predictor, EMA+VICReg, no goal NCE, fixed seed `5204`.
- H1 no-delta train/eval: `3795143`/`3795144`, same six variants and seed,
  but `model.predict_delta=false`.
- H1 hierarchical-beam add-on evals: `3795248` and `3795249`.
- H1-extra train: `3795246_0-10` plus replacement `3795327_11`. The original
  `3795246_11` OOMed at batch `8`; the comparable replacement used batch `4`
  with grad accumulation `2`, effective batch `8`.

Running:

- H1-extra evals `3795247_0-7,9,10` and replacement `3795328_11`.
- `3795247_8` completed. `3795247_11` was canceled after the superseded OOM.

Latest H1 results:

- H1 exact solve rate is `0/10` across completed rows.
- Best H1 `mpc_beam` row: no-delta `K16_LR5e4`, oracle changed-cell raw L2,
  depth `4`, remaining Hamming `6.6`. Depths `16/32/64` are `6.9-7.0`.
- Best predicted-goal changed-cell row on the same checkpoint is remaining
  Hamming `33.8`.
- Hierarchical beam is worse on these checkpoints: best remaining Hamming
  `28.8`.
- H1-extra has 443 partial rows so far, no exact solves. Best partial row:
  `rank_pairwise_both_action`, `mpc_beam`, oracle changed-cell raw L2,
  depth `4`, remaining Hamming `14.9`.

Storage cleanup at 2026-06-29 14:10 CEST:

- `$HPCVAULT/sequence-editing` is about `226GB`, down from about `949GB`.
- Deleted intermediate `checkpoint-[0-9]*.pt` files from completed non-active
  Grid Goal runs only. Kept final `checkpoint.pt`, configs, metrics,
  diagnostics, panels, and planner outputs.
- Active H1 debug/extra run roots were not pruned.
- `$WORK/.cache` was removed, freeing about `11.8GiB`. `$WORK/sequence-editing`
  remains about `19GB`.
- Common config: seed `5204`, batch `8`, 45k steps, LR `1e-4`,
  `affected_marker`, `predict_delta=false`, EMA+VICReg, no goal NCE,
  context-only goal predictor, temporal straightening on, dense base rollout
  horizons `[1,4,8,16]`.
- Variants: `rank_oracle_progress`, `rank_both_progress`,
  `rank_no_progress`, `rank_pairwise_oracle_action`,
  `rank_pairwise_both_action`, `rank_listwise_pred_action`,
  `rank_listwise_both_action`, `rank_no_action`, `hier_l4`,
  `hier_l4_l16_l32`, `hier_l4_l16_shared`, and
  `hier_l4_l16_hier_dense`.

Implemented and verified:

- conditional predicted goal `q(c,H0,Ht)`, recomputed at each MPC root state
- dense rollout supervision over intermediate base-predictor futures
- recursive dense supervision for high-level hierarchy predictors
- hierarchy levels with either separate predictors or one shared
  level-conditioned predictor
- hierarchical beam planner: high-level beam proposes latent subgoals, then
  primitive beam plans to the first subgoal
- delta-top-k local score probes for `k=1,3,5`
- progress-rank target switches: predicted, oracle, both, or none
- pairwise/listwise action-rank switches
- optional policy prior over primitive legal actions and fused macro-actions;
  planning can bias beam scores using `model.policy_prior_planning_weight`

Stage submit command used:

```bash
GRID_GOAL_STAGE=goal_conditioning TRAIN_CONCURRENCY=3 EVAL_CONCURRENCY=3 scripts/experiments/submit_grid_goal_next_wave.sh
```

Stages supported by the wrapper:

`goal_conditioning`, `dense_horizon`, `hierarchy_levels`,
`predictor_delta_topk`, `ranking_losses`, `hierarchical_planning`,
`policy_prior`.

Oversight scheduling command used:

```bash
GRID_GOAL_STAGE=goal_conditioning OVERSIGHT_COUNT=10 OVERSIGHT_INTERVAL_HOURS=12 OVERSIGHT_SUBMIT_NEXT=1 OVERSIGHT_CLEANUP=1 scripts/oversight/submit_grid_goal_oversight.sh
```

Oversight only reports by default. Set `OVERSIGHT_SUBMIT_NEXT=1` to let it
submit the next stage after the current stage is complete. Set
`OVERSIGHT_CLEANUP=1` for safe cache/failed-run cleanup. Do not set
`OVERSIGHT_DELETE_CHECKPOINTS=1` unless checkpoint deletion is explicitly
intended.

Verification from this pass:

- `source scripts/env.sh && pytest -q tests/test_grid_goal_jepa.py` -> pass
- `source scripts/env.sh && pytest -q tests/test_grid_goal_plan_regressions.py`
  -> pass
- `source scripts/env.sh && pytest -q` -> pass
- `source scripts/env.sh && python -m compileall -q puzzle_jepa scripts tests`
  -> pass
- `bash -n` over the new Slurm and oversight scripts -> pass

## Slurm Snapshot

Current sequence-editing status at 2026-06-25 11:56 CEST:

- Active next-wave jobs are listed above. Previous follow-up jobs are complete.
- Follow-up training/eval is complete:
  - original train array `3776065`: tasks `0,2,3,4` completed; tasks `1,5`
    OOMed at batch 8 and were superseded
  - replacement train array `3776086`: tasks `1,5` completed at batch 4
  - follow-up eval arrays `3776066`, `3776068`, `3776069`, `3776070`,
    `3776087`, and `3776088` completed
  - checkpoint-time eval array `3776072` completed
- Result files:
  - follow-up trained variants: 336 planner rows across 6 variants, beam/CEM
    planners, depths `4,16,32,64`, and six oracle/predicted score modes
  - checkpoint-time sweep: 240 planner rows across checkpoints
    `20k,30k,40k,50k,60k`, beam/CEM planners, depths `4,16,32,64`, and six
    score modes
- Best follow-up signal:
  `H1_hierarchy_dense_l4_l16` with `mpc_beam` and
  `oracle_goal_changed_cell_raw_euclidean_distance` solved `6/10` boards at
  beam depth `16` and had mean remaining Hamming `0.5`. Depths `32` and `64`
  solved `4/10` and `5/10`, respectively.
- All predicted-goal follow-up rows still solved `0/10`. The best predicted
  follow-up row had mean remaining Hamming `36.0`, so predicted-goal geometry
  remains the main blocker.
- Categorical CEM and hierarchical CEM did not solve any boards in this wave.
- Corrected action-conditioning/stability evals also finished:
  - main `planner_eval_latent`: 96/96 complete files, 1728 rows, zero solves
  - depth-64 `planner_eval_latent_depth64`: 96/96 complete files, 576 rows,
    zero solves
  - best action-suite row remains
    `R4_no_goal_nce/A6_affected_marker_delta/S4_ema_vicreg/D0_uniform` with
    `oracle_goal_distance`, mean remaining Hamming `5.8`
  - best predicted action-suite row remains much worse: mean remaining Hamming
    `35.1` with changed-cell predicted goal distance

Action-conditioning/stability suite state:

- Original training array: `3760074`, `grid_goal_act_train`, `0-95%32`,
  partitions `rtxpro6k,a100`, 24h limit.
- Original outcome: training had 29 completed tasks and 67 failed tasks.
- Failure reason: CUDA OOM on 40GB A100 nodes at batch 8. RTX Pro 6000 tasks
  completed.
- Stale eval `3760099` was canceled because it was `DependencyNeverSatisfied`.
- Rerun training array: `3768285`, failed indices only
  `24-26,28-38,42,44-95%16`, originally partition `rtxpro6k`.
- Training rerun `3768285` completed all 96 checkpoints. Tail RTX Pro tasks
  finished on 2026-06-23 around 17:32-18:40 CEST; A100 task `76` finished at
  15:14 CEST after `17:35:55`.
- Replacement eval array `3768300` was canceled and replaced by split evals so
  completed checkpoints evaluate immediately:
  - `3770937`: original completed tasks `0-23,27,39-41,43%32`
  - `3770953`: completed rerun tasks `24-26,28-38,42,44%32`
  - `3770954`-`3771004`: per-task evals for rerun tasks `45-95`, each
    dependency-held with explicit `afterok:<train-task-jobid>`
  - mapping file:
    `logs/grid_goal_act_eval_split_afterok_submit_20260622_2020.tsv`
- Monitor:
  ```bash
  squeue -j 3775750,3775751
  ```
- Eval status at 2026-06-24 10:36 CEST: prior evals are complete, but the
  2026-06-23 corrected rerun/depth64 submissions passed comma-separated
  variables through `sbatch --export=...`, so Slurm split them at commas.
  Current main outputs have 96 files but only 51 complete 18-row matrices, 16
  two-row matrices, and 29 one-row matrices; current depth64 outputs have 96
  one-row matrices.
- Current depth-32 result in observed complete rows: `0` solves across 306
  rows. Best row is still `remaining_hamming_mean=5.8` for
  `R4_no_goal_nce/A6_affected_marker_delta/S4_ema_vicreg/D0_uniform` with
  `oracle_goal_distance`. The same config reaches `9.2` remaining Hamming with
  changed-cell raw oracle L2, but predicted-goal variants remain much worse
  (`35.1-36.6` remaining Hamming for the best predicted rows).
- Corrected main eval rerun: `3775750`, RTX Pro-only, 8h limit, reruns
  indices `0,1,2,5-13,24-26,28,30-35,37,38,42-44,46-60,66-68%16`.
  This uses environment inheritance rather than comma values in `--export` and
  should restore 18-row `planner_eval_latent` matrices.
- Corrected depth64 eval: `3775751`, RTX Pro-only, 12h limit, indices
  `0-95%16`, beam width `16`, beam depth `64`, max steps `81`, same six score
  modes, output `planner_eval_latent_depth64`.
- Output root:
  `$PUZZLE_JEPA_WORK_ROOT/runs/grid_goal_action_suite/grid_goal_action_<base>_<action>_<stability>_<dynamics>/`.
- Scripts:
  `scripts/slurm/run_grid_goal_action_suite_train.slurm` and
  `scripts/slurm/run_grid_goal_action_suite_eval.slurm`.

Follow-up scripts are now submitted; script details:

- `scripts/slurm/run_grid_goal_followup_train.slurm`
  - array `0-5%6`, 24h, variants `F0_dense_k16`,
    `F1_dense_k32_detach8`, `H0_hierarchy_l4_l16`,
    `H1_hierarchy_dense_l4_l16`, `S0_scale_d384_dense`,
    `S1_deeper_d256_dense`
  - all use the current best recipe: `R4_no_goal_nce`,
    `affected_marker`, `predict_delta=true`, `EMA+VICReg`, uniform dynamics
  - default max steps `45000`; override with `TRAIN_MAX_STEPS`
- `scripts/slurm/run_grid_goal_followup_eval.slurm`
  - array `0-17%12`, 24h, one row per follow-up variant and planner group
  - planner groups: beam, categorical CEM, hierarchical CEM
  - hierarchical CEM skips non-hierarchical variants
  - default eval: latent rollout, 10 boards, beam width `16`, depths
    `4,16,32,64`, six oracle/predicted score modes
- `scripts/slurm/run_grid_goal_best_checkpoint_eval.slurm`
  - eval-only sweep for existing best action-suite checkpoint at
    `20k,30k,40k,50k,60k`
  - planner groups: beam and categorical CEM

Submitted follow-up jobs at 2026-06-24 11:54 CEST, all on `rtxpro6k`:

- Training array `3776065`, `grid_goal_fu_train`, array `0-5%6`, 24h.
- Dependency-held eval arrays:
  - `3776066`, array `0-2%3`, dependency `afterok:3776065_0`
  - `3776068`, array `6-8%3`, dependency `afterok:3776065_2`
  - `3776069`, array `9-11%3`, dependency `afterok:3776065_3`
  - `3776070`, array `12-14%3`, dependency `afterok:3776065_4`
- Existing-best checkpoint-time eval array `3776072`, `grid_goal_ckpt_eval`,
  array `0-9%10`, 24h, no dependency.
- Initial training tasks `3776065_1` (`F1_dense_k32_detach8`) and
  `3776065_5` (`S1_deeper_d256_dense`) OOMed quickly at batch 8 on RTX Pro.
  Their stale eval arrays `3776067` and `3776071` were canceled.
- Failed partial run directories were preserved as:
  - `grid_goal_followup_F1_dense_k32_detach8_failed_3776065_20260624_115750`
  - `grid_goal_followup_S1_deeper_d256_dense_failed_3776065_20260624_115750`
- Replacement training array `3776086`, array `1,5%2`, 24h, `BATCH_SIZE=4`,
  partition `rtxpro6k`.
- Replacement dependency-held eval arrays:
  - `3776087`, array `3-5%3`, dependency `afterok:3776086_1`
  - `3776088`, array `15-17%3`, dependency `afterok:3776086_5`
- Current state after resubmission: `3776065_0,2,3,4` and checkpoint eval
  tasks `3776072_0-6` are running; `3776072_7-9` are pending for resources;
  replacement train array `3776086_[1,5%2]` is pending on priority with
  estimated start `2026-06-24T22:06:08`; eval arrays `3776066`,
  `3776068`, `3776069`, `3776070`, `3776087`, and `3776088` are
  dependency-held.
- Output roots:
  - follow-up training/eval:
    `$PUZZLE_JEPA_WORK_ROOT/runs/grid_goal_followups/grid_goal_followup_<variant>/`
  - checkpoint-time eval:
    `$PUZZLE_JEPA_WORK_ROOT/runs/grid_goal_action_suite/grid_goal_action_R4_no_goal_nce_A6_affected_marker_delta_S4_ema_vicreg_D0_uniform/planner_eval_checkpoint_<step>_<planner>/`

Local verification for the follow-up implementation:

- `source scripts/env.sh && python -m compileall -q puzzle_jepa tests`
- Prior compact Python smoke over all six follow-up configs and
  beam/CEM/hierarchy planner paths completed in `8.68s` on CPU, but was too
  shallow.
- Audit regressions fixed at 2026-06-24 11:50 CEST:
  - `test_categorical_cem_mpc_handles_horizon_longer_than_remaining_blanks`
  - `test_hierarchical_cem_mpc_handles_subgoal_horizon_longer_than_remaining_blanks`
  - `test_rollout_diagnostics_include_configured_long_horizons`
- Fixes:
  - beam, categorical CEM, and hierarchical CEM now cap lookahead by remaining
    blanks so the simulated future never exceeds editable cells left
  - CEM sequence sampling stops safely if a sampled sequence fills the board
  - rollout diagnostics now include configured horizons such as h32
- Verification:
  - targeted audit tests: `3 passed`
  - `source scripts/env.sh && python -m compileall -q puzzle_jepa tests`
  - `source scripts/env.sh && pytest -q` -> `70 passed`

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
in training. The 60k-step suite completed in about 1.8 to 10.8 hours per
ablation, depending mainly on the multi-step horizon ablation.

Full suite state:

- Training array: `3748789`, `rtxpro6k`, array `0-12%13`, completed all 13
  ablations at 60k steps.
- Planner eval array: `3748790`, `rtxpro6k`, array `0-12%13`, failed
  immediately during checkpoint loading because PyTorch defaulted
  `torch.load` to `weights_only=True` and rejected numpy scalar metadata in
  the trusted local checkpoint payload.
- Fix: `puzzle_jepa/eval/grid_goal_planner_matrix.py::load_checkpoint` now
  passes `weights_only=False`; regression coverage was added for checkpoint
  payloads containing numpy scalar metadata.
- Training overrides: `TRAIN_MAX_STEPS=60000`, `BATCH_SIZE=8`,
  `GRADIENT_ACCUMULATION_STEPS=1`, `LEARNING_RATE=1e-4`
- Logs: `logs/grid_goal_train_3748789_<task>.out/.err` and
  `logs/grid_goal_plan_3748790_<task>.out/.err`
- Planner eval rerun: `3749458`, `rtxpro6k`, array `0-12%13`, running on
  nodes `a2041` and `a2843`; logs are
  `logs/grid_goal_plan_3749458_<task>.out/.err`.
- Interim rerun check at 15:02 CEST: all 13 tasks are still running after
  about 6h10m; all ablations emitted diagnostics; every ablation has completed
  3/64 planner rows: symbolic-reencode/oracle-goal/beam-width-1 at depths `8`,
  `16`, and `32`. Solve rate is `0.0` so far. Completed rows are flushed to
  JSONL and will be preserved if the full matrix hits the 24h wall.
- Small raw-distance probe submitted at 16:07 CEST:
  - checkpoints: `M0_full`, `R4_no_goal_nce`, `R1_no_context_masks`,
    `R6_no_action_rank`
  - jobs: `3750392`, `3750393`, `3750394`, `3750395`
  - settings: 8 examples, symbolic re-encode only, beam widths `4,16`, beam
    depths `8,16,32,64`, scores `oracle_goal_distance` and
    `oracle_goal_raw_euclidean_distance`
  - outputs:
    `$PUZZLE_JEPA_WORK_ROOT/runs/grid_goal_sudoku_<ablation>/planner_probe_bw4_16_raw_oracle_8ex/`

## Verify

```bash
source scripts/env.sh
pytest -q
python -m compileall -q puzzle_jepa configs
bash -n scripts/slurm/run_grid_goal_sudoku_ablation.slurm
bash -n scripts/slurm/run_grid_goal_sudoku_planner_eval.slurm
```

Current verification after fixing the planner checkpoint loader:

- `source scripts/env.sh && pytest -q tests/test_grid_goal_jepa.py`:
  `13 passed`
- `source scripts/env.sh && pytest -q tests/test_grid_goal_plan_regressions.py`:
  `19 passed`
- `source scripts/env.sh && pytest -q`: `32 passed`
- `source scripts/env.sh && python -m compileall -q puzzle_jepa configs tests`:
  passed
- Real checkpoint smoke load:
  `grid_goal_sudoku_M0_full/checkpoint.pt` loads on CPU and reports
  ablation `M0_full`, max steps `60000`.

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

## Eval Rerun

The training checkpoints are complete. Planner eval rerun `3749458` is active:

```bash
squeue -j 3749458
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

Submitted training settings:

- optimizer steps: `60000`
- microbatch size: `8`
- gradient accumulation: `1`
- effective batch size: `8` full trajectories per optimizer step
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
