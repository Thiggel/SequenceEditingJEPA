# Current Experiments

Last updated: 2026-07-02 08:34 CEST

## Minimal-Aux 5k Single-Factor Wave

This is now the active clean sweep. It uses the H1 `minimal_aux` result as the
base and changes one ingredient at a time, with all runs trained for `5000`
optimizer steps. The goal is to preserve the strong oracle global geometry
from `minimal_aux` while identifying which single addition helps predicted-goal
planning.

Slurm:

| Array | State | Notes |
|---|---:|---|
| `3803494` train `0-28%29` | completed | all 29 tasks completed with exit `0:0`; durations were about `17-74` minutes |
| `3803495` eval `0-28%29` | completed | all 29 tasks completed with exit `0:0`; durations were about `2-6.2` hours |

Final eval snapshot:

- `456 / 456` planner rows written, with no malformed rows.
- Eval used latent rollout only, beam width `16`, depths `{4,16}`, `8` boards.
- Oracle global latent-rollout planning is strong in several 5k variants.
- Predicted-goal planning remains `0/8` for every variant and score.

Best oracle rows:

| Variant | Planner | Score | Depth | Result |
|---|---|---|---:|---|
| `geom_oracle_progress` | `mpc_beam` | oracle raw L2 | 4 | `8/8`, h `0.0` |
| `goal_distance_field_distill` | `mpc_beam` | oracle raw L2 | 4 | `8/8`, h `0.0` |
| `rank_pairwise_oracle_action` | `mpc_beam` | oracle raw L2 | 4 | `8/8`, h `0.0` |
| `reg_sigreg` | `mpc_beam` | oracle raw L2 | 4 | `8/8`, h `0.0` |
| `base` | `mpc_beam` | oracle raw L2 | 4 | `7/8`, h `0.125` |
| `hier_l4_l16` | `mpc_beam` | oracle raw L2 | 4 | `7/8`, h `0.125` |

Best predicted-goal rows:

| Variant | Planner | Score | Depth | Result |
|---|---|---|---:|---|
| `reg_vicreg` | `mpc_beam` | predicted raw L2 | 16 | `0/8`, h `32.6` |
| `reg_vicreg_sigreg` | `hierarchical_beam` | predicted raw L2 | 4 | `0/8`, h `33.8` |
| `reg_sigreg` | `mpc_beam` | predicted raw L2 | 16 | `0/8`, h `35.2` |

Interpretation:

- The base `minimal_aux` recipe is viable: `7/8` under oracle global latent
  rollout after only 5k steps.
- SIGReg, oracle progress ranking, oracle pairwise action ranking, and
  distance-field distillation each reached `8/8` under oracle global scores.
- Predicted goals remain the bottleneck; no predicted-goal row solved.
- Dense horizon changes, no-stopgrad goal targets, and q(c,H0,Ht) were harmful
  in this pass.
- Dense horizon is not a clean "longer horizon is worse" readout: the base
  rolls horizons `[1,4,8,16]` with repeated intermediate supervision and
  terminal multi-horizon dynamics terms, while `dense_k*` uses a single `[K]`
  all-steps objective with different weighting and only starts rollouts where
  the full K-step future exists.
- `minimal_aux` included hierarchy `[4,16]`, but the good rows used
  `mpc_beam`; hierarchical-beam rows were poor.
- `q(c,H0,Ht)` and no-stopgrad goal targets are not isolated goal-head
  changes because they backpropagate goal loss into the state encoder.

Prepared but not submitted: `grid_goal_dense_exact` K=8 weighting probe with
`dense_rollout_variable_starts=true`, variants `uniform`, `inverse_sqrt`, and
geometric gamma `0.8`.

Implementation:

- code commit `617ad95`
- added `model.goal_target_mode=online_no_stopgrad`, which uses the online
  encoder for the true goal latent and does not detach the goal target
- added `model.goal_distance_field_weight`, which trains `D(f(s),q)` to match
  the oracle distance field from `D(f(s),f(g*))`
- added combined `regularizer=both`
- added fill-depth goal diagnostics:
  `goal_fill_{000,025,050,075,100}_*`, `q_to_initial_mse`,
  `q_to_half_mse`, and `q_to_goal_mse`
- verification: `source scripts/env.sh && pytest -q tests/` passed;
  `compileall` and shell syntax checks passed

Training base:

- `action_conditioning=affected_marker`
- `predict_delta=true`
- `regularizer=none`
- `use_ema_target_encoder=true`
- `goal_conditioning=context`
- `goal_target_mode=target_stopgrad`
- dense rollout weight `1.0`, horizons `[1,4,8,16]`
- hierarchy levels `[4,16]`, hierarchy loss `1.0`
- no temporal straightening, no progress rank, no action rank, no terminal
  corruption, no goal NCE
- LR `1e-4`, batch `8`, no grad accumulation, `5000` steps

Variants:

| Group | Variants |
|---|---|
| Calibration | `base` |
| Regularization/EMA | `reg_vicreg`, `reg_sigreg`, `reg_no_ema`, `reg_vicreg_no_ema`, `reg_sigreg_no_ema`, `reg_vicreg_sigreg`, `reg_vicreg_sigreg_no_ema` |
| Ranking | `rank_pairwise_pred_action`, `rank_listwise_pred_action`, `rank_pairwise_oracle_action`, `rank_listwise_oracle_action` |
| Geometry | `geom_temporal`, `geom_pred_progress`, `geom_oracle_progress` |
| Dense rollout | `dense_k1`, `dense_k2`, `dense_k4`, `dense_k8`, `dense_k16` |
| Hierarchy | `hier_none`, `hier_l4`, `hier_l16`, `hier_l4_l16`, `hier_l4_l16_l32` |
| Goal prediction | `goal_initial_current`, `goal_no_stopgrad`, `goal_initial_current_no_stopgrad`, `goal_distance_field_distill` |

Eval matrix:

- transition: `latent_rollout`
- planners: `mpc_beam`, plus `hierarchical_beam` for all hierarchy-trained
  variants
- beam width `16`, depths `{4,16}`
- `8` boards
- scores: oracle/predicted global normalized distance and oracle/predicted
  global raw L2

Superseded H1 eval jobs canceled to free GPUs:
`3799697`, `3801461`, `3801460`, `3801428`, `3801429`, and `3800229_4`.

## H1 Recipe Sweep

This sweep uses `grid_goal_followup_H1_hierarchy_dense_l4_l16` as the
latent-rollout anchor and changes one ingredient at a time. The goal is to stop
mixing action conditioning, scoring, weighting, auxiliary losses, and hierarchy
changes in the same run.

Slurm:

| Array | State | Notes |
|---|---:|---|
| `3799696` train `0-3,5,6` | completed | all six wrote checkpoints and metrics |
| `3799696` train `7-16` | completed | all original non-retry tasks finished by 13:16 CEST |
| `3799777` train `4` | node failed | replacement for failed `action_old_local_concat`; A100-80GB node `a0631` failed after about 3 minutes, not an OOM |
| `3800228` train `4` | completed | A100-80GB retry for `action_old_local_concat`, batch `4`, grad accumulation `2`; completed at 15:19 CEST |
| `3799697` eval | canceled | superseded by the minimal-aux 5k wave |
| `3800229` eval `4` | canceled | superseded by the minimal-aux 5k wave |
| `3800130` oversight | canceled | canceled at user request; no Wave 2 will be auto-submitted from this job |
| `3800223` health | completed | found only the known dtype failure and made no new submissions |
| `3801426` / `3801427` depth-32 triage `0-3,5,6` | mostly completed | early checkpoint depth-32 triage rows written |
| `3801461` / `3801460` depth-32 triage `7-16` | canceled/partial | superseded by the minimal-aux 5k wave; `3801460_13` had already failed because `hier_none` has no hierarchy |
| `3801428` / `3801429` depth-32 triage `4` | canceled/partial | superseded by the minimal-aux 5k wave |

Operational note: original train task `3799696_4` failed immediately from a
bf16/float dtype mismatch in `old_local_concat`. Code commit `69d5c78` fixes
the concat path and adds a regression test. Replacement train `3799777_4`
node-failed, and retry jobs `3800228_4`/`3800229_4` are now active for that
variant.

A100-80GB trial: pending replacement `3799777_4` was broadened to a safe
RTX/A100-80 node list and started on A100-80GB node `a0631`, but the node
failed quickly. This was not an OOM. The stale eval `3799778_4` was canceled,
and retry train/eval jobs `3800228_4`/`3800229_4` were submitted on
`a100 --constraint=a100_80 --exclude=a0631` with batch `4` and grad
accumulation `2`. The retry train is running on `a0934`. Pending original tasks
`3799696_7-16` were briefly broadened too, but that worsened the grouped ETA;
they were restored to RTX-only. Health job `3800223` completed without
submitting repairs; it saw only the known non-OOM dtype failure.

Oversight job `3800130` was canceled at user request before it ran, so no Wave
2 has been submitted. Eval jobs are too large to finish all rows within the
24h partition max; `grid_goal_planner_matrix.py` now resumes safely by
skipping completed matrix cells and appending after truncated JSONL tails, so
future repair jobs can continue partial eval matrices without overwriting rows.

Training basis for `anchor_h1`:

- `action_conditioning=affected_marker`
- `predict_delta=true`
- dense rollout horizons `[1,4,8,16]`, dense weight `1.0`
- hierarchy levels `[4,16]`, hierarchy loss `1.0`
- context-only goal predictor, no goal NCE
- EMA+VICReg, temporal straightening, progress rank, action rank, terminal
  corruption
- LR `1e-4`, batch `8`, no grad accumulation, `45000` optimizer steps

Variants:

| Group | Variants | Question |
|---|---|---|
| Anchor | `anchor_h1` | Can the current code reproduce the H1 latent-rollout signal? |
| Action | `action_token`, `action_local_feature`, `action_old_local_value`, `action_old_local_concat` | Is the bottleneck action grounding? |
| Dynamics weighting | `dynamics_affected`, `dynamics_affected_context` | Does local/row/column/block weighting improve transition geometry? |
| Auxiliary losses | `no_temporal`, `no_progress`, `no_action_rank`, `no_terminal_corrupt`, `no_vicreg`, `minimal_aux` | Which geometry-shaping losses help or hurt? |
| Hierarchy | `hier_none`, `hier_l4`, `hier_l16`, `hier_l4_l16_l32` | Does hierarchy itself improve geometry/planning? |

Eval matrix per checkpoint:

- planners: `mpc_beam`, plus `hierarchical_beam` when hierarchy exists
- transitions: `symbolic_reencode,latent_rollout`
- beam width `16`, depths `{4,16,32,64}`, 10 boards
- oracle and predicted variants of normalized full-board distance, raw L2,
  raw MSE, affected-token raw L2, and affected+context raw L2

Fast depth-32 triage was added to get useful H1 signal before the full matrix
finishes. It evaluates three mode families first: `mpc_beam+symbolic_reencode`,
`mpc_beam+latent_rollout`, and `hierarchical_beam+latent_rollout`. Scores are
oracle/predicted global normalized distance, global raw L2, and changed-cell
raw L2. Output dirs are `planner_eval_h1_triage_d32_mpc` and
`planner_eval_h1_triage_d32_hier`.

Implementation note: `affected_context` uses affected-cell weight `8`, Sudoku
row/column/3x3-block context weight `2`, and base weight `1`. This is the
Sudoku instance of the general affected-token/local-context recipe.

## Old-Local Fast Wave

This sweep tests whether the old Sudoku action interface, `old_local_value`
(inject the digit/value embedding into the edited cell), recovers the strong
oracle-goal planning signal. All runs use EMA+VICReg, temporal straightening,
predicted-goal progress monotonicity, `q(c,H0,Ht)` goal conditioning, LR
`1e-4`, batch `8`, no grad accumulation, and `5000` optimizer steps.

Slurm:

| Array | State | Notes |
|---|---:|---|
| `3797928` train `0-17` | completed | all 18 checkpoints written |
| `3797929` eval `0-4` | completed | dense-horizon variants are fully evaluated |
| `3797929` eval `5-17` | running | hierarchy and ranking variants are still writing rows; `.err` files empty |

Eval stopped at `1628 / 1984` expected planner rows (`82.1%`). Tasks `6-17`
hit the 24h wall. The first nonzero solve signal is still:
`rank_listwise_both_action` reaches `6/10` with symbolic re-encode and `2/10`
with latent rollout under oracle changed-cell raw L2. Predicted-goal rows
remain at `0/10` and about `48-49` remaining Hamming.

## Variants

| Group | Variants | Meaning |
|---|---|---|
| Dense horizon | `dense_k1,k4,k8,k16,k32` | supervise every rollout step up to that horizon, no hierarchy |
| Hierarchy | `hier_l4`, `hier_l4_l16`, `hier_l4_l16_l32`, `hier_l4_l16_shared`, `hier_l4_l16_hier_dense` | hierarchy depth/parameter sharing/dense high-level supervision |
| Ranking | `rank_*` | progress/action ranking target or mode ablations on top of `[4,16]` hierarchy |

## Training

| Variant | Params | Final loss | Dyn | Dense | Hier | Progress | Action |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense_k1 | 27.9M | 1.728 | 0.016 | 0.000 | 0.000 | 0.124 | 1.326 |
| dense_k4 | 27.9M | 1.711 | 0.014 | 0.010 | 0.000 | 0.122 | 1.347 |
| dense_k8 | 27.9M | 1.653 | 0.015 | 0.010 | -0.000 | 0.123 | 1.306 |
| dense_k16 | 27.9M | 1.714 | 0.017 | 0.010 | -0.000 | 0.124 | 1.313 |
| dense_k32 | 27.9M | 1.622 | 0.019 | 0.008 | 0.000 | 0.123 | 1.282 |
| hier_l4 | 33.2M | 1.673 | 0.013 | 0.007 | 0.007 | 0.118 | 1.351 |
| hier_l4_l16 | 37.5M | 1.696 | 0.014 | 0.008 | 0.011 | 0.121 | 1.344 |
| hier_l4_l16_l32 | 41.8M | 1.594 | 0.018 | 0.007 | 0.016 | 0.121 | 1.270 |
| hier_l4_l16_shared | 33.2M | 1.664 | 0.014 | 0.007 | 0.009 | 0.122 | 1.306 |
| hier_l4_l16_hier_dense | 37.5M | 1.636 | 0.014 | 0.008 | 0.016 | 0.120 | 1.305 |
| rank_oracle_progress | 37.5M | 1.695 | 0.015 | 0.008 | 0.012 | 0.123 | 1.317 |
| rank_both_progress | 37.5M | 1.728 | 0.016 | 0.008 | 0.013 | 0.128 | 1.351 |
| rank_no_progress | 37.5M | 1.505 | 0.015 | 0.009 | 0.012 | 0.000 | 1.313 |
| rank_pairwise_oracle_action | 37.5M | 1.511 | 0.014 | 0.007 | 0.011 | 0.128 | 1.149 |
| rank_pairwise_both_action | 37.5M | 1.611 | 0.012 | 0.006 | 0.009 | 0.127 | 1.277 |
| rank_listwise_pred_action | 37.5M | 3.527 | 0.023 | 0.013 | 0.017 | 0.148 | 3.015 |
| rank_listwise_both_action | 37.5M | 3.679 | 0.028 | 0.015 | 0.021 | 0.136 | 3.014 |
| rank_no_action | 37.5M | 0.167 | 0.007 | 0.004 | 0.004 | 0.115 | 0.000 |

## Partial Planner Results

H1 recipe now has `505` partial planner rows. The depth-32 triage has a strong
latent result: `minimal_aux` solved `10/10` with `hierarchical_beam +
latent_rollout` under oracle global normalized/raw L2 distance at depth 32.
The same variant solves `10/10` with `mpc_beam + symbolic_reencode` under
oracle global distance in broad and triage rows. The `mpc_beam +
latent_rollout` rows for `minimal_aux` have not appeared yet, so the
non-hierarchical latent-rollout comparison is still unknown. Predicted-goal
rows remain `0/10`.

| H1 recipe variant | Rows | Expected | Best current row |
|---|---:|---:|---|
| `minimal_aux` | 12 broad + 12 triage | 160 broad + 18 triage | `10/10`, h `0.0`, hierarchical latent, oracle global, depth 32 |
| `dynamics_affected_context` | 21 broad + 18 triage | 160 broad + 18 triage | `1/10`, h `4.4`, mpc latent, oracle raw L2, depth 32 |
| `dynamics_affected` | 21 broad + 18 triage | 160 broad + 18 triage | `1/10`, h `4.9`, symbolic, oracle changed-cell, depth 32 |
| `hier_l4_l16_l32` | 10 broad + 9 triage | 160 broad + 18 triage | `1/10`, h `5.9`, symbolic, oracle changed-cell, depth 32 |
| `action_token` | 30 broad + 18 triage | 160 broad + 18 triage | `0/10`, h `2.5`, mpc latent, oracle raw L2, depth 32 |

Old-local fast best rows so far:

| Variant | Rows | Expected | Best symbolic oracle | Best latent oracle | Best latent predicted |
|---|---:|---:|---|---|---|
| `dense_k1` | 64 | 64 | `0/10`, h `37.6` | `0/10`, h `44.4` | `0/10`, h `48.9` |
| `dense_k4` | 64 | 64 | `0/10`, h `40.0` | `0/10`, h `41.4` | `0/10`, h `48.9` |
| `dense_k8` | 64 | 64 | `0/10`, h `38.5` | `0/10`, h `42.8` | `0/10`, h `48.9` |
| `dense_k16` | 64 | 64 | `0/10`, h `38.7` | `0/10`, h `45.2` | `0/10`, h `49.0` |
| `dense_k32` | 64 | 64 | `0/10`, h `41.4` | `0/10`, h `41.8` | `0/10`, h `49.3` |
| `rank_oracle_progress` | 86 | 128 | `0/10`, h `13.1` | `0/10`, h `3.1` | `0/10`, h `48.3` |
| `rank_pairwise_oracle_action` | 86 | 128 | `0/10`, h `5.5` | `0/10`, h `3.8` | `0/10`, h `47.5` |
| `rank_listwise_pred_action` | 86 | 128 | `0/10`, h `4.8` | `0/10`, h `5.4` | `0/10`, h `49.1` |
| `rank_listwise_both_action` | 86 | 128 | `6/10`, h `0.4` | `2/10`, h `2.4` | `0/10`, h `48.5` |
| `rank_no_action` | 86 | 128 | `0/10`, h `6.1` | `0/10`, h `15.8` | `0/10`, h `49.5` |

## Early Diagnostics

| Variant | Oracle Spearman | Pred Spearman | Oracle top-1 | Pred top-1 | Drift h16 | Goal cosine |
|---|---:|---:|---:|---:|---:|---:|
| dense_k1 | 0.414 | 0.998 | 0.031 | 0.094 | 0.851 | 0.847 |
| dense_k4 | 0.935 | 0.997 | 0.125 | 0.094 | 0.054 | 0.859 |
| dense_k8 | 0.698 | 0.997 | 0.094 | 0.156 | 0.032 | 0.887 |
| dense_k16 | 0.531 | 0.998 | 0.062 | 0.125 | 0.038 | 0.849 |
| dense_k32 | 0.717 | 0.994 | 0.031 | 0.062 | 0.043 | 0.891 |
| rank_oracle_progress | 0.996 | 0.991 | 0.062 | 0.125 | 0.042 | 0.805 |
| rank_pairwise_oracle_action | -0.536 | 0.997 | 0.781 | 0.094 | 0.026 | 0.884 |
| rank_no_action | 0.487 | 0.997 | 0.188 | 0.219 | 0.019 | 0.979 |

## Interpretation

- The H1 recipe post-eval oversight was canceled and no Wave 2 was scheduled.
- The health oversight did run and made no submissions. It saw only the known
  non-OOM dtype failure from `3799696_4`.
- Dense rollout horizon alone does not recover the old local-action signal:
  `dense_k1` through `dense_k32` are complete and all solve `0/10`.
- The strongest current signal is ranking plus old-local action conditioning:
  `rank_listwise_both_action` solves `6/10` with symbolic re-encode and `2/10`
  with latent rollout under oracle changed-cell scoring.
- The H1 `minimal_aux` result changes the read: removing a large block of
  auxiliary losses can produce very strong oracle global geometry, including
  hierarchical latent-rollout planning. It still does not validate the
  predicted-goal planner, because predicted-goal rows remain poor.
- `minimal_aux` removes temporal straightening, progress ranking, action
  ranking, terminal corruption, and VICReg. It still keeps the core dynamics
  loss, dense rollout loss, hierarchy loss, goal MSE, EMA target encoder, and
  context-only goal predictor.
- The successful H1 `minimal_aux` predicted-goal head is context-only
  `q(c)`, so recomputing it at each planner step does not make it depend on
  the current board. The current-conditioned `q(c,H0,Ht)` behavior belongs to
  the old-local fast wave and next-wave configs, where predicted-goal planning
  is still failing.
