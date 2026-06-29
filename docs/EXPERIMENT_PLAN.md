# Experiment Plan

Last updated: 2026-06-29 13:20 CEST

## Next Wave: Staged Grid

Do not run the full Cartesian product. Use the staged scripts and advance only
when the current stage produces useful diagnostics.

Submission:

```bash
GRID_GOAL_STAGE=<stage> scripts/experiments/submit_grid_goal_next_wave.sh
```

Stages:

| Stage | Variants | Question |
| --- | --- | --- |
| `goal_conditioning` | context-only, `q(c,H0,Ht)`, `q(c,H0,Ht)` plus oracle progress | Does current-state-conditioned goal prediction close the predicted-goal gap? |
| `dense_horizon` | dense `K={2,4,8,16,32}` at fixed hierarchy `[2,4,8,16]` | How far does dense rollout improve geometry? |
| `hierarchy_levels` | `[]`, `[2]`, `[2,4]`, `[2,4,8]`, `[2,4,8,16]`, `[2,4,8,16,32]` | Does hierarchy improve geometry beyond dense rollout? |
| `predictor_delta_topk` | separate/shared predictors, residual/no residual, affected/top-k local training probe | Which predictor parameterization and local signal best preserve oracle solves? |
| `ranking_losses` | predicted/oracle/both/no progress, pairwise/listwise/no action rank | Which ranking signal improves branch discrimination? |
| `hierarchical_planning` | current best hierarchy config | Does hierarchical beam help when the hierarchy is actually used at planning time? |
| `policy_prior` | no prior, pairwise prior, listwise prior, stronger planning prior | Does a learned primitive/macro prior improve search efficiency without masking metric failures? |

Default eval per completed checkpoint:

- latent rollout only
- 10 examples
- beam width `16`
- depths `4,16,32,64`
- planners `mpc_beam` and `hierarchical_beam` when hierarchy levels exist
- oracle and predicted versions of normalized, changed-cell raw L2, and
  delta-top-k raw L2 (`k=1,3,5`)

Gate before broadening:

- predicted-goal local action ranking must improve materially
- oracle changed-cell/local solves should not regress from the H1 follow-up
  signal
- diagnostics should show whether the bottleneck is goal prediction, rollout
  drift, action discrimination, or search

## Grid-Token Goal-JEPA

The current plan replaces the CLS/vector-state LeWM architecture with a
full-grid token latent and no scalar value head.

Modules:

- Context encoder `C_omega(c)`: bidirectional transformer over Sudoku givens
  plus clue/editable/active masks.
- State encoder `f_theta(s, H_c)`: bidirectional self-attention over current
  board tokens plus cross-attention to cached context tokens.
- Markov predictor `P_phi(H_t, a_t, H_c)`: bidirectional transformer over one
  action token plus the current latent board tokens, with cross-attention to
  context. It sees only the current board latent, not a causal history.
- Goal predictor `q_eta(H_c)`: output-query decoder that predicts terminal
  board-token latents from context.
- Planner score: tokenwise normalized Euclidean distance
  `D(f_theta(s,H_c), q_eta(H_c))`.

There is no CLS token, value head, validity head, reachability head, or
dead-end head.

## Losses

The full model trains:

- multi-step dynamics MSE with self-rollout horizons `1,4,8,16`
- covariance SIGReg over active state tokens
- goal MSE against encoded true terminal board tokens
- goal InfoNCE over mean-pooled goal summaries
- progress ranking along successful trajectories only, selected by
  `oracle_mask`
- action ranking between encoded symbolic successors for target-consistent and
  wrong fill actions
- temporal straightening over valid three-frame trajectory triplets
- terminal corruption contrast against 1-5 digit corruptions

Temporal straightening computes curvature from adjacent latent velocities over
the full active grid-token latent and is independent of the predicted goal.

Follow-up objective knobs now implemented:

- dense future-state prediction: a rollout from `s_t` can supervise every
  intermediate future state `t+1...t+K`, not only the endpoint `t+K`
- optional truncated rollout gradients via `model.rollout_detach_interval`
- high-level hierarchy losses with one shared context/state encoder and
  multiple stride-specific predictors

## Ablations

## Action-Conditioning First Wave

Submitted as training array `3760074` and dependency-held eval array
`3760099`.

Grid:

- base recipes: `R4_no_goal_nce`, `R7_no_terminal_corrupt`
- action variants: `A0_action_token`, `A1_affected_marker`,
  `A2_local_action_feature`, `A3_action_cross_attention`, `A4_adaln_action`,
  `A5_action_token_delta`, `A6_affected_marker_delta`,
  `A7_local_action_feature_delta`
- stability variants: `S0_sigreg`, `S3_ema_sigreg`, `S4_ema_vicreg`
- dynamics weighting: `D0_uniform`, `D1_affected`

Eval for this wave uses latent rollout only, beam width `16`, depths
`4,16,32`, 10 boards, and normalized/raw/changed-cell metrics against oracle
and predicted goal latents.

## Follow-Up Wave

Submitted and completed after the follow-up audit regressions were fixed:

- Train script: `scripts/slurm/run_grid_goal_followup_train.slurm`
- Eval script: `scripts/slurm/run_grid_goal_followup_eval.slurm`
- Existing-checkpoint eval sweep:
  `scripts/slurm/run_grid_goal_best_checkpoint_eval.slurm`

All trained variants start from the current best recipe:
`R4_no_goal_nce/A6_affected_marker_delta/S4_ema_vicreg/D0_uniform`
(`affected_marker`, `predict_delta=true`, `EMA+VICReg`, uniform dynamics).

Variants:

| Run | Change |
| --- | --- |
| `F0_dense_k16` | Dense intermediate future-state loss, horizons `1,4,8,16` |
| `F1_dense_k32_detach8` | Dense loss, horizons `1,4,8,16,32`, detach rollout every 8 steps |
| `H0_hierarchy_l4_l16` | Hierarchy with stride-4 and stride-16 predictors |
| `H1_hierarchy_dense_l4_l16` | Hierarchy plus dense intermediate future-state loss |
| `S0_scale_d384_dense` | Wider model probe, `d_model=384`, dense loss, reduced default batch |
| `S1_deeper_d256_dense` | Deeper 256-wide model probe, dense loss |

Planning modes:

- `mpc_beam`: existing latent-rollout beam MPC
- `categorical_cem`: discrete CEM over legal Sudoku action sequences
- `hierarchical_cem`: high-level continuous latent-action CEM creates latent
  subgoals, then lower-level categorical CEM plans primitive actions to the
  next subgoal

Hierarchy follows the "Hierarchical Planning with Latent World Models" design
at the level needed here: a shared encoder/latent space, multiple temporal
predictors, latent macro-actions for high-level planning, and top-down subgoal
conditioning. There is no second state encoder.

Follow-up audit blockers:

- Fixed. Beam, categorical CEM, and hierarchical CEM cap lookahead by remaining
  blank cells, categorical CEM sampling stops safely after a sampled sequence
  fills the board, and rollout diagnostics include configured long horizons
  such as h32.

Follow-up outcome:

- `H1_hierarchy_dense_l4_l16` is the only nonzero solve result so far:
  `6/10` solved with `mpc_beam`, depth `16`, and oracle changed-cell raw L2.
- All predicted-goal rows still solved `0/10`.
- Categorical CEM and hierarchical CEM solved `0/10`.

## H1 Controlled Debug/Extra Wave

Active H1 debug sweeps:

- Delta sweep: train `3795127`, eval `3795128`.
- No-delta sweep: train `3795143`, eval `3795144`.
- Both use fixed seed `5204`, batch `8`, 45k steps, hierarchy `[4,16]`,
  `affected_marker`, EMA+VICReg, no goal NCE, context-only goal prediction,
  temporal straightening, and dense base rollout.
- Hierarchical-beam add-on evals: `3795248` for delta, `3795249` for
  no-delta.

H1-extra controlled wave:

- Train `3795246`, eval `3795247`.
- Common config: fixed seed `5204`, batch `8`, 45k steps, LR `1e-4`,
  `predict_delta=false`, `affected_marker`, dense horizons `[1,4,8,16]`,
  EMA+VICReg, no goal NCE, context-only goal predictor.
- Exception: `hier_l4_l16_hier_dense` OOMed at batch `8`; comparable
  replacement train/eval elements `3795327_11`/`3795328_11` run that one
  variant with batch `4` and grad accumulation `2`, preserving effective
  batch size `8`.
- Ranking variants: oracle/both/no progress rank; oracle/both/listwise/no
  action rank.
- Hierarchy variants: `[4]`, `[4,16,32]`, shared `[4,16]` predictor, and
  `[4,16]` with dense supervision on high-level predictors.
- Eval uses latent rollout, 10 boards, beam width `16`, depths `4,16,32,64`,
  six oracle/predicted normalized/raw/changed-cell score modes, and both
  `mpc_beam` and `hierarchical_beam`.

## Original Ablations

Run one peak LR (`1e-4`) and one seed per ablation. Use linear warmup for
`1000` optimizer steps, then cosine decay to `1e-5`.

| Run | Change |
| --- | --- |
| `M0_full` | Full Grid-Token Goal-JEPA |
| `R1_no_context_masks` | Remove explicit clue/editable context masks |
| `R2_mean_pooled_distance` | Replace tokenwise distance with mean-pooled distance |
| `R3_k1_only` | One-step dynamics only |
| `R3_k4` | Multi-step horizons `1,4` |
| `R3_k8` | Multi-step horizons `1,4,8` |
| `R3_k16` | Multi-step horizons `1,4,8,16` |
| `R4_no_goal_nce` | Remove goal InfoNCE |
| `R5_no_progress_rank` | Remove progress ranking |
| `R6_no_action_rank` | Remove action ranking |
| `R7_no_terminal_corrupt` | Remove terminal corruption contrast |
| `R8_no_sigreg` | Remove SIGReg |
| `R9_no_temporal_straightening` | Remove temporal straightening |

Training budget used for submitted suite:

- optimizer steps: `60000`
- microbatch size: `8`
- gradient accumulation: `1`
- effective batch size: `8` full trajectories per optimizer step

## Evaluation

Each completed checkpoint should run a separate eval job. The first
dependency-held eval array failed before planning on a checkpoint loader issue;
rerun eval from the completed checkpoints after the loader fix.

Planning matrix:

- MPC outer loop
- Beam search inner optimizer
- Beam widths `1,4,16,64`
- Beam depths `8,16,32,64`
- Scores: oracle goal distance and predicted goal distance
- Transitions: symbolic re-encode and latent rollout

Diagnostics record losses, latent geometry/effective rank, monotonicity,
top-positive action accuracy, near-goal corruption margin, concrete action
panels, predictor rollout drift by horizon, latent-rollout action ranking,
predicted-goal vs oracle-goal alignment, distance-vs-Hamming Spearman
correlation, action margins by fill depth, terminal corruption margins by
corruption size, planner solve rate, remaining Hamming, action-eval counts,
and timing.
