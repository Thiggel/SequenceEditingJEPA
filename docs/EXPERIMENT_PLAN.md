# Experiment Plan

Last updated: 2026-06-13 16:56 CEST

The active source-of-truth backlog lives in
`../sequence-editing-report/BACKLOG.md`. The clean Grid5-only plan/backlog/log
live in `../sequence-editing-report/GRID5_PLAN.md`,
`../sequence-editing-report/GRID5_BACKLOG.md`, and
`../sequence-editing-report/GRID5_LOG.md`.

## Active: Grid 6 Causal Trajectory JEPA

Goal: test whether a sequence-conditioned JEPA representation repairs the
Grid5 failure mode where compact single-state latents were monotone along gold
paths but bad at candidate-action ranking.

Submitted Slurm arrays:

- Training: `3739195_[0-1]`, completed cleanly
- Dependent eval: `3739196_[0-1]` with `afterok:3739195`, running on
  `a40/a1721` at 2026-06-13 22:53 CEST, elapsed `08:12:37`

Architecture:

- board stem receives current board, initial puzzle board, and clue mask
- causal encoder attends over past board/action history only
- target encoder is a frozen EMA copy of the online causal encoder
- action-chunk encoder compresses the intervening primitive actions for a
  requested horizon
- horizon predictor predicts the future target latent from the causal current
  latent and the action chunk
- loss is JEPA latent MSE + SIGReg + learned terminal `goal_energy`
- no anti-causal target encoder branch in this first run

Matrix:

| Task | Run | Horizons |
| ---: | --- | --- |
| 0 | `grid6_causal_traj_k1_d320` | `[1]` |
| 1 | `grid6_causal_traj_mh_d320` | `[1,2,4,8,16]` |

Fixed scale:

- `d_model=320`, action embedding `32`
- encoder layers `4`, predictor layers `4`, action-chunk layers `2`
- hidden/intermediate size `1280`
- trainable params `15.70M`; total params with EMA target `23.25M`
- rollout length `32`, batch size `192`, max steps `5000`
- rollout training mix `50%` oracle/correct and `50%` wrong/random mutable
  trajectories

Eval gate:

- planner axis: Beam, CEM, diagnostic MCTS
- transition axis: exact symbolic board application plus re-encode at horizon
  vs latent-only predictor rollout
- score axis: oracle solved-board latent distance (`latent_goal`) vs learned
  terminal energy (`goal_energy`)
- horizon axis: `4/8/16` and mean-prefix score over `1/2/4/8/16`

Pre-submit verification passed: compile, Slurm syntax, focused Grid6 pytest,
combined Grid6+Hydra pytest, one-step train smoke, and planner CLI smoke.

Training/eval read at 2026-06-13 22:53 CEST: both training tasks completed with empty
stderrs and finite final losses. K1 final eval loss is `0.01431`;
multi-horizon final eval loss is `0.02343`. The eval array is still running
with partial streamed JSONL records only so far: beam + symbolic re-encode +
oracle `latent_goal` solves `0/2`, with K1 mean remaining Hamming `44.0` at
h4/h8/h16 and multi-horizon `50.5` for oracle symbolic modes.

## Background: Grid 5 SIGReg Single-State JEPA

Goal: restart the Sudoku JEPA experiments around a compact single-state latent
with JEPA latent MSE plus SIGReg always enabled, and diagnose whether the latent
geometry is planner-ready.

Submitted Slurm array: `3722613_[0-23]`; completed cleanly with all tasks
exit `0:0`.

| Factor | Values |
| --- | --- |
| Encoder | one-hidden-layer board `mlp`; bidirectional `cls_transformer` |
| Predictor | one-hidden-layer `mlp`; causal `ar_transformer` over state/action sequence |
| Prediction target | full next latent; residual delta |
| Latent size | `32`, `64`, `128` |

Fixed defaults:

- `sigreg_weight=1.0`
- terminal-energy head enabled and trained in every model
- `action_size=16`
- rollout length `8`
- rollout training mix `50%` oracle/correct and `50%` wrong/random mutable
  trajectories
- batch size `512`, max steps `5000`

## Gate

Grid 5 failed the gate. The reason is not training/Slurm failure: all variants
produced metrics and diagnostics. The failure is representational/planning:
oracle latent distances are mostly monotone along gold trajectories, but
all-action ranking is poor and planning solves `0/16` under both oracle
`latent_goal` and learned `goal_energy` for every variant.

The decisive diagnostics to inspect are:

- Does SIGReg produce approximately healthy latent spread?
- Is oracle latent-goal distance monotone along oracle trajectories?
- Does oracle latent-goal distance rank adjacent/action candidates correctly?
- Does the learned terminal-energy head track and rank the same candidates?
- Does small enumerated beam planning solve or at least reduce remaining
  Hamming under oracle latent distance?

Observed answer: monotone trajectories mostly pass, action ranking and planning
fail. The best oracle variant is `grid5_sigreg_mlp_mlp_delta_z128` with mean
remaining Hamming `44.88`, latent monotone rate `0.992`, latent gold-action
top-1 `0.031`, and latent top-goal-value rate `0.156`.

Next decision: do not spend on more planner variants for this compact
single-state geometry until the action-ranking objective/representation is
changed. Grid 4Z has since completed and also failed, but it remains a useful
tokenized/local-control reference point for the next repair branch.

## Active Posthoc: Grid 5 MPC-CEM Lookahead

Completed as `3724325_[0-23]`.

Purpose: check whether the failed Grid 5 read was partly caused by the cheap
enumerated beam diagnostic rather than by the latent geometry. This follows the
LeWorldModel-style planning recipe more closely:

- CEM optimizes a sequence of symbolic Sudoku actions in latent space.
- Candidate sequences are rolled through the JEPA predictor.
- The final predicted latent is scored against the solved-board latent.
- MPC executes only the first action, updates the symbolic board, re-encodes,
  and replans.
- Horizons: `4`, `8`, `16`, `32`, `64`.

Artifacts:

- `diagnostics_mpc_cem/mpc_cem_summary.json`
- `diagnostics_mpc_cem/mpc_cem_records.jsonl`
- `diagnostics_mpc_cem/mpc_cem_root_actions.jsonl`
- `diagnostics_mpc_cem/mpc_cem_lookahead_examples.jsonl`

Result: failed. All 24 checkpoints solved `0` under every horizon and score.
Mean remaining Hamming improved only mildly with lookahead:

- h4: latent `52.98`, learned energy `53.41`
- h16: latent `51.90`, learned energy `52.48`
- h64: latent `51.49`, learned energy `51.48`

This means planner horizon alone did not rescue the original Grid 5 geometry.

## Active: Recursive Rollout Training Sweep

Delta-target sweep `3724413_[0-5]` completed cleanly.
Full-state-target sweep `3724500_[0-5]` completed cleanly.

Purpose: train the same recursive prediction mode that MPC-CEM uses. The base
Grid 5 model trained mostly teacher-forced one-step prediction over rollout
segments. The new loss adds recursive predicted-latent rollout supervision:

```text
z_t, a_t -> zhat_{t+1}
zhat_{t+1}, a_{t+1} -> zhat_{t+2}
...
```

The recursive loss compares each `zhat_{t+h}` to the encoded target
`z_{t+h}` for horizons up to K, from every valid start inside the sampled
rollout segment.

Matrix:

| Factor | Values |
| --- | --- |
| Predictor | `mlp`, `ar_transformer` |
| Recursive K | `2`, `4`, `8` |
| Target | delta in `3724413`; full-state in `3724500` |

Fixed base:

- encoder `mlp`
- latent size `128`
- recursive loss weight `1.0`
- K=8 uses 16-step sampled rollout segments; K=2/4 use 8-step segments
- each job runs standard diagnostics plus MPC-CEM horizons `4/8/16/32/64`

Result: failed solve gate. All 12 recursive variants solved `0` under oracle
`latent_goal` and learned `goal_energy` in MPC-CEM at horizons `4/8/16/32/64`;
terminal rate stayed `0.0`. Best MPC-CEM proximity was
`grid5_recursive_mlp_mlp_delta_z128_k2` with oracle `latent_goal` at h64, mean
remaining Hamming `49.88`. Best learned `goal_energy` proximity was
`grid5_recursive_mlp_ar_transformer_state_z128_k2` at h64, mean remaining
Hamming `50.50`.

Decision: do not submit more compact single-state Grid 5 planner variants
without changing the representation/objective. Recursive training did not
repair the action-ranking/planner geometry.

Follow-up symbolic re-encode probe confirms this decision. Even when candidate
futures are exact symbolic Sudoku boards and are re-encoded before scoring,
oracle `latent_goal` and learned `goal_energy` solve `0/4` across horizons
`8/16/32/64/full`. A perfect true-Hamming cost gets much closer but still does
not solve at the same CEM budget, so the next grid should not be just "more CEM"
on the same compact scorer. The useful next branches are a more LeWM-faithful
encoder/SIGReg setup, a direct action/constraint ranking objective, or a
hierarchical setup only after the low-level symbolic/re-encode scorer improves.

Grid 5C is now a negative gate for the compact single-state scorer. The first
full-matrix attempt exceeded wall time, but small probe `3728790` completed and
covered the planner/transition/score axes on the best Grid5B checkpoint. Oracle
`latent_goal` with `symbolic_reencode` improved one board from start Hamming
`55` to `37` under MCTS, but still solved `0/1`; latent rollout stayed near
the start (`53-55`), and learned `goal_energy` remained weak (`49-54`).

## Active: Grid 5B 10M Stabilizer Screen

Submitted as `3724634_[0-11]` via
`scripts/slurm/run_grid5b_10m_stabilizer_screen.slurm`. Original tasks `0-5`
hit Slurm `NODE_FAIL` on node `a2143` with empty stderr and were resubmitted as
`3724689_[0-5]` with that node excluded. The rerun and original tasks `6-11`
completed cleanly, so all 12 Grid5B runs have final diagnostics.

Purpose: test whether the compact single-state failure was mainly capacity or
stabilization, before spending on hierarchy.

Fixed large scale:

- latent size `512`
- hidden size `3072`
- action embedding size `64`
- one transformer layer for transformer encoder/predictor variants
- batch size `512`, max steps `5000`
- trainable params about `10.6M-13.4M`

Screen:

| Task | Run | Main Contrast |
| ---: | --- | --- |
| 0 | `canonical_sigreg_k4` | CLS encoder + AR predictor + full target + SIGReg + K4 |
| 1 | `canonical_ema_sigreg_k4` | task 0 plus EMA target encoder |
| 2 | `canonical_vicreg_k4` | task 0 with VICReg instead of SIGReg |
| 3 | `canonical_ema_vicreg_k4` | VICReg plus EMA target encoder |
| 4 | `canonical_sigreg_k1` | task 0 with one-step loss only |
| 5 | `canonical_ema_sigreg_k1` | EMA SIGReg one-step loss |
| 6 | `delta_sigreg_k4` | delta target instead of full target |
| 7 | `mlp_pred_sigreg_k4` | MLP predictor instead of AR predictor |
| 8 | `mlp_enc_sigreg_k4` | MLP encoder instead of CLS encoder |
| 9 | `oldbest_scaled_sigreg_k4` | scaled old best: MLP encoder + MLP predictor + delta |
| 10 | `oldbest_scaled_ema_sigreg_k4` | scaled old best plus EMA target |
| 11 | `oldbest_scaled_sigreg_k1` | scaled old best one-step loss |

Gate read: capacity/stabilization improved proximity but not exact solving.
Best symbolic oracle result is `grid5b_10m_canonical_ema_vicreg_k4`, h8 mean
remaining Hamming `41.00`, solve `0/4`; cheap beam oracle mean remaining
Hamming is `29.56` with latent top-goal-value rate `0.969`. Predicted-latent
MPC-CEM still solves `0` for all variants. Treat Grid5C as the deciding
planner/transition/score gate before launching any next experiment.

## Active: Grid 5C Planner Matrix

Submitted as dependent eval jobs using
`scripts/slurm/run_grid5c_planner_matrix_eval.slurm`:

- `3724691_[0-5]`, after `3724689`
- `3724698_[9-11]`, started immediately for completed old-best tasks
- `3724700_[6]`, after `3724634_6`
- `3724701_[7]`, after `3724634_7`
- `3724702_[8]`, after `3724634_8`

All full-matrix tasks timed out without usable summaries:
`3724691_[0-5]`, `3724698_[9-11]`, `3724700_6`, `3724701_7`, and
`3724702_8`. Checked stderrs contain only Slurm time-limit messages.

Purpose: test the planner axes requested after Grid 5B without retraining.
Every read is MPC: plan a horizon, execute the first symbolic action, update
the board, then replan.

Planner axis:

- `beam`
- `mcts`
- `nn_cem`: CEM in the continuous action-embedding space, decoded each step to
  the nearest currently valid mutable-cell symbolic action

Transition axis:

- `symbolic_reencode`: apply the candidate sequence to the exact Sudoku board,
  encode the horizon board, then score
- `latent_rollout`: keep the symbolic board only for valid action decoding, but
  score the recursively predicted latent after the horizon

Score axis:

- `latent_goal`: oracle solved-board latent MSE, lower is better
- `goal_energy`: learned terminal-energy head, lower is better

Gate: if symbolic re-encode plus oracle scoring works but latent rollout does
not, the failure is still predictor drift. If oracle symbolic re-encode fails
for all three optimizers, the compact scorer geometry/action parameterization
is the blocker. If learned energy fails while oracle works, the learned scorer
remains the blocker.

Actual small-probe read: oracle symbolic re-encode did not solve even on one
board, although it was less bad than latent rollout. A follow-up geometry probe
on `grid5b_10m_canonical_ema_vicreg_k4` found learned true-terminal top1
`0/16` among one-cell corrupt terminal boards, latent/Hamming nearest-neighbor
Spearman `0.133`, and best wrong action displacement beating the gold action's
goal-direction cosine in `84.4%` of sampled states. This follows the
oracle-symbolic-reencode-fails branch: do not scale the planner matrix or add
hierarchy on this compact scorer before repairing geometry/action ranking.

## Next Decision Tree After Grid 5C

Grid 5C identified the bottleneck as compact scorer geometry/action
parameterization. Do not submit a broad planner grid on this representation.

Candidate repair branch: causal trajectory JEPA. Drop the anti-causal
future-summary branch for now. The next clean test is a causal trajectory
encoder over past boards/actions, an EMA target copy of the same causal
encoder, and an action-chunk-conditioned predictor. Run exactly two first-pass
jobs: one-step JEPA+SIGReg and the same architecture with multi-horizon losses
for K in `{1,2,4,8,16}`. For each horizon, omit positions without K remaining
steps in the sampled trajectory. Gate on exact symbolic-board ranking and
MPC/beam/MCTS planning with symbolic re-encode and latent rollout.

If Grid 5C works under oracle `latent_goal` with `symbolic_reencode`:

- Treat the compact representation as potentially viable.
- Scale the best planner read to more boards, then compare exact solve rate,
  remaining Hamming, root goal-value rate, and runtime.
- If `latent_rollout` is still worse than `symbolic_reencode`, train for
  longer rollout fidelity: recursive K `8/16/32`, scheduled re-encoding,
  and predictor consistency against re-encoded horizon states.
- If `goal_energy` is worse than `latent_goal`, keep the world model but train
  a better learned scorer from the oracle action-ranking signal.

If Grid 5C works only for one optimizer:

- If `beam` wins, the best next planner is structured discrete search with
  better pruning, not continuous action optimization.
- If `mcts` wins, add progressive widening, cached re-encoded leaf scoring,
  and a cheap default rollout policy before scaling.
- If `nn_cem` wins, keep continuous action-embedding planning and test
  gradient/CEM hybrids plus vector-quantized action embeddings.

If Grid 5C does not work even for oracle `latent_goal` with
`symbolic_reencode`:

- Stop scaling compact single-state JEPA as-is; hierarchy would sit on a weak
  low-level scorer.
- Move to objectives that directly shape candidate-action ordering:
  action-conditioned advantage/ranking, multi-positive feasible successor
  contrastive learning, and verifier-style terminal/constraint heads.
- Reintroduce the tokenized/local representation as the control, because the
  old re-encoded oracle planner solved Sudoku and therefore isolates what the
  compact bottleneck lost.
- Keep symbolic true-Hamming/constraint scoring only as a diagnostic upper
  bound, not as the target recipe for Maze/ARC.

If Grid 5B capacity/stabilizer helps but still does not solve:

- Isolate the winning factor with small follow-up jobs rather than expanding a
  full factorial grid.
- Prefer the branch that improves exact symbolic-board action ranking, not the
  branch with only lower one-step JEPA loss.

Only after a low-level scorer passes the exact symbolic-board ranking gate
should hierarchy be retried. Then use HWM-style macro-action encoders and make
the top-level scorer rank reachable chunks/subgoals, not arbitrary latent
states.

## Historical

Old `grid0`-`grid4` configs and Slurm wrappers were removed from the active
tree. Historical results remain in `../sequence-editing-report`.
