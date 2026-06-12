# Experiment Plan

Last updated: 2026-06-12 15:54 CEST

The active backlog lives in `../sequence-editing-report/BACKLOG.md`.

## Active: Grid 5 SIGReg Single-State JEPA

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

Next decision: do not spend on planner variants for this compact single-state
geometry until the action-ranking objective/representation is changed, or wait
for the still-running tokenized Grid 4Z control to decide whether tokenized
local geometry remains the better base.

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

## Active: Grid 5B 10M Stabilizer Screen

Submitted as `3724634_[0-11]` via
`scripts/slurm/run_grid5b_10m_stabilizer_screen.slurm`. Original tasks `0-5`
hit Slurm `NODE_FAIL` on node `a2143` with empty stderr and were resubmitted as
`3724689_[0-5]` with that node excluded. Original tasks `9-11` and `8`
completed; tasks `6-7` were still running at the last check.

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

Gate: the first pass is not exact solve only. Read symbolic re-encode action
ranking, K=32 drift, and symbolic re-encode MPC-CEM before judging. If no
variant improves symbolic re-encode ranking/proximity, the compact latent path
needs a different objective or a tokenized/verifier control before hierarchy.

## Active: Grid 5C Planner Matrix

Submitted as dependent eval jobs using
`scripts/slurm/run_grid5c_planner_matrix_eval.slurm`:

- `3724691_[0-5]`, after `3724689`
- `3724698_[9-11]`, started immediately for completed old-best tasks
- `3724700_[6]`, after `3724634_6`
- `3724701_[7]`, after `3724634_7`
- `3724702_[8]`, after `3724634_8`; task `8` has started

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

## Next Decision Tree After Grid 5C

Do not submit the next training grid until Grid 5C identifies the bottleneck.

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
