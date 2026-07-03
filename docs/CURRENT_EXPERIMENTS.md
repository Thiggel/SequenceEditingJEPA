# Current Experiments

Last updated: 2026-07-03 13:08 CEST

Source of truth: `../sequence-editing-report/CURRENT_EXPERIMENTS.md`.

## Active: Metric/Value Geometry Ablation

This sweep tests whether a separate value metric projection can turn JEPA
latents into a planning distance that works without oracle terminal states.
It uses the current best clean dynamics base, `K8_smooth_count`, and evaluates
projected oracle-goal distance separately from projected predicted-goal
distance.

Implementation:

- `P_src(E(s))` and `P_goal(E(g))` metric projection heads, with asymmetric
  source/goal variants
- projected metric goal MSE for the goal decoder
- terminal-progress, hindsight future-state, and contrastive future-vs-bad
  metric losses
- Sudoku bad-state labels from wrong digits and duplicate row/col/box values
- bad-state BCE and bad-margin losses
- projected eval score modes route through the new metric heads

Scripts:

- `scripts/slurm/run_grid_goal_metric_geometry_train.slurm`
- `scripts/slurm/run_grid_goal_metric_geometry_eval.slurm`

Train/eval jobs:

| Variant | Train | Oracle projected eval | Predicted projected eval |
|---|---:|---:|---:|
| `FB_M0_goalpred_mse` | `3808505` | `3808506` | `3808507` |
| `FB_M1_terminal_progress_bad_fix1` | `3808548` | `3808549` | `3808550` |
| `FB_M2_hindsight_bad_fix1` | `3808551` | `3808552` | `3808553` |
| `FB_M3_contrastive_bad_fix1` | `3808554` | `3808555` | `3808556` |
| `FB_M4_terminal_progress_asym_fix1` | `3808557` | `3808558` | `3808559` |
| `FB_M5_hindsight_asym_fix1` | `3808560` | `3808561` | `3808562` |
| `SV_M0_goalpred_mse` | `3808523` | `3808524` | `3808525` |
| `SV_M1_terminal_progress_bad_fix1` | `3808563` | `3808564` | `3808565` |
| `SV_M2_hindsight_bad_fix1` | `3808566` | `3808567` | `3808568` |
| `SV_M3_contrastive_bad_fix1` | `3808569` | `3808570` | `3808571` |
| `SV_M4_terminal_progress_asym_fix1` | `3808572` | `3808573` | `3808574` |
| `SV_M5_hindsight_asym_fix1` | `3808575` | `3808576` | `3808577` |

Current state at 12:35 CEST:

- all 12 active training jobs completed with exit `0:0`
- all 24 oracle/predicted projected-distance eval jobs completed with exit
  `0:0`
- initial M1-M3 bad-state rows failed at step 1 with NaN gradient from
  differentiating `sqrt(distance)` at zero; the loss now uses an epsilon-safe
  square root and has a backward-gradient regression test
- superseded M1-M5 train/eval jobs `3808508`-`3808522` and `3808526`-`3808540`
  were canceled or superseded

Eval settings: `mpc_beam`, latent rollout, beam width `16`, depths `{4,16}`,
8 boards, one job for `oracle_goal_projected_euclidean_distance` and one job
for `predicted_goal_projected_euclidean_distance`.

Best rows:

| Variant | Best oracle projected | Best predicted projected |
|---|---|---|
| `FB_M0_goalpred_mse` | `0/8`, h `2.375` | `0/8`, h `44.5` |
| `FB_M1_terminal_progress_bad_fix1` | `0/8`, h `46.375` | `0/8`, h `49.875` |
| `FB_M2_hindsight_bad_fix1` | `8/8`, h `0.0` | `0/8`, h `40.875` |
| `FB_M3_contrastive_bad_fix1` | `0/8`, h `48.875` | `0/8`, h `48.875` |
| `FB_M4_terminal_progress_asym_fix1` | `0/8`, h `28.375` | `0/8`, h `44.125` |
| `FB_M5_hindsight_asym_fix1` | `8/8`, h `0.0` | `0/8`, h `44.25` |
| `SV_*` | `0/8`, h about `49-51` | `0/8`, h about `49-51` |

Interpretation: full-board hindsight metric supervision works as an oracle
projected planning geometry. Predicted-goal planning remains unsolved, and the
single-vector latent is not viable in this fast sweep.

## Active: Delta-JEPA / Single-State Ablation

Implementation and fidelity fixes are complete. The first Delta submission used
one-step dynamics and was superseded after the horizon ablation showed
`K8_smooth_count` is the current best base. The old Delta evals were canceled;
replacement K8/smooth-count train variants were submitted as independent Slurm
jobs so each eval can depend on its matching checkpoint.

Scripts:

- `scripts/slurm/run_grid_goal_delta_jepa_train.slurm`
- `scripts/slurm/run_grid_goal_delta_jepa_eval.slurm`

Grid:

- full board latent with Delta-JEPA LDAD, crossing meaningful dynamics target
  modes and goal regularizer off/on: 6 variants
- single hidden-state board latent with learned CLS encoder token, causal
  history predictor, Delta-JEPA, and goal regularizer off/on: 2 variants

Delta-JEPA defaults in these jobs: `dynamics_target_mode=online_no_stopgrad`,
no SIGReg/VICReg, `delta_action_weight=10`, LDAD horizons `[1,2,3,4,5]`,
K8 dense rollout with `dense_rollout_all_steps=true` and
`dense_rollout_weighting=smooth_count`, and no temporal/ranking/corruption
auxiliaries.

Train/eval jobs:

| Variant | Train | Oracle eval | Predicted eval |
|---|---:|---:|---:|
| `FB_online_noema_nogoal` | `3808387` | `3808388` | n/a |
| `FB_online_noema_goal` | `3808389` | `3808390` | `3808391` |
| `FB_stopgrad_noema_nogoal` | `3808392` | `3808393` | n/a |
| `FB_stopgrad_noema_goal` | `3808394` | `3808395` | `3808396` |
| `FB_stopgrad_ema_nogoal` | `3808397` | `3808398` | n/a |
| `FB_stopgrad_ema_goal` | `3808399` | `3808400` | `3808401` |
| `SV_online_nogoal` | `3808402` | `3808403` | n/a |
| `SV_online_goal` | `3808404` | `3808405` | `3808406` |

Eval is dependency-held per checkpoint and split by goal-distance mode:

- oracle rows use `oracle_goal_raw_euclidean_distance`
- predicted rows use `predicted_goal_raw_euclidean_distance` only for
  goal-trained variants
- output dirs are suffixed with `_oracle` or `_predicted`
- each eval uses `mpc_beam`, latent rollout plus symbolic re-encode, beam width
  `16`, depths `{4,16}`, and 8 boards

Current Slurm state at 13:08 CEST:

- replacement train jobs `3808387`, `3808389`, `3808392`, `3808394`,
  `3808397`, `3808399`, `3808402`, `3808404` completed successfully
- evals `3808388`, `3808390`, `3808391`, `3808393`, `3808395`, `3808396`,
  `3808398`, `3808400`, `3808401`, `3808403`, `3808405`, and `3808406`
  failed at checkpoint load because the evaluator was too strict about
  optional metric/bad-state heads added after the checkpoints
- the loader was fixed and covered by a regression test
- replacement evals are running: oracle `3808863`-`3808870`, predicted-goal
  `3808871`-`3808874`
- superseded one-step Delta evals `3808222`, `3808224`, `3808225`, `3808227`,
  `3808229`, `3808230`, `3808232`, `3808234`, `3808235`, `3808237`,
  `3808239`, `3808240` were canceled

Fidelity fixes:

- `delta_action_weight > 0` now requires non-empty `delta_action_horizons`.
- `use_ema_target_encoder=true` is rejected with `online_no_stopgrad`
  dynamics, because that combination would be a no-op.
- Single-state latent-rollout planning now passes growing state/action history
  to the causal predictor.
- Full-board Delta goal variants use `goal_conditioning=context_current`.

## Active: Horizon-Length Ablation

This sweep tests whether the multi-step dynamics horizon itself is the
important factor, using only the clean one-long-rollout path. It does not use
the old legacy multi-horizon rollout code.

Fixed base:

- dropout off: `model.dropout=0.0`
- no residual delta: `model.predict_delta=false`
- one recursive rollout: `model.dense_rollout_all_steps=true`
- no hierarchy: `model.hierarchy_levels=[]`
- context-goal MSE on, goal NCE off
- no SIGReg/VICReg, temporal straightening, progress rank, action rank, or
  terminal corruption
- seed `5204`, LR `1e-4`, batch `8`, `5000` steps

Grid:

| Horizon | Uniform job | Smooth/count job |
|---:|---|---|
| 1 | train `3807867`, eval `3807868` | train `3807869`, eval `3807870` |
| 2 | train `3807871`, eval `3807872` | train `3807873`, eval `3807874` |
| 3 | train `3807875`, eval `3807876` | train `3807877`, eval `3807878` |
| 4 | train `3807879`, eval `3807880` | train `3807881`, eval `3807882` |
| 8 | train `3807883`, eval `3807884` | train `3807885`, eval `3807886` |
| 16 | train `3807887`, eval `3807888` | train `3807889`, eval `3807890` |

Eval is flat latent-rollout MPC beam only: beam width `16`, depths `{4,16}`,
8 boards, oracle raw L2 and predicted raw L2.

Current state at 11:00 CEST:

- K1-K4 train/eval pairs completed successfully; eval rows are available.
- K8/K16 training completed successfully, but eval jobs `3807884`, `3807886`,
  `3807888`, and `3807890` failed at checkpoint load after the Delta-JEPA
  decoder was added. The loader instantiated a Delta decoder for these older
  horizon checkpoints and reported missing `delta_action_decoder.*` keys.
- The loader compatibility fix is applied. Repair evals are running:
  `K8_uniform` `3808345`, `K8_smooth_count` `3808346`, `K16_uniform`
  `3808347`, `K16_smooth_count` `3808348`.
- First repair rows: `K8_smooth_count` solves `8/8` with oracle raw L2 at
  depth 4; `K8_uniform` reaches h `3.75`; `K16_smooth_count` reaches h
  `4.375`; predicted-goal repair rows are still pending.

## Previous Sweep Takeaway

The completed dropout-off factorization sweep showed that oracle-goal planning
is recoverable, but predicted-goal planning remains `0/8` across all rows.
Dropout-off rescued exact-refactor, smooth/count, and old H8-only objectives,
but did not rescue uniform/gamma/K16 single-rollout losses.
