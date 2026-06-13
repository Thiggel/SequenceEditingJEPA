# Results

Last updated: 2026-06-13 04:58 CEST

Detailed historical results live in `../sequence-editing-report/RESULTS.md` and
`../sequence-editing-report/report.tex`.

Clean Grid5-only planning and running-experiment docs live in
`docs/GRID5_PLAN.md`, `docs/GRID5_BACKLOG.md`, and `docs/GRID5_LOG.md`; the
source-of-truth versions live in `../sequence-editing-report/`.

## Current Result

Grid 5B completed. Original tasks `0-5` hit Slurm `NODE_FAIL` on `a2143` and
were resubmitted as `3724689_[0-5]`; the rerun completed cleanly, and final
Grid5B stderrs checked so far are empty. Grid 5C planner matrix eval has not
yet produced usable full-matrix artifacts: tasks `6-11` timed out before
writing summaries, and tasks `0-5` were still running near their 12h limit at
2026-06-13 04:54 CEST. The eval now writes incremental records/summaries, and
small streaming probe `3728790` is running on a40 node a0124.

Grid5 oversight checks are scheduled every 6h for the next 2.5 days as
`3724789`-`3724798`. Dummy alias-path verification passed as `3724787`.

Grid 5 `3722613_[0-23]` completed cleanly. All 24 tasks exited `0:0`, all
stderr files are empty, and all expected diagnostics were written.

Solve gate failed for every variant:

- oracle `latent_goal` small beam planning: `0/16` solves for all 24 variants
- learned `goal_energy` small beam planning: `0/16` solves for all 24 variants
- best oracle remaining Hamming:
  `grid5_sigreg_mlp_mlp_delta_z128`, mean remaining Hamming `44.88`
- best learned-energy remaining Hamming:
  `grid5_sigreg_mlp_mlp_delta_z64`, mean remaining Hamming `48.19`

## Current Interpretation

The compact single-state Grid 5 representation does not yet learn a
planner-ready metric, even with SIGReg. The strongest variants have monotone
oracle latent distance along known gold trajectories, but they fail local
all-action ranking and therefore fail planning.

Representative best oracle variant:
`grid5_sigreg_mlp_mlp_delta_z128`.

- latent trajectory monotone rate: `0.992`
- learned-energy trajectory monotone rate: `0.992`
- oracle latent gold-action top-1: `0.031`
- oracle latent top action is any goal-correct value: `0.156`
- learned-energy gold-action top-1: `0.000`
- learned-energy top action is any goal-correct value: `0.063`

So SIGReg avoided trivial collapse and the gold path is mostly directionally
ordered, but the geometry still does not distinguish the correct next action
from adjacent/wrong actions reliably enough for planning.

## Loss-Curve Read

Grid 5 does not look like a simple job crash or totally unconverged run. Most
one-step prediction losses drop sharply by step `1000` and then plateau or
wiggle. CLS-transformer encoders have much lower SIGReg/eval total losses than
MLP encoders, but both families fail planning.

Aggregate final eval metrics:

- CLS-transformer encoder: pred `0.00501`, SIGReg `0.02515`, energy `0.00661`
- MLP encoder: pred `0.00543`, SIGReg `0.12707`, energy `0.04419`
- MLP predictor: pred `0.00508`, SIGReg `0.07333`, energy `0.02564`
- AR-transformer predictor: pred `0.00536`, SIGReg `0.07889`, energy `0.02516`
- delta target: pred `0.00449`, SIGReg `0.06829`, energy `0.03065`
- full-state target: pred `0.00595`, SIGReg `0.08393`, energy `0.02014`

Interpretation: delta prediction is easier for one-step dynamics; CLS
transformer gives healthier SIGReg geometry; neither is sufficient for local
action ranking.

## New Posthoc Control

Submitted Grid 5 posthoc MPC-CEM diagnostics as `3724325_[0-23]`. This is the
LeWorldModel-style planner control missing from the first Grid 5 read:
optimize latent action sequences with CEM, score final predicted latent against
the solved-board latent, execute one action, re-encode/replan, and sweep
horizons `4/8/16/32/64`.

Final read: still failed. All 24 original Grid 5 checkpoints solved `0` at
every horizon and score. Average remaining Hamming improved mildly with
lookahead, from about `53` at h4 to about `51.5` at h64, but no variant became
terminal or exact.

## Recursive Rollout Sweep

Grid 5 recursive rollout training completed cleanly: delta prediction
`3724413_[0-5]` and full-state prediction `3724500_[0-5]` all exited `0:0`
with empty checked stderr files.

Hypothesis: the compact latent may fail MPC-CEM partly because training is
mostly teacher-forced one-step prediction, while planning recursively feeds
predicted latents back into the predictor. The six new jobs add recursive
rollout loss with K `2/4/8`, crossed with MLP vs AR-transformer predictor.
Both use MLP encoder and latent size `128`; the two arrays compare delta vs
full-state prediction.

Result: failed. All 12 recursive variants solved `0` under oracle
`latent_goal` and learned `goal_energy` in MPC-CEM at horizons `4/8/16/32/64`.
Terminal rate stayed `0.0`.

Best proximity reads:

- best oracle `latent_goal`: `grid5_recursive_mlp_mlp_delta_z128_k2`, h64,
  mean remaining Hamming `49.88`
- best learned `goal_energy`: `grid5_recursive_mlp_ar_transformer_state_z128_k2`,
  h64, mean remaining Hamming `50.50`
- best cheap standard-diagnostic beam proximity among the recursive runs:
  `grid5_recursive_mlp_mlp_delta_z128_k8`, oracle mean remaining Hamming `39.6`

Interpretation: recursive rollout training improved some small beam proximity
signals, but it did not make MPC-CEM planning solve boards. The compact
single-state geometry still fails as a planner objective, even when trained in
the same recursive mode used by MPC-CEM.

## Symbolic Re-Encode Probe

Added and ran `scripts/analysis/grid5_symbolic_planning_probe.py` on CPU for
three representative checkpoints:

- `grid5_sigreg_mlp_mlp_delta_z128`
- `grid5_recursive_mlp_mlp_delta_z128_k2`
- `grid5_recursive_mlp_ar_transformer_state_z128_k2`

Artifacts:

- `$PUZZLE_JEPA_WORK_ROOT/analysis/grid5_symbolic_probe_20260612/`
- `$PUZZLE_JEPA_WORK_ROOT/analysis/grid5_symbolic_probe_state_20260612/`
- `$PUZZLE_JEPA_WORK_ROOT/analysis/grid5_symbolic_probe_true_hamming_20260612/`

The probe executes candidate futures symbolically, re-encodes the resulting
boards, and scores with oracle `latent_goal` or learned `goal_energy`, using a
fill-empty action space. It still solved `0/4` for every horizon
`8/16/32/64/full`; remaining Hamming stayed roughly `45-51`, which is close to
random filled-board quality from starts with about `55` blanks.

Random rollout drift is real but not the only blocker. The AR full-state
recursive checkpoint reduced K=32 drift to MSE `0.1369` / L2 `3.39`, versus
MSE `1.1407` / L2 `11.32` for the base MLP-delta checkpoint. Yet symbolic
re-encode planning still solved `0`. Therefore predictor drift matters, but
the encoder/scorer geometry itself is not ranking exact symbolic boards well.

Perfect-score sanity: using true Hamming as the symbolic CEM cost got much
closer at h8, mean remaining Hamming `7.5`, but still solved `0/4`. So flat
categorical CEM is also a weak Sudoku optimizer at this budget; however, the
latent/learned scores are the larger blocker because they remain near
`45-51` wrong cells.

## Grid 5B 10M Stabilizer Screen

Submitted 10M-scale stabilizer/capacity screen `3724634_[0-11]`. Original
tasks `0-5` failed with Slurm `NODE_FAIL` on `a2143` and empty stderr; they
were resubmitted as `3724689_[0-5]` excluding `a2143` and completed cleanly.

- Trainable params: `10.6M-13.4M`
- Stabilizers: SIGReg, EMA+SIGReg, VICReg, EMA+VICReg
- Other anchored contrasts: K1/K4, full/delta, MLP/CLS encoder, MLP/AR
  predictor
- Each job ran standard diagnostics, predicted-latent MPC-CEM, and symbolic
  re-encode MPC-CEM

Result: still no exact solve. The best symbolic oracle proximity is
`grid5b_10m_canonical_ema_vicreg_k4`, horizon 8, mean remaining Hamming
`41.00`, root goal-value rate `0.500`, solve `0/4`. Its cheap standard beam
diagnostic is stronger than earlier compact runs, with oracle mean remaining
Hamming `29.56`, latent gold-action top-1 `0.125`, and latent top-goal-value
rate `0.969`, but exact symbolic re-encode planning still fails. Predicted
latent MPC-CEM solves `0` for every Grid5B variant; best proximity is
`grid5b_10m_canonical_ema_sigreg_k4`, h64 `goal_energy`, mean remaining
Hamming `49.50`.

True-Hamming symbolic CEM reaches mean remaining Hamming `1.75` and solve
`1/4` for `canonical_ema_vicreg_k4`, `oldbest_scaled_ema_sigreg_k4`, and
`oldbest_scaled_sigreg_k4`. This shows the small symbolic optimizer can get
near a solution on these boards, but the learned/oracle latent scores remain
the larger blocker.

## Active Grid 5C

Added `puzzle_jepa/eval/grid5_planner_matrix.py` and
`scripts/slurm/run_grid5c_planner_matrix_eval.slurm`.

The matrix evaluates all Grid 5B checkpoints with MPC over:

- optimizer: `beam`, `mcts`, `nn_cem`
- transition: symbolic board application + horizon re-encode vs latent-only
  recursive predictor rollout
- score: oracle `latent_goal` vs learned `goal_energy`

Submitted jobs:

- `3724691_[0-5]` after rerun `3724689`
- `3724698_[9-11]` for completed old-best tasks; currently running on `a2143`
  with 8h limit, so monitor for repeat node failure
- `3724700_[6]`, `3724701_[7]`, `3724702_[8]`, each after its matching
  original Grid 5B task; `3724702_8` has started

Current state at 2026-06-12 22:52 CEST: all 12 Grid5C tasks are running.
`3724691_[0-5]` is on `a40`; `3724698_[9-11]`, `3724700_6`,
`3724701_7`, and `3724702_8` are on `rtxpro6k` node `a2143`. Stderrs are
empty, `sstat` shows CPU/RSS activity, and no planner summaries have been
written yet. `3724698_[9-11]` still has the old 8h wall time and may time out
before producing artifacts.

Verification passed: compile, Slurm syntax,
`pytest tests/test_grid5_sigreg.py -q`, and a tiny real-checkpoint CLI smoke.

## Diagnostics To Read First

For each Grid 5 run, read:

- `diagnostics/diagnostics.json`
- `diagnostics/trajectory_records.jsonl`
- `diagnostics/action_rank_records.jsonl`
- `diagnostics/action_rank_examples.jsonl`

Primary success signals before exact solve:

- latent `std_mean` near `1`, healthy `pairwise_distance_mean`, low
  `cov_offdiag_abs_mean`
- high `latent_monotone_rate`
- low `latent_gold_rank_mean` and high `latent_top_goal_value_rate`
- low learned `goal_energy_abs_error_mean`
- learned-energy action ranking close to oracle latent ranking
