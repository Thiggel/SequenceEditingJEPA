# Results

Last updated: 2026-06-13 12:36 CEST

Detailed historical results live in `../sequence-editing-report/RESULTS.md` and
`../sequence-editing-report/report.tex`.

Clean Grid5-only planning and running-experiment docs live in
`docs/GRID5_PLAN.md`, `docs/GRID5_BACKLOG.md`, and `docs/GRID5_LOG.md`; the
source-of-truth versions live in `../sequence-editing-report/`.

## Current Result

Grid 5B completed. Original tasks `0-5` hit Slurm `NODE_FAIL` on `a2143` and
were resubmitted as `3724689_[0-5]`; the rerun completed cleanly, and final
Grid5B stderrs checked so far are empty. Grid 5C full planner matrix eval did
not produce usable full-matrix artifacts: `3724691_[0-5]`,
`3724698_[9-11]`, `3724700_6`, `3724701_7`, and `3724702_8` all timed out.
The stderrs contain only Slurm time-limit messages.

Small streaming probe `3728790` completed cleanly and is the current Grid5C
read. On one board from `grid5b_10m_canonical_ema_vicreg_k4`, h8 MCTS +
`symbolic_reencode` + oracle `latent_goal` reduced remaining Hamming from `55`
to `37` but solved `0/1`; beam oracle symbolic reached `39`; latent-rollout
modes stayed `53-55`; learned `goal_energy` remained weak (`49-54`). A follow-up
geometry probe shows learned `goal_energy` true-terminal top1 `0/16` among
one-cell corrupt terminal boards, latent/Hamming nearest-neighbor Spearman
`0.133`, and best wrong action displacement beating gold goal-direction cosine
in `84.4%` of sampled states.

Grid5 oversight checks are scheduled every 6h as `3724789`-`3724798`.
`3724789`, `3724790`, and `3724791` completed cleanly; `3724792`-`3724798`
remain pending by begin time. Dummy alias-path verification passed as
`3724787`.

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

Stabilizer read: EMA is the only canonical stabilizer change that materially
improves local oracle action ranking. Canonical SIGReg K4 has symbolic oracle
remaining Hamming `48.00` and latent top-goal-value rate `0.125`; EMA+SIGReg
K4 improves these to `44.50` and `0.969`. VICReg alone improves latent health
but not planning (`48.25`, top-goal `0.156`). EMA+VICReg is the best symbolic
oracle run (`41.00`, top-goal `0.969`) but still solves `0/4`, and its learned
energy remains weak (`44.25`, learned top-goal `0.281`). Low drift is not
sufficient: VICReg and delta controls have lower K32 drift than EMA+VICReg but
worse symbolic planning.

## Grid 5C and Geometry Probe

Added `puzzle_jepa/eval/grid5_planner_matrix.py` and
`scripts/slurm/run_grid5c_planner_matrix_eval.slurm`.

The matrix evaluates all Grid 5B checkpoints with MPC over:

- optimizer: `beam`, `mcts`, `nn_cem`
- transition: symbolic board application + horizon re-encode vs latent-only
  recursive predictor rollout
- score: oracle `latent_goal` vs learned `goal_energy`

Submitted jobs:

- `3724691_[0-5]` after rerun `3724689`: timed out at 12h on `a40`
- `3724698_[9-11]`: timed out at 8h on `a2143`
- `3724700_6`, `3724701_7`, `3724702_8`: timed out at 12h on `a2143`

The full matrix was a runtime/artifact failure, not a Python failure: checked
stderrs contain only time-limit messages. The smaller streaming replacement
`3728790` completed in `01:03:08` on a40 node `a0124` and wrote:

- `$PUZZLE_JEPA_WORK_ROOT/runs/grid5b_10m_canonical_ema_vicreg_k4/diagnostics_planner_matrix_probe_20260613/planner_summary.json`
- `$PUZZLE_JEPA_WORK_ROOT/runs/grid5b_10m_canonical_ema_vicreg_k4/diagnostics_planner_matrix_probe_20260613/planner_records.jsonl`

Result: oracle symbolic re-encode improves proximity but does not solve even
one board. Best h8 mode is MCTS + oracle `latent_goal` +
`symbolic_reencode`, remaining Hamming `37` from start `55`, solve `0/1`.
Beam oracle symbolic gives `39`. Learned-energy symbolic gives `49` for
beam/MCTS and `54` for `nn_cem`. All latent-rollout modes remain `53-55`.

Added `scripts/analysis/grid5_geometry_probe.py` and ran it on the same
checkpoint. Artifact:
`$PUZZLE_JEPA_WORK_ROOT/analysis/grid5_geometry_probe_canonical_ema_vicreg_k4_20260613/`.
The audit supports the metric-mismatch diagnosis: one-cell corrupt terminal
boards can be very close to the true terminal latent (`p10` corrupt latent MSE
`0.00168`, mean minimum margin `0.00047`); learned `goal_energy` ranks the true
terminal top1 in `0/16`; latent/Hamming nearest-neighbor Spearman is only
`0.133`; and best wrong action displacement beats the gold action's
goal-direction cosine in `84.4%` of sampled states.

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
