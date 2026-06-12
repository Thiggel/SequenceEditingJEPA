# Results

Last updated: 2026-06-12 15:06 CEST

Detailed historical results live in `../sequence-editing-report/RESULTS.md` and
`../sequence-editing-report/report.tex`.

## Current Result

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
