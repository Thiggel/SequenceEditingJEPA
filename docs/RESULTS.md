# Results

Last updated: 2026-06-12 13:30 CEST

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
