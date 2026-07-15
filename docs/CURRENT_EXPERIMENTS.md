# Current Experiments

Source of truth: `../sequence-editing-report/CURRENT_EXPERIMENTS.md`.

Last updated: 2026-07-15 18:04 CEST

## What The Active Sweeps Test

Three controlled follow-ups to Wave 15 are active. Wave 16 calibrates
LDAD/VICReg/SIGReg weights and target contracts in a fixed jointly trained
`[1,10,100]` hierarchy. Wave 17 tests full `T={100,300,500}` trajectories,
dense supervision at every causal endpoint and autonomous anchor, and explicit
cross-level composition consistency. Wave 18 tests whether high-level CEM is
failing through off-support macro optimization, lower-level unreachability, or
both. Every model remains a single learned MLP CLS; there are no grid latents.

## Slurm State

| wave | stage | job | tasks | state at snapshot | output root |
| --- | --- | ---: | ---: | --- | --- |
| 16 objective weights | train | `3860384` | 231 | active, `%24` | `controlled_objective_weights_v1_steps20000` |
| 16 objective weights | probe | `3860385` | 231 | `aftercorr:3860384` | same |
| 17 dense trajectories | train | `3860420` | 42 | active, `%6` | `controlled_dense_trajectories_v1` |
| 17 dense trajectories | probe | `3860421` | 42 | `aftercorr:3860420` | same |
| 18 planner interfaces | eval | `3860422` | 48 | active, `%12` | `controlled_planner_interfaces_v1` |

GPU gates passed for the extreme objective (`3860374_230`), largest dense cell
after repair (`3860383_6`, about 20.6 GiB peak), and strongest planner mode
(`3860375_7`). Dense preflight `3860373_6` failed before training because the
evaluator used batch-shaped identity/change baselines for anchor-flattened
predictions; the implementation and regression test are repaired.

## Pending Results

No production result is yet interpretable. The analyzers require 231/231,
42/42, and 48/48 artifacts before final aggregation. The decisive outputs are
all frozen property probes, frozen-feature pixel reconstruction, effective
rank, dense rollout and hierarchy consistency, primitive realization, and
32-episode planning success/final error.
