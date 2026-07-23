# Current Experiments

Source of truth: `../sequence-editing-report/CURRENT_EXPERIMENTS.md`.

Last updated: 2026-07-23

## What This Wave Tests

Wave 17 is the only active gate. It tests whether complete trajectories,
supervision at every causal endpoint and autonomous anchor, and direct/composed
hierarchy consistency improve the single-CLS representation and make span-100
subgoals realizable. It compares `T={100,300,500}` and a constant-state-budget
batch axis for VICReg and EMA+VICReg. No grid latent is used.

## Current Slurm State

| stage | job | state/role |
| --- | ---: | --- |
| original dense trainers | `3860420` | 36 complete, 6 failed |
| original probes | `3860421` | 15 complete, 21 failed, 6 permanently dependency-blocked |
| attempted B64 trainer repair | `3862936` | 3 complete, but selected the wrong manifest rows |
| bounded-horizon probe repair | `3862939` | 24 complete |
| attempted B64 repair probes | `3862940` | 3 complete for the wrong repaired rows |

The probe defect requested a 400-action top-level rollout from `T100` data.
Probe horizons now mirror dense training: `[10,10,1]` for T100,
`[10,10,3]` for T300, and `[10,10,4]` for T500. The output root currently has
39/42 checkpoints and `probe_eval_v5.json` files. Repair array `3862936`
trained `vicreg_t300_b64_*`, but the missing cells are
`ema_vicreg_t300_b64_*`; a corrected three-cell trainer/probe repair is still
required. No partial dense result is promoted.

## Completed Objective Gate

All 231 trainers and probes in `3860384/3860385` completed. VICReg 1x gives
the best useful representation (rank `50.3`, shape `.291`, position `.550`,
foreground `.153`) but loses rank and foreground from initialization. VICReg
10x preserves rank (`125.6`) and improves foreground (`.248`) but has negative
position/relation R2. VICReg 100x raises rank to `162.3` without useful
semantics. LDAD and SIGReg combinations do not resolve this tradeoff.

## Completed Planner Gate

All 48 rows in `3860422` completed over three seeds and 32 episodes per seed.

| objective | baseline | hard support | best soft reachability |
| --- | ---: | ---: | ---: |
| VICReg | .510 | .323 | .490 |
| EMA+VICReg | .604 | .375 | .479 |

Every support/reachability intervention is worse than unconstrained CEM.
Off-support optimizer exploitation is not the primary observed bottleneck; the
tested constraints remove useful proposals rather than repairing hierarchy.
