# Current Experiments

Source of truth: `../sequence-editing-report/CURRENT_EXPERIMENTS.md`.

Last updated: 2026-07-16

## What This Wave Tests

Wave 17 is the only active gate. It tests whether complete trajectories,
supervision at every causal endpoint and autonomous anchor, and direct/composed
hierarchy consistency improve the single-CLS representation and make span-100
subgoals realizable. It compares `T={100,300,500}` and a constant-state-budget
batch axis for VICReg and EMA+VICReg. No grid latent is used.

## Active Slurm State

| stage | job | state/role |
| --- | ---: | --- |
| original dense trainers | `3860420` | 27 complete, 6 running, 6 pending, 3 A40 OOM |
| original probes | `3860421` | 6 complete; short-horizon failures; remaining dependencies active |
| exact B64 trainer repair | `3862936` | active on 96 GiB RTX Pro |
| failed-probe repair | `3862939` | active |
| B64 repair probes | `3862940` | dependency-held |

The probe defect requested a 400-action top-level rollout from `T100` data.
Probe horizons now mirror dense training: `[10,10,1]` for T100,
`[10,10,3]` for T300, and `[10,10,4]` for T500. No partial dense result is
promoted.

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
