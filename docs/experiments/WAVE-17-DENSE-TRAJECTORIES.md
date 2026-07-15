# Wave 17: Long-Trajectory Dense Hierarchy

## Question

Does training jointly on whole trajectories, with every causal endpoint and
every valid autonomous rollout anchor supervised, prevent representation and
hierarchy drift?

## Fixed Contract

- Exact two-object valid rigid motion and one learned 256D MLP CLS.
- End-to-end hierarchy `[1,10,100]`; all levels train jointly from step 0.
- Full teacher-forced causal prediction at every segment endpoint.
- Autonomous rollout profile `[10,10,4]` at every valid anchor.
- Unit cross-level consistency between a direct high-level transition and the
  corresponding composition of the next lower level.
- VICReg and EMA+stop-gradient+VICReg, three seeds each.

## Grid

Trajectory/ordinary-batch cells are `T100/B64`, `T300/B20`, and `T500/B12`,
each for 20,000 steps. A separate constant-processed-state axis at `T300`
uses `B8/50k`, `B16/25k`, `B32/12.5k`, and `B64/6.25k`. This gives 42 trainers
and 42 correlated probes.

## Execution

- Largest-cell GPU preflight: `3860383_6`, complete `0:0`, peak GPU memory
  about 20.6 GiB. An earlier preflight `3860373_6` exposed and led to repair
  of a dense-evaluation anchor-shape bug.
- Trainers: `3860420` (`0-41%6`)
- Correlated probes: `3860421`, dependency `aftercorr:3860420`
- Root: `$HPCVAULT/sequence-editing/runs/controlled_objects/controlled_dense_trajectories_v1`
- Status: active, submitted 2026-07-15

## Gate

Test whether longer context or batch/state budget improves frozen properties,
rank, reconstruction, autonomous rollout gains, direct/composed hierarchy
agreement, and primitive realization across all seeds.

