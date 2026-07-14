# Current Experiments

Source of truth: `../sequence-editing-report/CURRENT_EXPERIMENTS.md`.

Last updated: 2026-07-14 12:07 CEST

## Fixed Valid-Motion HWM VICReg Sweep

This is the only active experiment. It asks whether an EMA target plus the
right VICReg variance/covariance coefficients can prevent representation
collapse before hierarchy or planning is judged.

The world contains exactly two complete rigid objects on `16x16`. One
primitive `(row,col,transform)` command selects an object and atomically
translates or rotates it, changing a sparse set of pixels simultaneously. The
hierarchy counts valid environment actions, not a variable number of changed
cells: `[1,10,100]`. A shared `768 -> 256` MLP state encoder feeds separate
causal Transformer predictors; ordered action chunks pass through an 8D
nonlinear macro-action bottleneck. Every level receives four-step
teacher-forced and autonomous rollout supervision. EMA is `.99`.

| stage | job | cells | state |
| --- | ---: | ---: | --- |
| `[1]` | `3855790` | 48 | 16 running, remainder array-limited |
| `[1,10]` | `3855791` | 48 | dependency-held |
| `[1,10,100]` | `3855792` | 48 | dependency-held |
| probes and planning | `3855793` | 48 | dependency-held |

The 48 final cells cross variance coefficient `{.05,1,10,29.409}`, adjusted
covariance coefficient `{.1,1,10,17.866}`, and seeds `{1707,2707,3707}`. No
other model, trajectory, objective, or capacity axis is active.

## Current Results

No trained cell is complete yet. Prelaunch controls are:

| check | result |
| --- | --- |
| full repository tests | pass |
| three-stage CUDA smoke | job `3855783`, `COMPLETED 0:0` |
| valid primitive actions | `1.0` fraction |
| object area preservation | `1.0` fraction |
| changed cells per action | `2-6` in the smoke sample |
| symbolic/oracle geometry controls | exact on tested short paths/actions |

Primary evaluation uses exact-N=2 semantic, reconstruction, rollout, rank, and
planning probes. A separately labeled N=1/2/4/8 suite measures load
generalization. The random hidden motion-policy label is an intentionally
unobservable single-frame negative control and must not be read as a semantic
representation target.

Planning compares action-sequence retrieval, bounded CEM, support-regularized
CEM, and MPPI. All learned planners recursively pass the first coarse latent
prediction to the next finer model and replan after each executed primitive
action. Manual low-level subgoal control gates interpretation of high-level
planning; macro support and reachability AUROC diagnose off-manifold actions.

Task manifest:
`$HPCVAULT/sequence-editing/runs/controlled_objects/manifests/controlled_valid_hwm_vicreg_v1_steps20000_tasks.tsv`.
Outputs:
`$HPCVAULT/sequence-editing/runs/controlled_objects/controlled_valid_hwm_vicreg_v1_steps20000/`.
