# Experiment Backlog

Only Stage 0 is active. Later stages are gates, not permission to launch broad
grids automatically.

## Stage 0: Representation Rescue

| Priority | Experiment | Status | Decision gate |
| --- | --- | --- | --- |
| P0 | Valid rigid-motion HWM VICReg sweep | active: jobs `3855790`-`3855793` | Find a stable coefficient region with materially non-collapsed rank and positive semantic/reconstruction transfer across all three seeds. |
| P0 | Faithful planner and manual-subgoal evaluation | included in the same 48-cell evaluation | Manual low-level subgoal control must work before high-level planning is interpreted. |
| P0 | Historical storage cleanup | complete | Historical metrics/manifests/docs retained; obsolete checkpoints and caches removed. Keep only active sweep checkpoints. |

## Stage 1: Diagnose a Passing Representation

| Priority | Experiment | Status | Decision gate |
| --- | --- | --- | --- |
| P1 | Re-run top VICReg cells with five seeds | blocked on Stage 0 | Confirm rank and every claimed semantic gain; reject one-seed threshold effects. |
| P1 | Static versus rollout feature probes by level | already instrumented | Determine whether high-level prediction preserves or destroys the low-level factors. |
| P1 | Same-color/permutation-aware object probes | proposed | Remove the unique-color object-slot shortcut before claiming object identity. |

## Stage 2: Macro Manifold and Reachability

| Priority | Experiment | Status | Decision gate |
| --- | --- | --- | --- |
| P2 | Macro dimension `{2,4,8}` | blocked on Stage 1 | Select the smallest dimension that preserves endpoint prediction and low-level reachability. |
| P2 | Continuous macro versus support retrieval/codebook | blocked on Stage 1 | Separate world-model error from off-support optimizer exploitation. |
| P2 | Learned support energy or hard macro projection | blocked on support diagnostics | Require off-support AUROC and improved exact planning without harming on-support plans. |

## Stage 3: Trajectory Causality

| Priority | Experiment | Status | Decision gate |
| --- | --- | --- | --- |
| P3 | Valid rigid motion versus object-by-object construction | blocked on Stage 1 | Test whether representation factors depend on movement or coherent construction. |
| P3 | Hidden transform command versus explicit sparse pixel delta | proposed | Determine whether exact action information creates a non-object shortcut. |

## Stage 4: Broader Models

| Priority | Experiment | Status | Decision gate |
| --- | --- | --- | --- |
| P4 | Predictor family, capacity, LDAD, SIGReg | blocked | Re-open only after one fixed HWM recipe passes representation and manual-subgoal gates. |
| P4 | More objects and broader trajectory regimes | blocked | Scale only after N=2 remains stable across seeds and same-color controls. |

## Retired

The MLP pixel-edit factorial, broad object-dynamics grids, full-grid Delta
rows, and superseded controlled-HWM jobs are historical controls. Do not resume
them without a new explicit decision.
