# Experiment Backlog

Wave 15 is complete and its strict representation gate failed. Later stages
are gates, not permission to launch broad grids automatically.

## Stage 0: Representation Rescue

| Priority | Experiment | Status | Decision gate |
| --- | --- | --- | --- |
| P0 | Valid rigid-motion HWM VICReg sweep | complete: 48/48 cells; representation gate failed | No pair jointly preserves rank and improves semantic/foreground transfer across all seeds. |
| P0 | Correct jointly trained HWM objective comparison | complete: `3858542`/`3858543`, 36/36 each `0:0`; gate failed | VICReg improves spatial/semantic probes, but every objective loses substantial rank and foreground reconstruction. |
| P0 | Frozen-probe calibration | implemented in Wave 15 | Regression targets are standardized during fitting; compare frozen learned, matched-initial, and raw controls. |
| P0 | Conditional macro-support diagnostic repair | implemented in Wave 15 | Score macro support conditional on nearby states; require held-out support/reachability AUROC before planning use. |
| P0 | Planner sample-size repair | blocked: no Wave 15 objective passed | Run only after a representation-qualified objective exists. |
| P0 | Historical storage cleanup | complete | Historical metrics/manifests/docs retained; obsolete checkpoints and caches removed. Keep only active sweep checkpoints. |

## Stage 1: Diagnose a Passing Representation

| Priority | Experiment | Status | Decision gate |
| --- | --- | --- | --- |
| P1 | Re-run top VICReg cells with five seeds | blocked: Stage 0 gate failed | Confirm only after evaluation repairs identify a cell worth promoting. |
| P1 | Static versus rollout feature probes by level | already instrumented | Determine whether high-level prediction preserves or destroys the low-level factors. |
| P1 | Same-color/permutation-aware object probes | proposed | Remove the unique-color object-slot shortcut before claiming object identity. |
| P1 | Joint encoder training versus staged encoder freeze | folded into corrected P0 experiment | Joint training is the required default; retain staged freezing only as an explicitly labeled later ablation. |

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
