# Wave 18: Planner Interface Diagnostics

## Question

Is high-level planning failing because CEM proposes macro actions outside the
conditional action-code support, because proposed subgoals are not reachable
by the lower level, or both?

## Fixed Contract

- Retained Wave 15 VICReg and EMA+VICReg checkpoints, three seeds each.
- The same 32 deterministic 100-action episodes per row, 512 high-level CEM
  candidates, and Wilson uncertainty intervals.
- Eight planner modes: baseline, hard conditional support projection,
  reachability penalties `.1/1/10`, and hard projection plus each penalty.
- Reachability feedback re-scores top candidates by reduced-budget lower-level
  realization residual. No ensemble is included because independently trained
  latent coordinates do not define a directly comparable prediction variance.

## Execution

- Strongest-path GPU preflight: `3860375_7`, complete `0:0`.
- Planner array: `3860422` (`0-47%12`)
- Root: `$HPCVAULT/sequence-editing/runs/controlled_objects/controlled_planner_interfaces_v1`
- Status: complete 48/48, all exit `0:0`

## Gate

Hard projection improving success implicates support exploitation; lower-level
residual improving success implicates hierarchy-interface reachability. Report
exact success and final pixel error over identical episodes, not one-episode
rates.

## Results

| objective | planner | success | final pixel error |
| --- | --- | ---: | ---: |
| VICReg | baseline | .510 | .0133 |
| VICReg | hard support | .323 | .0235 |
| VICReg | reachability `.1/1/10` | .490/.490/.354 | .0147/.0159/.0189 |
| EMA+VICReg | baseline | .604 | .0140 |
| EMA+VICReg | hard support | .375 | .0228 |
| EMA+VICReg | reachability `.1/1/10` | .448/.479/.427 | .0152/.0159/.0177 |

Hard support and all hard-support/reachability combinations reduce success.
Soft reachability never beats baseline. Conditional support exploitation is
therefore not the primary failure in these checkpoints; the crude projection
and lower-level residual discard useful macro proposals. Unconstrained
hierarchical CEM is the retained planner baseline.
