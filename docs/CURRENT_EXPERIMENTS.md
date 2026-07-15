# Current Experiments

Source of truth: `../sequence-editing-report/CURRENT_EXPERIMENTS.md`.

Last updated: 2026-07-15 14:41 CEST

## Joint End-to-End HWM Objective Gate

Wave 15 is complete. Trainers `3858542` and correlated frozen probes `3858543`
both completed 36/36 with exit `0:0`. Every `[1,10,100]` level and the shared
256D MLP CLS encoder trained jointly from step 0; no grid latent or staged
freeze was used.

| objective | rank `/256` | presence BA | shape BA | position R2 | relation R2 | foreground IoU | prediction MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| online | 1.3 | .777 | .314 | .030 | -.124 | .012 | .0000002 |
| EMA | 3.2 | .793 | .313 | .468 | .230 | .020 | .0000002 |
| SIGReg | 9.2 | .657 | .261 | .439 | -2.627 | .042 | .040 |
| LDAD | 9.4 | .653 | .241 | .292 | -.925 | .063 | .0045 |
| VICReg | 33.3 | .807 | .304 | .705 | .511 | .116 | .0021 |
| EMA+VICReg | 47.6 | .782 | .279 | .531 | .086 | .155 | .0079 |

VICReg gives the strongest semantic/spatial representation; its position R2
remains `.698/.644/.579` after four predicted steps at spans 1/10/100.
EMA+VICReg retains the most rank and foreground. Nevertheless, initialization
has rank about 131 and foreground IoU about `.211`: every objective loses rank
and foreground reconstruction. Online/EMA prediction collapses almost to a
constant latent.

VICReg direct 10/100-action endpoint MSE is `.0020/.0030`, but primitive
realization MSE is `.164/13.315`. Repaired conditional support AUROC is about
`.998`, while level-2 reachability remains chance (`.515`). The model can tell
that a macro is off observed action support, but its long-horizon predicted
subgoal still cannot be reliably reached by the lower level.

No objective passes the strict representation gate, so no planning jobs were
submitted. Output and aggregate:
`$HPCVAULT/sequence-editing/runs/controlled_objects/controlled_joint_hwm_objectives_v1_steps20000/{summary.json,...}`.
