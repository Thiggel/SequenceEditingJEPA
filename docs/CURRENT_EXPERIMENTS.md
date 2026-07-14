# Current Experiments

Source of truth: `../sequence-editing-report/CURRENT_EXPERIMENTS.md`.

Last updated: 2026-07-14 08:14 CEST

## MLP Pixel-Edit HWM Grid

The only active controlled-object sweep uses one 768-to-256 RGB MLP latent,
atomic `(row,col,new_color)` actions, and a causal Transformer, Gated DeltaNet,
or nonlinear LSTM predictor. It crosses predictor `{transformer,gated_deltanet,lstm}`,
hierarchy `{[1],[1,4],[1,4,16],[1,2,4]}`, dense rollout `{1,2,4,8}`, loss
weighting `{uniform,0.9^i}`, exact object count `{1,2,4,8}`, and three seeds.
This is 1,152 final cells. There is no grid latent and no Transformer encoder.

Five correlated train arrays stage and freeze lower levels:

| stage | job | dependency | tasks |
|---|---:|---:|---:|
| `[1]` | `3851603` | none | 288 |
| `[1,4]` | `3851604` | `aftercorr:3851603` | 288 |
| `[1,4,16]` | `3851605` | `aftercorr:3851604` | 288 |
| `[1,2]` private stage | `3851606` | `aftercorr:3851603` | 288 |
| `[1,2,4]` | `3851607` | `aftercorr:3851606` | 288 |

Probe arrays `3851608`-`3851611` depend on the four final schedules and contain
1,152 tasks. Each checkpoint gets matched-initialization linear probes for
count, presence/color, shape, motion policy, area, position, pair relations,
and pixel-edit fields, plus a nonlinear pixel decoder trained on frozen
latents. Planning compares on-support retrieval, bounded CEM,
support-regularized CEM, and MPPI under recursive shared-latent subgoals.

Task manifest:
`$PUZZLE_JEPA_WORK_ROOT/runs/controlled_objects/manifests/controlled_mlp_hwm_v1_steps20000_tasks.tsv`.
Job manifest:
`$PUZZLE_JEPA_WORK_ROOT/runs/controlled_objects/manifests/controlled_mlp_hwm_v1_steps20000_array_jobs.tsv`.
Outputs:
`$PUZZLE_JEPA_WORK_ROOT/runs/controlled_objects/controlled_mlp_hwm_v1_steps20000/`.

At 08:14 CEST, 1,287/1,440 staged checkpoints and 694/1,152 final probes were
complete. `[1]` is complete; `[1,4]` has 287/288 checkpoints; deeper schedules
and their probes are still running. A home-filesystem exhaustion, one node
failure, and several administrator-canceled tasks broke correlated dependency
chains. Targeted repair arrays `3854953`-`3854964` and `3855009`-`3855014`
are running. Hydra metadata now goes into each vault run directory.

Balanced `[1]` versus `[1,4]` results are already negative for hierarchy.
Mean exact 16-edit planning falls from `.102` to `.041`, while retrieval pixel
error rises from `.052` to `.100`. At exact loads 1/2/4/8, flat planning is
`.326/.083/0/0`; `[1,4]` gives `.081/.083/0/0`. Rollout 1/2/4/8 changes flat
planning only from `.104/.104/.104/.097`; `0.9^i` weighting has no material
effect. MPPI lowers two-level pixel error but does not improve exact solves.

The 256-wide latent has collapsed to mean probe effective rank `9.4`. Count
balanced accuracy improves by `.103`, but shape remains at chance (`.201`),
position R2 is approximately zero, relation R2 is `-.121`, and foreground IoU
drops by `.080`. Motion-policy accuracy is high but policy ID is deterministically
tied to visible color, so it is not evidence of inferred dynamics.

The superseded jobs `3850642,3850645,3850648,3850656,3850658,3850660,
3850668,3850670,3850672` and probe jobs `3850878`-`3850989` in the old
controlled grid were explicitly canceled. Historical completed results remain
in the report repo and are not part of this grid.
