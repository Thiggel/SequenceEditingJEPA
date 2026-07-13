# Current Experiments

Source of truth: `../sequence-editing-report/CURRENT_EXPERIMENTS.md`.

Last updated: 2026-07-13 22:15 CEST

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

At 22:15 CEST, 20 `[1]` checkpoints and their 20 probe evaluations were
complete, later stages were releasing through `aftercorr`, and active logs had
no runtime errors. The first complete three-seed cell is Transformer, `[1]`,
rollout 1, uniform loss, and eight objects. Prediction beats identity by
`.00241-.00328`, primitive action top-1 is `1.0`, but learned fixed 16-edit
planning is `0.0` in every seed. Count and motion-policy probes improve from
initialization in every seed; shape is inconsistent, position/relation R2 are
negative, and foreground reconstruction worsens. This is an early flat-model
result, not yet evidence about hierarchy or rollout depth.

The superseded jobs `3850642,3850645,3850648,3850656,3850658,3850660,
3850668,3850670,3850672` and probe jobs `3850878`-`3850989` in the old
controlled grid were explicitly canceled. Historical completed results remain
in the report repo and are not part of this grid.
