# Results

Long-form results: `../sequence-editing-report/RESULTS.md`.

Last updated: 2026-07-14 08:14 CEST

The MLP pixel-edit HWM grid has 1,287/1,440 staged checkpoints and 694/1,152
final probe files. Flat `[1]` is complete and `[1,4]` is complete except one
filesystem-failed row, now resubmitted.

| schedule | train cells | exact planning | retrieval pixel error | effective rank |
|---|---:|---:|---:|---:|
| `[1]` | 288 | `.102` | `.052` | `10.80` |
| `[1,4]` | 287 | `.041` | `.100` | `10.76` |
| `[1,2,4]` partial | 271 | `.021` | `.097` | `10.80` |
| `[1,4,16]` partial | 152 | `.039` | `.102` | `10.71` |

At loads 1/2/4/8, `[1]` planning is `.326/.083/0/0`; `[1,4]` is
`.081/.083/0/0`. Deeper rollout slightly lowers flat pixel error but leaves
exact planning flat, and `0.9^i` weighting is neutral. MPPI lowers two-level
pixel error to about `.069` without increasing exact success.

Across all flat probes, count improves from initialization by `.103`, but
shape is chance-level `.201`, position R2 is approximately zero, relation R2
is `-.121`, and foreground IoU falls by `.080`. Mean probe effective rank is
only `9.4/256`. Motion-policy accuracy is color decoding because policy ID and
visible color are coupled; it is not semantic dynamics evidence.

Prelaunch verification passed the full CPU test suite, one-step Hydra CPU
training, and separate RTX Pro 6000 forward/backward training smokes for the
Transformer, Gated DeltaNet, and LSTM predictors. The Gated DeltaNet smoke
first exposed and then verified the FP32/BF16 kernel-boundary fix. Generated
trajectories with exact object counts `{1,2,4,8}` change exactly one pixel per
action and replay exactly.

Historical controlled results showed that wider Transformer/CLS models
improved color-indexed position and relation probes without producing reliable
hierarchical planning. Those jobs used the superseded encoder/action world and
are controls, not results for the active grid.
