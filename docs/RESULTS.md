# Results

Long-form results: `../sequence-editing-report/RESULTS.md`.

Last updated: 2026-07-14 12:07 CEST

The active valid-motion HWM VICReg sweep has no completed trained cells yet.
Its prelaunch three-stage CUDA smoke passed in job `3855783`; valid-action and
object-area-preservation fractions were both `1.0`, with `2-6` pixels changed
simultaneously per primitive action.

The superseded MLP pixel-edit factorial was canceled after 1,419 staged
checkpoints and 752 final probes. Its balanced flat versus `[1,4]` comparison
was negative for hierarchy: exact planning `.102` versus `.041`, with pixel
error `.052` versus `.100`. Mean effective rank was only `9.4/256`; count
improved, while shape, position, relation, and foreground reconstruction did
not. Those trajectories passed through partial erase/paint states and are not
evidence about valid rigid-motion HWM.

Per-wave run surfaces, results, and conclusions are indexed in
`docs/experiments/README.md`. The current backlog is `docs/BACKLOG.md`.
