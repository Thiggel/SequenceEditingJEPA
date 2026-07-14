# Wave 06: Minimal Objectives, Macro HWM, and Delta-JEPA

## Question

Which objective components preserve oracle geometry, and do Delta-JEPA/LDAD or
macro-HWM planning improve single-latent rollout?

## Runs

- Minimal-aux 29-variant train/eval arrays `3803494/3803495`, `456/456` rows.
- Clean17 train/eval pairs `3804755`-`3804788`, `76/76` rows.
- Macro-HWM jobs `3804951`-`3804974`, `40/40` rows.
- Objective factorization and dropout controls, followed by clean K8 horizon.
- Delta-JEPA replacements `3808387`-`3808404`, eval `3808863`-`3808874`.
- Metric/value geometry and LDAD-weight follow-ups.

## Results

Minimal-aux recovered oracle geometry (`8/8` for several rows), but every
predicted-goal row remained `0/8`. Macro-HWM had no solves; best remaining
Hamming was `22.5`. Full-board EMA-hybrid Delta could recover oracle latent
rollout, while paper-pure online Delta and every single-CLS Delta row failed.

## Conclusion

Oracle geometry is recoverable without the full auxiliary stack. Predicted
goals and single-vector dynamics remain the blockers. Historical full-grid
Delta rows are controls only and are excluded from the current no-grid scope.
