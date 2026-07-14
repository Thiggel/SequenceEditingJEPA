# Wave 04: Grid-Token Goal JEPA

## Question

Which action representation and stabilizer preserve useful local Sudoku
geometry under uninterrupted latent rollout?

## Runs

- Baseline Grid-Token Goal-JEPA: 13 ablations, 60k steps.
- Action/stability suite: `96` checkpoints from train array `3768285`.
- Corrected eval arrays `3775750` and `3775751`: `1,728` main rows plus `576`
  depth-64 rows, latent rollout, beam width 16, depths `4/16/32/64`.
- Axes covered action injection, delta/state prediction, EMA/VICReg and other
  stability recipes, and normalized/raw/changed-cell oracle or predicted goals.

## Results

Every action-suite row solved `0/10`. Best remaining Hamming was `5.8` for
`R4_no_goal_nce/A6_affected_marker_delta/S4_ema_vicreg/D0_uniform` under an
oracle goal. Its best predicted-goal readout was about `35.1` remaining cells.

## Conclusion

Affected-cell action grounding plus delta prediction and EMA/VICReg was the
best direction, but predicted goal geometry remained unusable.

Source: report `RESULTS.md`, section `Current Result`.
