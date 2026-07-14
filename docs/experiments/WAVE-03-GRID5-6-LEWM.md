# Wave 03: Grid5/6 and LeWM

## Question

Do SIGReg, recursive rollout, causal trajectory context, or a LeWorldModel
baseline produce a stable single-latent Sudoku world model?

## Runs

- Grid5 commits `c1e38de`-`7505dd7`: SIGReg, MPC/CEM diagnostics, recursive
  rollout, full-state rollout, symbolic probes, and a 10M-step stabilizer screen.
- Grid6 commits `6cfce28`-`7d3a91e`: causal-only trajectory JEPA.
- LeWM reset commits `373eb1e`-`d2c72db`: corrected training, masking, BN,
  BF16, planning diagnostics, and split planner evaluation.

## Results

The stabilizers changed latent statistics but did not yield a robust latent
planner. The LeWM reset reproduced the key negative geometry result: exact
symbolic and true-Hamming planning worked, while oracle latent distance and
learned scalar goal distance did not solve the recorded Sudoku sets.

## Conclusion

The project reset to a grid-token goal JEPA so action conditioning, goal
prediction, and world-model rollout could be tested independently.
