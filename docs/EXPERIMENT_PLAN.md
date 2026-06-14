# Experiment Plan

Last updated: 2026-06-14

## LeWorldModel Reset

Implement and evaluate one faithful LeWorldModel-style Sudoku JEPA.

Architecture:

- Encoder: 6-layer bidirectional transformer over the current 9x9 board only.
  Input tokens are digit embeddings summed with row, column, and 3x3-box
  embeddings. A CLS token is projected through a BatchNorm projector to the
  latent state.
- Predictor: 6-layer causal autoregressive transformer over encoded board
  states. Actions are `(row, col, digit)` embeddings with small component
  embeddings, projected to an AdaLN-zero condition at each predictor block.
- Loss: masked next-embedding MSE plus step-wise SIGReg. SIGReg uses 1024
  random projections, 17 Epps-Pulley knots over `[0, 3]`, and weight `0.1`.
- Value head: MLP on the current latent state, trained to regress the Euclidean
  distance from current latent to the solved-board latent.
- Game model: fill empty cells only. No overwrites, no clue mask input, no
  initial-board input.
- Projectors: LeWM-style `Linear -> BatchNorm -> GELU -> Linear` projectors
  after the encoder CLS output and predictor output.

Training sweep:

- Slurm: `scripts/slurm/run_lewm_sudoku_lr_sweep.slurm`
- Learning rates: `1e-6..9e-6`, `1e-5..9e-5`, `1e-4..7e-4`
- Batch size: `128`
- Trajectory frames: full puzzle trajectory by default (`training.num_frames:
  null`), padded per batch with masks
- Correct/random trajectory mix: `50/50`

Evaluation matrix:

- All neural planners run as MPC.
- Inner planners: greedy one-step, beam search, best-first/weighted A*,
  categorical CEM, sequence local search, progressive-widening UCT-MCTS.
- Calibration baseline: exact symbolic Sudoku solver.
- Transition variants: symbolic re-encode and latent rollout.
- Score variants: true Hamming oracle, oracle latent goal distance, predicted
  goal distance.
- Horizons: `4`, `8`, `16`, `32`, `64`.

Gate:

Pass only if at least one learned or oracle latent planner produces exact
solves under fill-only actions, and diagnostics show local action ranking is
better than the legacy compact-scorer failure mode.

The first LR submission `3740707` is cancelled/superseded because it used
8-frame training trajectories and pre-fix MCTS. Do not analyze it as the clean
LeWM baseline.
