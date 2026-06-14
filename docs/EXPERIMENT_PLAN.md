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
- Padding: masks remove padded frames from prediction/value/SIGReg losses and
  from encoder/predictor BatchNorm projector statistics.
- BatchNorm context: predictor projection is step-wise across sequence time,
  so unsupervised final/future outputs cannot change supervised prefix
  predictions. State embeddings are encoded from board sequences only; solved
  frames reuse their own state embedding as the exact goal target.

Evaluation matrix:

- All neural planners run as MPC.
- Inner planners: greedy one-step, beam search, best-first/weighted A*,
  categorical CEM, sequence local search, and UCT MCTS. With the default
  `mcts_branch_size > 0`, MCTS is score-pruned progressive UCT; set
  `mcts_branch_size=0` for unpruned progressive UCT.
- Calibration baseline: exact symbolic Sudoku solver.
- Transition variants: symbolic re-encode and latent rollout.
- Score variants: true Hamming oracle, oracle latent goal distance, predicted
  goal distance.
- Horizons: `4`, `8`, `16`, `32`, `64`.
- Latent rollout: MPC passes observed board/action history into the predictor,
  so latent rollout uses the same absolute fill-step context as training when
  it fits. If observed history plus requested horizon exceeds `max_history`, the
  scorer caps the effective lookahead rather than throwing. Score-pruned branch
  ranking and diagnostic projection panels use the same history context.

Diagnostics written automatically:

- Scalar losses: raw and weighted prediction/SIGReg/value losses, value MAE,
  RMSE, correlation, predicted-vs-target distance scale, early/middle/late
  transition MSE, and oracle-vs-random trajectory splits.
- Latent geometry: PCA CSV/SVG, covariance/variance/effective-rank summaries,
  random-projection normality checks, and optional t-SNE/UMAP CSVs when local
  packages are available.
- Trajectory diagnostics: oracle and learned value along true fill
  trajectories, monotonicity rates, stepwise distance drops, and per-step
  JSONL traces.
- Action diagnostics: local action ranking across fill fractions and horizons,
  pairwise gold-vs-wrong accuracy, top-is-gold rates, and concrete panels that
  print gold, same-cell wrong, nearby-cell, and far-cell actions with true
  Hamming, oracle latent distance, and learned goal-distance scores.
- Implementation-failure diagnostics: train-vs-eval terminal goal-distance,
  full-vs-truncated predictor BatchNorm deltas, no-history vs full-history
  latent-rollout rank divergence, branch-prune gold-action survival, and
  latent-rollout vs symbolic re-encode drift by horizon.
- Planner cost diagnostics: every planner-matrix row records wall-clock time,
  action-evaluation counts, and seconds per score call.

Gate:

Pass only if at least one learned or oracle latent planner produces exact
solves under fill-only actions, and diagnostics show local action ranking is
better than the legacy compact-scorer failure mode.

The first LR submission `3740707` is cancelled/superseded because it used
8-frame training trajectories and pre-fix MCTS. Do not analyze it as the clean
LeWM baseline.

Submit the replacement sweep only after the user says `go`.
