# Grid 5 Plan

This file is the clean Grid-5-era plan. The long historical record remains in
`../sequence-editing-report`; this file intentionally omits pre-Grid5 job
detail.

## Main Opinion

If the current compact JEPA fails even when candidate Sudoku boards are applied
symbolically and re-encoded before scoring, that should not be read as "JEPA
cannot work." It should be read as evidence that our current JEPA geometry is
not yet the geometry used by successful LeWorldModel-style planning.

Sudoku is deterministic and noiseless, but that does not automatically make the
latent planning metric easier. In continuous-control settings, smooth visual
similarity, reachable actions, and Euclidean latent proximity often correlate.
In Sudoku, one wrong digit can make a board globally invalid while looking very
close under a predictor-trained latent. Predictive MSE plus SIGReg can learn a
valid next-latent model while still making Euclidean distance poor for discrete
constraint satisfaction.

So the core suspicion is not "we need a random ranking loss." The suspicion is:

- latent MSE is self-referential because the target distance is defined by the
  same representation being trained;
- SIGReg/VICReg prevent collapse but do not impose reachability, constraint, or
  task-aligned distance;
- the compact single-vector bottleneck may encode board identity without making
  local edit directions geometrically meaningful;
- current action embeddings may not define a clean action manifold for CEM/GD;
- wrong/off-policy states may not be covered in the way planning generates
  them;
- the learned terminal-energy head may be asked to learn small differences in a
  latent metric that is itself not planner aligned.

Ranking, contrastive, advantage, and verifier losses are worth trying, but they
should be treated as ways to test and repair geometry, not as an ad hoc patch
to hide a broken world model.

## Grid 5C Gate

Grid 5C crosses:

- planner: `beam`, `mcts`, `nn_cem`
- transition: `symbolic_reencode`, `latent_rollout`
- score: oracle `latent_goal`, learned `goal_energy`

The important read is not just exact solve. Read:

- solve rate;
- mean remaining Hamming;
- root goal-value rate;
- first-step qualitative examples;
- symbolic re-encode vs latent rollout gap;
- oracle score vs learned score gap;
- runtime and failure mode by optimizer.

## If Grid 5C Works

If oracle `latent_goal` with `symbolic_reencode` works, the compact
representation is viable.

Next steps:

1. Scale the winning planner to more boards.
2. If `latent_rollout` fails while `symbolic_reencode` works, train better
   long-horizon predictor fidelity:
   - recursive K `8/16/32`;
   - scheduled re-encoding during training/eval;
   - consistency between predicted horizon latents and re-encoded horizon
     states;
   - EMA targets and normalization on the horizon targets.
3. If learned `goal_energy` fails while oracle `latent_goal` works, train a
   better learned scorer using the oracle action-ranking signal:
   - action-conditioned advantage;
   - local pairwise/listwise ranking;
   - multi-positive feasible successor contrastive learning;
   - terminal/verifier auxiliary head.
4. If only one optimizer works:
   - `beam`: invest in structured discrete search and pruning;
   - `mcts`: add progressive widening, cached leaf scoring, and a cheap default
     rollout policy;
   - `nn_cem`: continue continuous action-embedding planning, test gradient/CEM
     hybrids, and consider VQ action embeddings.

## If Grid 5C Does Not Work

If oracle `latent_goal` with `symbolic_reencode` fails across optimizers, do not
blindly scale compact single-state JEPA and do not add hierarchy on top of the
same low-level scorer.

Next steps:

1. Audit for bugs and mismatches against LeWorldModel:
   - target normalization;
   - predictor input/output normalization;
   - stop-gradient placement;
   - action embedding geometry;
   - EMA target use;
   - recurrent rollout training;
   - whether the planner optimizes in the same space the model was trained on.
2. Build geometry diagnostics:
   - distance spectrum for correct terminal boards versus one-cell corruptions;
   - monotonicity over exact symbolic trajectories;
   - nearest neighbors in latent space by board edit distance and constraint
     violation;
   - PCA/whitening/eigenvalue spectrum;
   - action-vector displacement consistency.
3. Try objective changes that still target a general JEPA recipe:
   - predictor-consistency over multi-step exact symbolic rollouts;
   - action-conditioned value/advantage head;
   - multi-positive future/reachable contrastive loss;
   - verifier-style terminal/constraint head as an auxiliary, not a Sudoku-only
     replacement;
   - tokenized/local representation as a control to identify what the compact
     bottleneck lost.

## Hierarchy Gate

Retry hierarchy only after the low-level scorer can rank exact symbolic
candidate boards. A high-level model can only generate useful subgoals if the
lower level has a reliable reachable-state metric. Otherwise hierarchy will
produce plausible but low-level-unusable latent states.
