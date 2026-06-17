# Experiment Plan

Last updated: 2026-06-17 08:48 CEST

## Grid-Token Goal-JEPA

The current plan replaces the CLS/vector-state LeWM architecture with a
full-grid token latent and no scalar value head.

Modules:

- Context encoder `C_omega(c)`: bidirectional transformer over Sudoku givens
  plus clue/editable/active masks.
- State encoder `f_theta(s, H_c)`: bidirectional self-attention over current
  board tokens plus cross-attention to cached context tokens.
- Markov predictor `P_phi(H_t, a_t, H_c)`: bidirectional transformer over one
  action token plus the current latent board tokens, with cross-attention to
  context. It sees only the current board latent, not a causal history.
- Goal predictor `q_eta(H_c)`: output-query decoder that predicts terminal
  board-token latents from context.
- Planner score: tokenwise normalized Euclidean distance
  `D(f_theta(s,H_c), q_eta(H_c))`.

There is no CLS token, value head, validity head, reachability head, or
dead-end head.

## Losses

The full model trains:

- multi-step dynamics MSE with self-rollout horizons `1,4,8,16`
- covariance SIGReg over active state tokens
- goal MSE against encoded true terminal board tokens
- goal InfoNCE over mean-pooled goal summaries
- progress ranking along successful trajectories only, selected by
  `oracle_mask`
- action ranking between encoded symbolic successors for target-consistent and
  wrong fill actions
- temporal straightening over valid three-frame trajectory triplets
- terminal corruption contrast against 1-5 digit corruptions

Temporal straightening computes curvature from adjacent latent velocities over
the full active grid-token latent and is independent of the predicted goal.

## Ablations

Run one peak LR (`1e-4`) and one seed per ablation. Use linear warmup for
`1000` optimizer steps, then cosine decay to `1e-5`.

| Run | Change |
| --- | --- |
| `M0_full` | Full Grid-Token Goal-JEPA |
| `R1_no_context_masks` | Remove explicit clue/editable context masks |
| `R2_mean_pooled_distance` | Replace tokenwise distance with mean-pooled distance |
| `R3_k1_only` | One-step dynamics only |
| `R3_k4` | Multi-step horizons `1,4` |
| `R3_k8` | Multi-step horizons `1,4,8` |
| `R3_k16` | Multi-step horizons `1,4,8,16` |
| `R4_no_goal_nce` | Remove goal InfoNCE |
| `R5_no_progress_rank` | Remove progress ranking |
| `R6_no_action_rank` | Remove action ranking |
| `R7_no_terminal_corrupt` | Remove terminal corruption contrast |
| `R8_no_sigreg` | Remove SIGReg |
| `R9_no_temporal_straightening` | Remove temporal straightening |

Training budget used for submitted suite:

- optimizer steps: `60000`
- microbatch size: `8`
- gradient accumulation: `1`
- effective batch size: `8` full trajectories per optimizer step

## Evaluation

Each completed checkpoint should run a separate eval job. The first
dependency-held eval array failed before planning on a checkpoint loader issue;
rerun eval from the completed checkpoints after the loader fix.

Planning matrix:

- MPC outer loop
- Beam search inner optimizer
- Beam widths `1,4,16,64`
- Beam depths `8,16,32,64`
- Scores: oracle goal distance and predicted goal distance
- Transitions: symbolic re-encode and latent rollout

Diagnostics record losses, latent geometry/effective rank, monotonicity,
top-positive action accuracy, near-goal corruption margin, concrete action
panels, predictor rollout drift by horizon, latent-rollout action ranking,
predicted-goal vs oracle-goal alignment, distance-vs-Hamming Spearman
correlation, action margins by fill depth, terminal corruption margins by
corruption size, planner solve rate, remaining Hamming, action-eval counts,
and timing.
