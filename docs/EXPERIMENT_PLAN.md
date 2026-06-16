# Experiment Plan

Last updated: 2026-06-16 15:05 CEST

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
- progress ranking along successful trajectories
- action ranking between target-consistent and wrong fill actions
- terminal corruption contrast against 1-5 digit corruptions

## Ablations

Run one LR (`1e-4`) and one seed per ablation:

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

## Evaluation

Each checkpoint should run a separate dependency-held eval job.

Planning matrix:

- MPC outer loop
- Beam search inner optimizer
- Beam widths `1,4,16,64`
- Beam depths `8,16,32,64`
- Scores: oracle goal distance and predicted goal distance
- Transitions: symbolic re-encode and latent rollout

Diagnostics record losses, latent geometry/effective rank, monotonicity,
top-positive action accuracy, near-goal corruption margin, concrete action
panels, planner solve rate, remaining Hamming, action-eval counts, and timing.
