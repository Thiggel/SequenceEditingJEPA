# Experiment Plan

Source of truth: `../sequence-editing-report/BACKLOG.md` and
`../sequence-editing-report/CURRENT_EXPERIMENTS.md`.

## Verifier-Free Compatibility / Progress Energy Plan

Implemented, audit-fixed, submitted.

Research questions:

- Can a learned compatibility energy replace the oracle solution latent during
  inference?
- Can a separate remaining-edit head provide progress without collapsing the
  dynamics latent?
- Does counterfactual successor ranking train the exact local discrimination
  needed by verifier-free MPC?
- Does a policy prior help search after the learned state score is calibrated?

Fixed base recipe:

- full `9x9` grid-token latent, not single-CLS
- dropout off
- EMA target encoder plus VICReg
- editable non-clue cells
- counterfactual branches
- dense K8 smooth/count rollout supervision
- affected-context dynamics weighting
- no terminal goal predictor, no waypoint predictor, no oracle goal score

Planned model additions:

- tokenwise `W`: wrong-commitment compatibility energy
- tokenwise `R`: remaining-edit / Hamming-to-solution head
- optional action prior over editable cell-value actions
- verifier-free planner score `alpha * W + beta * R - eta * log pi`

Planned diagnostics:

- W AUC and wrong-count MAE on same-fill and near-solution corruptions
- R MAE and Spearman correlation against editable distance to solution
- successor pairwise/listwise action-ranking accuracy on latent rollouts
- predicted-latent W/R calibration versus encoded symbolic successors
- no-verifier MPC solve rate, remaining Hamming, first wrong commitment, and
  action-evaluation count

Prepared scripts:

- `scripts/slurm/run_grid_goal_verifier_energy_train.slurm`
- `scripts/slurm/run_grid_goal_verifier_energy_eval.slurm`
- `scripts/experiments/submit_grid_goal_verifier_energy.sh`

Audit blockers fixed on 2026-07-06:

- `verifier_energy` MPC no longer encodes the oracle goal latent during setup.
- `_sample_rank_actions(..., allow_overwrite=True)` skips full filled-wrong
  sequence states no longer; overwrite mode samples states mismatched from the
  solved board.
- The listwise verifier-targeted policy prior ignores no-blank wrong boards, so
  it now adds the positive repair cell even when the board has no blanks.
- Single-latent compatibility loss can use count targets as BCE labels and go
  negative no longer; BCE labels are clamped to binary and count supervision
  remains separate.

Regression tests were added in `tests/test_grid_goal_jepa.py` and now pass.

Prepared variants:

| Variant | Purpose |
|---|---|
| `E0_base_oracle_sanity` | Preserve oracle raw-L2 baseline with no verifier heads. |
| `E1_compat_state` | Train only W on encoded states plus corruption negatives. |
| `E2_remaining_state` | Train only R on encoded states plus corruption states. |
| `E3_wr_state` | Train W+R on encoded states. |
| `E4_wr_predicted` | Add W/R supervision on one-step predicted successor latents. |
| `E5_wr_pairwise_rank` | Add pairwise successor ranking on predicted latents. |
| `E6_wr_listwise_rank` | Replace pairwise with listwise successor ranking. |
| `E7_wr_listwise_policy` | Add verifier-targeted policy prior and planning bias. |
| `E8_wr_no_counterfactual` | Remove counterfactual dynamics branches from the full scorer. |
| `E9_wr_no_corruption` | Remove synthetic corruption negatives from the full scorer. |
| `F0_full_score` | W+R, predicted-latent calibration, corruptions, listwise ranking. |
| `F1_full_policy` | F0 plus verifier-targeted policy prior. |

## Counterfactual Editable Weekend Wave

Prepared, not submitted.

Research questions:

- Does counterfactual branching teach the world model action dependence?
- Does allowing non-clue cell overwrites remove the irreversible fill-only
  geometry failure?
- Does receding-horizon waypoint prediction work better than one-shot terminal
  goal prediction?
- Do asymmetric/value metric heads improve non-oracle planning?
- Can Delta-JEPA work once action/data coverage are fixed?

Stages:

| Stage | Purpose |
|---|---|
| `S` | Data/action smoke tests: old data, counterfactual fill, editable repair, AdaLN marker, old-local conditioning. |
| `E` | EMA+VICReg base, hierarchy, and waypoint variants. |
| `D` | Delta-JEPA paired full-grid and single-CLS variants. |
| `V` | Asymmetric/value geometry variants. |
| `I` | Integrated winners, including paired Delta-JEPA if the Delta gate passes. |

Operational invariant: any Delta-JEPA row must be paired as `_grid` and
`_single`. This applies to the dedicated Delta stage and any later autonomous
follow-up or integrated Delta stage.
