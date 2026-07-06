# Experiment Plan

Source of truth: `../sequence-editing-report/BACKLOG.md` and
`../sequence-editing-report/CURRENT_EXPERIMENTS.md`.

## Verifier-Free Compatibility / Progress Energy Plan

Proposed, not implemented or submitted.

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
