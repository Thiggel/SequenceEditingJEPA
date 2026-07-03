# Experiment Plan

Source of truth: `../sequence-editing-report/BACKLOG.md` and
`../sequence-editing-report/CURRENT_EXPERIMENTS.md`.

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
