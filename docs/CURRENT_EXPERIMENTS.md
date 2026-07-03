# Current Experiments

Source of truth: `../sequence-editing-report/CURRENT_EXPERIMENTS.md`.

# Current Experiments

Last updated: 2026-07-03T21:43:12

## Active: Counterfactual Editable Weekend Wave

Research questions:
- Does counterfactual branching improve action dependence and latent-rollout top-action accuracy?
- Does making non-given cells editable reduce the irreversibility/asymmetric-distance failure mode?
- Which action conditioning works best once counterfactual branches and editable repairs are present?
- Does receding-horizon waypoint prediction beat one-shot terminal predicted-goal planning?
- Do asymmetric source/goal projections or value-guided quasi-distance improve non-oracle planning?
- Can Delta-JEPA work once data coverage and action conditioning are fixed?
- Which isolated winners combine constructively in an integrated recipe?

Current summary:
- run suffix: `_mb4ga2`
- oversight cadence: `6` hours
- repair evals enabled: `True`
- completed checkpoints: 0
- total planner rows: 0

Operational state:
- Active replacement train array: `3809648`, variants `0-29`, suffix `_mb4ga2`, `BATCH_SIZE=4`, `GRADIENT_ACCUMULATION_STEPS=2`, effective batch `8`.
- Dependency-held replacement eval jobs: `3809649`-`3809678`, one per train array element, same `_mb4ga2` suffix.
- Superseded pending 12h oversight jobs `3809682`-`3809700` were canceled.
- Active 6h oversight cadence: `3809723`-`3809742`, repair evals enabled, same `_mb4ga2` suffix.

Insights:

Variant table:

| Variant | Checkpoint | Rows | Best result |
|---|---:|---:|---|
| `S0_anchor_olddata` | `False` | `0` |  |
| `S1_counterfactual_fill` | `False` | `0` |  |
| `S2_counterfactual_edit` | `False` | `0` |  |
| `S3_counterfactual_edit_adaln` | `False` | `0` |  |
| `S4_counterfactual_edit_oldlocal` | `False` | `0` |  |
| `E0_base_cf_edit` | `False` | `0` |  |
| `E1_hierarchy_l4_l8_l16` | `False` | `0` |  |
| `E2_waypoint_h8` | `False` | `0` |  |
| `E3_waypoint_h16` | `False` | `0` |  |
| `E4_waypoint_h4_h8_h16` | `False` | `0` |  |
| `E5_waypoint_h16_hierarchy` | `False` | `0` |  |
| `D0_online_h1_grid` | `False` | `0` |  |
| `D0_online_h1_single` | `False` | `0` |  |
| `D1_online_ordered_h12345_grid` | `False` | `0` |  |
| `D1_online_ordered_h12345_single` | `False` | `0` |  |
| `D2_online_set_h12345_grid` | `False` | `0` |  |
| `D2_online_set_h12345_single` | `False` | `0` |  |
| `D3_hybrid_ema_h1_grid` | `False` | `0` |  |
| `D3_hybrid_ema_h1_single` | `False` | `0` |  |
| `D4_hybrid_ema_set_h12345_grid` | `False` | `0` |  |
| `D4_hybrid_ema_set_h12345_single` | `False` | `0` |  |
| `D5_ema_vicreg_no_ldad_grid` | `False` | `0` |  |
| `D5_ema_vicreg_no_ldad_single` | `False` | `0` |  |
| `V1_asym_hindsight` | `False` | `0` |  |
| `V2_iql_quasi` | `False` | `0` |  |
| `V3_waypoint_asym_hindsight` | `False` | `0` |  |
| `I0_integrated_waypoint_asym` | `False` | `0` |  |
| `I1_integrated_waypoint_iql` | `False` | `0` |  |
| `I2_integrated_best_delta_if_gate_passes_grid` | `False` | `0` |  |
| `I2_integrated_best_delta_if_gate_passes_single` | `False` | `0` |  |
