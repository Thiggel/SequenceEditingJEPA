# Current Experiments

Source of truth: `../sequence-editing-report/CURRENT_EXPERIMENTS.md`.

# Current Experiments

Last updated: 2026-07-04 09:41 CEST

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
- completed checkpoints: 30/30
- total planner rows: 167 so far
- eval status: 17/30 eval jobs completed, 13/30 still running
- error status: no non-empty stderr logs found for replacement train/eval/oversight jobs

Insights:
- Best current row: mpc_beam latent_rollout oracle_goal_raw_euclidean_distance solve=8/8 h=0.0
- Delta branch has eval rows; compare every grid variant against its single-CLS paired variant before promoting.
- Waypoint rows are present; prioritize predicted-waypoint versus oracle-waypoint gap before terminal predicted-goal variants.
- Direct 09:41 aggregation: oracle latent-rollout planning solves 8/8 for the S/E/V/I EMA+VICReg branches and for grid EMA/Delta-hybrid controls; paper-pure online Delta grid has only symbolic-reencode solves so far, and all single-CLS variants remain at 0/8.
- Direct 09:41 aggregation: predicted goal / predicted waypoint rows remain 0/8; best predicted-waypoint row is `E4_waypoint_h4_h8_h16` symbolic reencode at h 48.875.

Current direct stage summary:

| Stage | Rows so far | Variants with rows | Best current result |
|---|---:|---:|---|
| S data/action | 37 | 5 | `S1_counterfactual_fill`, mpc beam, latent rollout, oracle raw L2, depth 4: 8/8, h 0.0 |
| E hierarchy/waypoint | 44 | 6 | `E4_waypoint_h4_h8_h16`, waypoint beam, latent rollout, oracle waypoint raw L2, depth 4: 8/8, h 0.0 |
| D Delta-JEPA paired | 55 | 12 | grid variants can solve with oracle symbolic reencode or EMA/hybrid latent rollout; single-CLS variants remain 0/8 |
| V value/asym | 18 | 3 | `V3_waypoint_asym_hindsight`, waypoint beam, latent rollout, oracle waypoint raw L2, depth 4: 8/8, h 0.0 |
| I integrated | 13 | 4 | `I0_integrated_waypoint_asym`, waypoint beam, latent rollout, oracle waypoint raw L2, depth 4: 8/8, h 0.0 |

The variant table below is the last scheduled oversight snapshot from 03:43 and is stale for row counts after the direct 09:41 aggregation.

Variant table:

| Variant | Checkpoint | Rows | Best result |
|---|---:|---:|---|
| `S0_anchor_olddata` | `True` | `8` | mpc_beam / latent_rollout / oracle_goal_raw_euclidean_distance = 8/8, h 0.0 |
| `S1_counterfactual_fill` | `True` | `8` | mpc_beam / latent_rollout / oracle_goal_raw_euclidean_distance = 8/8, h 0.0 |
| `S2_counterfactual_edit` | `True` | `3` | mpc_beam / latent_rollout / oracle_goal_raw_euclidean_distance = 8/8, h 0.0 |
| `S3_counterfactual_edit_adaln` | `True` | `3` | mpc_beam / latent_rollout / oracle_goal_raw_euclidean_distance = 8/8, h 0.0 |
| `S4_counterfactual_edit_oldlocal` | `True` | `3` | mpc_beam / latent_rollout / oracle_goal_raw_euclidean_distance = 8/8, h 0.0 |
| `E0_base_cf_edit` | `True` | `5` | mpc_beam / latent_rollout / oracle_goal_raw_euclidean_distance = 8/8, h 0.0 |
| `E1_hierarchy_l4_l8_l16` | `True` | `3` | mpc_beam / latent_rollout / oracle_goal_raw_euclidean_distance = 8/8, h 0.0 |
| `E2_waypoint_h8` | `True` | `5` | waypoint_beam / latent_rollout / oracle_waypoint_raw_euclidean_distance = 8/8, h 0.0 |
| `E3_waypoint_h16` | `True` | `3` | waypoint_beam / latent_rollout / oracle_waypoint_raw_euclidean_distance = 8/8, h 0.0 |
| `E4_waypoint_h4_h8_h16` | `True` | `3` | waypoint_beam / latent_rollout / oracle_waypoint_raw_euclidean_distance = 8/8, h 0.0 |
| `E5_waypoint_h16_hierarchy` | `True` | `2` | waypoint_beam / latent_rollout / oracle_waypoint_raw_euclidean_distance = 8/8, h 0.0 |
| `D0_online_h1_grid` | `True` | `1` | mpc_beam / latent_rollout / oracle_goal_raw_euclidean_distance = 0/8, h 55.125 |
| `D0_online_h1_single` | `True` | `3` | mpc_beam / symbolic_reencode / oracle_goal_raw_euclidean_distance = 0/8, h 45.625 |
| `D1_online_ordered_h12345_grid` | `True` | `1` | mpc_beam / latent_rollout / oracle_goal_raw_euclidean_distance = 0/8, h 54.625 |
| `D1_online_ordered_h12345_single` | `True` | `1` | mpc_beam / latent_rollout / oracle_goal_raw_euclidean_distance = 0/8, h 55.375 |
| `D2_online_set_h12345_grid` | `False` | `0` |  |
| `D2_online_set_h12345_single` | `True` | `0` |  |
| `D3_hybrid_ema_h1_grid` | `False` | `0` |  |
| `D3_hybrid_ema_h1_single` | `True` | `0` |  |
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
