# Current Experiments

Source of truth: `../sequence-editing-report/CURRENT_EXPERIMENTS.md`.

# Current Experiments

Last updated: 2026-07-04T09:43:37

## Active: Counterfactual Editable Weekend Wave

Research questions:
- Does counterfactual branching improve action dependence and latent-rollout top-action accuracy?
- Does making non-given cells editable reduce the irreversibility/asymmetric-distance failure mode?
- Which action conditioning works best once counterfactual branches and editable repairs are present?
- Does receding-horizon waypoint prediction beat one-shot terminal predicted-goal planning?
- When predicted-waypoint solve rate is zero, are intermediate waypoints nevertheless locally correct, progressive, or trackable?
- Do asymmetric source/goal projections or value-guided quasi-distance improve non-oracle planning?
- Can Delta-JEPA work once data coverage and action conditioning are fixed?
- Which isolated winners combine constructively in an integrated recipe?

Current summary:
- run suffix: `_mb4ga2`
- oversight cadence: `6` hours
- repair evals enabled: `True`
- completed checkpoints: 30
- total planner rows: 168

Insights:
- Best current row: mpc_beam latent_rollout oracle_goal_raw_euclidean_distance solve=8/8 h=0.0
- Delta branch has eval rows; compare every grid variant against its single-CLS paired variant before promoting.
- Waypoint rows are present; prioritize predicted-waypoint versus oracle-waypoint gap before terminal predicted-goal variants.
- If predicted-waypoint solve rate is still zero, inspect waypoint quality directly: latent alignment to oracle future waypoints, Hamming progress after one tracked chunk, and trackability distance.

Required oversight diagnostics:
- Predicted waypoint latent alignment: compare `q_hat_H(s_t)` with oracle future waypoint `E(s_min(t+H,T))` by raw L2/cosine and report the oracle-vs-predicted gap.
- Predicted waypoint progress: after tracking `q_hat_H` for one MPC chunk, report Hamming/edit-distance improvement toward the solved board even if no full solve occurs.
- Predicted waypoint trackability: report `D(E(s_after_mpc), q_hat_H)` and compare it to `D(E(s_t), q_hat_H)`.
- Multi-horizon consistency: for multi-waypoint heads, check whether predicted H4/H8/H16 waypoints are closest to their matching oracle future horizon rather than mismatched horizons.
- Terminal-locality split: report waypoint quality separately for early, middle, and near-terminal states.

Variant table:

| Variant | Checkpoint | Rows | Best result |
|---|---:|---:|---|
| `S0_anchor_olddata` | `True` | `8` | mpc_beam / latent_rollout / oracle_goal_raw_euclidean_distance = 8/8, h 0.0 |
| `S1_counterfactual_fill` | `True` | `8` | mpc_beam / latent_rollout / oracle_goal_raw_euclidean_distance = 8/8, h 0.0 |
| `S2_counterfactual_edit` | `True` | `7` | mpc_beam / latent_rollout / oracle_goal_raw_euclidean_distance = 8/8, h 0.0 |
| `S3_counterfactual_edit_adaln` | `True` | `7` | mpc_beam / latent_rollout / oracle_goal_raw_euclidean_distance = 8/8, h 0.0 |
| `S4_counterfactual_edit_oldlocal` | `True` | `8` | mpc_beam / latent_rollout / oracle_goal_raw_euclidean_distance = 8/8, h 0.0 |
| `E0_base_cf_edit` | `True` | `8` | mpc_beam / latent_rollout / oracle_goal_raw_euclidean_distance = 8/8, h 0.0 |
| `E1_hierarchy_l4_l8_l16` | `True` | `7` | mpc_beam / latent_rollout / oracle_goal_raw_euclidean_distance = 8/8, h 0.0 |
| `E2_waypoint_h8` | `True` | `8` | waypoint_beam / latent_rollout / oracle_waypoint_raw_euclidean_distance = 8/8, h 0.0 |
| `E3_waypoint_h16` | `True` | `7` | waypoint_beam / latent_rollout / oracle_waypoint_raw_euclidean_distance = 8/8, h 0.0 |
| `E4_waypoint_h4_h8_h16` | `True` | `7` | waypoint_beam / latent_rollout / oracle_waypoint_raw_euclidean_distance = 8/8, h 0.0 |
| `E5_waypoint_h16_hierarchy` | `True` | `7` | waypoint_beam / latent_rollout / oracle_waypoint_raw_euclidean_distance = 8/8, h 0.0 |
| `D0_online_h1_grid` | `True` | `4` | mpc_beam / symbolic_reencode / oracle_goal_raw_euclidean_distance = 8/8, h 0.0 |
| `D0_online_h1_single` | `True` | `4` | mpc_beam / symbolic_reencode / oracle_goal_raw_euclidean_distance = 0/8, h 45.625 |
| `D1_online_ordered_h12345_grid` | `True` | `4` | mpc_beam / symbolic_reencode / oracle_goal_raw_euclidean_distance = 8/8, h 0.0 |
| `D1_online_ordered_h12345_single` | `True` | `4` | mpc_beam / symbolic_reencode / oracle_goal_raw_euclidean_distance = 0/8, h 44.625 |
| `D2_online_set_h12345_grid` | `True` | `4` | mpc_beam / symbolic_reencode / oracle_goal_raw_euclidean_distance = 7/8, h 0.125 |
| `D2_online_set_h12345_single` | `True` | `7` | mpc_beam / symbolic_reencode / oracle_goal_raw_euclidean_distance = 0/8, h 44.25 |
| `D3_hybrid_ema_h1_grid` | `True` | `4` | mpc_beam / latent_rollout / oracle_goal_raw_euclidean_distance = 8/8, h 0.0 |
| `D3_hybrid_ema_h1_single` | `True` | `8` | mpc_beam / symbolic_reencode / oracle_goal_raw_euclidean_distance = 0/8, h 43.0 |
| `D4_hybrid_ema_set_h12345_grid` | `True` | `4` | mpc_beam / latent_rollout / oracle_goal_raw_euclidean_distance = 8/8, h 0.0 |
| `D4_hybrid_ema_set_h12345_single` | `True` | `4` | mpc_beam / symbolic_reencode / oracle_goal_raw_euclidean_distance = 0/8, h 41.875 |
| `D5_ema_vicreg_no_ldad_grid` | `True` | `4` | mpc_beam / latent_rollout / oracle_goal_raw_euclidean_distance = 8/8, h 0.0 |
| `D5_ema_vicreg_no_ldad_single` | `True` | `4` | mpc_beam / symbolic_reencode / oracle_goal_raw_euclidean_distance = 0/8, h 45.25 |
| `V1_asym_hindsight` | `True` | `7` | mpc_beam / latent_rollout / oracle_goal_projected_euclidean_distance = 8/8, h 0.0 |
| `V2_iql_quasi` | `True` | `5` | mpc_beam / latent_rollout / predicted_goal_projected_euclidean_distance = 0/8, h 55.375 |
| `V3_waypoint_asym_hindsight` | `True` | `6` | waypoint_beam / latent_rollout / oracle_waypoint_raw_euclidean_distance = 8/8, h 0.0 |
| `I0_integrated_waypoint_asym` | `True` | `5` | waypoint_beam / latent_rollout / oracle_waypoint_raw_euclidean_distance = 8/8, h 0.0 |
| `I1_integrated_waypoint_iql` | `True` | `5` | waypoint_beam / latent_rollout / oracle_waypoint_raw_euclidean_distance = 8/8, h 0.0 |
| `I2_integrated_best_delta_if_gate_passes_grid` | `True` | `2` | waypoint_beam / latent_rollout / oracle_waypoint_raw_euclidean_distance = 0/8, h 55.125 |
| `I2_integrated_best_delta_if_gate_passes_single` | `True` | `1` | waypoint_beam / latent_rollout / oracle_waypoint_raw_euclidean_distance = 0/8, h 54.5 |
