# Current Experiments

Source of truth: `../sequence-editing-report/CURRENT_EXPERIMENTS.md`.

# Current Experiments

Last updated: 2026-07-06 08:28 CEST

## Active: Wide Single-CLS Oracle Probe

Purpose:
- Test whether the one-vector board latent failed because `d_model=256` was too narrow, or because the one-vector geometry/predictor is structurally wrong for Sudoku.
- Keep the readout oracle-only for now: no predicted goal, no waypoint, no value head.

Runs:

| Variant | Train | Eval | Stabilization |
|---|---:|---:|---|
| `W0_ema_vicreg_d1024` | `3815481` | `3815482` | EMA + VICReg, no LDAD |
| `W1_ema_ldad_set_d1024` | `3815483` | `3815484` | EMA + LDAD, no VICReg |
| `W2_ldad_vicreg_set_d1024` | `3815485` | `3815486` | LDAD + VICReg, no EMA |
| `W3_ldad_only_set_d1024` | `3815487` | `3815488` | LDAD only, no EMA/VICReg |

State at submission check:
- train jobs `3815481`, `3815483`, `3815485`, and `3815487` are running on `rtxpro6k` node `a2841`
- eval jobs `3815482`, `3815484`, `3815486`, and `3815488` are dependency-held

Common setup:
- `latent_representation=single`
- `d_model=1024`, `num_heads=16`, `distance_dim=256`
- 5k steps, effective batch 8 via micro-batch 2 and grad accumulation 4
- counterfactual editable data, K8 smooth/count dense rollout, affected-context dynamics weighting
- eval: oracle raw L2 only, `mpc_beam`, latent rollout and symbolic re-encode, depths `{4,16}`, 8 boards
- output root: `$WORK/sequence-editing` to avoid the current `$HPCVAULT` quota limit

Storage housekeeping:
- `$HPCVAULT/sequence-editing` was reduced from about `759G` to `5.9G`
- removed redundant `checkpoint-*.pt`, final old `checkpoint.pt`, HF-style model/optimizer artifacts, vault cache, and failed zero-byte single-wide files
- preserved lightweight configs, metrics, diagnostics, and planner/result records
- cancelled stale sequence-editing dependency-never-satisfied evals and old weekend oversight jobs; current wide-single train/eval jobs were not touched

Gate:
- If `W0` solves but LDAD-only variants fail, width helps single-CLS only with EMA/VICReg.
- If `W3` solves under latent rollout, prior Delta/single failure was probably capacity-limited.
- If all four fail, the single-vector board latent is likely structurally poor for this planner.

## Completed Context: Counterfactual Editable Weekend Wave

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
- total planner rows: 326
- final interpretation: oracle full-grid geometry works; predicted goals/waypoints and faithful macro hierarchy do not.

Insights:
- Best current row: mpc_beam latent_rollout oracle_goal_raw_euclidean_distance solve=8/8 h=0.0
- Delta branch has eval rows; compare every grid variant against its single-CLS paired variant before promoting.
- Waypoint rows are present; prioritize predicted-waypoint versus oracle-waypoint gap before terminal predicted-goal variants.
- Current waypoint rows may be flat-only; do not treat waypoint_beam solves as evidence that hierarchical MPC works.
- If predicted-waypoint solve rate is still zero, inspect waypoint quality directly: latent alignment to oracle future waypoints, Hamming progress after one tracked chunk, and trackability distance.
- Final weekend conclusion: counterfactual/editable training preserved strong oracle latent-rollout MPC but did not fix non-oracle planning. Macro CEM/MPPI hierarchy rows also failed even with oracle waypoints.

Required oversight diagnostics:
- predicted waypoint latent alignment: compare q_hat_H(s_t) with oracle future waypoint E(s_{min(t+H,T)}) by raw L2/cosine and report the oracle-vs-predicted gap
- predicted waypoint progress: after tracking q_hat_H for one MPC chunk, report Hamming/edit-distance improvement toward the solved board even if no full solve occurs
- predicted waypoint trackability: report D(E(s_after_mpc), q_hat_H) and compare it to D(E(s_t), q_hat_H)
- multi-horizon consistency: for multi-waypoint heads, check whether predicted H4/H8/H16 waypoints are increasingly closer to their matching oracle future waypoints than to mismatched horizons
- terminal-locality split: report waypoint quality separately for early, middle, and near-terminal states
- hierarchical waypoint rows must include macro-action CEM or MPPI tracking; flat waypoint_beam rows are only primitive-tracker baselines

Variant table:

| Variant | Checkpoint | Rows | Best result |
|---|---:|---:|---|
| `S0_anchor_olddata` | `True` | `8` | mpc_beam / latent_rollout / oracle_goal_raw_euclidean_distance = 8/8, h 0.0 |
| `S1_counterfactual_fill` | `True` | `8` | mpc_beam / latent_rollout / oracle_goal_raw_euclidean_distance = 8/8, h 0.0 |
| `S2_counterfactual_edit` | `True` | `8` | mpc_beam / latent_rollout / oracle_goal_raw_euclidean_distance = 8/8, h 0.0 |
| `S3_counterfactual_edit_adaln` | `True` | `8` | mpc_beam / latent_rollout / oracle_goal_raw_euclidean_distance = 8/8, h 0.0 |
| `S4_counterfactual_edit_oldlocal` | `True` | `8` | mpc_beam / latent_rollout / oracle_goal_raw_euclidean_distance = 8/8, h 0.0 |
| `E0_base_cf_edit` | `True` | `8` | mpc_beam / latent_rollout / oracle_goal_raw_euclidean_distance = 8/8, h 0.0 |
| `E1_hierarchy_l4_l8_l16` | `True` | `9` | mpc_beam / latent_rollout / oracle_goal_raw_euclidean_distance = 8/8, h 0.0 |
| `E2_waypoint_h8` | `True` | `8` | waypoint_beam / latent_rollout / oracle_waypoint_raw_euclidean_distance = 8/8, h 0.0 |
| `E3_waypoint_h16` | `True` | `8` | waypoint_beam / latent_rollout / oracle_waypoint_raw_euclidean_distance = 8/8, h 0.0 |
| `E4_waypoint_h4_h8_h16` | `True` | `8` | waypoint_beam / latent_rollout / oracle_waypoint_raw_euclidean_distance = 8/8, h 0.0 |
| `E5_waypoint_h16_hierarchy` | `True` | `25` | waypoint_beam / latent_rollout / oracle_waypoint_raw_euclidean_distance = 8/8, h 0.0 |
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
| `V1_asym_hindsight` | `True` | `8` | mpc_beam / latent_rollout / oracle_goal_projected_euclidean_distance = 8/8, h 0.0 |
| `V2_iql_quasi` | `True` | `8` | mpc_beam / latent_rollout / predicted_goal_projected_euclidean_distance = 0/8, h 55.375 |
| `V3_waypoint_asym_hindsight` | `True` | `31` | waypoint_beam / latent_rollout / oracle_waypoint_raw_euclidean_distance = 8/8, h 0.0 |
| `I0_integrated_waypoint_asym` | `True` | `31` | waypoint_beam / latent_rollout / oracle_waypoint_raw_euclidean_distance = 8/8, h 0.0 |
| `I1_integrated_waypoint_iql` | `True` | `30` | waypoint_hierarchical_beam / symbolic_reencode / oracle_waypoint_raw_euclidean_distance = 8/8, h 0.0 |
| `I2_integrated_best_delta_if_gate_passes_grid` | `True` | `27` | waypoint_beam / symbolic_reencode / oracle_waypoint_raw_euclidean_distance = 5/8, h 0.625 |
| `I2_integrated_best_delta_if_gate_passes_single` | `True` | `30` | waypoint_beam / symbolic_reencode / oracle_waypoint_raw_euclidean_distance = 0/8, h 24.125 |
