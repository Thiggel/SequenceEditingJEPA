# H1 Recipe Oversight Handoff

Last updated: 2026-06-30 22:07 CEST

## Purpose

The H1 recipe sweep is meant to recover and explain the original
`grid_goal_followup_H1_hierarchy_dense_l4_l16` latent-rollout signal without
mixing many moving parts at once.

Reference signal:

- `mpc_beam`
- `transition_mode=latent_rollout`
- `score_mode=oracle_goal_changed_cell_raw_euclidean_distance`
- beam width `16`, depth `16`
- `6/10` solved, remaining Hamming `0.5`

The new sweep should answer:

- whether current code can reproduce the H1 anchor
- whether action grounding is the bottleneck
- whether affected-token/local-context dynamics weighting helps
- whether temporal straightening, progress rank, action rank, terminal
  corruption, or VICReg help or hurt this geometry
- whether hierarchy contributes beyond dense rollout

## Current Jobs

- Main train array: `3799696`
- Main eval array: `3799697`
- Replacement concat train: `3799777_4` node-failed on A100 node `a0631`
- Retry concat train: `3800228_4`, A100-80GB, excluding `a0631`, batch `4`,
  grad accumulation `2`
- Retry concat eval: `3800229_4`, `afterok:3800228`
- After-eval oversight: `3800130`

Original task `3799696_4` failed from a bf16 dtype mismatch in
`old_local_concat`. This was fixed in commit `69d5c78`. First replacement task
`3799777_4` node-failed on A100 node `a0631` without evidence of OOM, so
`3800228_4`/`3800229_4` are the active retry jobs for that variant.

## First-Wave Variants

| Group | Variants | Interpretation |
|---|---|---|
| Anchor | `anchor_h1` | Current-code H1-compatible reproduction |
| Action | `action_token`, `action_local_feature`, `action_old_local_value`, `action_old_local_concat` | Action grounding ablation |
| Dynamics weighting | `dynamics_affected`, `dynamics_affected_context` | Local transition-loss weighting ablation |
| Auxiliary losses | `no_temporal`, `no_progress`, `no_action_rank`, `no_terminal_corrupt`, `no_vicreg`, `minimal_aux` | Geometry-shaping loss ablation |
| Hierarchy | `hier_none`, `hier_l4`, `hier_l16`, `hier_l4_l16_l32` | Hierarchy contribution |

Primary readout is oracle local latent-rollout planning:

- `planner=mpc_beam`
- `transition_mode=latent_rollout`
- score in
  `oracle_goal_changed_cell_raw_euclidean_distance`,
  `oracle_goal_affected_context_raw_euclidean_distance`

Predicted-goal rows are still important, but if oracle local planning fails,
the bottleneck is not the goal predictor alone.

## Oversight Policy

The oversight job runs after the eval arrays end. It should:

1. Summarize rows under
   `$PUZZLE_JEPA_WORK_ROOT/runs/grid_goal_h1_recipe`.
2. Record best overall and best oracle-local rows.
3. Resubmit missing evals for variants with checkpoints but incomplete planner
   matrices.
4. Choose the best action conditioning from the action variants using
   oracle-local latent-rollout rows.
5. Choose the best dynamics weighting from `anchor_h1`, `dynamics_affected`,
   and `dynamics_affected_context` using the same readout.
6. If enough variants have completed, submit Wave 2 with the chosen
   action/dynamics pair.

## Conditional Next Wave

Wave 2 uses `TRAIN_MAX_STEPS=20000` for faster morning signal and keeps the
same eval matrix. It is not a final solve-rate run; it is a targeted
interaction probe.

Wave 2 variants:

- chosen action + chosen dynamics anchor
- chosen action + `affected_context`
- chosen action without temporal straightening
- chosen action without progress rank
- chosen action without action rank
- chosen action with minimal auxiliary losses
- chosen action without hierarchy
- chosen action with `[4,16,32]` hierarchy

Interpretation:

- If old/local action conditioning wins, action grounding is the likely
  bottleneck.
- If `affected_context` wins, unchanged-cell dilution/local transition loss is
  likely important.
- If removing an auxiliary loss wins, that objective is distorting the
  planning geometry.
- If all rows remain far from solved even with oracle local scoring, prioritize
  exact legacy Grid3 reproduction and bridge experiments before more scaling.
