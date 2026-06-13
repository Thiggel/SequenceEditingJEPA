# Grid 5 Backlog

Clean Grid-5-only backlog and running experiment snapshot.

## Running

| Item | Jobs | Status | Read First |
| --- | --- | --- | --- |
| Grid 5B 10M stabilizer/capacity screen | Original `3724634_[0-11]`; rerun `3724689_[0-5]` | Complete. Original tasks `0-5` hit Slurm `NODE_FAIL` on `a2143`; rerun `0-5` completed cleanly. All 12 final runs wrote standard, MPC-CEM, and symbolic re-encode diagnostics. Best symbolic oracle read is `canonical_ema_vicreg_k4`, h8 mean remaining Hamming `41.00`, solve `0/4`. | Per-run `diagnostics/diagnostics.json`, `diagnostics_mpc_cem/mpc_cem_summary.json`, `diagnostics_symbolic_reencode/summary.json` |
| Grid 5C planner matrix | `3724691_[0-5]`, `3724698_[9-11]`, `3724700_[6]`, `3724701_[7]`, `3724702_[8]` | All original full-matrix tasks timed out. Stderrs contain only Slurm time-limit messages; no Python traceback. The eval now writes incremental `planner_records.jsonl` and `planner_summary.json` after each mode. | Per-run `diagnostics_planner_matrix/planner_summary.json` and `planner_records.jsonl` |
| Grid 5C small streaming probe | `3728790` | Complete, exit `0:0`, runtime `01:03:08` on a40 node `a0124`. One-board h8 result on `grid5b_10m_canonical_ema_vicreg_k4`: best read is MCTS + `symbolic_reencode` + oracle `latent_goal`, remaining Hamming `37` from start `55`, solve `0/1`; beam oracle symbolic gives `39`; all latent-rollout modes stay `53-55`; learned `goal_energy` symbolic stays `49-54`. | `$PUZZLE_JEPA_WORK_ROOT/runs/grid5b_10m_canonical_ema_vicreg_k4/diagnostics_planner_matrix_probe_20260613/` |
| Grid 5 geometry probe | local analysis | Complete. `scripts/analysis/grid5_geometry_probe.py` on `grid5b_10m_canonical_ema_vicreg_k4`: one-cell terminal corruptions are very close to the true solved latent, learned `goal_energy` true-terminal top1 `0/16`, latent/Hamming nearest-neighbor Spearman `0.133`, and best wrong action displacement beats gold cosine in `84.4%` of sampled states. | `$PUZZLE_JEPA_WORK_ROOT/analysis/grid5_geometry_probe_canonical_ema_vicreg_k4_20260613/summary.{json,md}` |
| Grid 5 oversight checks | `3724789`-`3724798` | `3724789`, `3724790`, and `3724791` completed cleanly; `3724792`-`3724798` are pending by `BeginTime`, so partition broadening would not help. | Logs `logs/grid5_watch_<jobid>.out/.err` and last messages |

## Immediate Analysis Tasks

1. Check whether any Grid 5B/5C tasks failed because of node/quota/runtime
   rather than code.
2. Treat the Grid5C probe as a negative gate for compact single-state planning:
   oracle symbolic re-encode improves proximity but does not solve even on one
   board; latent rollout and learned energy are weaker.
3. Do not submit a broad Grid5C rerun unless a new objective/representation
   first improves exact symbolic-board ranking.
4. Save qualitative examples for the best and worst reads.
5. Oversight jobs should update these Grid5 docs and the required report docs
   whenever they submit jobs, find failures, or change interpretation.

## Conditional Next Experiments

Only run these after Grid 5C is analyzed.

### If Oracle Symbolic Re-Encode Works

- Scale the winning planner to 32/64/128 boards.
- If latent rollout fails, submit longer rollout-fidelity jobs:
  - K `8/16/32`;
  - consistency to re-encoded horizon states;
  - EMA target on horizon states.
- If learned energy fails, submit scorer-only repairs:
  - action advantage;
  - listwise/pairwise action ranking;
  - multi-positive feasible-successor contrastive objective;
  - verifier auxiliary.

### If Only One Optimizer Works

- `beam`: structured pruning and larger beam/branch budgets.
- `mcts`: progressive widening, cached leaf scoring, default rollout policy.
- `nn_cem`: gradient/CEM hybrid and VQ action embeddings.

### If Oracle Symbolic Re-Encode Fails

- Audit implementation against LeWorldModel-style assumptions before adding
  losses:
  - normalization;
  - EMA target;
  - stop-gradient;
  - action manifold;
  - recurrent training/inference match.
- Run geometry probes:
  - true terminal vs corrupted terminal distance;
  - latent nearest-neighbor examples;
  - constraint-violation correlation;
  - action displacement consistency.
- Use tokenized/local representation as the positive control.

## Deferred

- Hierarchical JEPA: wait until low-level exact symbolic-board ranking works.
- Maze/ARC transfer: wait until Sudoku has a non-oracle learned scorer or a
  clearly working oracle-geometry mechanism.
- Large factorial sweeps: avoid until a small diagnostic passes.
