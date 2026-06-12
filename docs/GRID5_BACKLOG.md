# Grid 5 Backlog

Clean Grid-5-only backlog and running experiment snapshot.

## Running

| Item | Jobs | Status | Read First |
| --- | --- | --- | --- |
| Grid 5B 10M stabilizer/capacity screen | Original `3724634_[0-11]`; rerun `3724689_[0-5]` | Complete. Original tasks `0-5` hit Slurm `NODE_FAIL` on `a2143`; rerun `0-5` completed cleanly. All 12 final runs wrote standard, MPC-CEM, and symbolic re-encode diagnostics. Best symbolic oracle read is `canonical_ema_vicreg_k4`, h8 mean remaining Hamming `41.00`, solve `0/4`. | Per-run `diagnostics/diagnostics.json`, `diagnostics_mpc_cem/mpc_cem_summary.json`, `diagnostics_symbolic_reencode/summary.json` |
| Grid 5C planner matrix | `3724691_[0-5]`, `3724698_[9-11]`, `3724700_[6]`, `3724701_[7]`, `3724702_[8]` | All 12 planner eval tasks are running as of 2026-06-12 22:52 CEST. Stderrs are empty; no `planner_summary.json` files have been written yet. `3724698_[9-11]` retains the old 8h limit on `a2143` and has timeout risk. | Per-run `diagnostics_planner_matrix/planner_summary.json` and `planner_records.jsonl` |
| Grid 5 oversight checks | `3724789`-`3724798` | Scheduled every 6h from 2026-06-12 22:50 CEST through 2026-06-15 04:50 CEST on `a100mig`. Uses the local `cs` alias with medium reasoning. | Logs `logs/grid5_watch_<jobid>.out/.err` and last messages |

## Immediate Analysis Tasks

1. Check whether any Grid 5B/5C tasks failed because of node/quota/runtime
   rather than code.
2. For each completed Grid 5C checkpoint, tabulate:
   - best solve rate and mean remaining Hamming by planner;
   - symbolic re-encode vs latent rollout gap;
   - oracle `latent_goal` vs learned `goal_energy` gap;
   - root goal-value rate;
   - runtime by planner.
3. If Grid 5C times out before writing summaries, resubmit the smallest
   streaming/partial diagnostic for one representative checkpoint before any
   broad rerun.
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
