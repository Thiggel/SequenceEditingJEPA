# Runbook

Long-form handoff source of truth: `../sequence-editing-report`.

Last updated: 2026-07-15 18:04 CEST

Inspect the active jobs with:

```bash
squeue -j 3860384,3860385,3860420,3860421,3860422
sacct -j 3860384,3860385,3860420,3860421,3860422 \
  --format=JobID,JobName,State,ExitCode,Elapsed
```

Task/job manifests are under
`$HPCVAULT/sequence-editing/runs/controlled_objects/manifests/` with prefixes
`controlled_objective_weights_v1_steps20000`,
`controlled_dense_trajectories_v1`, and `controlled_planner_interfaces_v1`.

After a complete gate, use `scripts/analysis/analyze_controlled_objects_gates.py`
with mode `objective`, `dense`, or `planner`, its task manifest, output root,
and `--output <root>/summary.json`. Treat any listed missing run as an
incomplete result; do not average partial arrays into the report.
