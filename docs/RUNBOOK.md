# Runbook

Long-form handoff source of truth: `../sequence-editing-report`.

Last updated: 2026-07-16

Inspect the active jobs with:

```bash
squeue -j 3860420,3860421,3862936,3862939,3862940
sacct -j 3860420,3860421,3862936,3862939,3862940 \
  --format=JobID,JobName,State,ExitCode,Elapsed
```

Wave 16 and 18 aggregates are complete. Active Wave 17 task/job manifests are under
`$HPCVAULT/sequence-editing/runs/controlled_objects/manifests/` with prefixes
`controlled_dense_trajectories_v1`.

After a complete gate, use `scripts/analysis/analyze_controlled_objects_gates.py`
with mode `objective`, `dense`, or `planner`, its task manifest, output root,
and `--output <root>/summary.json`. Treat any listed missing run as an
incomplete result; do not average partial arrays into the report.
