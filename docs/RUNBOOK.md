# Runbook

Long-form handoff source of truth: `../sequence-editing-report`.

Last updated: 2026-07-23

Inspect the incomplete Wave 17 jobs with:

```bash
squeue -j 3860420,3860421,3862936,3862939,3862940
sacct -j 3860420,3860421,3862936,3862939,3862940 \
  --format=JobID,JobName,State,ExitCode,Elapsed
```

Wave 16 and 18 aggregates are complete. Wave 17 task/job manifests are under
`$HPCVAULT/sequence-editing/runs/controlled_objects/manifests/` with prefixes
`controlled_dense_trajectories_v1`.

The output root has 39/42 checkpoints and bounded-horizon probes. The attempted
B64 repair `3862936/3862940` selected `vicreg_t300_b64_*`; the actually missing
cells are `ema_vicreg_t300_b64_*`. Six original probe tasks in `3860421` are
permanently `DependencyNeverSatisfied`. Do not aggregate or delete this output
root until a corrected three-cell trainer/probe repair completes.

After a complete gate, use `scripts/analysis/analyze_controlled_objects_gates.py`
with mode `objective`, `dense`, or `planner`, its task manifest, output root,
and `--output <root>/summary.json`. Treat any listed missing run as an
incomplete result; do not average partial arrays into the report.
