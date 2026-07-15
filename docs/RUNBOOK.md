# Runbook

Long-form handoff source of truth: `../sequence-editing-report`.

Last updated: 2026-07-15 14:41 CEST

Wave 15 trainer array `3858542` and correlated probe array `3858543` both
completed 36/36 tasks with exit `0:0`. Check the archived state with:

```bash
squeue -j 3858542,3858543
sacct -j 3858542,3858543 --format=JobID,State,ExitCode,Elapsed
```

Task and job manifests:
`$HPCVAULT/sequence-editing/runs/controlled_objects/manifests/controlled_joint_hwm_objectives_v1_steps20000_{tasks,jobs}.tsv`.
Output root:
`$HPCVAULT/sequence-editing/runs/controlled_objects/controlled_joint_hwm_objectives_v1_steps20000/`.
Each successful trainer writes `checkpoint.pt` and `metrics.json`; its matching
probe writes `probe_eval_v5.json`.

After all probes finish, run:

```bash
source scripts/env.sh
ROOT="$PUZZLE_JEPA_WORK_ROOT/runs/controlled_objects"
python scripts/analysis/analyze_controlled_objects_joint.py \
  "$ROOT/manifests/controlled_joint_hwm_objectives_v1_steps20000_tasks.tsv" \
  "$ROOT/controlled_joint_hwm_objectives_v1_steps20000" \
  --output "$ROOT/controlled_joint_hwm_objectives_v1_steps20000/summary.json"
```

The representation gate failed. Do not launch planning or broader axes
automatically.
