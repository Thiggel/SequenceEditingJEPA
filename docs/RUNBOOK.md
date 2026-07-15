# Runbook

Long-form handoff source of truth: `../sequence-editing-report`.

Last updated: 2026-07-15 10:47 CEST

Active Wave 15 trainer array: `3858542` (`0-35%12`). Dependent correlated
probe array: `3858543`. Check with:

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
python scripts/analysis/analyze_controlled_objects_joint.py \
  "$PUZZLE_JEPA_WORK_ROOT/runs/controlled_objects/controlled_joint_hwm_objectives_v1_steps20000"
```

Do not launch planning or broader axes automatically. First apply the
three-seed representation gate documented in Wave 15.
