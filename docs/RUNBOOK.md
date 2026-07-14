# Runbook

Long-form handoff source of truth: `../sequence-editing-report`.

Last updated: 2026-07-14 12:07 CEST

## Active Sweep

Check the four arrays with:

```bash
squeue -j 3855790,3855791,3855792,3855793
```

Do not rerun the launcher while these arrays are active. The entry point is
`scripts/experiments/submit_controlled_objects_vicreg_hwm.sh`; it is a dry run
unless `SUBMIT=1`.

Task manifest:
`$HPCVAULT/sequence-editing/runs/controlled_objects/manifests/controlled_valid_hwm_vicreg_v1_steps20000_tasks.tsv`.
Job manifest:
`$HPCVAULT/sequence-editing/runs/controlled_objects/manifests/controlled_valid_hwm_vicreg_v1_steps20000_array_jobs.tsv`.
Output root:
`$HPCVAULT/sequence-editing/runs/controlled_objects/controlled_valid_hwm_vicreg_v1_steps20000/`.

Each stage writes `config.json`, `metrics.jsonl`, `metrics.json`, and
`checkpoint.pt`. Final evaluation writes `probe_eval_v4.json`. Aggregate after
all 48 evaluations finish:

```bash
python scripts/analysis/analyze_controlled_objects_vicreg_hwm.py \
  "$HPCVAULT/sequence-editing/runs/controlled_objects/manifests/controlled_valid_hwm_vicreg_v1_steps20000_tasks.tsv" \
  "$HPCVAULT/sequence-editing/runs/controlled_objects/controlled_valid_hwm_vicreg_v1_steps20000" \
  --output "$HPCVAULT/sequence-editing/runs/controlled_objects/controlled_valid_hwm_vicreg_v1_steps20000_summary.json"
```

The controlled jobs use `$HPCVAULT/sequence-editing/.venv`; launch scripts
clear inherited `$WORK` venv/cache paths. Keep the active sweep's checkpoints
and small runtime caches until evaluation completes. Historical checkpoints
and caches have already been removed.
