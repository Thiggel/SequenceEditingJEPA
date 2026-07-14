# Runbook

Long-form handoff source of truth: `../sequence-editing-report`.

Last updated: 2026-07-14 08:14 CEST

## Active Controlled Grid

Check the nine arrays with:

```bash
squeue -j 3851603,3851604,3851605,3851606,3851607,3851608,3851609,3851610,3851611
```

Task and array-job manifests are under
`$PUZZLE_JEPA_WORK_ROOT/runs/controlled_objects/manifests/` with prefix
`controlled_mlp_hwm_v1_steps20000`. Outputs use the same prefix under
`runs/controlled_objects/`.

Do not rerun the launcher while these arrays are active. The reproducible entry
point is `scripts/experiments/submit_controlled_objects_mlp_grid.sh`; it is a
dry run unless `SUBMIT=1`. It creates 288 base-axis rows, five correlated train
arrays, and four correlated probe arrays.

Each run writes `config.json`, `metrics.jsonl`, `metrics.json`, and
`checkpoint.pt`; probes write `probe_eval_v3.json`. Aggregate after completion:

```bash
python scripts/analysis/analyze_controlled_objects_mlp_grid.py \
  "$PUZZLE_JEPA_WORK_ROOT/runs/controlled_objects/manifests/controlled_mlp_hwm_v1_steps20000_tasks.tsv" \
  "$PUZZLE_JEPA_WORK_ROOT/runs/controlled_objects/controlled_mlp_hwm_v1_steps20000" \
  --output "$PUZZLE_JEPA_WORK_ROOT/runs/controlled_objects/controlled_mlp_hwm_v1_steps20000_summary.json"
```

The Gated DeltaNet uses `fla-core`'s chunk kernel in FP32 inside an
autocast-disabled boundary; removing that boundary reproduces a Triton
FP32/BF16 dot-product compilation failure on RTX Pro 6000.

Targeted repair arrays `3854953`-`3854964` and `3855009`-`3855014` cover
filesystem-, node-, cancellation-, and stuck-dependency gaps. Do not resubmit
their task IDs again. Hydra metadata is redirected to
`${OUTPUT_ROOT}/${RUN_NAME}/hydra` to avoid filling the home filesystem.
