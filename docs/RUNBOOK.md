# Runbook

Last updated: 2026-05-15

## Environment

Always use the shared environment and cache/proxy settings:

```bash
source scripts/env.sh
```

This activates `$WORK/.venv` and routes Hugging Face, Torch, and dataset caches
under `$WORK`.

## Check Jobs

```bash
squeue -u "$USER" -o "%.18i %.9P %.30j %.8T %.10M %.6D %R" | rg "seqedit|JOBID"
```

## Current iGSM Runs

Run directories:

```text
$SEQ_EDIT_JEPA_WORK_ROOT/runs/igsm_official_med_mask_action_conditioned_jepa_200k
$SEQ_EDIT_JEPA_WORK_ROOT/runs/igsm_official_med_mask_denoising_lm_200k
$SEQ_EDIT_JEPA_WORK_ROOT/runs/igsm_official_med_causal_lm_200k
```

Latest known status: `3605092_[0-2]` completed the resume to
`checkpoint-200000`; `3605200` produced the original trusted full-denoise eval;
`3605202` and `3605203` later completed under the corrected sampler path.

New x0 masked-diffusion runs submitted after the objective fix:

```text
3606650_[0-1]  iGSM x0 JEPA and denoising LM; running, likely needs resume
3606651        LANO replacement x0 JEPA; completed
3606652_[0-3]  LANO x0 matched-step mask ablations; completed
3607175_[0-1]  iGSM x0 resume continuation; running
3607176        iGSM x0 latest full-denoise eval; afterok:3607175
3612686_[0-5]  x0 inference-mode/commit-k eval; afterok:3607175
3612683_[0-3]  stepwise iGSM JEPA training; pending
3612684_[0-3]  stepwise iGSM JEPA resume; afterany:3612683
3612685_[0-4]  stepwise commit-k eval; afterok:3612684
```

Resume all three from their latest complete checkpoint:

```bash
sbatch scripts/slurm/run_igsm_official_200k_resume.slurm
```

Submit a chained resume after the current array:

```bash
sbatch --dependency=afterany:3605092 scripts/slurm/run_igsm_official_200k_resume.slurm
```

Submit the x0 objective iGSM rerun:

```bash
sbatch scripts/slurm/run_igsm_official_x0_200k.slurm
```

Resume the x0 objective iGSM rerun from latest complete checkpoints:

```bash
sbatch --dependency=afterany:3606650 scripts/slurm/run_igsm_official_x0_200k_resume.slurm
```

## iGSM Evaluation

Evaluate latest checkpoints only:

```bash
sbatch scripts/slurm/run_posthoc_igsm_latest_full_denoise.slurm
```

Evaluate all checkpoints:

```bash
sbatch scripts/slurm/run_posthoc_igsm_full_denoise.slurm
```

Evaluate latest x0 checkpoints:

```bash
sbatch scripts/slurm/run_posthoc_igsm_x0_latest_full_denoise.slurm
```

Evaluate x0 JEPA inference mode and commit count:

```bash
sbatch --dependency=afterok:3607175 scripts/slurm/run_posthoc_igsm_x0_commit_ablation.slurm
```

This array uses:

```text
0: JEPA policy_head, commit_k=schedule
1: JEPA predictor_decoder, commit_k=schedule
2: predictor_decoder, commit_k=1
3: predictor_decoder, commit_k=2
4: predictor_decoder, commit_k=5
5: predictor_decoder, commit_k=10
```

Both scripts use:

```text
seq_len = 1024
batches = 16
batch_size = 8
splits = eval, eval_op_20, eval_op_21, eval_op_22, eval_op_23
```

Both print qualitative samples. Denoising-style models print intermediate
states from the full-denoise sampler.

## Stepwise JEPA

Submit the stepwise iGSM JEPA ablation:

```bash
sbatch scripts/slurm/run_igsm_stepwise_jepa_200k.slurm
```

Resume it after the first allocation:

```bash
sbatch --dependency=afterany:3612683 scripts/slurm/run_igsm_stepwise_jepa_200k_resume.slurm
```

Evaluate stepwise mask JEPA with different commit counts:

```bash
sbatch --dependency=afterok:3612684 scripts/slurm/run_posthoc_igsm_stepwise_commit_ablation.slurm
```

The training array is:

```text
0: mask, target_mode=step, num_steps=20
1: mask, target_mode=step, num_steps=50
2: mask, target_mode=step, num_steps=64
3: replacement, target_mode=step, num_steps=64
```

## Manual Sample Printing

Causal:

```bash
python -m seq_edit_jepa.eval.debug_samples \
  --run-dir "$SEQ_EDIT_JEPA_WORK_ROOT/runs/igsm_official_med_causal_lm_200k" \
  --seq-len 1024 \
  --examples 1 \
  --device cuda
```

JEPA:

```bash
python -m seq_edit_jepa.eval.debug_samples \
  --run-dir "$SEQ_EDIT_JEPA_WORK_ROOT/runs/igsm_official_med_mask_action_conditioned_jepa_200k" \
  --seq-len 1024 \
  --examples 1 \
  --trace-steps 16 8 4 1 \
  --device cuda
```

## LANO

Replacement correction:

```bash
sbatch scripts/slurm/run_lano_replace.slurm
```

Mask ablations:

```bash
sbatch scripts/slurm/run_lano_mask_ablations.slurm
```

x0 replacement and matched-step x0 mask ablations:

```bash
sbatch scripts/slurm/run_lano_replace_x0.slurm
sbatch scripts/slurm/run_lano_mask_x0_ablations.slurm
```

## Validation

Fast local checks:

```bash
python -m py_compile seq_edit_jepa/eval/posthoc_igsm_ood.py
python -m pytest tests/test_causal_lm_baseline.py tests/test_tasks.py tests/test_modeling.py -q
python -m pytest tests/test_corruptors.py tests/test_full_denoise.py -q
```
