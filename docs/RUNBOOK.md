# Runbook

Last updated: 2026-05-19

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
3606650_[0-1]  iGSM x0 JEPA and denoising LM; timed out before 200k
3606651        LANO replacement x0 JEPA; completed
3606652_[0-3]  LANO x0 matched-step mask ablations; completed
3607175_[0-1]  iGSM x0 resume continuation; completed to checkpoint-200000
3607176        iGSM x0 latest full-denoise eval; completed
3612686_[0-5]  x0 inference-mode/commit-k eval; completed
3612683_[0-3]  stepwise first allocation; timed out cleanly after checkpoint-120000
3612684_[0-3]  stepwise resume; completed to checkpoint-200000
3612685_[0-4]  stepwise commit-k eval; dependency cleared, pending priority
3614356_[0-7]  partial stepwise latest-checkpoint commit/inference sweep; completed
3614357_[0-5]  partial stepwise oracle-goal latent MPC; completed
3615623_[0-2]  proposal/factorized/DLM-proposer diagnostics; completed
3615624_[0-15] larger oracle-MPC grid; completed
3615625_[0-1]  oracle-goal value-head training; completed
3615643_[0-1]  rollout-repair continuation; completed
3615630_[0-1]  rollout-repair continuation; cancelled/replaced after budget reduction
3615626_[0-1]  rollout-repair continuation; failed immediately, replaced
```

Latest x0 summary:

```text
$SEQ_EDIT_JEPA_WORK_ROOT/posthoc/igsm_ood/x0_latest_full_denoise_metrics_3607176.summary.json
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

This was already submitted as `3612686_[0-5]` and completed.

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

Evaluate the currently available partial stepwise checkpoints without waiting
for the resume/eval dependency chain:

```bash
sbatch scripts/slurm/run_posthoc_igsm_stepwise_partial_commit_ablation.slurm
```

This was submitted as `3614356_[0-7]`. It evaluates latest complete T20/T50
checkpoints only and sweeps:

```text
commit_k = schedule, 1, 2, 5
jepa_inference_mode = predictor_decoder, policy_head
```

Run the oracle-goal latent MPC diagnostic:

```bash
sbatch scripts/slurm/run_posthoc_igsm_stepwise_oracle_mpc.slurm
```

This was submitted as `3614357_[0-5]`. It evaluates latest complete T20/T50
checkpoints with:

```text
horizon = 1, 2, 4
candidates_per_step = 4
tokens_per_position = 1
max_steps = 160
score = negative latent MSE to EMA target encoder(clean, n=0)
oracle-token candidate injection = off/on
```

The oracle-token variants are upper bounds: they test the latent action scorer
when the correct clean token is guaranteed to be in the candidate set.

Run the new proposal/factorized diagnostics:

```bash
sbatch scripts/slurm/run_posthoc_igsm_stepwise_diagnostics.slurm
```

This was submitted as `3615623_[0-2]`:

```text
0: proposal coverage, top-M positions and top-K tokens
1: factorized model/oracle position-token ablations
2: DLM token proposer under the stepwise JEPA position/action policy
```

Run the larger MPC grid:

```bash
sbatch scripts/slurm/run_posthoc_igsm_stepwise_mpc_grid.slurm
```

This was submitted as `3615624_[0-15]`. It sweeps:

```text
horizon = 1, 2, 4
candidates_per_step = 4, 8, 16
tokens_per_position = 1, 5
rollout_mode = latent, reencode
```

Train the oracle-goal value head:

```bash
sbatch scripts/slurm/run_igsm_stepwise_value_head.slurm
```

Continue stepwise runs with rollout repair:

```bash
sbatch scripts/slurm/run_igsm_stepwise_rollout_repair.slurm
```

The value-head job trains only `model.value_head`; the rollout-repair job
initializes new T20/T50 runs from the latest complete stepwise checkpoints and
adds multi-step latent rollout loss.

Run the 2026-05-18 objective/capacity ablations:

```bash
sbatch scripts/slurm/run_igsm_objective_ablation_200k.slurm
```

Submitted as `3621296_[0-7]` on 2026-05-18. As of 2026-05-19 15:54 CEST,
all eight tasks are running after raising the live array throttle from `%2` to
`%8`. Resume array `3624400_[0-7]` is queued with
`afterany:3621296` because the slower variants will not hit 200k in one
allocation. Dependent MPC job `3621300_[0-1]` now waits on `3624400`, and its
array throttle was raised from `%1` to `%2`.

Array mapping:

```text
0  T20_high_ce: action op CE 2x, action-token CE 4x, decoder CE 2.0, SIGReg 0.1
1  T20_deep_decoder: 12-layer decoder, decoder CE 1.0, SIGReg 0.1
2  T20_deep_decoder_detach: same as 1, but decoder CE is detached from latents
3  T20_deep_policy: 12-layer policy head, matching the predictor depth
4  T20_soft_action_dyn: predictor trained on soft model-predicted actions
5  T20_lewm_no_dec: no decoder CE, stronger SIGReg, policy-head eval
6  T20_lewm_value: same as 5, plus oracle-latent value loss
7  x0_denoising_lm_deep_decoder: 18-layer encoder + 12-layer decoder DLM
```

Run the dependent LeWM-like MPC eval:

```bash
sbatch --dependency=afterany:3621296 scripts/slurm/run_posthoc_igsm_objective_mpc.slurm
```

Submitted as `3621300_[0-1]`. Task `0` uses oracle-goal latent scoring for the
no-decoder-CE run; task `1` uses the learned value head from the value run. Both
use horizon `2`, top `8` positions, top `5` tokens, latent rollout, and
`max_steps=220`.

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
python -m py_compile seq_edit_jepa/eval/posthoc_igsm_ood.py seq_edit_jepa/eval/oracle_mpc.py
python -m pytest tests/test_causal_lm_baseline.py tests/test_tasks.py tests/test_modeling.py -q
python -m pytest tests/test_corruptors.py tests/test_full_denoise.py -q
```
