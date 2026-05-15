# Results

Last updated: 2026-05-15

## LANO Mask, 200k Steps

LANO is currently the strongest signal that the implementation works as a
controlled denoising/editing system.

| Model | Exact Reconstruction | Token Accuracy | Grammar Validity | Parse-Depth Match | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Sequence-Edit JEPA | `62.79%` | `94.65%` | `56.30%` | `81.30%` | Also reports rollout MSE. |
| Denoising LM | `62.74%` | `95.50%` | `55.86%` | `83.78%` | Slightly higher token/parse metrics. |
| Causal LM | `0.00%` generation exact | `19.97%` generation token accuracy | `45.12%` generation grammar validity | not reported | Perplexity `3.45`; generation setup is not identical to denoising reconstruction. |

JEPA rollout metrics:

```text
rollout_mse_k2 = 0.0373
rollout_mse_k4 = 0.0758
```

## iGSM Full-Denoise Eval, `checkpoint-200000`

The latest trusted iGSM eval is:

```text
/home/atuin/c107fa/c107fa12/sequence-editing/posthoc/igsm_ood/latest_full_denoise_metrics_3605200.summary.json
```

It uses fixed causal generation, full denoising for denoising-style models, and
`128` examples per split/op setting.

| Model | ID | OOD ops | op20 | op21 | op22 | op23 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Causal LM | `88.3%` | `86.7%` | `25.8%` | `27.3%` | `15.6%` | `15.6%` |
| Sequence-Edit JEPA full denoise | `29.7%` | `28.1%` | `6.3%` | `12.5%` | `7.0%` | `6.3%` |
| Denoising LM full denoise | `0.0%` | `0.0%` | `0.0%` | `0.0%` | `0.0%` | `0.0%` |

Main read: JEPA clears the planned gate versus the plain denoising baseline, but
only weakly. The causal baseline remains much stronger on ID and medium OOD, and
then drops sharply at fixed long op counts.

Later reruns `3605202` and `3605203` completed after the sampler/reporting fix.
They preserve the causal and JEPA values above, but the denoising LM no longer
collapses to all-zero answer accuracy:

| Model | ID | OOD ops | op20 | op21 | op22 | op23 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Denoising LM, corrected sampler path | `22.7%` | `21.9%` | `5.5%` | `5.5%` | `4.7%` | `4.7%` |

Interpretation: suppressing `[MASK]` at sampling/eval matters. The old
denoising LM is still below JEPA and far below causal, but it was not purely
incapable once the sampler stopped committing masks.

## iGSM Qualitative Samples at `checkpoint-120000`

### Causal LM

The causal model can solve simple ID examples at `checkpoint-120000`:

```text
problem: How many Haemal Spines does Eel have?

gold:
<igsm_solution> ... Eel's Haemal Spines as Y; so Y = 13 * K = 13 * 17 = 14.
<igsm_answer> 14<eos>

causal:
<igsm_solution> ... Eel's Haemal Spines as h; so h = 13 * v = 13 * 17 = 14.
<igsm_answer> 14<eos>

answer_match: True
```

### Sequence-Edit JEPA

The JEPA model currently produces plausible trace shape but wrong arithmetic on
the same sample:

```text
n=16 start:
<igsm_solution><mask><mask>...

after n=8:
<igsm_solution> Define ... so ... = 13 * ... = 13 * 11 = ...

final:
<igsm_solution> Define Flounder's Postophyses as E; so E = 11.
Define Eel's Haemal Spines as I; so I = 13 * E = 13 * 11 = 15.
<igsm_answer> 11<eos>

expected_answer: 14
predicted_answer: 11
answer_match: False
```

### Denoising LM, Previous `checkpoint-120000` Sample

The denoising LM currently tends to keep masks under full denoising from a fully
masked solution:

```text
n=16 start:
<igsm_solution><mask><mask>...

after n=8:
still mostly masks

final:
<igsm_solution><mask><mask>...

predicted_answer: <missing>
answer_match: False
```

## LANO Replacement

LANO replacement job `3605204` completed, but replacement correction is not good
enough to move to iGSM replacement yet.

| Metric | Value |
| --- | ---: |
| Exact reconstruction | `37.3%` |
| Token accuracy | `90.2%` |
| Grammar validity | `23.5%` |
| Edit F1 | `0.214` |
| Edit precision | `0.806` |
| Edit recall | `0.125` |

The replacement model is conservative: when it edits, it is often right, but it
misses most required edits and produces poor grammar validity.

## LANO Mask Ablations

LANO mask ablations `3605205_[0-3]` completed. The denoising LM is strongest
among the short mask ablations, but the comparison is confounded because
denoising/action-conditioned runs used `8k` steps while policy-only/action-free
runs used `20k`.

## x0 Objective Status

- `3606650_[0-1]`: iGSM x0 JEPA and denoising LM are running, but will likely
  need the chained resume job to reach `200k`.
- `3607175_[0-1]`: chained iGSM x0 resume is running; latest complete
  checkpoints are currently `checkpoint-140000`, with logs progressing beyond
  `150000`.
- `3607176`: dependent latest-checkpoint iGSM x0 full-denoise eval is pending
  and will use JEPA `predictor_decoder` inference by default.
- `3612686_[0-5]`: x0 inference-mode/commit-k ablation is pending on
  `3607175`; it compares JEPA `policy_head` vs `predictor_decoder` and
  `commit_k=schedule/1/2/5/10`.
- `3606651`: LANO replacement x0 completed: exact `10.7%`, token accuracy
  `52.6%`, grammar validity `28.0%`, edit F1 `0.875`, precision `0.859`,
  recall `0.892`.
- `3606652_[0-3]`: matched-step LANO x0 mask ablations completed. Denoising LM
  is best on exact/grammar/token (`25.4%` exact, `63.3%` grammar, `72.9%`
  token); action-free JEPA is second on exact (`17.1%`); action-conditioned JEPA
  remains weaker on exact (`13.1%`).

## Stepwise JEPA Status

Submitted on 2026-05-15:

| Job ID | Purpose |
| --- | --- |
| `3612683_[0-3]` | Train stepwise mask JEPA at `T=20/50/64` and stepwise replacement JEPA at `T=64`. |
| `3612684_[0-3]` | Resume the same runs from latest checkpoints after the first allocation. |
| `3612685_[0-4]` | Evaluate stepwise mask JEPA with `commit_k=schedule/1/2/5/10`. |

The stepwise mask model differs from the x0 model by targeting the next
partially denoised state. It labels the deterministic reveal chunk as
`REPLACE`, labels the remaining editable positions as `KEEP`, and trains the
JEPA latent predictor toward the next-state target encoder. This is intended to
measure whether the action policy can learn a useful denoising order.
