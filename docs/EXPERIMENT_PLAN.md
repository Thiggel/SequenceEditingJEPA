# Experiment Plan

Last updated: 2026-05-15

## Current Priority

The immediate goal is now to separate three questions cleanly:

1. Does the x0 masked-diffusion baseline work when evaluated with the corrected
   sampler?
2. Does JEPA help when its latent predictor is actually used at inference time?
3. Does stepwise JEPA training improve action selection compared with direct x0
   JEPA?

The x0 masked-diffusion fix has been implemented for mask/replacement
corruptors, the denoising LM, and action-conditioned JEPA:

- corrupted examples now target clean `x0`;
- token CE is restricted to corrupted editable positions;
- `[MASK]` is suppressed as an editable token prediction;
- JEPA action conditioning is preserved, but actions now describe explicit
  REPLACE operations for the currently corrupted positions;
- mask/replacement training samples uniform corruption counts and uses larger
  denoising step counts in the new configs.

Additional 2026-05-15 fixes:

- JEPA full-denoise inference now defaults to `predictor_decoder`, i.e. policy
  actions condition the latent predictor and the decoder reads tokens from the
  predicted next latent state.
- The old JEPA policy-token path remains available as
  `jepa_inference_mode=policy_head` for ablation only.
- Full-denoise eval now supports fixed commit counts with `commit_k=1/2/5/10`
  in addition to the scheduled high-confidence commit sampler.
- Stepwise mask/replacement corruptors can use `target_mode: step`, where the
  next target is one visible deterministic reveal chunk rather than clean `x0`.

The previous full-denoise eval remains the trusted baseline snapshot:

- causal LM is the strongest iGSM baseline;
- JEPA beats the plain denoising LM under full denoising, but remains far from
  causal;
- the denoising LM still mostly leaves masks;
- LANO replacement exposes an edit-recall/objective issue.

Detailed denoising literature notes and proposed changes are in
`docs/DENOISING_LITERATURE.md`.

## Submitted Jobs

| Job ID | Purpose | Dependency | Notes |
| --- | --- | --- | --- |
| `3605092_[0-2]` | iGSM resume training | none | Completed; trained from `checkpoint-120000` to `checkpoint-200000`. |
| `3605200` | Latest-checkpoint iGSM full-denoise eval | none | Completed; trusted latest full-denoise eval. |
| `3605201_[0-2]` | Resume iGSM training with fixed eval/sample logging | `afterany:3605092` | Completed, effectively no-op because runs were already at `checkpoint-200000`. |
| `3605202` | Latest-checkpoint iGSM full-denoise eval after fixed resume | `afterany:3605201` | Completed; same causal/JEPA as `3605200`, denoising LM becomes nonzero under corrected sampler path. |
| `3605203` | All-checkpoints iGSM full-denoise eval after fixed resume | `afterany:3605201` | Completed; all-checkpoint curve artifact is available. |
| `3605204` | LANO replacement JEPA | none | Completed; poor edit recall, do not move to iGSM replacement yet. |
| `3605205_[0-3]` | LANO mask ablations | none | Completed; comparison confounded by unequal step counts. |
| `3606650_[0-1]` | iGSM x0 JEPA and denoising LM | none | Running; healthy, but full `200k` is unlikely to fit one 24h allocation. |
| `3606651` | LANO replacement x0 JEPA | none | Completed; much higher edit recall than pre-fix replacement, but exact/grammar still weak. |
| `3606652_[0-3]` | LANO x0 matched-step mask ablations | none | Completed; all four ablations used `20k` steps. |
| `3607175_[0-1]` | iGSM x0 resume continuation | `afterany:3606650` | Running; latest complete checkpoints at `140000`, logs have progressed beyond `150000`. |
| `3607176` | Latest-checkpoint iGSM x0 full-denoise eval | `afterok:3607175` | Pending; will use the corrected JEPA predictor-decoder default. |
| `3612683_[0-3]` | Stepwise iGSM JEPA training | none | Submitted: mask `T=20/50/64` plus replacement `T=64`. |
| `3612684_[0-3]` | Stepwise iGSM JEPA resume | `afterany:3612683` | Submitted to continue the stepwise runs if the first allocation times out. |
| `3612685_[0-4]` | Stepwise mask JEPA commit-k eval | `afterok:3612684` | Submitted for `commit_k=schedule/1/2/5/10`. |
| `3612686_[0-5]` | x0 JEPA/DLM inference-mode and commit-k eval | `afterok:3607175` | Submitted for JEPA policy-head vs predictor-decoder and `commit_k=schedule/1/2/5/10`. |

## iGSM Baseline Table

The first table to trust should compare:

```text
causal LM generation
denoising LM full denoise from masked solution + answer
Sequence-Edit JEPA full denoise from masked solution + answer
```

For each model report:

```text
ID answer accuracy
OOD op=20 answer accuracy
OOD op=21 answer accuracy
OOD op=22 answer accuracy
OOD op=23 answer accuracy
trace token accuracy
sample generations/traces
```

The main gate was passed only weakly: JEPA improves over the plain denoising
model under full denoising, but both results point to sampler/objective
debugging before adding planning.

## Denoising Debug Ablations

The first six objective/sampler fixes are implemented for the x0 configs. The
next submitted diagnostic is `3612686`, which compares:

```text
JEPA policy_head + scheduled commit
JEPA predictor_decoder + scheduled commit
JEPA/DLM predictor_decoder eval with commit_k = 1, 2, 5, 10
```

If JEPA or denoising still fail from fully masked traces after `3607176` and
`3612686`:

1. Compare `num_steps=32/64/128` and high-confidence commit vs random remasking.
2. Compare uniform exact-count masking with log-linear, cosine, and linear
   timestep sampling.
3. Re-run easier starts (`n=12`, `n=8`) as diagnostics, not as the main fix.
4. Add classifier-free guidance by sometimes dropping or masking the prompt.
5. Try block/semi-autoregressive diffusion for long iGSM traces.

## JEPA Ablations

The next JEPA-specific ablation is already submitted in `3612683`:

```text
stepwise mask JEPA, num_steps=20
stepwise mask JEPA, num_steps=50
stepwise mask JEPA, num_steps=64
stepwise replacement JEPA, num_steps=64
```

The stepwise mask runs should answer whether explicit KEEP/REPLACE supervision
helps the policy learn an edit order. Replacement is an intermediate test for
visible-token correction; it is not yet the full variable-length insert/delete
MDP.

Run these after the stepwise and x0 commit-k results:

```text
action-conditioned JEPA
action-free JEPA
policy/token only, lambda_dyn = 0
no SIGReg, lambda_sig = 0
rollout K=2
```

Primary question:

```text
Does action-conditioned latent prediction improve OOD iGSM and LANO rollout
metrics compared with action-free denoising?
```

## LANO Next Steps

LANO is the clean diagnostic task. The next LANO runs are:

```text
LANO replacement correction
LANO action-conditioned vs action-free
LANO policy-only vs JEPA dynamics
```

Only move to iGSM replacement after LANO replacement shows that the model can
detect wrong visible tokens, not only fill masks.

Immediate LANO follow-up: rerun the mask ablations with matched step counts and
then fix replacement recall by training/evaluating on corrupted-position CE/F1
rather than letting visible-token copying dominate.
