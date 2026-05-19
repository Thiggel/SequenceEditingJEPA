# Experiment Plan

Last updated: 2026-05-19

## Current Priority

The immediate goal is now to separate three questions cleanly:

1. Does the x0 masked-diffusion baseline work when evaluated with the corrected
   sampler?
2. Does JEPA help when its latent predictor is actually used at inference time?
3. Does stepwise JEPA training improve action selection compared with direct x0
   JEPA?
4. Is the current stepwise JEPA failure caused by greedy/chunked inference, or
   are the latent dynamics themselves not useful for planning?

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

The previous full-denoise eval remains the trusted baseline snapshot, but the
new x0 eval changes the denoising conclusion:

- causal LM is the strongest iGSM baseline;
- old JEPA beat the old plain denoising LM under full denoising, but this was
  partly because the old denoising objective/sampler were wrong;
- x0 denoising LM is now the strongest denoising-style iGSM baseline;
- x0 JEPA with predictor-decoder inference does not yet beat x0 denoising LM;
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
| `3606650_[0-1]` | iGSM x0 JEPA and denoising LM | none | Timed out before 200k; continued by `3607175`. |
| `3606651` | LANO replacement x0 JEPA | none | Completed; much higher edit recall than pre-fix replacement, but exact/grammar still weak. |
| `3606652_[0-3]` | LANO x0 matched-step mask ablations | none | Completed; all four ablations used `20k` steps. |
| `3607175_[0-1]` | iGSM x0 resume continuation | `afterany:3606650` | Completed; produced `checkpoint-200000` for x0 JEPA and x0 denoising LM. |
| `3607176` | Latest-checkpoint iGSM x0 full-denoise eval | `afterok:3607175` | Completed; x0 denoising LM beats x0 JEPA, causal remains strongest. |
| `3612683_[0-3]` | Stepwise iGSM JEPA training | none | All four timed out cleanly; latest saved checkpoint is `120000`. |
| `3612684_[0-3]` | Stepwise iGSM JEPA resume | `afterany:3612683` | Completed; produced `checkpoint-200000` for T20/T50/T64 mask and T64 replacement. |
| `3612685_[0-4]` | Stepwise mask JEPA commit-k eval | `afterok:3612684` | Dependency cleared; pending on priority for `commit_k=schedule/1/2/5/10`. |
| `3612686_[0-5]` | x0 JEPA/DLM inference-mode and commit-k eval | `afterok:3607175` | Completed. `policy_head` helps JEPA ID but hurts long OOD; `commit_k=1` is most interesting for DLM long-op splits. |
| `3614356_[0-7]` | Partial stepwise latest-checkpoint inference sweep | none | Completed. Small-k and `policy_head` help slightly but do not rescue long fixed-op generation. |
| `3614357_[0-5]` | Oracle-goal latent MPC diagnostic | none | Completed. Oracle-token injection solves small ID/OOD diagnostics, pointing to proposal/token bottlenecks. |
| `3615623_[0-2]` | Stepwise proposal/factorized diagnostics | none | Completed. Position coverage is decent; token choice is the larger bottleneck. |
| `3615624_[0-15]` | Larger oracle-MPC grid | none | Completed; re-encode MPC helps small T20 ID/OOD diagnostics but does not solve long fixed-op rows. |
| `3615625_[0-1]` | Oracle-goal value-head training | none | Completed; value heads fit the oracle latent-energy target, deployable MPC eval is still pending later jobs. |
| `3615643_[0-1]` | Stepwise rollout-repair continuation | none | Completed. T20 ID improved, fixed long-op OOD remains weak. |
| `3621296_[0-7]` | Objective/capacity ablations | none | All eight tasks are running as of 2026-05-19 15:54 CEST after raising the array throttle from `%2` to `%8`. Covers high CE, deep decoder, deep policy, soft-action dynamics, LeWM-like no-decoder CE/value, and deep DLM baseline. |
| `3624400_[0-7]` | Objective/capacity ablation resume | `afterany:3621296` | Submitted because the first allocation will not reach 200k for the slower variants. Resumes each run from its latest complete checkpoint. |
| `3621300_[0-1]` | LeWM-like MPC posthoc eval | `afterany:3624400` | Pending dependency. Uses oracle-goal scoring for no-value and learned value-head scoring for value run. |
| `3615630_[0-1]` | Stepwise rollout-repair continuation | none | Cancelled/replaced after confirming the original 80k-step budget was too slow for one allocation. |
| `3615626_[0-1]` | Stepwise rollout-repair continuation | none | Failed immediately due to a Hydra override path; replaced by `3615630`, then by `3615643`. |

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

The x0 rerun changes this read. The fairer denoising LM is now stronger than
JEPA:

| Model | ID | OOD ops | op20 | op21 | op22 | op23 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Causal LM | `88.3%` | `86.7%` | `25.8%` | `27.3%` | `15.6%` | `15.6%` |
| x0 JEPA predictor-decoder | `32.8%` | `28.9%` | `3.9%` | `5.5%` | `3.9%` | `7.8%` |
| x0 Denoising LM | `42.2%` | `29.7%` | `9.4%` | `8.6%` | `6.2%` | `9.4%` |

This means the next JEPA claim must come from the stepwise action-selection and
planning experiments, not from the current x0 JEPA result.

## Denoising Debug Ablations

The first six objective/sampler fixes are implemented for the x0 configs. The
next submitted diagnostic is `3612686`, which compares:

```text
JEPA policy_head + scheduled commit
JEPA predictor_decoder + scheduled commit
JEPA/DLM predictor_decoder eval with commit_k = 1, 2, 5, 10
```

`3607176` and `3612686` are complete. The ablation did not rescue x0 JEPA:
`policy_head` improved JEPA ID (`35.9%`) but reduced long-op OOD, and
`predictor_decoder + schedule` remains the best JEPA OOD setting (`28.9%`).
For the denoising LM, `commit_k=1` is worth following up because it improved
several fixed long-op splits. If JEPA or denoising still fail from fully masked
traces after the larger follow-up:

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

Early 50k periodic logs show that stepwise one-step edit learning is healthy
(`edit_f1` about `0.98` for `T=20` and `0.93` for `T=50`), but full-denoise
answer accuracy is still near zero. Do not over-interpret one-step iGSM answer
accuracy for stepwise runs, because the target is a partially denoised next
state rather than a full clean solution.

Two posthoc diagnostics were added on 2026-05-16 instead of waiting passively
for 200k:

- `3614356`: latest partial-checkpoint sweep of `commit_k=schedule/1/2/5` and
  `jepa_inference_mode=predictor_decoder/policy_head` on T20/T50. This tests
  whether stepwise collapse is mainly chunked inference or predictor-decoder
  corruption.
- `3614357`: oracle-goal latent MPC. It scores candidate one-action rollouts by
  distance to the EMA target encoding of the clean sequence. This is not a
  deployable sampler; it is a diagnostic for whether the JEPA dynamics contain
  useful planning signal.

The first diagnostic set is now clearer. The partial commit sweep does not
rescue stepwise JEPA. Oracle-MPC shows latent-energy selection can help when the
correct token is in the candidate set. Proposal coverage shows top-k candidate
sets often contain the correct action, especially for T50. Factorized oracles
show that using the model position with the oracle token gives high ID/OOD
accuracy, while using the model token with oracle position remains weak. The
deployable bottleneck is therefore token choice and action scoring, not just
position proposal.

The next seven diagnostics are implemented/submitted:

| # | Experiment | Purpose | Job |
| ---: | --- | --- | --- |
| 1 | Proposal coverage | Does the policy put the oracle position in top-M and the oracle token in top-K? | `3615623_0` |
| 2 | Factorized oracle | Separate position errors from token errors: model/model, oracle-position/model-token, model-position/oracle-token, oracle/oracle. | `3615623_1` |
| 3 | Larger candidate MPC | Test whether more positions/tokens and longer horizons expose useful latent planning signal. | `3615624` |
| 4 | Re-encode vs latent rollout | Compare pure latent rollout against materializing actions and re-encoding after each simulated step. | `3615624` |
| 5 | Learned value/energy head | Train a small value head to estimate oracle-goal latent energy. | `3615625` |
| 6 | Rollout repair | Continue stepwise training with multi-step latent rollout loss, initialized from latest T20/T50. | `3615643` |
| 7 | Stronger token proposer | Use the x0 denoising LM as a token proposer under the stepwise JEPA position/action policy. | `3615623_2` |

2026-05-18 objective/capacity ablations were submitted as `3621296_[0-7]` to
test the current token bottleneck hypotheses directly:

| Array | Config suffix | Exact change |
| ---: | --- | --- |
| `0` | `T20_high_ce` | Keep T20 stepwise JEPA; set `lambda_action_op=2.0`, `lambda_action_token=4.0`, `lambda_tok=2.0`, `lambda_sig=0.1`. |
| `1` | `T20_deep_decoder` | Keep T20 stepwise JEPA; add `decoder_layers=12`, set `lambda_tok=1.0`, `lambda_sig=0.1`. |
| `2` | `T20_deep_decoder_detach` | Same as array `1`, plus `detach_token_head=true`, so decoder CE trains only the decoder readout. |
| `3` | `T20_deep_policy` | Keep T20 stepwise JEPA; set `policy_layers=12` to match predictor depth, and `lambda_sig=0.1`. |
| `4` | `T20_soft_action_dyn` | Keep T20 stepwise JEPA; set `predictor_action_source=predicted_soft`, `predictor_action_temperature=1.0`, `lambda_action_token=2.0`, `lambda_sig=0.1`. |
| `5` | `T20_lewm_no_dec` | LeWM-like JEPA: set `lambda_tok=0.0`, `lambda_sig=0.2`, keep action-token CE, and evaluate via `policy_head`. |
| `6` | `T20_lewm_value` | Same as array `5`, plus joint oracle-latent value training with `lambda_val=1.0`, `lambda_value_token=0.25`, `detach_value_head=true`. |
| `7` | `x0_denoising_lm_deep_decoder` | Denoising baseline, not JEPA: `target_mode=x0`, `num_steps=64`, `encoder_layers=18`, `decoder_layers=12`. |

Dependent posthoc job `3621300_[0-1]` evaluates the LeWM-like runs with latent
MPC:

| Array | Source run | Scoring |
| ---: | --- | --- |
| `0` | `T20_lewm_no_dec` | Oracle-goal latent energy, horizon `2`, top `8` positions, top `5` tokens, latent rollout, max `220` actions. |
| `1` | `T20_lewm_value` | Learned value-head score with the same MPC grid. |

Still useful after these finish:

```text
action-conditioned JEPA vs action-free JEPA
policy/token only, lambda_dyn = 0
no SIGReg, lambda_sig = 0
full insert/delete MDP after replacement is healthy
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
