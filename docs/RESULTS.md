# Results

Last updated: 2026-05-19

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

- `3606650_[0-1]`: initial iGSM x0 JEPA and denoising LM allocation timed out
  before 200k.
- `3607175_[0-1]`: chained iGSM x0 resume completed successfully on
  2026-05-16, producing `checkpoint-200000` for both x0 runs.
- `3607176`: latest-checkpoint iGSM x0 full-denoise eval completed
  successfully. Summary:
  `/home/atuin/c107fa/c107fa12/sequence-editing/posthoc/igsm_ood/x0_latest_full_denoise_metrics_3607176.summary.json`.
- `3612686_[0-5]`: x0 inference-mode/commit-k ablation is now dependency
  cleared and completed; it compares JEPA `policy_head` vs `predictor_decoder`
  and `commit_k=schedule/1/2/5/10`.
- `3606651`: LANO replacement x0 completed: exact `10.7%`, token accuracy
  `52.6%`, grammar validity `28.0%`, edit F1 `0.875`, precision `0.859`,
  recall `0.892`.
- `3606652_[0-3]`: matched-step LANO x0 mask ablations completed. Denoising LM
  is best on exact/grammar/token (`25.4%` exact, `63.3%` grammar, `72.9%`
  token); action-free JEPA is second on exact (`17.1%`); action-conditioned JEPA
  remains weaker on exact (`13.1%`).

### iGSM x0 Full-Denoise Eval, `checkpoint-200000`

Job `3607176` used the corrected JEPA `predictor_decoder` inference path and
scheduled high-confidence commits.

| Model | ID | OOD ops | op20 | op21 | op22 | op23 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Causal LM | `88.3%` | `86.7%` | `25.8%` | `27.3%` | `15.6%` | `15.6%` |
| x0 JEPA predictor-decoder | `32.8%` | `28.9%` | `3.9%` | `5.5%` | `3.9%` | `7.8%` |
| x0 Denoising LM | `42.2%` | `29.7%` | `9.4%` | `8.6%` | `6.2%` | `9.4%` |

Main read:

- The x0 objective substantially improves the denoising LM over the old
  corrected-sampler denoising LM (`22.7%` ID to `42.2%` ID).
- The x0 denoising LM is now the strongest denoising-style iGSM baseline.
- JEPA with predictor-decoder inference is not yet helping over the x0
  denoising LM; it improves ID slightly over old JEPA (`29.7%` to `32.8%`) but
  is weaker on most long fixed-op splits.
- Causal generation remains far stronger on ID and medium OOD, though all
  methods still struggle at fixed op counts `20..23`.

Qualitatively, x0 JEPA fills masks but still produces repetitive, malformed
proof text and often misses the answer span on long examples. The x0 denoising
LM is cleaner and more accurate than JEPA, but exact sequence match remains
near zero.

### iGSM x0 Commit/Inference Ablation

Job `3612686_[0-5]` completed on 2026-05-16. The most informative settings:

| Setting | Model | ID | OOD ops | op20 | op21 | op22 | op23 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `policy_head`, schedule | JEPA | `35.9%` | `25.0%` | `1.6%` | `2.3%` | `1.6%` | `0.8%` |
| `predictor_decoder`, schedule | JEPA | `32.8%` | `28.9%` | `3.9%` | `5.5%` | `3.9%` | `7.8%` |
| `predictor_decoder`, `k=1` | JEPA | `32.8%` | `23.4%` | `7.0%` | `6.3%` | `3.9%` | `5.5%` |
| schedule | Denoising LM | `42.2%` | `29.7%` | `9.4%` | `8.6%` | `6.2%` | `9.4%` |
| `k=1` | Denoising LM | `43.0%` | `28.1%` | `7.8%` | `16.4%` | `10.2%` | `10.2%` |

Main read: the old `policy_head` path uses less active capacity and improves
JEPA ID, but it makes long-op OOD worse. Using the latent predictor at inference
is still the best x0 JEPA OOD setting, but it does not beat the denoising LM.
For the denoising LM, one-token commits are worth rerunning at larger sample
size because the fixed long-op splits improved despite similar aggregate OOD.

## Report Artifacts

Generated artifacts for the report live in `docs/assets/`:

- `igsm_train_loss.svg` / `.png`: HF trainer loss curves for the major iGSM runs.
- `igsm_full_denoise_answer_accuracy.svg` / `.png`: periodic full-denoise answer
  accuracy curves.
- `igsm_full_denoise_token_accuracy.svg` / `.png`: periodic full-denoise token accuracy
  curves.
- `igsm_generation_samples.md`: latest periodic qualitative sample excerpts.
- `posthoc_summary.md`: compact tables extracted from posthoc summary JSONs.

## Stepwise JEPA Status

Submitted on 2026-05-15:

| Job ID | Purpose |
| --- | --- |
| `3612683_[0-3]` | Train stepwise mask JEPA at `T=20/50/64` and stepwise replacement JEPA at `T=64`. |
| `3612684_[0-3]` | Resume the same runs from latest checkpoints after the first allocation. |
| `3612685_[0-4]` | Evaluate stepwise mask JEPA with `commit_k=schedule/1/2/5/10`. |

Status on 2026-05-19 09:44 CEST:

- `3612683_0` (`T=20` mask) and `3612683_1` (`T=50` mask) timed out cleanly
  after saving through `checkpoint-120000`.
- `3612683_2` (`T=64` mask) and `3612683_3` (`T=64` replacement) timed out
  cleanly after saving through `checkpoint-120000`.
- `3612684_[0-3]` completed successfully and produced `checkpoint-200000` for
  T20/T50/T64 mask and T64 replacement stepwise JEPA.
- `3612685_[0-4]` final stepwise commit eval is dependency-cleared but still
  pending on priority.
- `3614356_[0-7]` completed the partial T20/T50 `commit_k=schedule/1/2/5`
  sweep under both `predictor_decoder` and `policy_head`.
- `3614357_[0-5]` completed the first oracle-goal latent MPC diagnostic.
- New submitted follow-ups:
  - `3615623_[0-2]`: proposal coverage, factorized oracle, and DLM-proposer
    diagnostics completed.
  - `3615624_[0-15]`: larger oracle-MPC grid with latent vs materialized
    re-encode rollouts completed.
  - `3615625_[0-1]`: lightweight oracle-goal value-head training on T20/T50 is
    complete.
  - `3615643_[0-1]`: rollout-repair continuation initialized from latest
    T20/T50 stepwise checkpoints with a 20k-step diagnostic budget completed.

Early 50k periodic metrics:

| Run | Full-Denoise ID Answer | Full-Denoise OOD op20/op23 Answer | Full-Denoise ID Token Acc. | One-Step Edit F1 | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| Step mask `T=20` | `0.0%` | `0.0% / 0.0%` | `70.2%` | `0.979` | One-step action learning is strong, but generation is not working yet. |
| Step mask `T=50` | `3.1%` | `0.0% / 0.0%` | `67.6%` | `0.929` | One-step metrics are healthier than full-denoise generation. |

The one-step iGSM answer metrics in these periodic logs are not a reliable
generation score because the stepwise target is a partially denoised next state,
not a full clean solution. The full-denoise metrics are the relevant generation
signal, and they are still weak at 50k.

The stepwise mask model differs from the x0 model by targeting the next
partially denoised state. It labels the deterministic reveal chunk as
`REPLACE`, labels the remaining editable positions as `KEEP`, and trains the
JEPA latent predictor toward the next-state target encoder. This is intended to
measure whether the action policy can learn a useful denoising order.

The current hypothesis is that stepwise collapse may be an inference/planning
problem rather than a local action-learning problem. `3614356` tests whether
single-token commits avoid incompatible simultaneous edits. `3614357` tests an
upper-bound planner: candidate actions are scored by latent energy to the clean
target encoding, so any improvement there means the dynamics carry useful
planning information even if the deployable sampler does not yet know how to
score states.

### Partial Stepwise Commit Sweep

The partial sweep used the latest available checkpoints at job start, so some
settings saw `checkpoint-80000`, `100000`, or `120000`. It is diagnostic, not a
final curve. Main read: single-token or small-k commits help a little, and the
smaller policy head is sometimes better than the predictor-decoder path, but
neither rescues long fixed-op generation.

| Run / mode | ID | OOD ops | Best fixed-op note |
| --- | ---: | ---: | --- |
| T20 `policy_head`, `k=1` | `9.4%` | `10.9%` | op20 `1.6%`, op21-23 `0.0%` |
| T20 `policy_head`, schedule | `6.2%` | `10.9%` | fixed op20-23 all `0.0%` |
| T20 `predictor_decoder`, `k=1` | `6.2%` | `6.2%` | fixed op20-23 all `0.0%` |
| T50 `predictor_decoder`, `k=5` | `1.6%` | `3.1%` | op20 `4.7%`, op21 `3.1%`, op23 `1.6%` |

### Oracle-Goal Latent MPC

Without oracle-token injection, oracle MPC mostly remains near `0%` answer
accuracy, although it improves token accuracy. With the correct token inserted
into the candidate set, it can solve ID and medium OOD examples (`T20` ID
`100%`, OOD ops `75%` on the small diagnostic batches). This strongly suggests
that the proposal/token head is a bottleneck. Fixed op20/op23 rows are not
directly comparable because the diagnostic capped generation at `160` actions,
leaving many masks (`57%` remaining for op20, `66%` for op23).

Next diagnostics therefore focus on proposal coverage, factorized
position-vs-token oracles, larger token candidate sets, re-encode rollouts, and
a learned value/energy head.

### Stepwise Proposal Diagnostics

Job `3615623_[0-2]` completed on 2026-05-18.

Proposal coverage is better than greedy generation suggested. For T50, the
oracle position is almost always in the top-16 positions (`100%` on ID/OOD ops,
`99.7%` on op20, `98.1%` on op23), and the oracle token is usually in the
top-20 (`97.8%` ID, `97.2%` OOD ops, about `93%` on op20/op23). T20 is weaker
on long fixed-op splits, especially pair recall at top-16-by-top-20 (`76.8%`
op20, `65.5%` op23).

The factorized oracle result is sharper:

| Setting | T20 ID | T20 OOD ops | T50 ID | T50 OOD ops |
| --- | ---: | ---: | ---: | ---: |
| model position + model token | `9.4%` | `17.2%` | `0.0%` | `3.1%` |
| model position + oracle token | `93.8%` | `71.9%` | `93.8%` | `67.2%` |
| oracle position + model token | `9.4%` | `15.6%` | `0.0%` | `3.1%` |
| oracle position + oracle token | `93.8%` | `67.2%` | `93.8%` | `67.2%` |

Main read: the model is often choosing serviceable positions; exact token
prediction is the bigger deployable bottleneck. This matches the DLM-proposer
diagnostic: replacing JEPA tokens with the x0 denoising LM token proposer raises
ID/OOD ops to `39.1%/26.6%` for T20 and `37.5%/29.7%` for T50, still below the
oracle-token upper bound.

The long fixed-op rows in the factorized diagnostics are capped by
`max_steps=220`, leaving many masks (`43%` op20, `51%` op23), so their answer
accuracy is not a clean generation score.

The larger MPC grid `3615624_[0-15]` completed. The best T20 re-encode settings
reach `50.0%` answer accuracy on the small ID/OOD diagnostic splits (`h2 m16
k5` for ID, `h2 m8 k5` for OOD ops) with token accuracy around `91.7%/93.7%`.
Pure latent rollout is weaker. This still does not solve long fixed-op cases:
op20/op23 answer accuracy stays `0.0%`, with many masks left under the current
`max_steps=220` cap. T50 also reaches high token accuracy under re-encode MPC,
but answer accuracy remains `0.0%` on these small diagnostic splits.

The value-head diagnostic `3615625_[0-1]` also completed. The small heads fit
oracle-goal latent energy well enough as a regression target (`loss=0.0079` for
T20, `0.0183` for T50 at step `5000`), but this is not yet an end-to-end
sampler result. The deployable test is the dependent LeWM-like MPC job after the
new objective/capacity ablations finish.

### Rollout Repair

Job `3615643_[0-1]` completed the 20k-step rollout-repair diagnostic. It is not
a full replacement for the 200k runs, but it is a useful signal:

| Run | ID full-denoise answer | op20 | op23 | ID token acc. | Edit F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| T20 rollout repair | `17.2%` | `0.8%` | `2.3%` | `73.0%` | `98.2%` |
| T50 rollout repair | `3.1%` | `0.0%` | `0.0%` | `71.7%` | `96.2%` |

Rollout repair helps T20 ID relative to the running base periodic evals, but it
does not solve fixed long-op OOD.

### Objective/Capacity Ablations

Job `3621296_[0-7]` was submitted on 2026-05-18. As of 2026-05-19 15:54 CEST,
all eight tasks are running. The array throttle was raised from `%2` to `%8`,
so the remaining tasks are no longer blocked by `JobArrayTaskLimit`. Because
the first two tasks are too slow to reach 200k in one allocation, resume array
`3624400_[0-7]` is queued with
`afterany:3621296`, and MPC eval `3621300_[0-1]` now waits for `3624400`. The
array tests higher
token/action CE, 12-layer contextual decoder heads with and without detach,
12-layer policy heads, soft model-predicted action conditioning, LeWM-like
no-decoder-CE objectives with and without a value head, and a large deep-decoder
denoising LM baseline. These are intended to separate three failure modes:
insufficient token CE pressure, insufficient active policy/decoder capacity, and
the gold-action train/inference mismatch in the latent predictor.

| Job | Run | Exact change | Readout question |
| --- | --- | --- | --- |
| `3621296_0` | `T20_high_ce` | `lambda_action_op=2.0`, `lambda_action_token=4.0`, `lambda_tok=2.0`, `lambda_sig=0.1`. | Was CE pressure too weak? |
| `3621296_1` | `T20_deep_decoder` | Add `decoder_layers=12`, set `lambda_tok=1.0`, `lambda_sig=0.1`. | Does contextual token decoding fix the token bottleneck? |
| `3621296_2` | `T20_deep_decoder_detach` | Same as `_1`, plus `detach_token_head=true`. | Is decoder CE helping readout or harming latent dynamics? |
| `3621296_3` | `T20_deep_policy` | Set `policy_layers=12`. | Was the active policy path under-capacity? |
| `3621296_4` | `T20_soft_action_dyn` | Set `predictor_action_source=predicted_soft`, `lambda_action_token=2.0`, `lambda_sig=0.1`. | Does differentiable self-conditioned dynamics reduce train/inference mismatch? |
| `3621296_5` | `T20_lewm_no_dec` | Set `lambda_tok=0.0`, `lambda_sig=0.2`, evaluate with `policy_head`. | Can a LeWM-like no-decoder-CE objective learn useful dynamics? |
| `3621296_6` | `T20_lewm_value` | Same as `_5`, plus `lambda_val=1.0`, `lambda_value_token=0.25`. | Can a learned latent value score replace oracle energy? |
| `3621296_7` | `x0_denoising_lm_deep_decoder` | DLM with `target_mode=x0`, `num_steps=64`, `encoder_layers=18`, `decoder_layers=12`. | Does a high-capacity DLM beat JEPA variants? |

Dependent posthoc job `3621300_[0-1]` will evaluate the LeWM-like runs with
latent MPC after `3621296`: oracle-goal scoring for the no-value run and learned
value-head scoring for the value run.

Early live signal from the first two array tasks:

| Job | Run | Latest live point | 50k full-denoise ID | 50k full-denoise op20/op21/op22/op23 | 50k ID token acc. | 50k edit F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `3621296_0` | `T20_high_ce` | checkpoint `120000`, latest log around step `120000`, loss `2.28` | `9.4%` at 100k | `0.0% / 0.0% / 0.0% / 3.1%` at 100k | `71.3%` | `0.942` |
| `3621296_1` | `T20_deep_decoder` | checkpoint `80000`, latest log around step `90000`, loss `0.64` | `3.1%` at 50k | `3.1% / 0.0% / 3.1% / 3.1%` at 50k | `70.0%` | `0.973` |

Main read so far: both runs are technically healthy, but neither has produced a
scientific win yet. High-CE now has a nonzero ID bump at 100k, but long fixed-op
accuracy remains near zero; deep-decoder had nonzero low-sample 50k metrics but
has not reached its next periodic eval.
