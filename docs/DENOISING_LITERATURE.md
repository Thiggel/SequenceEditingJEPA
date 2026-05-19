# Denoising Diffusion Literature Notes

Last updated: 2026-05-19

## Implementation Status

Implemented on 2026-05-14:

- mask and replacement corruptors now build `x_t -> x0` examples;
- `target_mask` marks only corrupted editable positions for token CE;
- `target_n=0` makes JEPA's EMA target encoder represent the clean state;
- action targets are explicit REPLACE actions for currently corrupted positions,
  preserving JEPA action conditioning without hidden reveal-order supervision;
- denoising LM and JEPA token CE suppress `[MASK]` as a clean editable
  prediction;
- full-denoise sampling suppresses `[MASK]` before confidence scoring/argmax;
- new configs use uniform corruption counts and larger denoising step counts.

Submitted validation runs:

```text
3606650_[0-1]  iGSM x0 JEPA/denoising, num_steps=64; timed out before 200k
3606651        LANO replacement x0 JEPA; completed
3606652_[0-3]  LANO x0 matched-step mask ablations; completed
3607175_[0-1]  iGSM x0 resume continuation; completed to checkpoint-200000
3607176        iGSM x0 latest full-denoise eval; completed
3612686_[0-5]  x0 inference-mode/commit-k ablation; completed
3612683_[0-3]  stepwise iGSM JEPA mask/replacement ablation; timed out after 120k
3612684_[0-3]  dependent stepwise resume; completed to checkpoint-200000
3612685_[0-4]  dependent stepwise commit-k eval; pending priority
3614356_[0-7]  partial stepwise latest-checkpoint commit/inference sweep; completed
3614357_[0-5]  partial stepwise oracle-goal latent MPC diagnostic; completed
3615623_[0-2]  proposal/factorized/DLM-proposer diagnostics; completed
3615624_[0-15] larger oracle-MPC grid; completed
3615625_[0-1]  oracle-goal value-head training; completed
3615643_[0-1]  rollout-repair continuation; completed
3621296_[0-7]  objective/capacity ablations; all tasks running after throttle raised to %8
3624400_[0-7]  objective/capacity ablation resume; afterany:3621296
3621300_[0-1]  LeWM-like MPC posthoc eval; afterany:3624400, throttle raised to %2
```

Implemented on 2026-05-15:

```text
JEPA full denoise uses predictor_decoder by default.
The old policy_head path is retained as an explicit ablation.
Full denoise supports commit_k=schedule/1/2/5/10.
Mask/replacement corruptors support target_mode=step for stepwise JEPA.
```

## Executive Summary

The pre-fix denoising LM was closer to a one-step `x_n -> x_{n-1}` masked
reconstruction model than to LLaDA/MDLM/D3PM-style masked diffusion. The biggest
problem is not the cosine schedule by itself. The main mismatch is that the
model was trained to predict `prev_ids`, which often still contained `[MASK]`,
and the loss was computed over all non-padding tokens. This taught the model to
copy visible tokens and keep many masked tokens masked.

The stronger masked diffusion language models instead train a clean-token
predictor: given a partially masked sequence `x_t`, predict `x0` on masked
positions only. The sampler, not the network token head, decides how many
positions remain masked at the next denoising level.

The iGSM x0 result validates the first part of that diagnosis: the x0 denoising
LM improves from `22.7%` ID answer accuracy under the corrected old sampler to
`42.2%` ID answer accuracy at `checkpoint-200000`. It is now the strongest
denoising-style baseline. The x0 JEPA predictor-decoder result is weaker
(`32.8%` ID), so the current evidence does not show that the action-conditioned
latent predictor helps on iGSM generation.

The stepwise JEPA diagnostics sharpen the next question. Partial commit sweeps
remain weak, but oracle-token latent MPC can solve small ID/OOD diagnostics.
Proposal coverage is often good at top-k, especially for T50, and factorized
oracles show that model positions plus oracle tokens are strong while oracle
positions plus model tokens are weak. The active bottleneck is therefore exact
token choice/action scoring more than the cosine noising schedule alone.

## Pre-Fix Implementation

Relevant files:

- `seq_edit_jepa/data/corruptors/mask_corruptor.py`
- `seq_edit_jepa/data/corruptors/base.py`
- `seq_edit_jepa/models/denoising_lm.py`
- `seq_edit_jepa/models/seq_edit_jepa.py`
- `seq_edit_jepa/eval/full_denoise.py`

Pre-fix forward corruption:

1. Sample integer `n` uniformly from `1..num_steps`.
2. Compute `gamma(n)`, currently linear or `sin(pi n / (2T))^2`.
3. Pick one random permutation of editable positions.
4. Mask exactly `floor(gamma(n) * editable_count)` positions for `input_ids`.
5. Mask exactly `floor(gamma(n - 1) * editable_count)` positions for `prev_ids`.
6. The oracle action is `REPLACE` only on the hidden subset `mask_n & ~mask_prev`.

Pre-fix denoising LM:

- Bidirectional Transformer with timestep embedding.
- Predicts token logits for every position.
- Trains CE against `prev_ids` on all attention positions.
- Therefore some masked input positions are explicitly trained to output
  `[MASK]`.

Pre-fix full denoise sampler:

- Starts with all editable positions set to `[MASK]`.
- At each `n`, predicts logits for all positions.
- Commits top-confidence remaining positions according to the schedule.
- Does not ban `[MASK]` from the committed token distribution.

This explains the observed failure mode: the model can be correct, under its own
training objective, when it predicts `[MASK]` for many masked positions.

## Literature Snapshot

### D3PM

D3PM defines discrete diffusion as a Markov corruption process with structured
transition matrices. One important transition is the absorbing `[MASK]` process:
tokens either stay unchanged or transition to `[MASK]`, and `[MASK]` is
absorbing. D3PM also shows that absorbing-state diffusion connects diffusion,
BERT-style MLMs, generative masked LMs, and autoregressive models.

Key implementation lesson: use an `x0` parameterization or equivalent clean-data
prediction. For absorbing diffusion, the reverse step can be derived from the
clean-token prediction plus the known forward process. The model should not be
asked to infer an arbitrary hidden reveal order from identical mask tokens.

### MDLM and MD4

MDLM and MD4 simplify masked diffusion. They show that the masked diffusion
objective can be written as a weighted mixture or integral of masked-token CE
losses. MDLM uses a substitution-style parameterization for absorbing-state
diffusion and reports state-of-the-art diffusion perplexity on LM1B/OpenWebText
among diffusion models at the time. MD4 similarly frames the continuous-time
masked diffusion objective as weighted clean-token CE and supports generalized
masking schedules.

Key implementation lesson: train a masked-token clean predictor and normalize the
loss over masked positions. Schedule details matter for variance and sampling,
but they are secondary to the parameterization and target.

### SEDD

SEDD estimates discrete score ratios rather than clean tokens directly. It is a
more general discrete diffusion formulation and is competitive with GPT-2-scale
autoregressive models in perplexity and sampling quality. It supports absorbing
and uniform transition matrices and uses score-based reverse samplers.

Key implementation lesson: if we want a plain denoising baseline quickly, SEDD is
probably more engineering than needed. But SEDD is the right reference if we want
a general discrete diffusion baseline that can revise wrong visible tokens rather
than only unmask.

### LLaDA and SMDM

LLaDA scales masked diffusion language modeling to LLM scale. It samples a mask
ratio `t` in `[0, 1]`, masks tokens independently at that ratio, and trains a
Transformer to predict all masked tokens. In SFT, prompt tokens remain visible
and only response tokens are masked. At inference it starts from a fully masked
response, predicts all masks in parallel, and remasks a scheduled fraction of
tokens, often using low-confidence remasking.

SMDM is a direct predecessor: it studies scaling laws for masked diffusion on
text and uses unsupervised classifier-free guidance for conditional inference.

Key implementation lesson: for iGSM, the closest setup is LLaDA SFT-style
conditional generation: keep the problem prompt visible, mask solution/answer
tokens, and train clean-token CE only on masked solution/answer tokens.

### Recent SOTA Direction

The public frontier has moved beyond "train from scratch with random masking":

- Dream 7B reports stronger open diffusion LLM results using AR-model
  initialization and context-adaptive token-level noise rescheduling.
- LLaDA2.0 scales diffusion LMs to 16B/100B MoE by converting pretrained AR
  models with a staged block/full-sequence/block diffusion training scheme.
- DMax targets aggressive parallel decoding by training on-policy recovery from
  both masked inputs and erroneous predictions, then decoding in a soft
  self-refinement space.

For this repo, the practical takeaway is not to chase 100B-scale machinery. It
is to first implement the modern masked-diffusion objective, then optionally add
AR initialization, block diffusion, and on-policy self-correction.

## Main Mismatches In Our Baseline

1. `prev_ids` target instead of `x0` target.
   The target sequence can still contain `[MASK]`, so the model is rewarded for
   keeping masks.

2. Loss over all non-padding positions.
   Visible tokens dominate the CE and reward copying. SOTA masked diffusion
   losses focus on masked positions.

3. Hidden reveal-order supervision.
   `mask_n & ~mask_prev` depends on a random permutation hidden from `x_n`.
   Predicting which identical mask token should reveal at this step is largely
   unidentifiable.

4. `[MASK]` is allowed as an editable prediction.
   The sampler can commit `[MASK]` as if it were a clean token.

5. Few denoising steps.
   `num_steps=16` is coarse for long iGSM traces. LLaDA notes quality improves
   when sampling steps approach response length, and MDLM examples often use far
   more reverse steps for language modeling.

6. Schedule is not the first-order issue.
   The cosine cumulative mask ratio is defensible as a schedule. The objective
   and sampler did not match the absorbing-state clean-token
   formulation used by LLaDA/MDLM/D3PM.

## Recommended Changes

### Priority 0: Make The Denoising LM A Real Masked Diffusion Baseline

Implement a new corruption/training path for denoising:

- Sample a mask ratio `t`, preferably continuous uniform in `[0, 1]`, or sample
  a uniform mask count for low-variance exact-count masking.
- Build `input_ids = x_t` by masking editable positions.
- Set `labels = clean_ids` only for `input_ids == mask_token_id` and editable
  positions; use `-100` elsewhere.
- Normalize CE by the number of masked editable tokens.
- During training and sampling, set the `[MASK]` logit to `-inf` for editable
  clean-token prediction. Also consider banning `<pad>` and `<unk>` on editable
  positions.
- At inference, copy visible/non-editable tokens from the input and only fill
  currently masked editable positions.

Minimal sampler:

1. Start from fully masked editable positions.
2. For each step, predict clean tokens for all remaining masks.
3. Score by max non-mask probability.
4. Commit the highest-confidence positions needed to reach the next target mask
   count.
5. Leave the rest masked.
6. At the last step, force-fill all remaining masks with non-mask predictions.

This is close to LLaDA/MaskGIT low-confidence remasking, expressed as
high-confidence committing.

### Priority 1: Fix JEPA To Avoid Hidden Reveal Targets

The pre-fix JEPA action head had the same hidden reveal-order problem, and the
pre-2026-05-15 full-denoise evaluator also bypassed the JEPA predictor at
inference time. Both issues now have implementation hooks:

- x0 JEPA trains all currently corrupted positions as `REPLACE` and can be
  evaluated either with the old `policy_head` path or the intended
  `predictor_decoder` path;
- stepwise JEPA uses `target_mode: step`, where a deterministic visible chunk is
  revealed, those positions are labeled `REPLACE`, and unrevealed editable
  positions are labeled `KEEP`;
- fixed `commit_k` evaluation measures whether committing one token at a time is
  more stable than committing multiple high-confidence tokens simultaneously.

Options, from simplest to more ambitious:

- Treat all currently masked editable positions as replace candidates and train
  token CE to `x0` there.
- If the latent predictor needs a one-step action, sample an explicit reveal mask
  and provide that mask as part of the action input. The model can condition on
  the proposed edit set; it should not infer it.
- Train latent dynamics toward either the clean-state target encoding or a
  schedule-defined `x_s` target whose revealed positions are externally sampled.
- Keep `[MASK]` disallowed for editable clean-token prediction in the JEPA token
  head too.

The action-conditioned idea can still be tested, but the action should represent
a proposed edit set, not an oracle hidden random reveal.

Current implementation detail: the stepwise reveal set is deterministic
left-to-right among currently corrupted editable positions. This makes the
action target identifiable from `x_t` and position. A later variant can replace
this with a visible priority signal or with a learned proposal/action search, but
we should avoid a hidden random reveal order as the supervised label.

The first planning diagnostic is deliberately oracle-scored. Given current
state latent `h_t`, candidate action `a_t`, and predicted latent
`P(h_t, a_t, t)`, `seq_edit_jepa/eval/oracle_mpc.py` scores actions by negative
MSE to the EMA target encoding of the clean sequence. This is not deployable
because it uses `x0`, but it answers whether the latent predictor contains
useful planning signal. If oracle MPC improves while greedy/full-denoise does
not, the next target is a learned goal-conditioned energy or value scorer. If it
does not improve, the JEPA dynamics/objective need retraining changes before
more elaborate planning.

2026-05-18 implementation update: the JEPA objective now supports split
operation/action-token CE weights, optional Transformer decoder layers before
the LM head, soft predicted-action conditioning for the latent predictor, and a
joint oracle-latent value loss. The soft predicted-action path avoids a
Gumbel-softmax dependency: it feeds expected op/token embeddings into the
predictor, with the token embedding gated by the predicted `REPLACE`
probability. This is the simplest differentiable test of the train-inference
mismatch where the predictor previously saw only gold actions during training.

The submitted `3621296` ablations instantiate those ideas as follows:

- `3621296_0`: higher op CE, action-token CE, decoder CE, and SIGReg.
- `3621296_1`: 12-layer contextual decoder before the LM head.
- `3621296_2`: same contextual decoder, but detached from the latent state.
- `3621296_3`: 12-layer policy head so action selection has predictor-scale
  capacity.
- `3621296_4`: soft model-predicted actions feed the predictor during training,
  letting latent MSE gradients reach the policy through expected action
  embeddings.
- `3621296_5`: LeWM-like no decoder CE, with action-token CE retained as the
  discrete token proposer.
- `3621296_6`: LeWM-like no decoder CE plus an oracle-goal value head.
- `3621296_7`: a large x0 denoising-LM baseline with an 18-layer encoder and
  12-layer decoder.

The dependent `3621300` MPC eval compares oracle-goal latent scoring against the
learned value-head score on the two LeWM-like runs.

### Priority 2: Schedules And Step Counts

After the objective is fixed, compare:

- uniform `t` with independent Bernoulli masks;
- uniform exact mask count;
- log-linear MDLM-style schedule;
- cosine cumulative mask ratio;
- linear cumulative mask ratio.

Use more denoising steps for iGSM full denoise. Suggested first sweep:

```text
num_steps: 16, 32, 64, 128
sampler: high-confidence commit vs random remask
start: fully masked, n=12, n=8
```

The easier starts are diagnostics. The real target should remain fully masked
solution/answer generation.

### Priority 3: Stronger SOTA-Style Variants

Once the baseline no longer leaves masks:

- Add classifier-free guidance by sometimes dropping/masking the prompt during
  training and combining conditional/unconditional logits at sampling.
- Add block or semi-autoregressive diffusion for long traces to reduce fixed
  length and coherence issues.
- Initialize from the causal iGSM model if feasible, or at least share token
  embeddings and comparable Transformer blocks.
- For replacement correction, add DMax-style/on-policy self-correction: train on
  visible wrong tokens and model-generated wrong tokens, not only `[MASK]`.

## Suggested Next Experiments

1. Read `3614356` to determine whether stepwise collapse is caused by committing
   too many tokens per inference step or by the predictor-decoder path itself.
2. Read `3614357` to determine whether oracle latent planning helps current
   partial stepwise checkpoints.
3. Complete `3612683`/`3612684`/`3612685` to compare stepwise JEPA with
   `num_steps=20/50/64` and the same commit-k sampler sweep.
4. If x0 denoising remains weak, sweep `num_steps=32/64/128` for the denoising
   LM with the fixed objective and sampler.
5. If stepwise JEPA improves action ordering, add action-free and policy-only
   stepwise controls.
6. Design the variable-length edit MDP for `REPLACE`, `KEEP`, `DELETE`, and
   `INSERT_AFTER`; this needs gap/action supervision and cannot be represented
   faithfully by the current fixed-length action tensors alone.

Success criteria:

- mask-token rate after full denoise is `0%`;
- iGSM ID answer accuracy is nonzero and clearly above the old denoising LM;
- LANO exact/token/grammar metrics do not regress versus the old short-mask
  denoising baseline;
- JEPA should beat the fixed denoising LM only after the denoising LM is a fair
  modern baseline.

## References

- D3PM: Structured Denoising Diffusion Models in Discrete State-Spaces,
  https://arxiv.org/abs/2107.03006
- SEDD: Discrete Diffusion Modeling by Estimating the Ratios of the Data
  Distribution, https://arxiv.org/abs/2310.16834
- MD4: Simplified and Generalized Masked Diffusion for Discrete Data,
  https://arxiv.org/abs/2406.04329
- MDLM: Simple and Effective Masked Diffusion Language Models,
  https://arxiv.org/abs/2406.07524
- SMDM: Scaling up Masked Diffusion Models on Text,
  https://arxiv.org/abs/2410.18514
- LLaDA: Large Language Diffusion Models, https://arxiv.org/abs/2502.09992
- Dream 7B: Diffusion Large Language Models,
  https://arxiv.org/abs/2508.15487
- LLaDA2.0: Scaling Up Diffusion Language Models to 100B,
  https://arxiv.org/abs/2512.15745
- DMax: Aggressive Parallel Decoding for dLLMs,
  https://arxiv.org/abs/2604.08302
- MaskGIT: Masked Generative Image Transformer,
  https://arxiv.org/abs/2202.04200
