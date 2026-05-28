# Legacy Sequence-Editing JEPA Work

Last updated: 2026-05-26

This document freezes the sequence-editing/iGSM/LANO line before the pivot to objective
puzzle worlds. The active direction after this point is maze and Sudoku-Extreme
reasoning with HRM/TRM/PTRM baselines and an action-conditioned JEPA world model.
The legacy source tree, configs, scripts, logs, and old handoff assets were
archived outside the active repo at `../legacy-sequence-editing` on
2026-05-26.

## Final Slurm Status

As of the last checked handoff snapshot on 2026-05-26 11:16 CEST, there were no
active `seqedit` Slurm jobs, no training jobs, no posthoc jobs, and no scheduled
oversight jobs. The scheduled `cs exec` oversight jobs completed cleanly.

## Methodology That Was Tested

The old repo explored latent and token denoising over text reasoning traces. The
core setup corrupted prompt-plus-reasoning sequences, trained models to repair
or denoise them, and evaluated whether iterative denoising could recover final
iGSM answers, especially on OOD operation-count splits.

Implemented branches included:

- x0 denoising LM: bidirectional denoising from masked editable reasoning/answer
  tokens directly to clean `x0`.
- x0 JEPA: action-conditioned latent predictor plus token decoder/policy heads.
- stepwise JEPA: predict local edit actions and future latent states over
  denoising schedules.
- visible replacement editing: corrupt clean tokens with visible in-vocabulary
  replacement noise and train direct repair.
- visible insert/delete editing: add replacement, deletion, and insert-after
  noise so the model could repair length changes.
- fully unrolled mask JEPA: train over entire noising histories from fully
  masked editable tokens to clean text, carrying the predicted latent state
  forward recurrently.
- diagnostic MPC/value/proposal sweeps: evaluate whether action position,
  oracle token injection, DLM proposal tokens, or oracle-goal latent search could
  rescue stepwise JEPA.

Important implementation fixes made during the line:

- Corruptors target clean `x0`, not the immediate corrupted state unless a
  stepwise target mode is explicitly requested.
- Token CE is restricted to corrupted editable positions.
- `[MASK]` is suppressed as an editable token prediction.
- JEPA inference can decode from predictor latents rather than only policy-token
  heads.
- Full-denoise eval supports fixed commit counts (`commit_k=1/2/5/10`) and
  scheduled confidence commits.
- Unrolled JEPA added OOM controls: microbatching, gradient accumulation,
  activation checkpointing, no path-logit stacking by default, and truncated BPTT
  by detaching recurrent latents every K steps.

## Final Experiment Ledger

| Area | Job(s) | State | Main result |
| --- | --- | --- | --- |
| Causal iGSM baseline | `3605092_[0-2]`, `3605200`, `3605202`, `3605203` | Completed | Strongest iGSM model: about `88.3%` ID, `86.7%` OOD ops, fixed op20-23 mean about `21.1%`. |
| x0 JEPA/DLM rerun | `3606650_[0-1]`, `3607175_[0-1]`, `3607176`, `3612686_[0-5]` | Completed after resume | x0 DLM beats x0 JEPA. x0 DLM latest full-denoise about `42.2%` ID, `29.7%` OOD ops; x0 JEPA best ID `35.9%` with policy head and OOD ops `28.9%` with predictor decoder. |
| x0 DLM level-field k=1 | `3650873` | Completed | `commit_k=1`, mask start, `checkpoint-200000`: ID answer `42.97%`; op20/op21/op22/op23 answer `7.81%`/`16.41%`/`10.16%`/`10.16%`; fixed-op average `11.13%`. |
| Stepwise JEPA training | `3612683_[0-3]`, `3612684_[0-3]` | Timed out then completed after resume | Produced `checkpoint-200000` for T20/T50/T64 mask and T64 replacement. Local edit/position signal existed, but long fixed-op answer accuracy stayed weak. |
| Stepwise inference sweeps | `3614356_[0-7]`, `3614357_[0-5]`, `3615623_[0-2]`, `3615624_[0-15]` | Completed | Oracle token plus model position solved small diagnostics (`93.8%` ID, `67-72%` OOD ops), showing token choice was the main deployable bottleneck. DLM-token proposer was the best stepwise route (`T20` about `39.1%/26.6%` ID/OOD ops). |
| Value heads and rollout repair | `3615625_[0-1]`, `3615643_[0-1]` | Completed | Value heads fit oracle latent-energy targets, but deployable fixed-op generation remained weak. Rollout repair improved T20 ID only modestly; fixed long-op OOD remained poor. |
| Objective/capacity ablations | `3621296_[0-7]`, `3624400_[0-7]`, `3621300_[0-1]` | Cancelled/stopped | Live results stayed weak; no fixed-op win emerged, so compute was stopped. |
| LANO visible replacement | `3643150_0` | Completed | One-step exact `31.25%`, edit F1 `0.8793`; iterative visible denoise exact `16.41%`, token `42.30%`, grammar `63.28%`. |
| iGSM visible replacement 100k | `3643150_1`, `3647005_1` | First failed on quota; vault rerun completed | `100k` final: ID full-denoise `39.84%`, fixed op20-23 full-denoise average `7.42%`, exact `0.0%`. |
| iGSM visible replacement 200k | `3651299 -> 3651300 -> 3651301` | Completed | Continued to `checkpoint-200000`; posthoc `commit_k=1`, visible-noise start: ID answer `36.72%`; op20/op21/op22/op23 `4.69%`/`10.94%`/`3.91%`/`3.12%`; fixed average `5.66%`. Below x0 DLM on answer accuracy. |
| First visible replacement 200k chain | `3651262 -> 3651263 -> 3651264` | Failed/failed/cancelled | Hydra override used `experiment.resume_from_checkpoint` instead of grouped key `long.experiment.resume_from_checkpoint`. Superseded by `3651299 -> 3651300 -> 3651301`. |
| Visible insert/delete 200k | `3651265 -> 3651266 -> 3655533`; `3651267 -> 3658466` | Timeout/failed/completed; stale posthoc cancelled; corrected posthoc completed | Corrected resume reached `checkpoint-200000`. Final posthoc k=1: ID answer `20.31%`, token `75.23%`, exact `1.56%`; op20/op21/op22/op23 `3.12%`/`3.12%`/`4.69%`/`2.34%`; fixed average `3.32%`. Below x0 DLM and replacement-only. |
| Fully unrolled mask JEPA T64-ish | `3655157` | Cancelled | Stable and no OOM, but about `23-24s/step`; no checkpoint before walltime because first save was `20000` steps. Cancelled as unproductive. |
| Practical unrolled mask JEPA T12 | `3656513` | Completed | `12000/12000` in `15:44:57`; retained `checkpoint-10250` through `checkpoint-12000`; ID answer `0.00%`, token `69.27%`, exact `0.00%`; op20-23 answer all `0.00%`; loss `6.395`. Samples collapsed to masks/repeated `=` tokens. |
| Oversight jobs | `3655191`-`3655194`, `3656514`-`3656518` | Completed | Autonomous `cs exec` checks parsed job state/results and kept handoffs current. Final oversight `3656518` completed on 2026-05-25 10:51 CEST. |

## Main Results To Remember

| Direction | Best evidence | Interpretation |
| --- | --- | --- |
| Causal LM | About `88.3%` ID and `86.7%` OOD ops | Still the only strong iGSM baseline. |
| x0 DLM | `42.97%` ID and fixed op20-23 average `11.13%` under `commit_k=1` | Strongest non-causal denoising baseline. |
| x0 JEPA | Best ID `35.9%`; OOD ops `28.9%` in another mode | Real signal, but no win over DLM. |
| Stepwise JEPA | Oracle-token diagnostic `93.8%` ID and `67-72%` OOD ops | Position/action signal existed; token proposal was the bottleneck. |
| Visible replacement | `36.72%` ID and fixed average `5.66%` at matched 200k posthoc | Learned local repair, but did not beat x0 DLM and exact reconstruction stayed near zero. |
| Visible insert/delete | `20.31%` ID and fixed average `3.32%` | Infrastructure worked, but length-edit training was weaker than replacement-only and x0 DLM. |
| Fully unrolled JEPA | T64 stable but too slow; T12 completed with `0.00%` answer accuracy | Full-history recurrent denoising was not a viable use of compute for iGSM traces. |

## Interpretation And Insights

1. The denoising-trace objective is fundamentally noisy: the latent "state of the
   world" is not objective, and many reasoning traces can be textually different
   while implying the same answer. This makes JEPA-style latent prediction hard
   to diagnose.
2. x0 DLM should be treated as the old line's main non-causal baseline. JEPA
   variants had useful signals but did not beat it on answer accuracy.
3. Stepwise JEPA learned where to edit better than what token to emit. The oracle
   token diagnostic was decisive: with token choice supplied, model-chosen
   positions could solve many examples.
4. Visible-token repair improved local edit F1 and token accuracy, but answer
   accuracy and exact reconstruction were not strong enough. Extra 100k training
   from 100k to 200k did not help.
5. Insert/delete support is technically useful but not empirically justified by
   iGSM results. It increased task difficulty without improving answer metrics.
6. Fully unrolled noising histories are operationally possible but not viable:
   the T64-style run did not OOM, yet throughput was too slow; the T12 pilot
   checkpointed cleanly but had no answer-accuracy signal.
7. A windowed K-step unroll (`K=2/4`) was identified as a more viable JEPA
   variant than full-history unroll, but it was not implemented before the
   pivot.

## Artifacts

Important output roots and summaries:

- x0 DLM k=1 level-field summary:
  `/home/vault/c107fa/c107fa12/sequence-editing/posthoc/igsm_ood/level_x0_dlm_mask_k1_3650873.summary.json`
- visible replacement 200k k=1 summary:
  `/home/vault/c107fa/c107fa12/sequence-editing/posthoc/igsm_ood/level_visible_edit200k_random_k1_3651301.summary.json`
- visible insert/delete 200k k=1 summary:
  `/home/vault/c107fa/c107fa12/sequence-editing/posthoc/igsm_ood/level_visible_insert_delete_edit_k1_3658466.summary.json`
- unrolled T12 JEPA run root:
  `/home/vault/c107fa/c107fa12/sequence-editing/runs/igsm_official_med_unrolled_mask_jepa_T12_12k`
- visible replacement run root:
  `/home/vault/c107fa/c107fa12/sequence-editing/runs/igsm_official_med_visible_replacement_edit_100k`
- visible insert/delete run root:
  `/home/vault/c107fa/c107fa12/sequence-editing/runs/igsm_official_med_visible_insert_delete_edit_200k`

## What Carries Forward

Carry forward the engineering lessons, not the denoising-trace task:

- Keep objective state representations. Sudoku boards and maze grids are much
  better JEPA substrates than free-form reasoning traces.
- Prefer action scoring over action generation at first. Sudoku and maze action
  spaces are small enough to enumerate.
- Keep the JEPA model decoder-free initially: encode current board, condition a
  predictor on a symbolic action, and train against the next board latent.
- Use oracle goal states early for diagnosis: rank actions by predicted latent
  distance to the encoded goal, then later replace the oracle with learned goal
  prediction or a verifier.
- Preserve strong baselines. HRM, TRM, and PTRM should be reproduced first on
  Sudoku-Extreme and Maze-Hard so the JEPA direction is compared against the
  relevant recursive-reasoning line.
