# Results

Long-form results: `../sequence-editing-report/RESULTS.md`.

Last updated: 2026-07-13 22:15 CEST

The MLP pixel-edit HWM grid was submitted as train arrays `3851603`-`3851607`
and probe arrays `3851608`-`3851611`. At 22:15 CEST, 20 base checkpoints and
20 probes were complete with no active-log errors.

The first complete three-seed cell is Transformer, hierarchy `[1]`, rollout 1,
uniform loss, and eight objects. Prediction loss is
`.00989/.01052/.01141`, predictor-over-identity gain is
`.00276/.00241/.00328`, and primitive action top-1 is `1.0/1.0/1.0`.
Nevertheless, learned fixed 16-edit planning is `0/0/0`; retrieval final pixel
error is `.0771/.0752/.0732`. Count and motion-policy balanced accuracy improve
from initialization in every seed, but shape does not, position/relation R2
remain negative, and foreground reconstruction decreases. The weighted
rollout-1 duplicate is metric-identical because `lambda^i` cannot change a
single supervised step.

Prelaunch verification passed the full CPU test suite, one-step Hydra CPU
training, and separate RTX Pro 6000 forward/backward training smokes for the
Transformer, Gated DeltaNet, and LSTM predictors. The Gated DeltaNet smoke
first exposed and then verified the FP32/BF16 kernel-boundary fix. Generated
trajectories with exact object counts `{1,2,4,8}` change exactly one pixel per
action and replay exactly.

Historical controlled results showed that wider Transformer/CLS models
improved color-indexed position and relation probes without producing reliable
hierarchical planning. Those jobs used the superseded encoder/action world and
are controls, not results for the active grid.
