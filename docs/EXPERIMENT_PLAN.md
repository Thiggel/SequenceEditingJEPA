# Experiment Plan

Source of truth: `../sequence-editing-report/BACKLOG.md` and
`../sequence-editing-report/CURRENT_EXPERIMENTS.md`.

The active controlled experiment is a full factorial over predictor family,
four hierarchy schedules, dense rollout depth, uniform versus `0.9^i` loss,
exact object count, and three seeds. It uses only a single MLP-encoded latent;
no grid latent or visual Transformer is permitted.

Higher levels are staged from same-task lower checkpoints, freeze the encoder
and trained lower predictors, and learn ordered concat-plus-linear pixel-edit
macro actions. Teacher-forced causal prediction and autonomous rollout are
deeply supervised at every active level. Planning recursively passes the first
coarse prediction as the next finer subgoal and replans after each edit.

Primary decisions:

1. Does hierarchy improve fixed-horizon planning and endpoint prediction over
   `[1]`, especially `[1,4,16]` versus stride control `[1,2,4]`?
2. Does dense rollout improve autonomous future prediction and semantic
   transfer, and does `0.9^i` prevent long-horizon degradation?
3. Which causal predictor retains object properties as load rises from one to
   eight objects?
4. Do reconstruction, linear properties, and planning move together, or does
   the latent retain task-relevant object state without becoming a pixel map?

Do not add LDAD or another representation axis until this grid identifies a
working hierarchy/planning regime.
