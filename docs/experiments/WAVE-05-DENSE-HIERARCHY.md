# Wave 05: Dense Rollout and Hierarchy

## Question

Can multi-step deep supervision and hierarchy turn the best action-grounded
Sudoku model into a working latent planner?

## Runs

- Follow-up dense/hierarchy/capacity wave: 336 planner rows.
- `H1_hierarchy_dense_l4_l16`, flat dense K16/K32, hierarchy-only, d384, and
  deeper d256 variants.
- Weekend goal/dense jobs `3780027/3780028`, `3782967/3782968`, and duplicate
  `3784073/3784074`.
- H1 debug jobs `3795127/3795128`, `3795143/3795144`; hierarchical add-ons
  `3795248/3795249`; H1-extra `3795246`, replacement `3795327`.

## Results

`H1_hierarchy_dense_l4_l16` was the first latent-rollout signal: oracle
changed-cell scoring solved `6/10` at depth 16, but all predicted-goal rows
were `0/10`. Controlled H1 reruns did not reproduce exact solves; best
no-delta H1 remained `6.6` cells away. Dense horizon alone was negative.

## Conclusion

Hierarchy can help when given an oracle local metric, but neither dense rollout
nor hierarchy repairs bad learned goal geometry. The result did not establish
paper-faithful HWM because macro planning and representation quality remained
confounded.
