# Wave 13: MLP Pixel-Edit HWM

## Question

Do predictor family, hierarchy, dense rollout, weighting, or object load rescue
a 256-dimensional MLP latent under atomic pixel-edit dynamics?

## Runs

Train arrays `3851603`-`3851607`, probe arrays `3851608`-`3851611`, and repair
jobs `3854953`-`3855014`. The factorial crossed predictor
`{Transformer,Gated DeltaNet,LSTM}`, schedules
`{[1],[1,4],[1,4,16],[1,2,4]}`, rollout `{1,2,4,8}`, weighting
`{uniform,.9^i}`, exact N `{1,2,4,8}`, and three seeds: 1,152 final cells and
1,440 staged trainers. At cancellation, 1,419 checkpoints and 752 final probe
files existed. All 52 remaining controlled jobs were canceled on 2026-07-14.

## Results

Balanced `[1]` versus `[1,4]` exact planning was `.102` versus `.041`; pixel
error `.052` versus `.100`. Flat planning at N1/2/4/8 was
`.326/.083/0/0`. Effective rank collapsed to `9.4/256`; count improved, while
shape, position, relations, and foreground reconstruction did not.

## Conclusion

The model accurately predicted a collapsed count/color summary. About three
quarters of primitive states were partial paint/erase objects, so fixed
hierarchy waypoints were not valid rigid-object states. The wave was stopped,
not completed.
