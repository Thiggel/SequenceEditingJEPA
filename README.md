# Sequence-Edit JEPA

This branch implements a minimal action-conditioned latent world model for sequence editing:

```text
whole sequence state -> edit action -> latent transition -> revised sequence
```

The first fully runnable experiments are fixed-length demasking and replacement correction on controlled LANO/CFG-style strings and synthetic iGSM-style arithmetic traces. Variable-length insert/delete editing is implemented at the symbolic action layer and exposed for later model extensions.

Use the shared cluster environment:

```bash
source ./scripts/env.sh
python -m pytest
python -m seq_edit_jepa.train.hydra_train --config-name smoke_lano_mask
```

Training uses Hydra config names, a Hugging Face `Trainer` subclass, and HF-compatible `PreTrainedModel` checkpoints. Each run saves:

```text
$SEQ_EDIT_JEPA_WORK_ROOT/runs/<experiment>/
  config.yaml
  metrics.json
  checkpoint.pt          # compatibility checkpoint
  model/                 # save_pretrained directory
  tokenizer/
```

The default architecture is a bidirectional sequence-edit Transformer with RoPE, RMSNorm, QK-norm self-attention, SwiGLU MLPs, sinusoidal timestep conditioning, an edit-policy MLP head, and an action-conditioned latent predictor. Debug sequence prints are controlled by the `debug:` config block.

SLURM launchers live under `scripts/slurm/` and use the same `$WORK/.venv`, proxy, and cache settings copied from `../T-JEPA`. Runtime outputs go under `SEQ_EDIT_JEPA_WORK_ROOT`, defaulting to `$WORK/sequence-editing`.

Current experiment tracking:

```text
docs/EXPERIMENT_PLAN.md
docs/RESULTS.md
docs/DENOISING_LITERATURE.md
docs/RUNBOOK.md
docs/sequence_edit_jepa_report.tex
```

Latest submitted experiment batch:

```text
3621296_[0-7]  objective/capacity ablations
3624400_[0-7]  objective/capacity ablation resume
3621300_[0-1]  dependent LeWM-like latent-MPC eval after resume
```

The ablation batch covers higher action/token CE, contextual decoder heads,
detached decoder readout, deeper policy heads, differentiable soft predicted
action conditioning, LeWM-like no-decoder-CE objectives with and without a value
head, and a large deep-decoder denoising-LM baseline. See
`docs/EXPERIMENT_PLAN.md` and `docs/RUNBOOK.md` for the exact array mapping.
