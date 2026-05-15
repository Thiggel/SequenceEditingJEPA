from __future__ import annotations

import torch
from transformers import Trainer, TrainerCallback

from seq_edit_jepa.models import SequenceEditJEPA


class EMAUpdateCallback(TrainerCallback):
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is not None and hasattr(model, "update_target_encoder"):
            model.update_target_encoder()
        return control


class SeqEditTrainer(Trainer):
    def __init__(self, *args, task=None, corruptor=None, seq_len: int | None = None, rollout_steps: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.task = task
        self.corruptor = corruptor
        self.seq_len = seq_len
        self.rollout_steps = int(rollout_steps)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss
        lambda_roll = float(getattr(model.config, "lambda_roll", 0.0))
        if (
            lambda_roll > 0.0
            and self.rollout_steps > 0
            and isinstance(model, SequenceEditJEPA)
            and self.task is not None
            and self.corruptor is not None
            and self.seq_len is not None
        ):
            clean = self.task.sample_batch(inputs["input_ids"].shape[0], self.seq_len, split="train", device=inputs["input_ids"].device)
            path = self.corruptor.sample_path(clean, rollout_steps=self.rollout_steps)
            roll = model.rollout_loss(path)
            loss = loss + lambda_roll * roll
            model.last_loss_components["loss/rollout"] = float(roll.detach().cpu())
        return (loss, outputs) if return_outputs else loss
