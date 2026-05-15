from __future__ import annotations

import torch
from transformers import TrainerCallback

from seq_edit_jepa.actions.action_types import Op
from seq_edit_jepa.actions.apply import apply_fixed_actions
from seq_edit_jepa.losses import suppress_token_logits


class DebugSequenceCallback(TrainerCallback):
    def __init__(
        self,
        task,
        corruptor,
        tokenizer,
        seq_len: int,
        every_steps: int = 0,
        rollout_steps: int = 0,
        max_actions: int = 24,
    ):
        self.task = task
        self.corruptor = corruptor
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.every_steps = int(every_steps)
        self.rollout_steps = int(rollout_steps)
        self.max_actions = int(max_actions)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.every_steps <= 0 or model is None:
            return control
        if state.global_step != 1 and state.global_step % self.every_steps != 0:
            return control
        if not state.is_local_process_zero:
            return control
        was_training = bool(model.training)
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            clean = self.task.sample_batch(1, self.seq_len, split="eval", device=device)
            batch = self.corruptor.sample_pair(clean)
            output = model(batch)
            if output.op_logits is not None and output.token_logits is not None:
                token_logits = suppress_token_logits(output.token_logits, [int(getattr(model.config, "mask_token_id", self.tokenizer.mask_token_id))])
                pred_ids = apply_fixed_actions(
                    batch.input_ids,
                    output.op_logits.argmax(dim=-1),
                    token_logits.argmax(dim=-1),
                )
                pred_ops = output.op_logits.argmax(dim=-1)
                pred_tokens = token_logits.argmax(dim=-1)
            else:
                token_logits = suppress_token_logits(output.logits, [int(getattr(model.config, "mask_token_id", self.tokenizer.mask_token_id))])
                pred_ids = batch.input_ids.clone()
                target_mask = batch.target_mask if batch.target_mask is not None else batch.prev_attention_mask.bool()
                pred_ids[target_mask.bool()] = token_logits.argmax(dim=-1)[target_mask.bool()]
                pred_ops = None
                pred_tokens = None
            lines = [
                "",
                f"[seq-edit debug] global_step={state.global_step} n={int(batch.n[0].item())}",
                f"clean:       {self._decode(batch.clean_ids[0], batch.attention_mask[0])}",
                f"current s_n: {self._decode(batch.input_ids[0], batch.attention_mask[0])}",
                f"oracle act:  {self._format_actions(batch.action_ops[0], batch.action_tokens[0], batch.editable_mask[0])}",
                f"oracle next: {self._decode(batch.prev_ids[0], batch.prev_attention_mask[0])}",
                f"model act:   {self._format_actions(pred_ops[0], pred_tokens[0], batch.editable_mask[0]) if pred_ops is not None else '<token decoder only>'}",
                f"model next:  {self._decode(pred_ids[0], batch.prev_attention_mask[0])}",
            ]
            if self.rollout_steps > 0:
                lines.extend(self._format_rollout(device))
            print("\n".join(lines), flush=True)
        if was_training:
            model.train()
        return control

    def _format_rollout(self, device: torch.device) -> list[str]:
        clean = self.task.sample_batch(1, self.seq_len, split="eval", device=device)
        path = self.corruptor.sample_path(clean, rollout_steps=self.rollout_steps)
        lines = ["oracle rollout:"]
        for index in range(len(path.action_ops)):
            lines.append(f"  s_{int(path.n_values[index][0].item())}: {self._decode(path.states[index][0], path.attention_masks[index][0])}")
            lines.append(
                f"    A: {self._format_actions(path.action_ops[index][0], path.action_tokens[index][0], path.editable_mask[0])}"
            )
        lines.append(f"  s_{int(path.n_values[-1][0].item())}: {self._decode(path.states[-1][0], path.attention_masks[-1][0])}")
        return lines

    def _decode(self, ids: torch.Tensor, mask: torch.Tensor) -> str:
        values = ids[mask.bool()].detach().cpu().tolist()
        if hasattr(self.tokenizer, "decode"):
            return str(self.tokenizer.decode(values, skip_special_tokens=False))
        return " ".join(str(value) for value in values)

    def _format_actions(self, ops: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor) -> str:
        if ops is None:
            return "<none>"
        chunks = []
        valid_positions = torch.where(mask.bool().detach().cpu())[0].tolist()
        ops_cpu = ops.detach().cpu()
        tokens_cpu = tokens.detach().cpu()
        for pos in valid_positions:
            op = int(ops_cpu[pos].item())
            if op == int(Op.KEEP):
                continue
            token = int(tokens_cpu[pos].item())
            chunks.append(f"{Op(op).name}@{pos}:{self._token_name(token)}")
            if len(chunks) >= self.max_actions:
                chunks.append("...")
                break
        return ", ".join(chunks) if chunks else "KEEP all"

    def _token_name(self, token_id: int) -> str:
        if hasattr(self.tokenizer, "convert_ids_to_tokens"):
            return str(self.tokenizer.convert_ids_to_tokens(int(token_id)))
        return str(token_id)
