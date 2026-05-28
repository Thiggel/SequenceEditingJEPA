from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from puzzle_jepa.models.layers import GridEncoder, TransformerStack


@dataclass(slots=True)
class RecursiveReasonerOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor
    q_logits: torch.Tensor
    preds: torch.Tensor
    components: dict[str, torch.Tensor]


class _AnswerHeadMixin:
    def _loss(
        self,
        logits: torch.Tensor,
        q_logits: torch.Tensor,
        labels: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
        if labels is None:
            return None, {}
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="mean")
        with torch.no_grad():
            exact = (logits.argmax(dim=-1) == labels).flatten(1).all(dim=-1).float()
        q_loss = F.binary_cross_entropy_with_logits(q_logits, exact)
        loss = ce + 0.5 * q_loss
        return loss, {"loss/ce": ce.detach(), "loss/q": q_loss.detach(), "exact": exact.mean()}


class HRMReasoner(nn.Module, _AnswerHeadMixin):
    """Minimal HRM scaffold with separate high- and low-frequency recurrent modules."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 128,
        intermediate_size: int = 256,
        num_heads: int = 4,
        input_layers: int = 1,
        h_layers: int = 2,
        l_layers: int = 2,
        h_cycles: int = 2,
        l_cycles: int = 2,
        max_height: int = 30,
        max_width: int = 30,
        task_vocab_size: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.h_cycles = int(h_cycles)
        self.l_cycles = int(l_cycles)
        self.input_encoder = GridEncoder(
            vocab_size,
            hidden_size,
            intermediate_size,
            input_layers,
            num_heads,
            max_height,
            max_width,
            task_vocab_size,
            dropout,
        )
        self.low = TransformerStack(l_layers, hidden_size, intermediate_size, num_heads, dropout)
        self.high = TransformerStack(h_layers, hidden_size, intermediate_size, num_heads, dropout)
        self.h_init = nn.Parameter(torch.zeros(hidden_size))
        self.l_init = nn.Parameter(torch.zeros(hidden_size))
        self.out = nn.Linear(hidden_size, vocab_size)
        self.q = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor | None = None, task_ids: torch.Tensor | None = None) -> RecursiveReasonerOutput:
        x = self.input_encoder(inputs, task_ids=task_ids)
        batch, length, hidden = x.shape
        z_h = self.h_init.view(1, 1, hidden).expand(batch, length, hidden)
        z_l = self.l_init.view(1, 1, hidden).expand(batch, length, hidden)
        for h_step in range(self.h_cycles):
            for _ in range(self.l_cycles):
                z_l = self.low(z_l + z_h + x)
            if h_step + 1 < self.h_cycles:
                z_h = self.high(z_h + z_l)
        z_h = self.high(z_h + z_l)
        logits = self.out(z_h).reshape(inputs.shape[0], inputs.shape[1], inputs.shape[2], -1)
        q_logits = self.q(z_h.mean(dim=1)).squeeze(-1)
        loss, components = self._loss(logits, q_logits, labels)
        return RecursiveReasonerOutput(loss, logits, q_logits, logits.argmax(dim=-1), components)


class TRMReasoner(nn.Module, _AnswerHeadMixin):
    """Tiny Recursive Model scaffold: one small network updates latent z and answer y."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 128,
        intermediate_size: int = 256,
        num_heads: int = 4,
        input_layers: int = 1,
        recurrent_layers: int = 2,
        h_cycles: int = 3,
        l_cycles: int = 6,
        max_height: int = 30,
        max_width: int = 30,
        task_vocab_size: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.h_cycles = int(h_cycles)
        self.l_cycles = int(l_cycles)
        self.input_encoder = GridEncoder(
            vocab_size,
            hidden_size,
            intermediate_size,
            input_layers,
            num_heads,
            max_height,
            max_width,
            task_vocab_size,
            dropout,
        )
        self.core = TransformerStack(recurrent_layers, hidden_size, intermediate_size, num_heads, dropout)
        self.y_init = nn.Parameter(torch.zeros(hidden_size))
        self.z_init = nn.Parameter(torch.zeros(hidden_size))
        self.out = nn.Linear(hidden_size, vocab_size)
        self.q = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor | None = None, task_ids: torch.Tensor | None = None) -> RecursiveReasonerOutput:
        y, z = self.rollout_latents(inputs, task_ids=task_ids)
        logits = self.out(y).reshape(inputs.shape[0], inputs.shape[1], inputs.shape[2], -1)
        q_logits = self.q(y.mean(dim=1)).squeeze(-1)
        loss, components = self._loss(logits, q_logits, labels)
        return RecursiveReasonerOutput(loss, logits, q_logits, logits.argmax(dim=-1), components)

    def rollout_latents(
        self,
        inputs: torch.Tensor,
        task_ids: torch.Tensor | None = None,
        depth: int | None = None,
        noise_std: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_encoder(inputs, task_ids=task_ids)
        batch, length, hidden = x.shape
        y = self.y_init.view(1, 1, hidden).expand(batch, length, hidden)
        z = self.z_init.view(1, 1, hidden).expand(batch, length, hidden)
        outer_steps = int(depth or self.h_cycles)
        for _ in range(outer_steps):
            for _ in range(self.l_cycles):
                if noise_std:
                    z = z + torch.randn_like(z) * float(noise_std)
                z = self.core(x + y + z)
            y = self.core(y + z)
        return y, z


class PTRMSampler(nn.Module):
    """Inference-time stochastic rollout wrapper for a TRMReasoner."""

    def __init__(self, model: TRMReasoner, rollouts: int = 8, depth: int | None = None, noise_std: float = 0.2):
        super().__init__()
        self.model = model
        self.rollouts = int(rollouts)
        self.depth = depth
        self.noise_std = float(noise_std)
        if self.rollouts <= 0:
            raise ValueError("rollouts must be positive.")

    @torch.no_grad()
    def forward(self, inputs: torch.Tensor, task_ids: torch.Tensor | None = None) -> RecursiveReasonerOutput:
        batch = inputs.shape[0]
        repeated_inputs = inputs.repeat_interleave(self.rollouts, dim=0)
        repeated_tasks = None if task_ids is None else task_ids.repeat_interleave(self.rollouts)
        y, _ = self.model.rollout_latents(
            repeated_inputs,
            task_ids=repeated_tasks,
            depth=self.depth,
            noise_std=self.noise_std,
        )
        logits = self.model.out(y).reshape(batch, self.rollouts, inputs.shape[1], inputs.shape[2], -1)
        q_logits = self.model.q(y.mean(dim=1)).reshape(batch, self.rollouts)
        best = q_logits.argmax(dim=1)
        selected_logits = logits[torch.arange(batch, device=inputs.device), best]
        selected_q = q_logits[torch.arange(batch, device=inputs.device), best]
        return RecursiveReasonerOutput(None, selected_logits, selected_q, selected_logits.argmax(dim=-1), {})
