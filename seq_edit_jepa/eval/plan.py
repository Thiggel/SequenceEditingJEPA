from __future__ import annotations

import torch


@torch.no_grad()
def score_candidate_latents(policy_logprob: torch.Tensor, value: torch.Tensor, cost: torch.Tensor, beta: float, eta: float) -> torch.Tensor:
    return policy_logprob + float(beta) * value - float(eta) * cost


@torch.no_grad()
def choose_best_candidate(scores: torch.Tensor) -> torch.Tensor:
    return scores.argmax(dim=-1)
