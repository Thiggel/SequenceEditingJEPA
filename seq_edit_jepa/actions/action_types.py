from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import torch


class Op(IntEnum):
    KEEP = 0
    REPLACE = 1
    DELETE = 2
    INSERT = 3
    MASK = 4
    INSERT_NOISE = 5


@dataclass(frozen=True)
class EditAction:
    op: Op
    pos: int
    token: Optional[int] = None

    def require_token(self) -> int:
        if self.token is None:
            raise ValueError(f"{self.op.name} at position {self.pos} requires a token.")
        return int(self.token)


@dataclass
class FixedActions:
    """Parallel fixed-length edit supervision for stages 1 and 2."""

    ops: torch.Tensor
    tokens: torch.Tensor

    def to(self, device: torch.device | str) -> "FixedActions":
        return FixedActions(ops=self.ops.to(device), tokens=self.tokens.to(device))
