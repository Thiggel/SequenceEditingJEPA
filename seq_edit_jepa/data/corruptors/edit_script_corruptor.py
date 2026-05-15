from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from seq_edit_jepa.actions.action_types import EditAction, Op
from seq_edit_jepa.actions.apply import apply_action
from seq_edit_jepa.actions.invert import invert_destructive_action


@dataclass
class EditScript:
    states: list[list[int]]
    destructive_actions: list[EditAction]
    inverse_actions: list[EditAction]


class EditScriptCorruptor:
    """Symbolic variable-length corruption for stage 3 tests and extensions."""

    def __init__(self, vocab_ids: Sequence[int], mask_token_id: int, seed: int = 0):
        self.vocab_ids = [int(token) for token in vocab_ids]
        self.mask_token_id = int(mask_token_id)
        self.rng = random.Random(seed)

    def sample_script(self, clean_tokens: Sequence[int], num_steps: int) -> EditScript:
        states = [list(int(token) for token in clean_tokens)]
        destructive: list[EditAction] = []
        inverses: list[EditAction] = []
        current = states[0]
        for _ in range(num_steps):
            action = self._sample_destructive(current)
            inverse = invert_destructive_action(current, action)
            current = apply_action(current, action, mask_token_id=self.mask_token_id)
            states.append(current)
            destructive.append(action)
            inverses.append(inverse)
        return EditScript(states=states, destructive_actions=destructive, inverse_actions=inverses)

    def _sample_destructive(self, sequence: Sequence[int]) -> EditAction:
        ops = [Op.MASK, Op.REPLACE, Op.INSERT_NOISE]
        if sequence:
            ops.append(Op.DELETE)
        op = self.rng.choice(ops)
        if op == Op.INSERT_NOISE:
            return EditAction(op, self.rng.randint(0, len(sequence)), self.rng.choice(self.vocab_ids))
        pos = self.rng.randrange(len(sequence))
        if op == Op.REPLACE:
            return EditAction(op, pos, self.rng.choice(self.vocab_ids))
        if op == Op.MASK:
            return EditAction(op, pos, self.mask_token_id)
        return EditAction(op, pos)
