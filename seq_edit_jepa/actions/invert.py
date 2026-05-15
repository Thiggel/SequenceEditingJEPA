from __future__ import annotations

from typing import Sequence

from seq_edit_jepa.actions.action_types import EditAction, Op


def invert_destructive_action(before: Sequence[int], action: EditAction) -> EditAction:
    """Return the inverse edit for one destructive action.

    `before` is the state before the destructive action was applied, so it
    contains the clean token needed to undo masks, replacements, and deletions.
    """

    pos = int(action.pos)
    if action.op == Op.MASK:
        return EditAction(Op.REPLACE, pos, int(before[pos]))
    if action.op == Op.REPLACE:
        return EditAction(Op.REPLACE, pos, int(before[pos]))
    if action.op == Op.DELETE:
        return EditAction(Op.INSERT, pos, int(before[pos]))
    if action.op == Op.INSERT_NOISE:
        return EditAction(Op.DELETE, pos)
    raise ValueError(f"Action {action.op.name} is not a destructive corruption action.")
