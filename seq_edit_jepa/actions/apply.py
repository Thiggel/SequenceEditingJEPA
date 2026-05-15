from __future__ import annotations

from typing import Iterable, List, Sequence

import torch

from seq_edit_jepa.actions.action_types import EditAction, Op


def apply_action(sequence: Sequence[int], action: EditAction, mask_token_id: int | None = None) -> List[int]:
    output = list(int(token) for token in sequence)
    pos = int(action.pos)
    if action.op == Op.KEEP:
        return output
    if action.op == Op.REPLACE:
        _check_existing_position(output, pos)
        output[pos] = action.require_token()
        return output
    if action.op == Op.MASK:
        _check_existing_position(output, pos)
        if mask_token_id is None:
            output[pos] = action.require_token()
        else:
            output[pos] = int(mask_token_id)
        return output
    if action.op == Op.DELETE:
        _check_existing_position(output, pos)
        del output[pos]
        return output
    if action.op in {Op.INSERT, Op.INSERT_NOISE}:
        if pos < 0 or pos > len(output):
            raise IndexError(f"Insert gap {pos} is outside sequence length {len(output)}.")
        output.insert(pos, action.require_token())
        return output
    raise ValueError(f"Unsupported edit op: {action.op}")


def apply_actions(sequence: Sequence[int], actions: Iterable[EditAction], mask_token_id: int | None = None) -> List[int]:
    """Apply actions with positions interpreted in the original sequence.

    Replacements/masks happen first, deletions are applied right-to-left, and
    insertions are applied left-to-right at original gaps.
    """

    replacements: list[EditAction] = []
    deletions: list[EditAction] = []
    insertions: list[EditAction] = []
    for action in actions:
        if action.op in {Op.REPLACE, Op.MASK, Op.KEEP}:
            replacements.append(action)
        elif action.op == Op.DELETE:
            deletions.append(action)
        elif action.op in {Op.INSERT, Op.INSERT_NOISE}:
            insertions.append(action)
        else:
            raise ValueError(f"Unsupported edit op: {action.op}")

    output = list(int(token) for token in sequence)
    for action in replacements:
        output = apply_action(output, action, mask_token_id=mask_token_id)
    for action in sorted(deletions, key=lambda item: item.pos, reverse=True):
        output = apply_action(output, action, mask_token_id=mask_token_id)
    inserted = 0
    for action in sorted(insertions, key=lambda item: item.pos):
        shifted = EditAction(action.op, action.pos + inserted, action.token)
        output = apply_action(output, shifted, mask_token_id=mask_token_id)
        inserted += 1
    return output


def apply_fixed_actions(input_ids: torch.Tensor, ops: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    output = input_ids.clone()
    replace_mask = ops == int(Op.REPLACE)
    output[replace_mask] = tokens[replace_mask]
    return output


def _check_existing_position(sequence: Sequence[int], pos: int) -> None:
    if pos < 0 or pos >= len(sequence):
        raise IndexError(f"Position {pos} is outside sequence length {len(sequence)}.")
