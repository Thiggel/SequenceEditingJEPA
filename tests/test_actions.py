from seq_edit_jepa.actions import EditAction, Op, apply_action, apply_actions, invert_destructive_action
from seq_edit_jepa.data.corruptors.edit_script_corruptor import EditScriptCorruptor


def test_apply_and_invert_destructive_actions():
    clean = [10, 11, 12]
    destructive = EditAction(Op.DELETE, 1)
    corrupted = apply_action(clean, destructive)
    inverse = invert_destructive_action(clean, destructive)
    assert corrupted == [10, 12]
    assert apply_action(corrupted, inverse) == clean

    destructive = EditAction(Op.INSERT_NOISE, 2, 99)
    corrupted = apply_action(clean, destructive)
    inverse = invert_destructive_action(clean, destructive)
    assert corrupted == [10, 11, 99, 12]
    assert apply_action(corrupted, inverse) == clean


def test_parallel_apply_actions_original_positions():
    clean = [1, 2, 3, 4]
    actions = [EditAction(Op.REPLACE, 1, 20), EditAction(Op.DELETE, 3), EditAction(Op.INSERT, 0, 99)]
    assert apply_actions(clean, actions) == [99, 1, 20, 3]


def test_edit_script_round_trip():
    corruptor = EditScriptCorruptor(vocab_ids=[7, 8, 9], mask_token_id=0, seed=3)
    script = corruptor.sample_script([1, 2, 3, 4], num_steps=8)
    current = script.states[-1]
    for inverse in reversed(script.inverse_actions):
        current = apply_action(current, inverse)
    assert current == script.states[0]
