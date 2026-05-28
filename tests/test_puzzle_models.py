import numpy as np
import torch

from puzzle_jepa.data import MazeWorld, SudokuWorld, collate_transitions, sample_oracle_transition
from puzzle_jepa.data.worlds import PuzzleExample
from puzzle_jepa.eval.diagnostics import evaluate_latent_drift, oracle_action_sequence
from puzzle_jepa.models import ActionConditionedWorldModel, HRMReasoner, PTRMSampler, TRMReasoner


SUDOKU_PUZZLE = (
    "530070000"
    "600195000"
    "098000060"
    "800060003"
    "400803001"
    "700020006"
    "060000280"
    "000419005"
    "000080079"
)
SUDOKU_SOLUTION = (
    "534678912"
    "672195348"
    "198342567"
    "859761423"
    "426853791"
    "713924856"
    "961537284"
    "287419635"
    "345286179"
)


def _sudoku_batch(batch_size=2):
    world = SudokuWorld()
    example = world.example_from_strings(SUDOKU_PUZZLE, SUDOKU_SOLUTION)
    transitions = [sample_oracle_transition(world, example, np.random.default_rng(i)) for i in range(batch_size)]
    return world, collate_transitions(transitions)


def _maze_batch(batch_size=2):
    world = MazeWorld(height=5, width=5)
    state = world.from_lines(["S   #", "### #", "#   #", "# ###", "#   G"])
    goal = world.from_lines(["Sooo#", "###o#", "#ooo#", "#o###", "#oooG"])
    example = type("Example", (), {"state": state, "goal": goal})()
    transitions = [sample_oracle_transition(world, example, np.random.default_rng(i)) for i in range(batch_size)]
    return world, collate_transitions(transitions)


def test_action_conditioned_jepa_forward_backward_sudoku():
    world, batch = _sudoku_batch()
    model = ActionConditionedWorldModel(
        vocab_size=world.vocab_size,
        hidden_size=32,
        intermediate_size=64,
        encoder_layers=1,
        predictor_layers=1,
        num_heads=4,
        max_height=9,
        max_width=9,
        task_vocab_size=2,
        action_value_vocab_size=10,
    )
    output = model(batch.states, batch.actions, batch.next_states)
    assert torch.isfinite(output.loss)
    assert output.pred_latents.shape == output.target_latents.shape == (2, 81, 32)
    output.loss.backward()
    assert model.encoder.token_embedding.weight.grad is not None
    model.sync_target()


def test_action_conditioned_jepa_scores_maze_actions():
    world, batch = _maze_batch()
    model = ActionConditionedWorldModel(
        vocab_size=world.vocab_size,
        hidden_size=32,
        intermediate_size=64,
        encoder_layers=1,
        predictor_layers=1,
        num_heads=4,
        max_height=5,
        max_width=5,
        task_vocab_size=2,
        action_value_vocab_size=5,
    )
    actions = world.legal_actions(batch.states[0].numpy())[:4]
    scores = model.score_actions_to_goal(batch.states[0], actions, batch.goals[0], world.task_id)
    assert scores.shape == (4,)
    assert torch.isfinite(scores).all()
    state_scores = model.score_states_to_goal(batch.states, batch.goals, batch.actions[:, 0])
    assert state_scores.shape == (2,)
    assert torch.isfinite(state_scores).all()


def test_predict_latent_from_latent_matches_state_path():
    world, batch = _sudoku_batch()
    model = ActionConditionedWorldModel(
        vocab_size=world.vocab_size,
        hidden_size=32,
        intermediate_size=64,
        encoder_layers=1,
        predictor_layers=1,
        num_heads=4,
        max_height=9,
        max_width=9,
        task_vocab_size=2,
        action_value_vocab_size=10,
        dropout=0.0,
    )
    model.eval()
    direct = model.predict_latent(batch.states, batch.actions)
    encoded = model.encoder(batch.states, task_ids=batch.actions[:, 0])
    latent_path = model.predict_latent_from_latent(encoded, batch.actions, height=9, width=9)
    assert torch.allclose(direct, latent_path)


def test_diagnostics_oracle_sequence_and_drift_smoke():
    world = MazeWorld(height=5, width=5)
    state = world.from_lines(["S   #", "### #", "#   #", "# ###", "#   G"])
    goal = world.from_lines(["Sooo#", "###o#", "#ooo#", "#o###", "#oooG"])
    example = PuzzleExample(state, goal)
    actions = oracle_action_sequence(world, example, np.random.default_rng(0))
    current = state.copy()
    for action in actions:
        current = world.apply(current, action)
    assert np.array_equal(current, goal)

    model = ActionConditionedWorldModel(
        vocab_size=world.vocab_size,
        hidden_size=32,
        intermediate_size=64,
        encoder_layers=1,
        predictor_layers=1,
        num_heads=4,
        max_height=5,
        max_width=5,
        task_vocab_size=2,
        action_value_vocab_size=5,
        dropout=0.0,
    )
    records, summary = evaluate_latent_drift(
        model,
        world,
        [example],
        np.random.default_rng(1),
        num_examples=1,
        horizons=[1, 2],
        max_unroll_steps=2,
    )
    assert records
    assert summary["count"] == len(records)


def test_hrm_forward_backward():
    world, batch = _sudoku_batch()
    model = HRMReasoner(
        vocab_size=world.vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        input_layers=1,
        h_layers=1,
        l_layers=1,
        h_cycles=2,
        l_cycles=2,
        max_height=9,
        max_width=9,
        task_vocab_size=2,
    )
    output = model(batch.states, labels=batch.goals, task_ids=batch.actions[:, 0])
    assert output.loss is not None
    assert output.logits.shape == (2, 9, 9, world.vocab_size)
    output.loss.backward()
    assert model.input_encoder.token_embedding.weight.grad is not None


def test_trm_forward_backward_and_ptrm_sampling():
    world, batch = _sudoku_batch()
    model = TRMReasoner(
        vocab_size=world.vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        input_layers=1,
        recurrent_layers=1,
        h_cycles=2,
        l_cycles=2,
        max_height=9,
        max_width=9,
        task_vocab_size=2,
    )
    output = model(batch.states, labels=batch.goals, task_ids=batch.actions[:, 0])
    assert output.loss is not None
    assert output.logits.shape == (2, 9, 9, world.vocab_size)
    output.loss.backward()
    sampler = PTRMSampler(model, rollouts=3, depth=2, noise_std=0.05)
    sampled = sampler(batch.states, task_ids=batch.actions[:, 0])
    assert sampled.loss is None
    assert sampled.preds.shape == (2, 9, 9)
    assert sampled.q_logits.shape == (2,)
