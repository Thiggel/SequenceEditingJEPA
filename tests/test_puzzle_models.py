import numpy as np
import torch

from puzzle_jepa.data import (
    MazeWorld,
    SudokuWorld,
    collate_rollouts,
    collate_transitions,
    sample_oracle_rollout_transition,
    sample_oracle_transition,
)
from puzzle_jepa.data.worlds import PuzzleExample
from puzzle_jepa.eval.diagnostics import (
    evaluate_cem_planning,
    evaluate_hierarchical_subgoal_cem_planning,
    evaluate_latent_drift,
    evaluate_latent_planning,
    evaluate_paired_reset_planning,
    evaluate_reencoded_planning,
    oracle_action_sequence,
)
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


def test_rollout_loss_backpropagates_through_predictor():
    world = SudokuWorld()
    example = world.example_from_strings(SUDOKU_PUZZLE, SUDOKU_SOLUTION)
    rollouts = [
        sample_oracle_rollout_transition(world, example, np.random.default_rng(seed), steps=3)
        for seed in range(2)
    ]
    batch = collate_rollouts(rollouts)
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
        use_task_embedding=False,
        use_selected_cell_marker=False,
    )
    output = model.rollout_loss(batch.states, batch.actions, batch.target_states)
    assert torch.isfinite(output.loss)
    assert output.pred_latents.shape == output.target_latents.shape == (2, 81, 32)
    output.loss.backward()
    assert model.predictor.layers[0].attn.in_proj_weight.grad is not None


def test_optional_task_and_selected_cell_conditioning_preserve_shapes():
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
        use_task_embedding=False,
        use_selected_cell_marker=False,
    )
    output = model(batch.states, batch.actions, batch.next_states)
    assert output.pred_latents.shape == (2, 81, 32)


def test_local_value_action_injection_uses_selected_position_without_row_col_embeddings():
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
        action_injection="local_value",
        use_task_embedding=False,
        use_selected_cell_marker=False,
    )
    output = model(batch.states, batch.actions, batch.next_states)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert model.value_embedding.weight.grad is not None
    assert model.row_embedding.weight.grad is None
    assert model.col_embedding.weight.grad is None


def test_residual_prediction_and_weighted_loss_backpropagate():
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
        action_injection="local_value",
        predict_residual=True,
    )
    weights = torch.zeros_like(batch.states, dtype=torch.float32)
    rows = batch.actions[:, 1]
    cols = batch.actions[:, 2]
    weights[torch.arange(batch.states.shape[0]), rows, cols] = 1.0
    output = model(batch.states, batch.actions, batch.next_states, loss_weights=weights)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert model.predictor.layers[0].attn.in_proj_weight.grad is not None


def test_goal_energy_cls_and_hierarchy_losses_backpropagate():
    world = SudokuWorld()
    example = world.example_from_strings(SUDOKU_PUZZLE, SUDOKU_SOLUTION)
    transitions = [sample_oracle_transition(world, example, np.random.default_rng(seed)) for seed in range(2)]
    rollouts = [
        sample_oracle_rollout_transition(world, example, np.random.default_rng(seed), steps=4)
        for seed in range(2)
    ]
    batch = collate_transitions(transitions)
    rollout_batch = collate_rollouts(rollouts)
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
        action_injection="local_value",
        use_cls_token=True,
        use_goal_energy_head=True,
        hierarchy_levels=3,
        hierarchy_span=2,
    )
    assert batch.clue_masks is not None
    initial_states = batch.states * batch.clue_masks.to(dtype=batch.states.dtype)
    output = model(
        batch.states,
        batch.actions,
        batch.next_states,
        goals=batch.goals,
        initial_states=initial_states,
        goal_energy_weight=0.5,
    )
    hierarchy = model.hierarchy_loss(rollout_batch.states, rollout_batch.actions, rollout_batch.target_states)
    assert torch.isfinite(output.loss)
    assert torch.isfinite(hierarchy.loss)
    assert output.pred_latents.shape == output.target_latents.shape == (2, 82, 32)
    assert "loss/goal_energy_mse" in output.components
    assert "loss/hierarchy_level_2_h4_mse" in hierarchy.components
    abstract_action = model.encode_hierarchy_action(rollout_batch.actions[:, :4], level=2)
    assert abstract_action.shape == (2, 32)
    latent = model.encoder(batch.states, task_ids=batch.actions[:, 0])
    macro_pred = model.predict_latent_from_abstract_action(latent, abstract_action, level=2)
    assert macro_pred.shape == latent.shape
    (output.loss + hierarchy.loss).backward()
    assert model.goal_energy_head[-1].weight.grad is not None
    assert model.higher_action_encoders[-1].stack.layers[0].attn.in_proj_weight.grad is not None
    assert model.higher_predictors[-1].layers[0].attn.in_proj_weight.grad is not None


def test_diagnostics_return_latent_and_reencoded_planning_records():
    world = SudokuWorld()
    example = world.example_from_strings(SUDOKU_PUZZLE, SUDOKU_SOLUTION)
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
        action_injection="local_value",
        use_task_embedding=False,
        use_selected_cell_marker=False,
    )
    rng = np.random.default_rng(0)
    latent_summary, latent_records = evaluate_latent_planning(
        model,
        world,
        [example],
        rng,
        num_examples=1,
        max_steps=2,
        branch_size=1,
        beam_size=1,
    )
    reencoded_summary, reencoded_records = evaluate_reencoded_planning(
        model,
        world,
        [example],
        rng,
        num_examples=1,
        max_steps=2,
        branch_size=1,
        beam_size=1,
    )
    reset_summary, reset_records = evaluate_paired_reset_planning(
        model,
        world,
        [example],
        rng,
        num_examples=1,
        max_steps=2,
        branch_size=1,
        beam_size=1,
        reset_cadences=[2],
    )
    assert latent_summary["step_energy"]["count"] == 1.0
    assert reencoded_summary["terminal_energy"]["count"] == 1.0
    assert reset_summary["reset_every_2"]["terminal_energy"]["count"] == 1.0
    assert latent_records[0]["planner"] == "latent"
    assert reencoded_records[0]["planner"] == "reencoded"
    assert {record["variant"] for record in reset_records} == {"latent_no_reset", "reset_every_2", "reencoded"}
    assert {record["example_index"] for record in reset_records} == {0}
    assert len(latent_records[0]["final_state"]) == 9
    assert {"row", "col", "pred", "goal"} <= set(latent_records[0]["mismatches"][0])


def test_cem_planning_records_with_goal_energy_head():
    world = SudokuWorld()
    example = world.example_from_strings(SUDOKU_PUZZLE, SUDOKU_SOLUTION)
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
        action_injection="local_value",
        use_cls_token=True,
        use_goal_energy_head=True,
        hierarchy_levels=2,
        hierarchy_span=2,
    )
    summary, records = evaluate_cem_planning(
        model,
        world,
        [example],
        np.random.default_rng(0),
        num_examples=1,
        max_steps=2,
        population_size=4,
        elite_frac=0.5,
        iterations=1,
        smoothing=0.7,
        score_mode="goal_energy",
    )
    assert summary["goal_energy"]["count"] == 1.0
    assert records[0]["planner"] == "cem"
    assert records[0]["score_mode"] == "goal_energy"
    assert len(records[0]["final_state"]) == 9
    hierarchical_summary, hierarchical_records = evaluate_cem_planning(
        model,
        world,
        [example],
        np.random.default_rng(1),
        num_examples=1,
        max_steps=2,
        population_size=4,
        elite_frac=0.5,
        iterations=1,
        smoothing=0.7,
        score_mode="hierarchical_latent_goal",
        hierarchy_level=1,
    )
    assert hierarchical_summary["hierarchical_latent_goal"]["count"] == 1.0
    assert hierarchical_records[0]["hierarchy_level"] == 1.0

    subgoal_summary, subgoal_records = evaluate_hierarchical_subgoal_cem_planning(
        model,
        world,
        [example],
        np.random.default_rng(2),
        num_examples=1,
        max_steps=2,
        hierarchy_level=1,
        macro_horizon=1,
        high_population_size=3,
        low_population_size=3,
        elite_frac=0.5,
        iterations=1,
        smoothing=0.7,
        execute_steps=1,
        prior_samples=2,
    )
    assert subgoal_summary["oracle_latent_subgoal"]["count"] == 1.0
    assert subgoal_records[0]["planner"] == "hierarchical_subgoal_cem"
    assert subgoal_records[0]["hierarchy_level"] == 1.0


def test_reset_planning_can_use_goal_energy_head():
    world = SudokuWorld()
    example = world.example_from_strings(SUDOKU_PUZZLE, SUDOKU_SOLUTION)
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
        action_injection="local_value",
        use_cls_token=True,
        use_goal_energy_head=True,
    )
    summary, records = evaluate_paired_reset_planning(
        model,
        world,
        [example],
        np.random.default_rng(0),
        num_examples=1,
        max_steps=2,
        branch_size=1,
        beam_size=1,
        reset_cadences=[2],
        planning_score="goal_energy",
    )
    assert summary["reset_every_2"]["terminal_energy"]["count"] == 1.0
    assert {record["planning_score"] for record in records} == {"goal_energy"}


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
