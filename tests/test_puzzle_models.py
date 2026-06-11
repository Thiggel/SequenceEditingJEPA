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
from puzzle_jepa.data.worlds import WorldAction
from puzzle_jepa.eval.diagnostics import (
    MCTSNode,
    backup_mcts_value,
    build_mcts_tree,
    clue_mask_for_planning,
    evaluate_cem_planning,
    evaluate_hierarchical_subgoal_cem_planning,
    evaluate_latent_drift,
    evaluate_latent_planning,
    evaluate_mpc_cem_planning,
    evaluate_mcts_planning,
    evaluate_paired_reset_planning,
    evaluate_reencoded_planning,
    evaluate_recursive_hierarchical_subgoal_planning,
    estimate_macro_action_prior,
    high_level_subgoal_cem,
    mcts_root_debug_record,
    mcts_ucb_score,
    oracle_action_sequence,
    score_leaf_state,
    score_symbolic_states_to_goal,
    select_mcts_root_child,
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


def test_global_mlp_jepa_uses_one_latent_token_and_small_action_embedding():
    world, batch = _sudoku_batch()
    model = ActionConditionedWorldModel(
        vocab_size=world.vocab_size,
        hidden_size=32,
        intermediate_size=64,
        encoder_layers=2,
        predictor_layers=2,
        num_heads=4,
        max_height=9,
        max_width=9,
        task_vocab_size=2,
        action_value_vocab_size=10,
        encoder_type="global_mlp",
        action_embedding_dim=8,
        use_task_embedding=False,
        use_selected_cell_marker=False,
        use_goal_energy_head=True,
    )
    assert model.row_embedding.embedding_dim == 8
    output = model(
        batch.states,
        batch.actions,
        batch.next_states,
        goals=batch.goals,
        initial_states=batch.states,
        goal_energy_weight=0.5,
    )
    assert torch.isfinite(output.loss)
    assert output.pred_latents.shape == output.target_latents.shape == (2, 1, 32)
    scores = model.score_states_to_goal(batch.states, batch.goals, batch.actions[:, 0])
    assert scores.shape == (2,)
    output.loss.backward()
    assert model.encoder.mlp[0].weight.grad is not None
    assert model.predictor.net[0].weight.grad is not None
    assert model.goal_energy_head[-1].weight.grad is not None


def test_global_mlp_hierarchy_uses_mlp_action_sequence_encoders():
    world = SudokuWorld()
    example = world.example_from_strings(SUDOKU_PUZZLE, SUDOKU_SOLUTION)
    rollouts = [
        sample_oracle_rollout_transition(world, example, np.random.default_rng(seed), steps=4)
        for seed in range(2)
    ]
    rollout_batch = collate_rollouts(rollouts)
    model = ActionConditionedWorldModel(
        vocab_size=world.vocab_size,
        hidden_size=32,
        intermediate_size=64,
        encoder_layers=2,
        predictor_layers=2,
        num_heads=4,
        max_height=9,
        max_width=9,
        task_vocab_size=2,
        action_value_vocab_size=10,
        encoder_type="global_mlp",
        action_embedding_dim=8,
        use_task_embedding=False,
        use_selected_cell_marker=False,
        use_goal_energy_head=True,
        hierarchy_levels=3,
        hierarchy_span=2,
        macro_action_dim=6,
    )
    output = model.hierarchy_loss(rollout_batch.states, rollout_batch.actions, rollout_batch.target_states)
    assert torch.isfinite(output.loss)
    assert output.pred_latents.shape == output.target_latents.shape == (2, 1, 32)
    abstract_action = model.encode_hierarchy_action(rollout_batch.actions[:, :4], level=2)
    assert abstract_action.shape == (2, 6)
    decoded = model.decode_hierarchy_action(abstract_action, level=2)
    assert decoded.shape == (2, 8)
    latents = model.encoder(rollout_batch.states, task_ids=rollout_batch.actions[:, 0, 0])
    macro_pred = model.predict_latent_from_abstract_action(latents, abstract_action, level=2)
    assert macro_pred.shape == latents.shape
    output.loss.backward()
    assert model.higher_action_encoders[-1].net[0].weight.grad is not None
    assert model.higher_predictors[-1].net[0].weight.grad is not None


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


def test_macro_action_bottleneck_and_vq_are_used_by_hierarchy_and_planner():
    world = SudokuWorld()
    example = world.example_from_strings(SUDOKU_PUZZLE, SUDOKU_SOLUTION)
    rollouts = [
        sample_oracle_rollout_transition(world, example, np.random.default_rng(seed), steps=4)
        for seed in range(2)
    ]
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
        use_macro_action_value_head=True,
        hierarchy_levels=3,
        hierarchy_span=2,
        macro_action_dim=4,
        use_macro_action_vq=True,
        macro_action_codebook_size=8,
    )
    abstract_action = model.encode_hierarchy_action(rollout_batch.actions[:, :4], level=2)
    assert abstract_action.shape == (2, 4)
    assert model.decode_hierarchy_action(abstract_action, level=2).shape == (2, 32)

    hierarchy = model.hierarchy_loss(rollout_batch.states, rollout_batch.actions, rollout_batch.target_states)
    assert torch.isfinite(hierarchy.loss)
    assert "loss/hierarchy_level_2_action_vq" in hierarchy.components
    macro_value = model.macro_action_value_loss(
        rollout_batch.states,
        rollout_batch.states,
        rollout_batch.actions[:, :4],
        rollout_batch.goals,
        level=2,
    )
    assert torch.isfinite(macro_value.loss)

    task_ids = rollout_batch.actions[:, 0, 0]
    latent = model.encoder(rollout_batch.states, task_ids=task_ids)
    macro_pred = model.predict_latent_from_abstract_action(latent, abstract_action, level=2)
    assert macro_pred.shape == latent.shape
    (hierarchy.loss + macro_value.loss).backward()
    assert model.macro_action_codebooks[-1].embedding.weight.grad is not None
    assert model.higher_action_decoders[-1][1].weight.grad is not None

    rng = np.random.default_rng(0)
    clue_mask = clue_mask_for_planning(world, example.state)
    prior = estimate_macro_action_prior(
        model,
        world,
        example.state,
        example.goal,
        clue_mask,
        rng,
        hierarchy_level=2,
        samples=2,
    )
    assert prior is not None
    assert prior.shape[1] == 4
    task_id = torch.full((1,), world.task_id, dtype=torch.long)
    current = torch.as_tensor(example.state[None, ...], dtype=torch.long)
    goal = torch.as_tensor(example.goal[None, ...], dtype=torch.long)
    current_latent = model.encoder(current, task_ids=task_id)
    goal_latent = model.target_encoder(goal, task_ids=task_id)
    plan = high_level_subgoal_cem(
        model,
        current_latent,
        goal_latent,
        rng,
        hierarchy_level=2,
        macro_horizon=2,
        population_size=4,
        elite_frac=0.5,
        iterations=1,
        smoothing=0.5,
        prior=prior,
        initial_latent=current_latent,
        score_mode="latent_goal",
    )
    assert plan["latent_action_sequence"].shape == (2, 4)
    assert plan["subgoal_latent"].shape == current_latent.shape


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
        use_macro_action_value_head=True,
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

    mpc_summary, mpc_records = evaluate_mpc_cem_planning(
        model,
        world,
        [example],
        np.random.default_rng(7),
        num_examples=1,
        max_steps=2,
        horizon=2,
        execute_steps=1,
        population_size=4,
        elite_frac=0.5,
        iterations=1,
        smoothing=0.7,
        score_mode="goal_energy",
    )
    assert mpc_summary["goal_energy"]["count"] == 1.0
    assert mpc_records[0]["planner"] == "mpc_cem"
    assert mpc_records[0]["replans"] >= 1.0

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
    assert subgoal_summary["latent_goal_subgoal"]["count"] == 1.0
    assert subgoal_records[0]["planner"] == "hierarchical_subgoal_cem"
    assert subgoal_records[0]["hierarchy_level"] == 1.0
    assert subgoal_records[0]["high_score_mode"] == "latent_goal"

    value_subgoal_summary, value_subgoal_records = evaluate_hierarchical_subgoal_cem_planning(
        model,
        world,
        [example],
        np.random.default_rng(3),
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
        high_score_mode="goal_value",
    )
    assert value_subgoal_summary["goal_value_subgoal"]["count"] == 1.0
    assert value_subgoal_records[0]["high_score_mode"] == "goal_value"

    macro_subgoal_summary, macro_subgoal_records = evaluate_hierarchical_subgoal_cem_planning(
        model,
        world,
        [example],
        np.random.default_rng(4),
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
        high_score_mode="macro_action_advantage",
    )
    assert macro_subgoal_summary["macro_action_advantage_subgoal"]["count"] == 1.0
    assert macro_subgoal_records[0]["high_score_mode"] == "macro_action_advantage"

    recursive_summary, recursive_records = evaluate_recursive_hierarchical_subgoal_planning(
        model,
        world,
        [example],
        np.random.default_rng(5),
        num_examples=1,
        max_steps=1,
        hierarchy_level=1,
        macro_horizon=1,
        high_population_size=3,
        low_population_size=3,
        elite_frac=0.5,
        iterations=1,
        smoothing=0.7,
        execute_steps=1,
        prior_samples=2,
        high_score_mode="goal_value",
        optimizer="cem",
    )
    assert recursive_summary["goal_value_cem_recursive_subgoal"]["count"] == 1.0
    assert recursive_records[0]["planner"] == "recursive_hierarchical_subgoal"
    assert recursive_records[0]["high_score_mode"] == "goal_value"
    assert recursive_records[0]["optimizer"] == "cem"

    gd_summary, gd_records = evaluate_recursive_hierarchical_subgoal_planning(
        model,
        world,
        [example],
        np.random.default_rng(6),
        num_examples=1,
        max_steps=1,
        hierarchy_level=1,
        macro_horizon=1,
        high_population_size=3,
        low_population_size=3,
        elite_frac=0.5,
        iterations=1,
        smoothing=0.7,
        execute_steps=1,
        prior_samples=2,
        high_score_mode="latent_goal",
        optimizer="gd_reachability",
        gd_steps=2,
        gd_lr=0.01,
        reachability_weight=0.1,
    )
    assert gd_summary["latent_goal_gd_reachability_recursive_subgoal"]["count"] == 1.0
    assert gd_records[0]["optimizer"] == "gd_reachability"


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


def test_goal_energy_and_goal_value_have_opposite_score_orientation():
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

    def fake_predict(states, initial_states, task_ids):
        del initial_states, task_ids
        return torch.arange(states.shape[0], dtype=torch.float32, device=states.device)

    model.predict_goal_energy = fake_predict  # type: ignore[method-assign]
    states = [example.state.copy(), example.goal.copy()]
    energy_scores = score_symbolic_states_to_goal(
        model,
        world,
        states,
        example.goal,
        example.state,
        planning_score="goal_energy",
    )
    value_scores = score_symbolic_states_to_goal(
        model,
        world,
        states,
        example.goal,
        example.state,
        planning_score="goal_value",
    )
    assert energy_scores == [0.0, -1.0]
    assert value_scores == [0.0, 1.0]


def test_action_advantage_scores_actions_with_higher_is_better():
    world = SudokuWorld()
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
        use_action_value_head=True,
    )
    actions = [WorldAction(0, 0, 1), WorldAction(0, 1, 9)]

    def fake_action_value(states, initial_states, action_tensor):
        del states, initial_states
        return action_tensor[:, 3].to(torch.float32)

    model.predict_action_value = fake_action_value  # type: ignore[method-assign]
    scores = model.score_actions_with_value_head(
        torch.zeros((9, 9), dtype=torch.long),
        torch.zeros((9, 9), dtype=torch.long),
        actions,
        world.task_id,
    )
    assert scores.argmax().item() == 1


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


class _HammingEnergyModel(torch.nn.Module):
    use_goal_energy_head = True

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def encoder(self, states, task_ids=None):
        del task_ids
        return states.reshape(states.shape[0], -1, 1).float() / 9.0 + self.anchor * 0.0

    def target_encoder(self, states, task_ids=None):
        return self.encoder(states, task_ids=task_ids)

    def predict_goal_energy(self, states, initial_states, task_id):
        del initial_states, task_id
        goal = torch.as_tensor(_NEAR_SOLVED_GOAL, dtype=states.dtype, device=states.device)
        return (states != goal).reshape(states.shape[0], -1).float().sum(dim=1)


_NEAR_SOLVED_GOAL = SudokuWorld.from_string(SUDOKU_SOLUTION)
_NEAR_SOLVED_PUZZLE = _NEAR_SOLVED_GOAL.copy()
_NEAR_SOLVED_PUZZLE[0, 2] = 0


def test_mcts_backup_and_ucb_score_are_self_contained():
    root = MCTSNode(
        state=np.zeros((1, 1), dtype=np.int64),
        parent=None,
        action=None,
        depth=0,
        untried_actions=[],
        children={},
    )
    child = MCTSNode(
        state=np.ones((1, 1), dtype=np.int64),
        parent=root,
        action=WorldAction(0, 0, 1),
        depth=1,
        untried_actions=[],
        children={},
    )
    root.children[child.action] = child
    backup_mcts_value(child, 3.0)
    backup_mcts_value(child, 1.0)

    assert root.visits == 2
    assert child.visits == 2
    assert child.mean_value == 2.0
    assert mcts_ucb_score(root, child, exploration=0.0) == 2.0


def test_mcts_leaf_scoring_uses_correct_sign_for_energy():
    world = SudokuWorld()
    model = _HammingEnergyModel()
    wrong_score = score_leaf_state(
        model,
        world,
        _NEAR_SOLVED_PUZZLE,
        _NEAR_SOLVED_GOAL,
        _NEAR_SOLVED_PUZZLE,
        score_mode="goal_energy",
    )
    correct_score = score_leaf_state(
        model,
        world,
        _NEAR_SOLVED_GOAL,
        _NEAR_SOLVED_GOAL,
        _NEAR_SOLVED_PUZZLE,
        score_mode="goal_energy",
    )

    assert correct_score > wrong_score
    assert correct_score == 0.0
    assert wrong_score == -1.0


def test_mcts_tree_with_hamming_energy_prefers_goal_write():
    world = SudokuWorld()
    model = _HammingEnergyModel()
    clue_mask = world.clue_mask_from_puzzle(_NEAR_SOLVED_PUZZLE)
    root = build_mcts_tree(
        model,
        world,
        _NEAR_SOLVED_PUZZLE,
        _NEAR_SOLVED_GOAL,
        _NEAR_SOLVED_PUZZLE,
        clue_mask,
        np.random.default_rng(0),
        simulations=64,
        max_depth=1,
        score_mode="goal_energy",
        exploration=0.5,
        expansion_actions=9,
    )
    child = select_mcts_root_child(root)

    assert child is not None
    assert child.action == WorldAction(0, 2, int(_NEAR_SOLVED_GOAL[0, 2]))
    assert child.visits > 1
    assert child.mean_value == 0.0


def test_mcts_expansion_cap_allows_tree_to_reach_deeper_nodes():
    world = SudokuWorld()
    model = _HammingEnergyModel()
    puzzle = _NEAR_SOLVED_GOAL.copy()
    puzzle[0, 2] = 0
    puzzle[0, 3] = 0
    clue_mask = world.clue_mask_from_puzzle(puzzle)
    root = build_mcts_tree(
        model,
        world,
        puzzle,
        _NEAR_SOLVED_GOAL,
        puzzle,
        clue_mask,
        np.random.default_rng(3),
        simulations=12,
        max_depth=2,
        score_mode="goal_energy",
        exploration=1.0,
        expansion_actions=2,
    )

    assert len(root.children) == 2
    assert any(child.children for child in root.children.values())


def test_mcts_debug_record_reports_action_ranking_details():
    world = SudokuWorld()
    model = _HammingEnergyModel()
    clue_mask = world.clue_mask_from_puzzle(_NEAR_SOLVED_PUZZLE)
    root = build_mcts_tree(
        model,
        world,
        _NEAR_SOLVED_PUZZLE,
        _NEAR_SOLVED_GOAL,
        _NEAR_SOLVED_PUZZLE,
        clue_mask,
        np.random.default_rng(1),
        simulations=64,
        max_depth=1,
        score_mode="goal_energy",
        exploration=0.5,
        expansion_actions=9,
    )
    record = mcts_root_debug_record(
        model,
        world,
        root,
        _NEAR_SOLVED_GOAL,
        _NEAR_SOLVED_PUZZLE,
        step=0,
        score_mode="goal_energy",
        debug_actions=4,
    )

    assert record["root_visits"] == 64
    assert record["expanded_actions"] == 9
    assert record["best_writes_goal_value"] is True
    assert record["actions"][0]["leaf_score"] == 0.0
    assert record["actions"][0]["oracle_leaf_energy"] == 0.0


def test_mcts_planning_solves_one_blank_board_with_reencoded_leaf_scoring(tmp_path):
    world = SudokuWorld()
    model = _HammingEnergyModel()
    example = PuzzleExample(_NEAR_SOLVED_PUZZLE, _NEAR_SOLVED_GOAL)
    summary, records, debug = evaluate_mcts_planning(
        model,
        world,
        [example],
        np.random.default_rng(2),
        num_examples=1,
        max_steps=1,
        simulations=64,
        max_depth=1,
        score_mode="goal_energy",
        exploration=0.5,
        expansion_actions=9,
        debug_examples=1,
        debug_actions=4,
        stream_dir=tmp_path,
    )

    assert summary["goal_energy"]["solve_rate"] == 1.0
    assert records[0]["solved"] == 1.0
    assert records[0]["steps"] == 1.0
    assert debug[0]["best_writes_goal_value"] is True
    assert len((tmp_path / "mcts_planning_records.jsonl").read_text().splitlines()) == 1
    assert len((tmp_path / "mcts_debug_records.jsonl").read_text().splitlines()) == 1


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
