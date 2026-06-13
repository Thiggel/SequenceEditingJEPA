import importlib.util
import numpy as np
import sys
import torch
from hydra import compose, initialize_config_dir

from pathlib import Path

from puzzle_jepa.data import SudokuWorld, collate_rollouts, sample_oracle_rollout_transition
from puzzle_jepa.eval.grid5_diagnostics import candidate_actions
from puzzle_jepa.eval.grid5_mpc_cem_diagnostics import cem_optimize_action_sequence, score_latent_rollouts
from puzzle_jepa.eval.grid5_planner_matrix import (
    action_embedding_matrix,
    beam_search_plan_once,
    decode_nearest_action,
    mcts_plan_once,
    nearest_neighbor_cem_plan_once,
    run_closed_loop,
    write_planner_outputs,
)
from puzzle_jepa.models import SigRegActionJEPA, sigreg_loss, vicreg_loss


_PROBE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "grid5_symbolic_planning_probe.py"
_PROBE_SPEC = importlib.util.spec_from_file_location("grid5_symbolic_planning_probe", _PROBE_PATH)
assert _PROBE_SPEC is not None and _PROBE_SPEC.loader is not None
_PROBE = importlib.util.module_from_spec(_PROBE_SPEC)
sys.modules[_PROBE_SPEC.name] = _PROBE
_PROBE_SPEC.loader.exec_module(_PROBE)
apply_sequences_vectorized = _PROBE.apply_sequences_vectorized
score_states = _PROBE.score_states


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


def _rollout_batch(steps=3, batch_size=2):
    world = SudokuWorld()
    example = world.example_from_strings(SUDOKU_PUZZLE, SUDOKU_SOLUTION)
    rollouts = [
        sample_oracle_rollout_transition(world, example, np.random.default_rng(seed), steps=steps)
        for seed in range(batch_size)
    ]
    return world, collate_rollouts(rollouts)


def test_sigreg_penalizes_degenerate_embeddings_more_than_gaussian():
    torch.manual_seed(0)
    degenerate = torch.zeros(256, 16)
    gaussian = torch.randn(256, 16)
    degenerate_loss = sigreg_loss(degenerate, projections=64, knots=16)
    gaussian_loss = sigreg_loss(gaussian, projections=64, knots=16)
    assert gaussian_loss < degenerate_loss


def test_vicreg_penalizes_degenerate_embeddings_more_than_gaussian():
    torch.manual_seed(0)
    degenerate = torch.zeros(256, 16)
    gaussian = torch.randn(256, 16)
    degenerate_loss = vicreg_loss(degenerate)
    gaussian_loss = vicreg_loss(gaussian)
    assert gaussian_loss < degenerate_loss


def test_grid5_variants_produce_single_latent_and_backpropagate():
    world, batch = _rollout_batch(steps=3, batch_size=2)
    for encoder_type in ("mlp", "cls_transformer"):
        for predictor_type in ("mlp", "ar_transformer"):
            for predict_delta in (False, True):
                model = SigRegActionJEPA(
                    vocab_size=world.vocab_size,
                    latent_size=32,
                    encoder_type=encoder_type,
                    predictor_type=predictor_type,
                    predict_delta=predict_delta,
                    encoder_hidden_size=64,
                    predictor_hidden_size=64,
                    transformer_layers=1,
                    predictor_layers=1,
                    num_heads=4,
                    max_rollout_steps=3,
                    sigreg_projections=8,
                    sigreg_knots=4,
                )
                output = model.rollout_loss(batch.states, batch.actions, batch.target_states, batch.goals)
                assert torch.isfinite(output.loss)
                assert torch.isfinite(output.teacher_forced_loss)
                assert output.recursive_loss.item() == 0.0
                assert output.pred_latents.shape == output.target_latents.shape == (2, 3, 32)
                output.loss.backward()
                assert any(param.grad is not None for param in model.encoder.parameters())
                assert any(param.grad is not None for param in model.goal_energy_head.parameters())


def test_grid5_recursive_rollout_loss_backpropagates_through_multistep_predictions():
    world, batch = _rollout_batch(steps=5, batch_size=2)
    for predictor_type in ("mlp", "ar_transformer"):
        model = SigRegActionJEPA(
            vocab_size=world.vocab_size,
            latent_size=16,
            encoder_type="mlp",
            predictor_type=predictor_type,
            encoder_hidden_size=32,
            predictor_hidden_size=32,
            predictor_layers=1,
            num_heads=4,
            max_rollout_steps=5,
            sigreg_projections=8,
            sigreg_knots=4,
        )
        output = model.rollout_loss(
            batch.states,
            batch.actions,
            batch.target_states,
            batch.goals,
            recursive_steps=4,
            recursive_weight=1.0,
        )
        assert torch.isfinite(output.loss)
        assert output.recursive_loss.item() > 0.0
        assert output.prediction_loss.item() > output.teacher_forced_loss.item()
        output.loss.backward()
        assert any(param.grad is not None for param in model.predictor.parameters())


def test_grid5_ema_target_encoder_is_frozen_and_updates():
    world, batch = _rollout_batch(steps=3, batch_size=2)
    model = SigRegActionJEPA(
        vocab_size=world.vocab_size,
        latent_size=16,
        encoder_type="mlp",
        predictor_type="mlp",
        encoder_hidden_size=32,
        predictor_hidden_size=32,
        target_encoder_momentum=0.5,
        sigreg_projections=8,
        sigreg_knots=4,
    )
    assert model.target_encoder is not None
    assert not any(param.requires_grad for param in model.target_encoder.parameters())
    output = model.rollout_loss(batch.states, batch.actions, batch.target_states, batch.goals)
    output.loss.backward()
    assert any(param.grad is not None for param in model.encoder.parameters())
    assert not any(param.grad is not None for param in model.target_encoder.parameters())
    with torch.no_grad():
        next(model.encoder.parameters()).add_(1.0)
    before = next(model.target_encoder.parameters()).detach().clone()
    model.update_target_encoder()
    after = next(model.target_encoder.parameters()).detach()
    assert not torch.allclose(before, after)


def test_ar_transformer_predictor_is_causal_over_training_sequence():
    torch.manual_seed(0)
    model = SigRegActionJEPA(
        vocab_size=10,
        latent_size=16,
        encoder_type="mlp",
        predictor_type="ar_transformer",
        encoder_hidden_size=32,
        predictor_hidden_size=32,
        predictor_layers=1,
        num_heads=4,
        max_rollout_steps=4,
        sigreg_projections=8,
        sigreg_knots=4,
    )
    model.eval()
    latents = torch.randn(2, 4, 16)
    actions = torch.tensor(
        [
            [[0, 0, 0, 1], [0, 0, 1, 2], [0, 0, 2, 3], [0, 0, 3, 4]],
            [[0, 1, 0, 1], [0, 1, 1, 2], [0, 1, 2, 3], [0, 1, 3, 4]],
        ],
        dtype=torch.long,
    )
    baseline = model.predict_sequence(latents, actions)
    changed_latents = latents.clone()
    changed_actions = actions.clone()
    changed_latents[:, 2:] = torch.randn_like(changed_latents[:, 2:]) * 100.0
    changed_actions[:, 2:, 1:] = torch.tensor([8, 8, 9])
    changed = model.predict_sequence(changed_latents, changed_actions)
    assert torch.allclose(baseline[:, :2], changed[:, :2], atol=1.0e-5)
    assert not torch.allclose(baseline[:, 3], changed[:, 3])


def test_grid5_hydra_config_composes():
    repo_root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(version_base=None, config_dir=str(repo_root / "configs" / "puzzle")):
        cfg = compose(config_name="grid5_sudoku_sigreg")
    assert cfg.model.sigreg_weight == 1.0
    assert cfg.model.action_size == 16
    assert cfg.training.goal_energy_weight == 1.0
    assert cfg.training.recursive_rollout_steps == 1
    assert cfg.training.recursive_rollout_weight == 0.0
    assert cfg.model.stabilizer_type == "sigreg"
    assert cfg.model.target_encoder_momentum == 0.0


def test_grid5_mpc_cem_components_return_valid_sequences():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    world = SudokuWorld()
    example = world.example_from_strings(SUDOKU_PUZZLE, SUDOKU_SOLUTION)
    clue_mask = world.clue_mask_from_puzzle(example.state)
    actions = candidate_actions(world, example.state, clue_mask)[:12]
    model = SigRegActionJEPA(
        vocab_size=world.vocab_size,
        latent_size=16,
        encoder_type="mlp",
        predictor_type="mlp",
        encoder_hidden_size=32,
        predictor_hidden_size=32,
        max_rollout_steps=4,
        sigreg_projections=8,
        sigreg_knots=4,
    )
    action_tensor = torch.as_tensor(
        np.stack([[action.as_array(world.task_id) for action in actions[:3]] for _ in range(2)]),
        dtype=torch.long,
    )
    scores = score_latent_rollouts(
        model,
        example.state,
        example.goal,
        example.state,
        action_tensor,
        score_mode="latent_goal",
        device=torch.device("cpu"),
    )
    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()

    plan = cem_optimize_action_sequence(
        model,
        world,
        example.state,
        example.goal,
        example.state,
        actions,
        horizon=3,
        score_mode="latent_goal",
        candidates=16,
        elites=4,
        iterations=2,
        smoothing=0.7,
        rng=rng,
        device=torch.device("cpu"),
    )
    assert len(plan["indices"]) == 3
    assert all(0 <= index < len(actions) for index in plan["indices"])
    assert np.isfinite(plan["score"])


def test_grid5_mpc_cem_uses_ar_history_beyond_predictor_window():
    torch.manual_seed(0)
    world = SudokuWorld()
    example = world.example_from_strings(SUDOKU_PUZZLE, SUDOKU_SOLUTION)
    clue_mask = world.clue_mask_from_puzzle(example.state)
    actions = candidate_actions(world, example.state, clue_mask)[:6]
    model = SigRegActionJEPA(
        vocab_size=world.vocab_size,
        latent_size=16,
        encoder_type="mlp",
        predictor_type="ar_transformer",
        encoder_hidden_size=32,
        predictor_hidden_size=32,
        predictor_layers=1,
        num_heads=4,
        max_rollout_steps=4,
        sigreg_projections=8,
        sigreg_knots=4,
    )
    action_tensor = torch.as_tensor(
        np.stack([[actions[(batch + step) % len(actions)].as_array(world.task_id) for step in range(6)] for batch in range(2)]),
        dtype=torch.long,
    )
    scores = score_latent_rollouts(
        model,
        example.state,
        example.goal,
        example.state,
        action_tensor,
        score_mode="goal_energy",
        device=torch.device("cpu"),
    )
    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()


def test_grid5_symbolic_probe_applies_action_sequences_exactly():
    world = SudokuWorld()
    example = world.example_from_strings(SUDOKU_PUZZLE, SUDOKU_SOLUTION)
    actions = [
        world.legal_actions(example.state, clue_mask=world.clue_mask_from_puzzle(example.state))[0],
        world.legal_actions(example.state, clue_mask=world.clue_mask_from_puzzle(example.state))[1],
    ]
    sampled = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
    boards = apply_sequences_vectorized(example.state, actions, sampled)
    assert boards.shape == (2, 9, 9)
    for batch, order in enumerate(sampled):
        expected = example.state.copy()
        for action_index in order:
            action = actions[int(action_index)]
            expected[action.row, action.col] = action.value
        np.testing.assert_array_equal(boards[batch], expected)


def test_grid5_symbolic_probe_true_hamming_score_orders_exact_goal_best():
    world = SudokuWorld()
    example = world.example_from_strings(SUDOKU_PUZZLE, SUDOKU_SOLUTION)
    model = SigRegActionJEPA(vocab_size=world.vocab_size, latent_size=16)
    states = np.stack([example.goal, example.state])
    scores = score_states(
        model,
        states,
        example.goal,
        example.state,
        "true_hamming",
        torch.device("cpu"),
    )
    assert scores[0] == 0.0
    assert scores[1] > scores[0]


def test_grid5_planner_matrix_optimizers_return_valid_actions():
    torch.manual_seed(0)
    world = SudokuWorld()
    example = world.example_from_strings(SUDOKU_PUZZLE, SUDOKU_SOLUTION)
    clue_mask = world.clue_mask_from_puzzle(example.state)
    model = SigRegActionJEPA(
        vocab_size=world.vocab_size,
        latent_size=16,
        encoder_type="mlp",
        predictor_type="mlp",
        encoder_hidden_size=32,
        predictor_hidden_size=32,
        max_rollout_steps=4,
        sigreg_projections=8,
        sigreg_knots=4,
    )
    common = dict(
        model=model,
        world=world,
        board=example.state,
        goal_np=example.goal,
        initial_np=example.state,
        clue_mask=clue_mask,
        horizon=2,
        transition_mode="symbolic_reencode",
        score_mode="latent_goal",
        action_mode="mutable_overwrite",
        device=torch.device("cpu"),
    )
    beam = beam_search_plan_once(beam_width=2, branch_size=4, **common)
    mcts = mcts_plan_once(simulations=8, branch_size=4, exploration=1.0, **common)
    nn_cem = nearest_neighbor_cem_plan_once(
        candidates=8,
        elites=2,
        iterations=2,
        smoothing=0.7,
        seed=0,
        **common,
    )
    for plan in (beam, mcts, nn_cem):
        action = plan["action"]
        assert action is not None
        assert not clue_mask[action.row, action.col]
        assert action.value != int(example.state[action.row, action.col])
        assert np.isfinite(plan["leaf_score"])


def test_grid5_planner_matrix_writes_incremental_outputs(tmp_path):
    records = [{"optimizer": "beam", "solved": False, "remaining_hamming": 12}]
    summary = {"modes": {"beam_symbolic_reencode_latent_goal_h8": {"solves": 0.0}}}

    write_planner_outputs(tmp_path, records, summary)

    assert (tmp_path / "planner_records.jsonl").read_text().strip()
    assert "beam_symbolic_reencode_latent_goal_h8" in (tmp_path / "planner_summary.json").read_text()


def test_grid5_nearest_neighbor_decode_uses_action_embedding_space():
    world = SudokuWorld()
    example = world.example_from_strings(SUDOKU_PUZZLE, SUDOKU_SOLUTION)
    clue_mask = world.clue_mask_from_puzzle(example.state)
    model = SigRegActionJEPA(
        vocab_size=world.vocab_size,
        latent_size=16,
        encoder_hidden_size=32,
        predictor_hidden_size=32,
        sigreg_projections=8,
        sigreg_knots=4,
    )
    actions = candidate_actions(world, example.state, clue_mask)
    target = actions[5]
    embeddings = action_embedding_matrix(model, world, [target], torch.device("cpu"))
    decoded = decode_nearest_action(
        model,
        world,
        example.state,
        clue_mask,
        embeddings[0],
        action_mode="mutable_overwrite",
        device=torch.device("cpu"),
    )
    assert decoded is not None
    assert (decoded.row, decoded.col, decoded.value) == (target.row, target.col, target.value)


def test_grid5_closed_loop_runs_all_planner_axes():
    torch.manual_seed(0)
    world = SudokuWorld()
    example = world.example_from_strings(SUDOKU_PUZZLE, SUDOKU_SOLUTION)
    model = SigRegActionJEPA(
        vocab_size=world.vocab_size,
        latent_size=16,
        encoder_type="mlp",
        predictor_type="ar_transformer",
        encoder_hidden_size=32,
        predictor_hidden_size=32,
        predictor_layers=1,
        num_heads=4,
        max_rollout_steps=4,
        sigreg_projections=8,
        sigreg_knots=4,
    )
    for optimizer in ("beam", "mcts", "nn_cem"):
        for transition_mode in ("symbolic_reencode", "latent_rollout"):
            result = run_closed_loop(
                model,
                world,
                example,
                optimizer=optimizer,
                transition_mode=transition_mode,
                score_mode="goal_energy",
                action_mode="mutable_overwrite",
                horizon=2,
                max_steps=2,
                beam_width=2,
                branch_size=4,
                mcts_simulations=8,
                mcts_exploration=1.0,
                nn_cem_candidates=8,
                nn_cem_elites=2,
                nn_cem_iterations=2,
                nn_cem_smoothing=0.7,
                seed=0,
                device=torch.device("cpu"),
            )
            assert result["steps"] <= 2
            assert result["remaining_hamming"] >= 0
            assert isinstance(result["root_goal_value"], bool)
