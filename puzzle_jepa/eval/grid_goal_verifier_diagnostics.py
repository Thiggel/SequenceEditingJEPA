from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from puzzle_jepa.data.worlds import PuzzleExample, WorldAction
from puzzle_jepa.eval.grid_goal_planner_matrix import load_checkpoint, load_eval_examples
from puzzle_jepa.models.grid_goal_jepa import GridTokenGoalJEPA
from puzzle_jepa.planning.grid_goal_planner import ACTION_VOCAB


@torch.no_grad()
def run_verifier_diagnostics(
    model: GridTokenGoalJEPA,
    examples: list[PuzzleExample],
    *,
    device: torch.device,
    seed: int = 0,
    max_examples: int = 16,
    max_actions: int = 128,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    compatibility_pos = []
    compatibility_neg = []
    remaining_pred = []
    remaining_true = []
    action_top1 = []
    action_top5 = []
    action_true_score_gap = []
    rollout_score_gap = []
    for example in examples[:max_examples]:
        puzzle = np.asarray(example.state, dtype=np.int64)
        goal = np.asarray(example.goal, dtype=np.int64)
        clue_mask = puzzle != 0
        editable_mask = ~clue_mask
        active_mask = np.ones_like(clue_mask, dtype=bool)
        context = _encode_context(model, puzzle, clue_mask, editable_mask, active_mask, device=device)
        positives = [
            puzzle,
            _solution_subset_board(puzzle, goal, editable_mask, rng, fill_fraction=0.35),
            _solution_subset_board(puzzle, goal, editable_mask, rng, fill_fraction=0.70),
            goal,
        ]
        negatives = [_corrupt_one(board, goal, editable_mask) for board in positives]
        pos_scores, pos_remaining = _score_boards(model, positives, context, clue_mask, editable_mask, active_mask, device)
        neg_scores, neg_remaining = _score_boards(model, negatives, context, clue_mask, editable_mask, active_mask, device)
        compatibility_pos.extend(pos_scores.tolist())
        compatibility_neg.extend(neg_scores.tolist())
        remaining_pred.extend(pos_remaining.tolist())
        remaining_pred.extend(neg_remaining.tolist())
        remaining_true.extend([_remaining_count(board, goal, editable_mask) for board in positives])
        remaining_true.extend([_remaining_count(board, goal, editable_mask) for board in negatives])

        parent = _solution_subset_board(puzzle, goal, editable_mask, rng, fill_fraction=0.45)
        acc = _successor_action_accuracy(
            model,
            parent,
            goal,
            context,
            clue_mask,
            editable_mask,
            active_mask,
            device=device,
            max_actions=max_actions,
        )
        if acc is not None:
            action_top1.append(acc["top1"])
            action_top5.append(acc["top5"])
            action_true_score_gap.append(acc["true_score_gap"])
            rollout_score_gap.append(acc["rollout_score_gap"])
    return {
        "compatibility_auc": _pairwise_auc_less(np.asarray(compatibility_pos), np.asarray(compatibility_neg)),
        "compatibility_pos_mean": float(np.mean(compatibility_pos)) if compatibility_pos else 0.0,
        "compatibility_neg_mean": float(np.mean(compatibility_neg)) if compatibility_neg else 0.0,
        "remaining_mae": float(np.mean(np.abs(np.asarray(remaining_pred) - np.asarray(remaining_true))))
        if remaining_pred
        else 0.0,
        "remaining_spearman": _spearman(np.asarray(remaining_pred), np.asarray(remaining_true)),
        "successor_top1": float(np.mean(action_top1)) if action_top1 else 0.0,
        "successor_top5": float(np.mean(action_top5)) if action_top5 else 0.0,
        "successor_true_score_gap": float(np.mean(action_true_score_gap)) if action_true_score_gap else 0.0,
        "successor_rollout_score_gap": float(np.mean(rollout_score_gap)) if rollout_score_gap else 0.0,
        "examples": float(min(max_examples, len(examples))),
    }


def _encode_context(
    model: GridTokenGoalJEPA,
    puzzle: np.ndarray,
    clue_mask: np.ndarray,
    editable_mask: np.ndarray,
    active_mask: np.ndarray,
    *,
    device: torch.device,
) -> torch.Tensor:
    return model.encode_context(
        torch.as_tensor(puzzle[None], dtype=torch.long, device=device),
        torch.as_tensor(clue_mask[None], dtype=torch.bool, device=device),
        torch.as_tensor(editable_mask[None], dtype=torch.bool, device=device),
        torch.as_tensor(active_mask[None], dtype=torch.bool, device=device),
    )


def _score_boards(
    model: GridTokenGoalJEPA,
    boards: list[np.ndarray],
    context: torch.Tensor,
    clue_mask: np.ndarray,
    editable_mask: np.ndarray,
    active_mask: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    board_t = torch.as_tensor(np.stack(boards), dtype=torch.long, device=device)
    count = board_t.shape[0]
    context_t = context.expand(count, -1, -1)
    clue_t = torch.as_tensor(clue_mask[None], dtype=torch.bool, device=device).expand(count, -1, -1)
    edit_t = torch.as_tensor(editable_mask[None], dtype=torch.bool, device=device).expand(count, -1, -1)
    active_t = torch.as_tensor(active_mask[None], dtype=torch.bool, device=device).expand(count, -1, -1)
    latents = model.encode_state(board_t, context_t, clue_t, edit_t, active_t)
    return (
        model.compatibility_energy(latents, active_t).detach().cpu().numpy(),
        model.remaining_edit_count(latents, active_t).detach().cpu().numpy(),
    )


def _successor_action_accuracy(
    model: GridTokenGoalJEPA,
    board: np.ndarray,
    goal: np.ndarray,
    context: torch.Tensor,
    clue_mask: np.ndarray,
    editable_mask: np.ndarray,
    active_mask: np.ndarray,
    *,
    device: torch.device,
    max_actions: int,
) -> dict[str, float] | None:
    editable_actions = [action for action in ACTION_VOCAB if editable_mask[action.row, action.col]]
    corrective_actions = [
        action
        for action in editable_actions
        if (board[action.row, action.col] != goal[action.row, action.col])
        and action.value == int(goal[action.row, action.col])
    ]
    negative_actions = [action for action in editable_actions if action not in corrective_actions]
    actions = [*corrective_actions, *negative_actions[: max(0, max_actions - len(corrective_actions))]]
    corrective = np.asarray(
        [
            (board[action.row, action.col] != goal[action.row, action.col])
            and action.value == int(goal[action.row, action.col])
            for action in actions
        ],
        dtype=bool,
    )
    if not bool(corrective.any()):
        return None
    parent_t = torch.as_tensor(board[None], dtype=torch.long, device=device)
    active_t = torch.as_tensor(active_mask[None], dtype=torch.bool, device=device)
    clue_t = torch.as_tensor(clue_mask[None], dtype=torch.bool, device=device)
    edit_t = torch.as_tensor(editable_mask[None], dtype=torch.bool, device=device)
    parent = model.encode_state(parent_t, context, clue_t, edit_t, active_t).expand(len(actions), -1, -1)
    action_t = torch.as_tensor([[a.row, a.col, a.value] for a in actions], dtype=torch.long, device=device)
    context_t = context.expand(len(actions), -1, -1)
    active_batch = active_t.expand(len(actions), -1, -1)
    pred = model.predict_next(parent, action_t, context_t)
    scores = model.verifier_score(pred, active_batch).detach().cpu().numpy()
    true_scores = np.asarray([_true_successor_score(board, goal, editable_mask, action, model) for action in actions])
    predicted_order = np.argsort(scores)
    true_order = np.argsort(true_scores)
    return {
        "top1": float(corrective[predicted_order[0]]),
        "top5": float(bool(corrective[predicted_order[:5]].any())),
        "true_score_gap": float(true_scores[predicted_order[0]] - true_scores[true_order[0]]),
        "rollout_score_gap": float(scores[predicted_order[0]] - scores[true_order[0]]),
    }


def _solution_subset_board(
    puzzle: np.ndarray,
    goal: np.ndarray,
    editable_mask: np.ndarray,
    rng: np.random.Generator,
    *,
    fill_fraction: float,
) -> np.ndarray:
    board = puzzle.copy()
    positions = np.argwhere(editable_mask)
    count = int(round(float(fill_fraction) * len(positions)))
    if count > 0:
        chosen = rng.choice(len(positions), size=count, replace=False)
        for index in np.atleast_1d(chosen):
            row, col = positions[int(index)]
            board[int(row), int(col)] = int(goal[int(row), int(col)])
    return board


def _corrupt_one(board: np.ndarray, goal: np.ndarray, editable_mask: np.ndarray) -> np.ndarray:
    out = board.copy()
    candidates = np.argwhere((board != 0) & editable_mask)
    if len(candidates) == 0:
        candidates = np.argwhere(editable_mask)
    row, col = (int(x) for x in candidates[0])
    out[row, col] = int(goal[row, col] % 9) + 1
    return out


def _true_successor_score(
    board: np.ndarray,
    goal: np.ndarray,
    editable_mask: np.ndarray,
    action: WorldAction,
    model: GridTokenGoalJEPA,
) -> float:
    successor = board.copy()
    successor[action.row, action.col] = action.value
    wrong = int(np.count_nonzero((successor != 0) & (successor != goal) & editable_mask))
    remaining = _remaining_count(successor, goal, editable_mask)
    return float(model.verifier_score_alpha * wrong + model.verifier_score_beta * remaining)


def _remaining_count(board: np.ndarray, goal: np.ndarray, editable_mask: np.ndarray) -> int:
    return int(np.count_nonzero((board != goal) & editable_mask))


def _pairwise_auc_less(positive: np.ndarray, negative: np.ndarray) -> float:
    if positive.size == 0 or negative.size == 0:
        return 0.0
    return float((positive[:, None] < negative[None, :]).mean())


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return 0.0
    ar = _rankdata(a)
    br = _rankdata(b)
    denom = float(np.std(ar) * np.std(br))
    if denom <= 0.0:
        return 0.0
    return float(np.mean((ar - ar.mean()) * (br - br.mean())) / denom)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=16)
    parser.add_argument("--max-actions", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_checkpoint(args.checkpoint, device)
    examples = load_eval_examples(config, limit=args.examples)
    metrics = run_verifier_diagnostics(
        model,
        examples,
        device=device,
        seed=int(args.seed),
        max_examples=int(args.examples),
        max_actions=int(args.max_actions),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(json.dumps(metrics, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
