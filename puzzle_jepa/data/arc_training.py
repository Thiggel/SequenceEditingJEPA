from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from puzzle_jepa.data.arc import (
    ARCGrid,
    ARCEpisode,
    grid_distance,
    grid_exact,
    iter_leave_one_out_episodes,
    make_initial_arc_candidates,
)
from puzzle_jepa.data.arc_actions import (
    ARCAction,
    apply_arc_action,
    episode_candidate_shapes,
    episode_palette,
    generate_arc_actions,
)
from puzzle_jepa.data.arc_proposals import ARCProposal, build_arc_sources, extract_arc_proposals


ARC_ACTION_OPS = (
    "target",
    "initial",
    "set_canvas",
    "set_cell",
    "fill_bbox",
    "fill_mask",
    "complete_bbox_corners",
    "complete_rectangle",
    "copy_patch",
    "crop",
    "recolor",
    "translate",
    "reflect",
    "rotate",
    "partition_map",
    "scale_source",
    "scale_patch",
    "apply_color_map",
    "render_color_mask",
)
ARC_ACTION_OP_TO_ID = {name: index for index, name in enumerate(ARC_ACTION_OPS)}
ARC_FEATURE_DIM = len(ARC_ACTION_OPS) + 16


@dataclass(frozen=True, slots=True)
class ARCCandidateRecord:
    episode: ARCEpisode
    current: ARCGrid
    candidate: ARCGrid
    label: float
    action: ARCAction | None
    action_features: np.ndarray
    current_distance: int
    candidate_distance: int


@dataclass(frozen=True, slots=True)
class ARCBatch:
    context_inputs: torch.Tensor
    context_outputs: torch.Tensor
    context_mask: torch.Tensor
    query: torch.Tensor
    query_active: torch.Tensor
    candidate: torch.Tensor
    candidate_active: torch.Tensor
    current: torch.Tensor
    current_active: torch.Tensor
    labels: torch.Tensor
    action_features: torch.Tensor


def episodes_from_tasks(tasks) -> list[ARCEpisode]:
    episodes: list[ARCEpisode] = []
    for task in tasks:
        episodes.extend(iter_leave_one_out_episodes(task))
    return episodes


def sample_arc_candidate_record(
    episodes: list[ARCEpisode],
    rng: np.random.Generator,
    *,
    oracle_shape: bool = False,
    include_cell_actions: bool = True,
    max_actions: int = 800,
    positive_probability: float = 0.25,
    best_action_probability: float = 0.5,
) -> ARCCandidateRecord:
    if not episodes:
        raise ValueError("Cannot sample ARC records without episodes.")
    episode = episodes[int(rng.integers(0, len(episodes)))]
    initial_candidates = make_initial_arc_candidates(episode, oracle_shape=oracle_shape)
    current = initial_candidates[int(rng.integers(0, len(initial_candidates)))]
    current_distance = grid_distance(current, episode.target_output)
    if rng.random() < positive_probability:
        return ARCCandidateRecord(
            episode=episode,
            current=current,
            candidate=episode.target_output,
            label=1.0,
            action=None,
            action_features=arc_action_features(None, current=current, candidate=episode.target_output, proposals={}),
            current_distance=current_distance,
            candidate_distance=0,
        )

    proposals = extract_arc_proposals(episode.context, episode.query_input, current)
    sources = build_arc_sources(episode.context, episode.query_input, current)
    actions = generate_arc_actions(
        episode.context,
        episode.query_input,
        current,
        proposals=proposals,
        candidate_shapes=episode_candidate_shapes(
            episode.context,
            episode.query_input,
            oracle_shape=episode.target_output.shape if oracle_shape else None,
        ),
        palette=episode_palette(episode.context, episode.query_input, current),
        include_cell_actions=include_cell_actions,
        max_actions=max_actions,
    )
    candidates: list[tuple[int, ARCAction, ARCGrid]] = []
    for action in actions:
        try:
            next_grid = apply_arc_action(current, action, proposals=proposals, sources=sources)
        except ValueError:
            continue
        candidates.append((grid_distance(next_grid, episode.target_output), action, next_grid))
    if not candidates:
        return ARCCandidateRecord(
            episode=episode,
            current=current,
            candidate=current,
            label=0.0,
            action=ARCAction(op="initial", params={}, label="initial"),
            action_features=arc_action_features(ARCAction(op="initial", params={}, label="initial"), current=current, candidate=current, proposals={}),
            current_distance=current_distance,
            candidate_distance=current_distance,
        )
    candidates.sort(key=lambda item: item[0])
    if rng.random() < best_action_probability:
        distance, action, candidate = candidates[0]
    else:
        distance, action, candidate = candidates[int(rng.integers(0, len(candidates)))]
    return ARCCandidateRecord(
        episode=episode,
        current=current,
        candidate=candidate,
        label=1.0 if grid_exact(candidate, episode.target_output) else 0.0,
        action=action,
        action_features=arc_action_features(action, current=current, candidate=candidate, proposals=proposals),
        current_distance=current_distance,
        candidate_distance=distance,
    )


def collate_arc_records(
    records: list[ARCCandidateRecord],
    *,
    max_context: int = 4,
    device: str | torch.device = "cpu",
) -> ARCBatch:
    if not records:
        raise ValueError("Cannot collate an empty ARC record list.")
    context_inputs = []
    context_outputs = []
    context_masks = []
    queries = []
    query_active = []
    candidates = []
    candidate_active = []
    currents = []
    current_active = []
    labels = []
    features = []
    for record in records:
        cin = np.zeros((max_context, 30, 30), dtype=np.int64)
        cout = np.zeros((max_context, 30, 30), dtype=np.int64)
        cmask = np.zeros((max_context,), dtype=bool)
        for index, example in enumerate(record.episode.context[:max_context]):
            in_values, _ = example.input.padded()
            out_values, _ = example.output.padded() if example.output is not None else example.input.padded()
            cin[index] = in_values
            cout[index] = out_values
            cmask[index] = True
        q_values, q_active = record.episode.query_input.padded()
        cand_values, cand_active = record.candidate.padded()
        current_values, current_mask = record.current.padded()
        context_inputs.append(cin)
        context_outputs.append(cout)
        context_masks.append(cmask)
        queries.append(q_values)
        query_active.append(q_active)
        candidates.append(cand_values)
        candidate_active.append(cand_active)
        currents.append(current_values)
        current_active.append(current_mask)
        labels.append(float(record.label))
        features.append(record.action_features.astype(np.float32))
    return ARCBatch(
        context_inputs=torch.as_tensor(np.stack(context_inputs), dtype=torch.long, device=device),
        context_outputs=torch.as_tensor(np.stack(context_outputs), dtype=torch.long, device=device),
        context_mask=torch.as_tensor(np.stack(context_masks), dtype=torch.bool, device=device),
        query=torch.as_tensor(np.stack(queries), dtype=torch.long, device=device),
        query_active=torch.as_tensor(np.stack(query_active), dtype=torch.bool, device=device),
        candidate=torch.as_tensor(np.stack(candidates), dtype=torch.long, device=device),
        candidate_active=torch.as_tensor(np.stack(candidate_active), dtype=torch.bool, device=device),
        current=torch.as_tensor(np.stack(currents), dtype=torch.long, device=device),
        current_active=torch.as_tensor(np.stack(current_active), dtype=torch.bool, device=device),
        labels=torch.as_tensor(labels, dtype=torch.float32, device=device),
        action_features=torch.as_tensor(np.stack(features), dtype=torch.float32, device=device),
    )


def arc_action_features(
    action: ARCAction | None,
    *,
    current: ARCGrid,
    candidate: ARCGrid,
    proposals: dict[str, ARCProposal],
) -> np.ndarray:
    features = np.zeros((ARC_FEATURE_DIM,), dtype=np.float32)
    op = "target" if action is None else action.op
    features[ARC_ACTION_OP_TO_ID.get(op, 0)] = 1.0
    offset = len(ARC_ACTION_OPS)
    features[offset + 0] = current.height / 30.0
    features[offset + 1] = current.width / 30.0
    features[offset + 2] = candidate.height / 30.0
    features[offset + 3] = candidate.width / 30.0
    features[offset + 4] = len(current.color_set()) / 10.0
    features[offset + 5] = len(candidate.color_set()) / 10.0
    if action is not None:
        params = action.params
        proposal = proposals.get(str(params.get("proposal_id", "")))
        if proposal is not None:
            row0, col0, row1, col1 = proposal.bbox
            features[offset + 6] = proposal.area / 900.0
            features[offset + 7] = row0 / 30.0
            features[offset + 8] = col0 / 30.0
            features[offset + 9] = row1 / 30.0
            features[offset + 10] = col1 / 30.0
            features[offset + 11] = len(proposal.colors) / 10.0
        for key, slot in (("color", 12), ("to_color", 12), ("from_color", 13), ("factor", 14)):
            if key in params:
                features[offset + slot] = float(params[key]) / (10.0 if key != "factor" else 4.0)
        if "dr" in params or "dc" in params:
            features[offset + 15] = (float(params.get("dr", 0)) + float(params.get("dc", 0))) / 30.0
    return features
