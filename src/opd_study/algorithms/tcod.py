"""Temporal curriculum masks from TCOD (arXiv:2604.24005v3)."""

from __future__ import annotations

import torch
from torch import Tensor

from opd_study.masks import shifted_causal_tensors
from opd_study.math import forward_kl_from_logits
from opd_study.types import LossOutput, TeacherSignals, TrajectoryBatch


def curriculum_depth(
    optimizer_step: int, *, start_depth: int, pacing_steps: int, maximum_depth: int
) -> int:
    """Return ``min(start + floor(step/pacing), maximum)``."""

    if optimizer_step < 0:
        raise ValueError("optimizer_step must be non-negative")
    if min(start_depth, pacing_steps, maximum_depth) < 1:
        raise ValueError("curriculum depths and pacing_steps must be positive")
    if start_depth > maximum_depth:
        raise ValueError("start_depth cannot exceed maximum_depth")
    return min(start_depth + optimizer_step // pacing_steps, maximum_depth)


def temporal_curriculum_mask(
    trajectories: TrajectoryBatch, *, depth: int, direction: str
) -> Tensor:
    """Select early F2B or late B2F turns from a correctly sourced trajectory.

    For B2F, callers must supply trajectories whose earlier history came from the
    teacher/successful prefix.  A mask alone cannot manufacture that state distribution.
    """

    if trajectories.turn_ids is None:
        raise ValueError("TCOD requires turn_ids")
    if depth < 1:
        raise ValueError("depth must be positive")
    turns = trajectories.turn_ids
    valid = trajectories.response_mask & torch.ge(turns, 0)
    if not valid.any().item():
        raise ValueError("TCOD found no response turns")
    if direction == "f2b":
        return valid & torch.lt(turns, depth)
    if direction == "b2f":
        maximum_turn = torch.where(valid, turns, torch.full_like(turns, -1)).amax(
            dim=1, keepdim=True
        )
        return valid & torch.gt(turns, maximum_turn - depth)
    raise ValueError("direction must be 'f2b' or 'b2f'")


def tcod_loss(
    student_logits: Tensor,
    trajectories: TrajectoryBatch,
    signals: TeacherSignals,
    *,
    depth: int,
    direction: str,
) -> LossOutput:
    if signals.logits is None or signals.logits.shape != student_logits.shape:
        raise ValueError("TCOD requires matching teacher logits")
    curriculum_mask = temporal_curriculum_mask(
        trajectories, depth=depth, direction=direction
    )
    _, _, _, causal_response_mask = shifted_causal_tensors(
        student_logits,
        trajectories.token_ids,
        trajectories.attention_mask,
        trajectories.response_mask,
    )
    effective_shifted = causal_response_mask & curriculum_mask[:, 1:]
    shifted_loss = forward_kl_from_logits(
        signals.logits[:, :-1], student_logits[:, :-1]
    )
    denominator = effective_shifted.sum()
    if denominator.item() == 0:
        raise ValueError("the selected TCOD curriculum window contains no tokens")
    loss = (shifted_loss * effective_shifted).sum() / denominator
    token_loss = shifted_loss.new_zeros(trajectories.token_ids.shape)
    effective_mask = torch.zeros_like(trajectories.response_mask)
    token_loss[:, 1:] = shifted_loss
    effective_mask[:, 1:] = effective_shifted
    return LossOutput(
        loss=loss,
        token_loss=token_loss,
        effective_mask=effective_mask,
        metrics={
            f"tcod_{direction}/loss": float(loss.detach().cpu()),
            f"tcod_{direction}/tokens": float(effective_mask.sum().cpu()),
            f"tcod_{direction}/depth": float(depth),
        },
    )
