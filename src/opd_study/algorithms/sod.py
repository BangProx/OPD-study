"""Step-wise OPD weighting from SOD (arXiv:2605.07725v3)."""

from __future__ import annotations

import torch
from torch import Tensor

from opd_study.masks import shifted_causal_tensors
from opd_study.math import forward_kl_from_logits
from opd_study.types import LossOutput, TeacherSignals, TrajectoryBatch


def step_divergence_weights(
    student_logits: Tensor,
    teacher_logits: Tensor,
    token_ids: Tensor,
    response_mask: Tensor,
    step_ids: Tensor,
    *,
    epsilon: float = 1e-6,
    delta: float = 0.2,
) -> tuple[Tensor, Tensor]:
    """Compute detached SOD ``d_k`` and cumulative-ratio ``w_k`` per token.

    ``d_k`` is the mean absolute sampled-token log-probability gap within a step.
    ``w_k = min(prod_{u<k}(d_u+eps)/(d_{u+1}+eps), 1+delta)``.
    """

    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must match")
    if token_ids.shape != response_mask.shape or token_ids.shape != step_ids.shape:
        raise ValueError("token_ids, response_mask and step_ids must match")
    if epsilon <= 0 or delta < 0:
        raise ValueError("epsilon must be positive and delta non-negative")
    student_log = torch.log_softmax(student_logits.float(), dim=-1)
    teacher_log = torch.log_softmax(teacher_logits.float(), dim=-1)
    sampled_student = student_log[:, :-1].gather(
        -1, token_ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)
    sampled_teacher = teacher_log[:, :-1].gather(
        -1, token_ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)
    gaps = (sampled_student - sampled_teacher).abs().detach()
    shifted_steps = step_ids[:, 1:]
    shifted_mask = response_mask[:, 1:] & (shifted_steps >= 0)
    token_weights = torch.zeros_like(gaps)
    token_divergences = torch.zeros_like(gaps)
    for batch_index in range(token_ids.shape[0]):
        ids = torch.unique(shifted_steps[batch_index][shifted_mask[batch_index]], sorted=True)
        previous_divergence: Tensor | None = None
        cumulative = gaps.new_tensor(1.0)
        for step_id in ids.tolist():
            mask = shifted_mask[batch_index] & (shifted_steps[batch_index] == step_id)
            divergence = gaps[batch_index][mask].mean()
            if previous_divergence is not None:
                cumulative = cumulative * (previous_divergence + epsilon) / (
                    divergence + epsilon
                )
            weight = torch.minimum(cumulative, gaps.new_tensor(1.0 + delta))
            token_divergences[batch_index][mask] = divergence
            token_weights[batch_index][mask] = weight
            previous_divergence = divergence
    return token_divergences.detach(), token_weights.detach()


def sod_loss(
    student_logits: Tensor,
    trajectories: TrajectoryBatch,
    signals: TeacherSignals,
    *,
    epsilon: float = 1e-6,
    delta: float = 0.2,
) -> LossOutput:
    """SOD's step-weighted distillation term (without its separate GRPO term)."""

    if trajectories.step_ids is None:
        raise ValueError("SOD requires step_ids")
    if signals.logits is None or signals.logits.shape != student_logits.shape:
        raise ValueError("SOD requires matching teacher logits")
    divergences, weights = step_divergence_weights(
        student_logits,
        signals.logits,
        trajectories.token_ids,
        trajectories.response_mask,
        trajectories.step_ids,
        epsilon=epsilon,
        delta=delta,
    )
    _, _, _, prediction_mask = shifted_causal_tensors(
        student_logits,
        trajectories.token_ids,
        trajectories.attention_mask,
        trajectories.response_mask,
    )
    shifted_loss = forward_kl_from_logits(
        signals.logits[:, :-1], student_logits[:, :-1]
    )
    effective_weights = weights * prediction_mask
    denominator = effective_weights.sum()
    if denominator.item() <= 0:
        raise ValueError("SOD has no positively weighted response tokens")
    loss = (shifted_loss * effective_weights).sum() / denominator
    token_loss = shifted_loss.new_zeros(trajectories.token_ids.shape)
    full_weights = shifted_loss.new_zeros(trajectories.token_ids.shape)
    token_loss[:, 1:] = shifted_loss
    full_weights[:, 1:] = effective_weights
    active_divergences = divergences[prediction_mask]
    return LossOutput(
        loss=loss,
        token_loss=token_loss,
        effective_mask=full_weights,
        metrics={
            "sod/loss": float(loss.detach().cpu()),
            "sod/tokens": float(prediction_mask.sum().cpu()),
            "sod/mean_step_divergence": float(active_divergences.mean().cpu()),
            "sod/mean_weight": float(effective_weights[prediction_mask].mean().cpu()),
            "sod/distillation_only": 1.0,
        },
    )
