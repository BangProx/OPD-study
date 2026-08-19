"""Clean-room educational OPD² objective from arXiv:2607.15161v1."""

from __future__ import annotations

import torch
from torch import Tensor

from opd_study.masks import shifted_causal_tensors
from opd_study.math import masked_mean
from opd_study.types import LossOutput, TeacherSignals, TrajectoryBatch


def _student_expectation(
    values: Tensor, student_log_probs: Tensor, top_k: int | None
) -> Tensor:
    if top_k is None:
        return (student_log_probs.exp() * values).sum(dim=-1)
    vocabulary_size = values.shape[-1]
    if not 1 <= top_k <= vocabulary_size:
        raise ValueError(f"top_k must be within [1, {vocabulary_size}]")
    indices = student_log_probs.topk(top_k, dim=-1).indices
    selected_student = student_log_probs.gather(-1, indices)
    selected_values = values.gather(-1, indices)
    renormalized = torch.softmax(selected_student, dim=-1)
    return (renormalized * selected_values).sum(dim=-1)


def opd2_loss(
    student_logits: Tensor,
    trajectories: TrajectoryBatch,
    teacher_signals: TeacherSignals,
    teacher_base_signals: TeacherSignals,
    *,
    centering_top_k: int | None = None,
) -> LossOutput:
    """Centered teacher-minus-base delta advantage with OPD-direction gating.

    This follows the paper's three essential choices: delta reward, action-independent
    centering, and the joint-sign condition.  It is intentionally not presented as a
    reproduction of the official large-scale TRL/GRPO recipe.
    """

    teacher = teacher_signals.logits
    teacher_base = teacher_base_signals.logits
    if teacher is None or teacher_base is None:
        raise ValueError("OPD² requires teacher and teacher-base logits")
    if teacher.shape != student_logits.shape or teacher_base.shape != student_logits.shape:
        raise ValueError("teacher, teacher-base and student logits must match")
    shifted_student, target_ids, _, prediction_mask = shifted_causal_tensors(
        student_logits,
        trajectories.token_ids,
        trajectories.attention_mask,
        trajectories.response_mask,
    )
    student_log = torch.log_softmax(shifted_student.float(), dim=-1)
    teacher_log = torch.log_softmax(teacher[:, :-1].float(), dim=-1)
    base_log = torch.log_softmax(teacher_base[:, :-1].float(), dim=-1)
    gather_index = target_ids.unsqueeze(-1)
    student_selected = student_log.gather(-1, gather_index).squeeze(-1)
    teacher_selected = teacher_log.gather(-1, gather_index).squeeze(-1)
    base_selected = base_log.gather(-1, gather_index).squeeze(-1)

    opd_rewards = teacher_log - student_log
    delta_rewards = teacher_log - base_log
    centered_opd = (
        teacher_selected
        - student_selected
        - _student_expectation(opd_rewards, student_log, centering_top_k)
    )
    centered_delta = (
        teacher_selected
        - base_selected
        - _student_expectation(delta_rewards, student_log, centering_top_k)
    )
    gate = (centered_delta.detach() * centered_opd.detach()) > 0
    advantage = torch.where(gate, centered_delta.detach(), torch.zeros_like(centered_delta))
    shifted_loss = -advantage * student_selected
    loss = masked_mean(shifted_loss, prediction_mask)
    token_loss = shifted_loss.new_zeros(trajectories.token_ids.shape)
    effective_mask = torch.zeros_like(trajectories.response_mask)
    token_loss[:, 1:] = shifted_loss
    effective_mask[:, 1:] = prediction_mask
    active = gate & prediction_mask
    return LossOutput(
        loss=loss,
        token_loss=token_loss,
        effective_mask=effective_mask,
        metrics={
            "opd2/loss": float(loss.detach().cpu()),
            "opd2/tokens": float(effective_mask.sum().cpu()),
            "opd2/gate_rate": float(active.sum().cpu() / prediction_mask.sum().cpu()),
            "opd2/mean_delta_advantage": float(
                masked_mean(centered_delta.detach(), prediction_mask).cpu()
            ),
        },
    )
