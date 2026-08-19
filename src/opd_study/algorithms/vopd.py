"""Clean-room vOPD estimator from arXiv:2605.07865v1, Eqs. 9--15."""

from __future__ import annotations

import torch
from torch import Tensor

from opd_study.masks import shifted_causal_tensors
from opd_study.math import masked_mean, reverse_kl_from_logits
from opd_study.types import LossOutput, TeacherSignals, TrajectoryBatch


def _top_k_reverse_kl(
    teacher_logits: Tensor, student_logits: Tensor, top_k: int
) -> Tensor:
    vocabulary_size = student_logits.shape[-1]
    if not 1 <= top_k <= vocabulary_size:
        raise ValueError(f"top_k must be within [1, {vocabulary_size}]")
    indices = student_logits.topk(top_k, dim=-1).indices
    student_selected = student_logits.gather(-1, indices)
    teacher_selected = teacher_logits.gather(-1, indices)
    return reverse_kl_from_logits(teacher_selected, student_selected)


def vopd_loss(
    student_logits: Tensor,
    trajectories: TrajectoryBatch,
    signals: TeacherSignals,
    *,
    baseline_top_k: int | None = None,
) -> LossOutput:
    """Sampled OPD with a detached, action-independent negative-RKL baseline.

    The sampled reward is ``log p_teacher(y) - log p_student(y)``.  The OPD value
    function is its student expectation, i.e. negative reverse KL.  Therefore the
    advantage adds reverse KL to the reward.  Only sampled-token log-probabilities
    receive gradients; the full/top-k baseline is detached.
    """

    if signals.logits is None or signals.logits.shape != student_logits.shape:
        raise ValueError("vOPD requires detached teacher logits matching student logits")
    shifted_student, target_ids, _, prediction_mask = shifted_causal_tensors(
        student_logits,
        trajectories.token_ids,
        trajectories.attention_mask,
        trajectories.response_mask,
    )
    shifted_teacher = signals.logits[:, :-1]
    student_log = torch.log_softmax(shifted_student.float(), dim=-1)
    teacher_log = torch.log_softmax(shifted_teacher.float(), dim=-1)
    student_selected = student_log.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    teacher_selected = teacher_log.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    reward = (teacher_selected - student_selected).detach()
    if baseline_top_k is None:
        reverse_kl = reverse_kl_from_logits(shifted_teacher, shifted_student)
    else:
        reverse_kl = _top_k_reverse_kl(
            shifted_teacher, shifted_student, baseline_top_k
        )
    advantage = (reward + reverse_kl.detach()).detach()
    shifted_loss = -advantage * student_selected
    loss = masked_mean(shifted_loss, prediction_mask)
    token_loss = shifted_loss.new_zeros(trajectories.token_ids.shape)
    effective_mask = torch.zeros_like(trajectories.response_mask)
    token_loss[:, 1:] = shifted_loss
    effective_mask[:, 1:] = prediction_mask
    return LossOutput(
        loss=loss,
        token_loss=token_loss,
        effective_mask=effective_mask,
        metrics={
            "vopd/loss": float(loss.detach().cpu()),
            "vopd/tokens": float(effective_mask.sum().cpu()),
            "vopd/mean_reward": float(masked_mean(reward, prediction_mask).cpu()),
            "vopd/mean_advantage": float(masked_mean(advantage, prediction_mask).cpu()),
            "vopd/baseline_top_k": float(
                student_logits.shape[-1] if baseline_top_k is None else baseline_top_k
            ),
        },
    )
