"""Small, equation-shaped implementations of SFT, KD, GKD and OPD losses."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from opd_study.masks import shifted_causal_tensors
from opd_study.math import (
    cross_entropy_from_logits,
    forward_kl_from_logits,
    generalized_jsd_from_logits,
    masked_mean,
    reverse_kl_from_logits,
)
from opd_study.types import LossOutput, TeacherSignals, TrajectoryBatch


def _restore_token_alignment(
    shifted_values: Tensor, shifted_mask: Tensor, sequence_length: int
) -> tuple[Tensor, Tensor]:
    """Place next-token losses at the target-token positions in ``[B, T]``."""

    batch_size = shifted_values.shape[0]
    token_values = shifted_values.new_zeros((batch_size, sequence_length))
    effective_mask = torch.zeros(
        (batch_size, sequence_length), dtype=torch.bool, device=shifted_mask.device
    )
    token_values[:, 1:] = shifted_values
    effective_mask[:, 1:] = shifted_mask
    return token_values, effective_mask


def _metrics(loss: Tensor, mask: Tensor, *, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}/loss": float(loss.detach().cpu()),
        f"{prefix}/tokens": float(mask.sum().detach().cpu()),
    }


def supervised_fine_tuning_loss(
    student_logits: Tensor,
    trajectories: TrajectoryBatch,
) -> LossOutput:
    """Hard-label next-token cross entropy on demonstration response tokens."""

    shifted_logits, target_ids, _, prediction_mask = shifted_causal_tensors(
        student_logits,
        trajectories.token_ids,
        trajectories.attention_mask,
        trajectories.response_mask,
    )
    shifted_loss = cross_entropy_from_logits(shifted_logits, target_ids)
    loss = masked_mean(shifted_loss, prediction_mask)
    token_loss, effective_mask = _restore_token_alignment(
        shifted_loss, prediction_mask, trajectories.token_ids.shape[1]
    )
    return LossOutput(
        loss=loss,
        token_loss=token_loss,
        effective_mask=effective_mask,
        metrics=_metrics(loss, effective_mask, prefix="sft"),
    )


def _teacher_shifted_logits(
    signals: TeacherSignals, trajectories: TrajectoryBatch, student_logits: Tensor
) -> Tensor:
    if signals.logits is None:
        raise ValueError("full-distribution distillation requires teacher logits")
    if signals.logits.shape != student_logits.shape:
        raise ValueError(
            "teacher and student logits must have the same [B, T, V] shape; "
            f"got {signals.logits.shape} and {student_logits.shape}"
        )
    if signals.logits.requires_grad:
        raise ValueError("teacher logits must be detached")
    return signals.logits[:, :-1]


def off_policy_kd_loss(
    student_logits: Tensor,
    trajectories: TrajectoryBatch,
    signals: TeacherSignals,
    *,
    divergence: str = "forward_kl",
    temperature: float = 1.0,
) -> LossOutput:
    """Distribution matching on fixed demonstration states.

    ``temperature**2`` is applied, as in classical KD, so changing temperature does
    not trivially shrink gradients.  At temperature 1 it is an exact no-op.
    """

    shifted_student, _, _, prediction_mask = shifted_causal_tensors(
        student_logits,
        trajectories.token_ids,
        trajectories.attention_mask,
        trajectories.response_mask,
    )
    shifted_teacher = _teacher_shifted_logits(signals, trajectories, student_logits)
    if divergence == "forward_kl":
        shifted_loss = forward_kl_from_logits(
            shifted_teacher, shifted_student, temperature=temperature
        )
    elif divergence == "reverse_kl":
        shifted_loss = reverse_kl_from_logits(
            shifted_teacher, shifted_student, temperature=temperature
        )
    else:
        raise ValueError("divergence must be 'forward_kl' or 'reverse_kl'")
    shifted_loss = torch.mul(shifted_loss, temperature**2)
    loss = masked_mean(shifted_loss, prediction_mask)
    token_loss, effective_mask = _restore_token_alignment(
        shifted_loss, prediction_mask, trajectories.token_ids.shape[1]
    )
    return LossOutput(
        loss=loss,
        token_loss=token_loss,
        effective_mask=effective_mask,
        metrics=_metrics(loss, effective_mask, prefix="off_policy_kd"),
    )


def generalized_kd_loss(
    student_logits: Tensor,
    trajectories: TrajectoryBatch,
    signals: TeacherSignals,
    *,
    beta: float = 0.5,
    temperature: float = 1.0,
) -> LossOutput:
    """Generalized JSD objective from GKD on whatever states were collected.

    State-source mixing (the GKD lambda) belongs in collection, not in this loss.
    Keeping those mechanisms separate makes on-policy/off-policy ablations auditable.
    """

    shifted_student, _, _, prediction_mask = shifted_causal_tensors(
        student_logits,
        trajectories.token_ids,
        trajectories.attention_mask,
        trajectories.response_mask,
    )
    shifted_teacher = _teacher_shifted_logits(signals, trajectories, student_logits)
    shifted_loss = torch.mul(
        generalized_jsd_from_logits(
            shifted_teacher,
            shifted_student,
            beta=beta,
            temperature=temperature,
        ),
        temperature**2,
    )
    loss = masked_mean(shifted_loss, prediction_mask)
    token_loss, effective_mask = _restore_token_alignment(
        shifted_loss, prediction_mask, trajectories.token_ids.shape[1]
    )
    return LossOutput(
        loss=loss,
        token_loss=token_loss,
        effective_mask=effective_mask,
        metrics={
            **_metrics(loss, effective_mask, prefix="gkd"),
            "gkd/beta": beta,
        },
    )


def sampled_reverse_kl_loss(
    student_logits: Tensor,
    trajectories: TrajectoryBatch,
    signals: TeacherSignals,
) -> LossOutput:
    """Score-function estimate of reverse KL on student-sampled response tokens.

    For ``y ~ p_student``, the detached advantage is
    ``log p_student(y) - log p_teacher(y)``.  Multiplying it by the current student
    log-probability gives the reverse-KL policy gradient in expectation.  This is a
    high-variance estimator; full-distribution OPD is the recommended toy default.
    """

    if signals.logits is None:
        raise ValueError("sampled reverse KL requires teacher logits")
    shifted_student, target_ids, _, prediction_mask = shifted_causal_tensors(
        student_logits,
        trajectories.token_ids,
        trajectories.attention_mask,
        trajectories.response_mask,
    )
    shifted_teacher = _teacher_shifted_logits(signals, trajectories, student_logits)
    student_selected = torch.log_softmax(shifted_student.float(), dim=-1).gather(
        -1, target_ids.unsqueeze(-1)
    ).squeeze(-1)
    teacher_selected = torch.log_softmax(shifted_teacher.float(), dim=-1).gather(
        -1, target_ids.unsqueeze(-1)
    ).squeeze(-1)
    advantage = (student_selected - teacher_selected).detach()
    if trajectories.student_logprobs is not None:
        old_selected = trajectories.student_logprobs[:, 1:]
        if not torch.isfinite(old_selected[prediction_mask]).all().item():
            raise ValueError("rollout-time student log-probabilities must be finite")
    shifted_loss = advantage * student_selected
    loss = masked_mean(shifted_loss, prediction_mask)
    token_loss, effective_mask = _restore_token_alignment(
        shifted_loss, prediction_mask, trajectories.token_ids.shape[1]
    )
    return LossOutput(
        loss=loss,
        token_loss=token_loss,
        effective_mask=effective_mask,
        metrics={
            **_metrics(loss, effective_mask, prefix="sampled_reverse_kl"),
            "sampled_reverse_kl/mean_advantage": float(
                masked_mean(advantage, prediction_mask).detach().cpu()
            ),
        },
    )


def on_policy_distillation_loss(
    student_logits: Tensor,
    trajectories: TrajectoryBatch,
    signals: TeacherSignals,
    *,
    estimator: str = "full_reverse_kl",
    beta: float = 1.0,
    temperature: float = 1.0,
) -> LossOutput:
    """OPD objective evaluated on trajectories sampled from the current student."""

    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if estimator == "full_reverse_kl":
        return generalized_kd_loss(
            student_logits,
            trajectories,
            signals,
            beta=beta,
            temperature=temperature,
        )
    if estimator == "sampled_reverse_kl":
        if temperature != 1.0:
            raise ValueError("sampled_reverse_kl currently requires temperature=1")
        return sampled_reverse_kl_loss(student_logits, trajectories, signals)
    raise ValueError("estimator must be 'full_reverse_kl' or 'sampled_reverse_kl'")
