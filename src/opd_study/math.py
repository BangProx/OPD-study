"""Numerically stable categorical losses used throughout the course.

The public functions name the teacher and student arguments explicitly.  This avoids
the most common OPD implementation bug: silently reversing the KL direction.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def _validate_logits(teacher_logits: Tensor, student_logits: Tensor) -> None:
    if teacher_logits.shape != student_logits.shape:
        raise ValueError(
            "teacher_logits and student_logits must have the same shape; "
            f"got {tuple(teacher_logits.shape)} and {tuple(student_logits.shape)}"
        )
    if teacher_logits.ndim < 1:
        raise ValueError("logits must have at least a vocabulary dimension")


def log_probs(logits: Tensor, *, temperature: float = 1.0) -> Tensor:
    """Return float32 log probabilities along the last (vocabulary) dimension."""

    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"temperature must be finite and > 0, got {temperature!r}")
    return F.log_softmax(logits.float() / temperature, dim=-1)


def entropy_from_logits(logits: Tensor, *, temperature: float = 1.0) -> Tensor:
    """Compute categorical entropy, preserving all dimensions except vocabulary."""

    log_p = log_probs(logits, temperature=temperature)
    return -(log_p.exp() * log_p).sum(dim=-1)


def cross_entropy_from_logits(
    student_logits: Tensor,
    target_ids: Tensor,
    *,
    temperature: float = 1.0,
) -> Tensor:
    """Compute unreduced hard-label cross entropy."""

    if student_logits.shape[:-1] != target_ids.shape:
        raise ValueError(
            "target_ids must match every non-vocabulary logits dimension; "
            f"got {tuple(student_logits.shape)} and {tuple(target_ids.shape)}"
        )
    log_p = log_probs(student_logits, temperature=temperature)
    return -log_p.gather(dim=-1, index=target_ids.long().unsqueeze(-1)).squeeze(-1)


def forward_kl_from_logits(
    teacher_logits: Tensor,
    student_logits: Tensor,
    *,
    temperature: float = 1.0,
) -> Tensor:
    """Return ``KL(teacher || student)`` without reducing token/batch dimensions."""

    _validate_logits(teacher_logits, student_logits)
    teacher_log_p = log_probs(teacher_logits, temperature=temperature)
    student_log_p = log_probs(student_logits, temperature=temperature)
    return (teacher_log_p.exp() * (teacher_log_p - student_log_p)).sum(dim=-1)


def reverse_kl_from_logits(
    teacher_logits: Tensor,
    student_logits: Tensor,
    *,
    temperature: float = 1.0,
) -> Tensor:
    """Return ``KL(student || teacher)`` without reducing token/batch dimensions."""

    _validate_logits(teacher_logits, student_logits)
    teacher_log_p = log_probs(teacher_logits, temperature=temperature)
    student_log_p = log_probs(student_logits, temperature=temperature)
    return (student_log_p.exp() * (student_log_p - teacher_log_p)).sum(dim=-1)


def generalized_jsd_from_logits(
    teacher_logits: Tensor,
    student_logits: Tensor,
    *,
    beta: float = 0.5,
    temperature: float = 1.0,
) -> Tensor:
    """Return the GKD generalized JSD with explicit KL boundary behavior.

    This follows Eq. (1) of Agarwal et al. (arXiv:2306.13649v3).  The course
    convention matches the official TRL reference: ``beta=0`` is forward KL,
    ``beta=1`` is reverse KL, and intermediate values compare teacher/student to
    ``beta * teacher + (1-beta) * student``.
    """

    _validate_logits(teacher_logits, student_logits)
    if not math.isfinite(beta) or not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must be finite and in [0, 1], got {beta!r}")
    if beta == 0.0:
        return forward_kl_from_logits(
            teacher_logits, student_logits, temperature=temperature
        )
    if beta == 1.0:
        return reverse_kl_from_logits(
            teacher_logits, student_logits, temperature=temperature
        )

    teacher_log_p = log_probs(teacher_logits, temperature=temperature)
    student_log_p = log_probs(student_logits, temperature=temperature)
    beta_tensor = torch.as_tensor(beta, dtype=teacher_log_p.dtype, device=teacher_log_p.device)
    mixture_log_p = torch.logsumexp(
        torch.stack(
            [
                teacher_log_p + torch.log(beta_tensor),
                student_log_p + torch.log1p(-beta_tensor),
            ]
        ),
        dim=0,
    )
    teacher_term = (teacher_log_p.exp() * (teacher_log_p - mixture_log_p)).sum(dim=-1)
    student_term = (student_log_p.exp() * (student_log_p - mixture_log_p)).sum(dim=-1)
    return beta_tensor * teacher_term + (1.0 - beta_tensor) * student_term


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Mean over a boolean/weight mask, failing on an empty effective mask."""

    if values.shape != mask.shape:
        raise ValueError(
            f"values and mask must have identical shapes, got {values.shape} and {mask.shape}"
        )
    weights = mask.to(dtype=values.dtype)
    denominator = weights.sum()
    if denominator.detach().item() <= 0:
        raise ValueError("cannot reduce an empty effective mask")
    return (values * weights).sum() / denominator
