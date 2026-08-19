"""Clean-room support metrics from Rethinking OPD (arXiv:2604.13016v2)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from opd_study.math import entropy_from_logits


@dataclass(frozen=True)
class SupportDiagnostics:
    overlap_ratio: float
    overlap_token_advantage: float
    absolute_entropy_gap: float
    student_top_k_mass: float
    teacher_top_k_mass: float
    evaluated_tokens: int


def support_diagnostics(
    student_logits: Tensor,
    teacher_logits: Tensor,
    mask: Tensor,
    *,
    top_k: int,
) -> SupportDiagnostics:
    """Measure top-k overlap and alignment only on explicitly selected states."""

    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must match")
    if mask.shape != student_logits.shape[:-1]:
        raise ValueError("mask must match non-vocabulary logits dimensions")
    vocabulary_size = student_logits.shape[-1]
    if not 1 <= top_k <= vocabulary_size:
        raise ValueError(f"top_k must be within [1, {vocabulary_size}]")
    selected_mask = mask.bool().reshape(-1)
    if not selected_mask.any().item():
        raise ValueError("support diagnostics require at least one selected token")
    student_log = torch.log_softmax(student_logits.float(), dim=-1).reshape(
        -1, vocabulary_size
    )[selected_mask]
    teacher_log = torch.log_softmax(teacher_logits.float(), dim=-1).reshape(
        -1, vocabulary_size
    )[selected_mask]
    student_top = student_log.topk(top_k, dim=-1).indices
    teacher_top = teacher_log.topk(top_k, dim=-1).indices
    overlaps: list[float] = []
    advantages: list[float] = []
    for student_row, teacher_row, student_ids, teacher_ids in zip(
        student_log, teacher_log, student_top, teacher_top, strict=False
    ):
        common = sorted(set(student_ids.tolist()) & set(teacher_ids.tolist()))
        overlaps.append(len(common) / top_k)
        if common:
            indices = torch.tensor(common, dtype=torch.long, device=student_row.device)
            student_common = torch.log_softmax(student_row[indices], dim=-1)
            teacher_common = torch.log_softmax(teacher_row[indices], dim=-1)
            token_advantage = student_common.exp() * (teacher_common - student_common)
            advantages.append(float(token_advantage.mean().cpu()))
        else:
            advantages.append(float("nan"))
    finite_advantages = [value for value in advantages if value == value]
    student_mass = student_log.exp().gather(-1, student_top).sum(dim=-1).mean()
    teacher_mass = teacher_log.exp().gather(-1, teacher_top).sum(dim=-1).mean()
    entropy_gap = (
        entropy_from_logits(student_log) - entropy_from_logits(teacher_log)
    ).abs().mean()
    return SupportDiagnostics(
        overlap_ratio=sum(overlaps) / len(overlaps),
        overlap_token_advantage=(
            sum(finite_advantages) / len(finite_advantages)
            if finite_advantages
            else float("nan")
        ),
        absolute_entropy_gap=float(entropy_gap.cpu()),
        student_top_k_mass=float(student_mass.cpu()),
        teacher_top_k_mass=float(teacher_mass.cpu()),
        evaluated_tokens=int(selected_mask.sum().item()),
    )
