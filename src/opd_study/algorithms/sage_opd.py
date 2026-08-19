"""Clean-room SAGE-OPD selective weighting from arXiv:2606.19659v1."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from opd_study.masks import shifted_causal_tensors
from opd_study.math import reverse_kl_from_logits
from opd_study.types import LossOutput, TeacherSignals, TrajectoryBatch


def intervention_labels_to_weights(
    labels: Sequence[Sequence[str]], *, weak_weight: float = 0.5
) -> Tensor:
    """Map Skip/Weak/Strong to 0/alpha/1; unparseable labels fail closed to 1."""

    if not 0.0 < weak_weight < 1.0:
        raise ValueError("weak_weight must be in (0, 1)")
    mapping = {"skip": 0.0, "weak": weak_weight, "strong": 1.0}
    rows = [
        [mapping.get(label.strip().lower(), 1.0) for label in row]
        for row in labels
    ]
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("labels must be a non-empty rectangular matrix")
    return torch.tensor(rows, dtype=torch.float32)


def proxy_intervention_from_token_agreement(
    student_logits: Tensor,
    teacher_logits: Tensor,
    trajectories: TrajectoryBatch,
) -> Tensor:
    """Deterministic mini-backend proxy for the paper's semantic teacher judge.

    Full SAGE-OPD obtains Skip/Weak/Strong from environment failures plus a separate
    teacher judgment query.  The offline toy backend has neither natural-language
    semantic equivalence nor an external judge, so it maps sampled-token teacher top-1
    agreement to Skip (>80%), Weak (>0%), or Strong (0%).  Reports label this proxy.
    """

    if trajectories.turn_ids is None:
        raise ValueError("proxy intervention requires turn_ids")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must match")
    turns = trajectories.turn_ids[:, 1:]
    response = trajectories.response_mask[:, 1:] & (turns >= 0)
    sampled = trajectories.token_ids[:, 1:]
    teacher_top = teacher_logits[:, :-1].argmax(dim=-1)
    agrees = teacher_top == sampled
    number_of_turns = int(turns[response].max().item()) + 1
    interventions = torch.ones(
        (turns.shape[0], number_of_turns),
        dtype=torch.float32,
        device=student_logits.device,
    )
    for batch_index in range(turns.shape[0]):
        for turn_id in range(number_of_turns):
            mask = response[batch_index] & (turns[batch_index] == turn_id)
            if not mask.any().item():
                continue
            agreement = agrees[batch_index][mask].float().mean()
            if agreement > 0.8:
                interventions[batch_index, turn_id] = 0.0
            elif agreement > 0.0:
                interventions[batch_index, turn_id] = 0.5
    return interventions.detach()


def sage_token_weights(
    teacher_logits: Tensor,
    trajectories: TrajectoryBatch,
    intervention: Tensor,
    *,
    epsilon: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    """Return normalized token weights and per-turn mean teacher top-1 confidence."""

    if trajectories.turn_ids is None:
        raise ValueError("SAGE-OPD requires turn_ids")
    if teacher_logits.shape[:2] != trajectories.token_ids.shape:
        raise ValueError("teacher logits must match trajectory batch/time dimensions")
    if intervention.ndim != 2 or intervention.shape[0] != trajectories.token_ids.shape[0]:
        raise ValueError("intervention must have shape [B, turns]")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if (torch.lt(intervention, 0) | torch.gt(intervention, 1)).any().item():
        raise ValueError("intervention weights must be in [0, 1]")

    probabilities = torch.softmax(teacher_logits[:, :-1].float(), dim=-1)
    top_one = probabilities.amax(dim=-1).detach()
    shifted_turns = trajectories.turn_ids[:, 1:]
    response = trajectories.response_mask[:, 1:] & (shifted_turns >= 0)
    token_selective = torch.zeros_like(top_one)
    intervention = intervention.to(top_one.device)
    turn_confidence = torch.zeros(
        intervention.shape, dtype=top_one.dtype, device=top_one.device
    )
    for batch_index in range(trajectories.token_ids.shape[0]):
        for turn_id in torch.unique(
            shifted_turns[batch_index][response[batch_index]], sorted=True
        ).tolist():
            if turn_id >= intervention.shape[1]:
                raise ValueError("turn_id exceeds the intervention matrix")
            mask = response[batch_index] & (shifted_turns[batch_index] == turn_id)
            confidence = top_one[batch_index][mask].mean()
            turn_confidence[batch_index, turn_id] = confidence
            token_selective[batch_index][mask] = (
                intervention[batch_index, turn_id] * confidence
            )
    number_of_tokens = response.sum().to(top_one.dtype)
    weighted_tokens = token_selective[response].sum()
    if number_of_tokens.item() <= 0:
        raise ValueError("SAGE-OPD found no response tokens")
    # Eq. 12: keep the dense-OPD loss scale while preserving relative weights.
    normalization = number_of_tokens / torch.clamp(weighted_tokens, min=epsilon)
    if weighted_tokens.item() == 0:
        normalization = normalization.new_tensor(0.0)
    return (token_selective * normalization).detach(), turn_confidence.detach()


def sage_opd_loss(
    student_logits: Tensor,
    trajectories: TrajectoryBatch,
    signals: TeacherSignals,
) -> LossOutput:
    """Selective reverse-KL OPD weighted by intervention times teacher confidence."""

    if signals.logits is None or signals.intervene is None:
        raise ValueError("SAGE-OPD requires teacher logits and intervention weights")
    if signals.logits.shape != student_logits.shape:
        raise ValueError("teacher and student logits must match")
    weights, confidence = sage_token_weights(
        signals.logits, trajectories, signals.intervene
    )
    _, _, _, prediction_mask = shifted_causal_tensors(
        student_logits,
        trajectories.token_ids,
        trajectories.attention_mask,
        trajectories.response_mask,
    )
    shifted_loss = reverse_kl_from_logits(
        signals.logits[:, :-1], student_logits[:, :-1]
    )
    effective_weights = weights * prediction_mask
    number_of_tokens = prediction_mask.sum()
    loss = (shifted_loss * effective_weights).sum() / number_of_tokens
    token_loss = shifted_loss.new_zeros(trajectories.token_ids.shape)
    full_weights = shifted_loss.new_zeros(trajectories.token_ids.shape)
    token_loss[:, 1:] = shifted_loss
    full_weights[:, 1:] = effective_weights
    return LossOutput(
        loss=loss,
        token_loss=token_loss,
        effective_mask=full_weights,
        metrics={
            "sage_opd/loss": float(loss.detach().cpu()),
            "sage_opd/tokens": float(number_of_tokens.cpu()),
            "sage_opd/active_weight_fraction": float(
                (effective_weights > 0).sum().cpu() / number_of_tokens.cpu()
            ),
            "sage_opd/mean_teacher_confidence": float(
                confidence[confidence > 0].mean().cpu()
                if torch.gt(confidence, 0).any()
                else 0.0
            ),
        },
    )
