"""Typed tensor containers shared by algorithms and training backends."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from torch import Tensor


def _require_shape(name: str, tensor: Tensor, shape: tuple[int, ...]) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")


@dataclass(frozen=True)
class TrajectoryBatch:
    """A padded batch of token trajectories.

    ``student_logprobs`` stores rollout-time snapshots and must never require a
    gradient. Update-time logits are recomputed from ``token_ids``.
    """

    token_ids: Tensor
    attention_mask: Tensor
    response_mask: Tensor
    prompt_lengths: Tensor
    student_logprobs: Tensor | None = None
    turn_ids: Tensor | None = None
    step_ids: Tensor | None = None
    terminal: Tensor | None = None
    policy_version: int = 0

    def __post_init__(self) -> None:
        if self.token_ids.ndim != 2:
            raise ValueError(f"token_ids must be rank 2 [B, T], got {self.token_ids.shape}")
        batch_size, sequence_length = self.token_ids.shape
        token_shape = (batch_size, sequence_length)
        _require_shape("attention_mask", self.attention_mask, token_shape)
        _require_shape("response_mask", self.response_mask, token_shape)
        _require_shape("prompt_lengths", self.prompt_lengths, (batch_size,))
        if self.token_ids.dtype.is_floating_point:
            raise TypeError("token_ids must use an integer dtype")
        if self.attention_mask.device != self.token_ids.device:
            raise ValueError("attention_mask and token_ids must be on the same device")
        if self.response_mask.device != self.token_ids.device:
            raise ValueError("response_mask and token_ids must be on the same device")
        if self.prompt_lengths.device != self.token_ids.device:
            raise ValueError("prompt_lengths and token_ids must be on the same device")
        attention = self.attention_mask.bool()
        if sequence_length > 1 and (~attention[:, :-1] & attention[:, 1:]).any().item():
            raise ValueError("attention_mask must be a contiguous prefix followed by padding")
        if (self.response_mask.bool() & ~self.attention_mask.bool()).any().item():
            raise ValueError("response_mask cannot include padding/non-attended tokens")
        if torch.lt(self.prompt_lengths, 1).any().item() or torch.gt(
            self.prompt_lengths, sequence_length
        ).any().item():
            raise ValueError("prompt_lengths must be within [1, sequence_length]")
        for name, tensor in (
            ("student_logprobs", self.student_logprobs),
            ("turn_ids", self.turn_ids),
            ("step_ids", self.step_ids),
        ):
            if tensor is not None:
                _require_shape(name, tensor, token_shape)
                if tensor.device != self.token_ids.device:
                    raise ValueError(f"{name} and token_ids must be on the same device")
        if self.student_logprobs is not None and self.student_logprobs.requires_grad:
            raise ValueError("rollout-time student_logprobs must be detached")
        if self.terminal is not None:
            _require_shape("terminal", self.terminal, (batch_size,))
            if self.terminal.device != self.token_ids.device:
                raise ValueError("terminal and token_ids must be on the same device")


@dataclass(frozen=True)
class TeacherSignals:
    """Teacher outputs evaluated without gradient tracking."""

    logits: Tensor | None = None
    logprobs: Tensor | None = None
    confidence: Tensor | None = None
    intervene: Tensor | None = None

    def __post_init__(self) -> None:
        for name, tensor in (
            ("logits", self.logits),
            ("logprobs", self.logprobs),
            ("confidence", self.confidence),
            ("intervene", self.intervene),
        ):
            if tensor is not None and tensor.requires_grad:
                raise ValueError(f"teacher {name} must be detached")


@dataclass(frozen=True)
class LossOutput:
    """A scalar optimization target with token-level audit information."""

    loss: Tensor
    token_loss: Tensor
    effective_mask: Tensor
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.loss.ndim != 0:
            raise ValueError(f"loss must be scalar, got shape {tuple(self.loss.shape)}")
        if self.token_loss.shape != self.effective_mask.shape:
            raise ValueError("token_loss and effective_mask must have the same shape")
