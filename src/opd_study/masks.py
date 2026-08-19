"""Causal-language-model mask helpers with explicit token alignment."""

from __future__ import annotations

import torch
from torch import Tensor


def causal_attention_mask(
    sequence_length: int, *, device: torch.device | str | None = None
) -> Tensor:
    """Return a boolean mask where ``True`` means attention is disallowed."""

    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    return torch.triu(
        torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=device),
        diagonal=1,
    )


def shifted_causal_tensors(
    logits: Tensor,
    token_ids: Tensor,
    attention_mask: Tensor,
    response_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Align position ``t`` logits with token ``t+1`` labels and masks."""

    if logits.ndim != 3 or token_ids.ndim != 2:
        raise ValueError("expected logits [B, T, V] and token_ids [B, T]")
    if logits.shape[:2] != token_ids.shape:
        raise ValueError("logits batch/time dimensions must match token_ids")
    if attention_mask.shape != token_ids.shape or response_mask.shape != token_ids.shape:
        raise ValueError("attention_mask and response_mask must match token_ids")
    if token_ids.shape[1] < 2:
        raise ValueError("causal LM training requires at least two tokens")
    prediction_mask = attention_mask[:, 1:].bool() & response_mask[:, 1:].bool()
    return logits[:, :-1], token_ids[:, 1:], attention_mask[:, 1:].bool(), prediction_mask
