"""Training loops and fair-comparison utilities."""

from opd_study.training.advanced import train_advanced_distillation
from opd_study.training.core import (
    TrainResult,
    limit_response_tokens,
    train_distillation,
    train_sft,
)

__all__ = [
    "TrainResult",
    "limit_response_tokens",
    "train_advanced_distillation",
    "train_distillation",
    "train_sft",
]
