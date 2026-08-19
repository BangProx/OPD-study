"""Auditable single-process training loops for the educational mini backend."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from opd_study.algorithms import (
    generalized_kd_loss,
    off_policy_kd_loss,
    on_policy_distillation_loss,
    score_teacher,
    supervised_fine_tuning_loss,
)
from opd_study.data import ArithmeticExample, CharacterTokenizer, collate_examples
from opd_study.models import TinyCausalLM
from opd_study.training.optim import clip_gradient_norm
from opd_study.types import LossOutput, TrajectoryBatch


@dataclass(frozen=True)
class TrainResult:
    """Small immutable record used by reports, cards and integration tests."""

    history: tuple[dict[str, Any], ...]
    optimizer_steps: int
    response_tokens: int
    wall_seconds: float
    optimizer_state: dict[str, Any]


def limit_response_tokens(batch: TrajectoryBatch, maximum_tokens: int) -> TrajectoryBatch:
    """Select the first N response targets in stable row-major order."""

    if maximum_tokens < 1:
        raise ValueError("maximum_tokens must be positive")
    available = int(batch.response_mask.sum().item())
    if available < maximum_tokens:
        raise ValueError(
            f"batch has {available} response tokens but {maximum_tokens} are required"
        )
    flat_mask = batch.response_mask.flatten()
    response_indices = torch.nonzero(flat_mask, as_tuple=False).squeeze(-1)
    limited_flat = torch.zeros_like(flat_mask)
    limited_flat[response_indices[:maximum_tokens]] = True
    limited_mask = limited_flat.view_as(batch.response_mask)
    return TrajectoryBatch(
        token_ids=batch.token_ids,
        attention_mask=batch.attention_mask,
        response_mask=limited_mask,
        prompt_lengths=batch.prompt_lengths,
        student_logprobs=batch.student_logprobs,
        turn_ids=batch.turn_ids,
        step_ids=batch.step_ids,
        terminal=batch.terminal,
        policy_version=batch.policy_version,
    )


def _fixed_batches(
    examples: Sequence[ArithmeticExample],
    tokenizer: CharacterTokenizer,
    *,
    steps: int,
    batch_size: int,
    tokens_per_step: int,
    device: torch.device,
    start_step: int = 0,
) -> tuple[TrajectoryBatch, ...]:
    if not examples:
        raise ValueError("training examples must not be empty")
    batches: list[TrajectoryBatch] = []
    for step in range(start_step, start_step + steps):
        start = (step * batch_size) % len(examples)
        selected = [examples[(start + offset) % len(examples)] for offset in range(batch_size)]
        batch = collate_examples(selected, tokenizer, device=device)
        batches.append(limit_response_tokens(batch, tokens_per_step))
    return tuple(batches)


def _optimize(
    model: nn.Module,
    batches: Sequence[TrajectoryBatch],
    objective: Callable[[TrajectoryBatch], LossOutput],
    *,
    learning_rate: float,
    algorithm_name: str,
    optimizer_state: dict[str, Any] | None = None,
    start_step: int = 0,
) -> TrainResult:
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    history: list[dict[str, Any]] = []
    response_tokens = 0
    started = time.perf_counter()
    model.train()
    for step, batch in enumerate(batches, start=start_step + 1):
        optimizer.zero_grad(set_to_none=True)
        output = objective(batch)
        if not torch.isfinite(output.loss).item():
            raise FloatingPointError(f"non-finite loss at step {step}")
        output.loss.backward()
        gradient_norm = clip_gradient_norm(model.parameters(), 1.0)
        optimizer.step()
        step_tokens = int(output.effective_mask.sum().item())
        response_tokens += step_tokens
        history.append(
            {
                "step": float(step),
                "loss": float(output.loss.detach().cpu()),
                "gradient_norm": float(gradient_norm.detach().cpu()),
                "response_tokens": float(response_tokens),
                "wall_seconds": time.perf_counter() - started,
                "algorithm": algorithm_name,
            }
        )
    return TrainResult(
        history=tuple(history),
        optimizer_steps=len(batches),
        response_tokens=response_tokens,
        wall_seconds=time.perf_counter() - started,
        optimizer_state=optimizer.state_dict(),
    )


def train_sft(
    model: TinyCausalLM,
    examples: Sequence[ArithmeticExample],
    tokenizer: CharacterTokenizer,
    *,
    steps: int,
    batch_size: int,
    tokens_per_step: int,
    learning_rate: float = 3e-4,
    optimizer_state: dict[str, Any] | None = None,
    start_step: int = 0,
) -> TrainResult:
    """Train hard-label SFT with an exact response-token budget per update."""

    device = next(model.parameters()).device
    batches = _fixed_batches(
        examples,
        tokenizer,
        steps=steps,
        batch_size=batch_size,
        tokens_per_step=tokens_per_step,
        device=device,
        start_step=start_step,
    )
    return _optimize(
        model,
        batches,
        lambda batch: supervised_fine_tuning_loss(
            model(batch.token_ids, batch.attention_mask), batch
        ),
        learning_rate=learning_rate,
        algorithm_name="sft",
        optimizer_state=optimizer_state,
        start_step=start_step,
    )


def train_distillation(
    student: TinyCausalLM,
    teacher: TinyCausalLM,
    examples: Sequence[ArithmeticExample],
    tokenizer: CharacterTokenizer,
    *,
    algorithm: str,
    steps: int,
    batch_size: int,
    tokens_per_step: int,
    learning_rate: float = 3e-4,
    beta: float = 0.5,
    lambda_on_policy: float = 1.0,
    rollout_temperature: float = 1.0,
    generator: torch.Generator | None = None,
    optimizer_state: dict[str, Any] | None = None,
    start_step: int = 0,
) -> TrainResult:
    """Train off-policy KD, GKD or OPD with explicit state-source selection."""

    if algorithm not in {"off_policy_kd", "gkd", "opd"}:
        raise ValueError("algorithm must be off_policy_kd, gkd or opd")
    if not 0.0 <= lambda_on_policy <= 1.0:
        raise ValueError("lambda_on_policy must be in [0, 1]")
    device = next(student.parameters()).device
    if next(teacher.parameters()).device != device:
        raise ValueError("teacher and student must be on the same device")
    teacher.requires_grad_(False)

    fixed = _fixed_batches(
        examples,
        tokenizer,
        steps=steps,
        batch_size=batch_size,
        tokens_per_step=tokens_per_step,
        device=device,
        start_step=start_step,
    )
    source_generator = generator or torch.Generator(device="cpu").manual_seed(0)
    optimizer = torch.optim.AdamW(student.parameters(), lr=learning_rate)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    history: list[dict[str, Any]] = []
    response_tokens = 0
    started = time.perf_counter()
    student.train()
    for step, fixed_batch in enumerate(fixed, start=start_step + 1):
        use_on_policy = algorithm == "opd"
        if algorithm == "gkd":
            use_on_policy = bool(
                torch.rand((), generator=source_generator).item() < lambda_on_policy
            )
        if use_on_policy:
            start = ((step - 1) * batch_size) % len(examples)
            prompts = [
                examples[(start + offset) % len(examples)].prompt
                for offset in range(batch_size)
            ]
            from opd_study.algorithms import collect_student_trajectories

            tokens_per_sequence = math.ceil(tokens_per_step / batch_size)
            collected = collect_student_trajectories(
                student,
                prompts,
                tokenizer,
                max_new_tokens=tokens_per_sequence,
                min_new_tokens=tokens_per_sequence,
                temperature=rollout_temperature,
                generator=source_generator,
            )
            batch = limit_response_tokens(collected, tokens_per_step)
        else:
            batch = fixed_batch
        signals = score_teacher(teacher, batch)
        student_logits = student(batch.token_ids, batch.attention_mask)
        if algorithm == "off_policy_kd":
            output = off_policy_kd_loss(student_logits, batch, signals)
        elif algorithm == "gkd":
            output = generalized_kd_loss(student_logits, batch, signals, beta=beta)
        else:
            output = on_policy_distillation_loss(student_logits, batch, signals)
        if not torch.isfinite(output.loss).item():
            raise FloatingPointError(f"non-finite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        output.loss.backward()
        gradient_norm = clip_gradient_norm(student.parameters(), 1.0)
        optimizer.step()
        step_tokens = int(output.effective_mask.sum().item())
        response_tokens += step_tokens
        history.append(
            {
                "step": float(step),
                "loss": float(output.loss.detach().cpu()),
                "gradient_norm": float(gradient_norm.detach().cpu()),
                "response_tokens": float(response_tokens),
                "wall_seconds": time.perf_counter() - started,
                "algorithm": algorithm,
                "on_policy": float(use_on_policy),
            }
        )
    return TrainResult(
        history=tuple(history),
        optimizer_steps=steps,
        response_tokens=response_tokens,
        wall_seconds=time.perf_counter() - started,
        optimizer_state=optimizer.state_dict(),
    )
