"""Dynamic-rollout loops for approved modern and multi-turn OPD variants."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import torch

from opd_study.algorithms import (
    collect_multiturn_trajectories,
    collect_student_trajectories,
    opd2_loss,
    sage_opd_loss,
    score_teacher,
    sod_loss,
    tcod_loss,
    vopd_loss,
)
from opd_study.algorithms.sage_opd import proxy_intervention_from_token_agreement
from opd_study.algorithms.tcod import curriculum_depth
from opd_study.data import ArithmeticExample, CharacterTokenizer
from opd_study.models import TinyCausalLM
from opd_study.training.core import TrainResult
from opd_study.training.optim import clip_gradient_norm
from opd_study.types import TeacherSignals


def train_advanced_distillation(
    student: TinyCausalLM,
    teacher: TinyCausalLM,
    examples: Sequence[ArithmeticExample],
    tokenizer: CharacterTokenizer,
    *,
    algorithm: str,
    steps: int,
    batch_size: int,
    number_of_turns: int = 3,
    tokens_per_turn: int = 2,
    learning_rate: float = 3e-4,
    teacher_base: TinyCausalLM | None = None,
    generator: torch.Generator | None = None,
    optimizer_state: dict[str, Any] | None = None,
    start_step: int = 0,
    curriculum_total_steps: int | None = None,
) -> TrainResult:
    """Train vOPD/OPD²/TCOD/SOD/SAGE with freshly collected student trajectories."""

    allowed = {"vopd", "opd2", "tcod_f2b", "tcod_b2f", "sod", "sage_opd"}
    if algorithm not in allowed:
        raise ValueError(f"advanced algorithm must be one of {sorted(allowed)}")
    if min(steps, batch_size, number_of_turns, tokens_per_turn) < 1:
        raise ValueError("step, batch, turn and token counts must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not examples:
        raise ValueError("examples must not be empty")
    if algorithm == "opd2" and teacher_base is None:
        raise ValueError("OPD² requires a teacher_base model")
    device = next(student.parameters()).device
    if next(teacher.parameters()).device != device:
        raise ValueError("teacher and student must share a device")
    if teacher_base is not None and next(teacher_base.parameters()).device != device:
        raise ValueError("teacher_base and student must share a device")
    teacher.requires_grad_(False)
    if teacher_base is not None:
        teacher_base.requires_grad_(False)
    sampling_generator = generator or torch.Generator(device="cpu").manual_seed(0)
    optimizer = torch.optim.AdamW(student.parameters(), lr=learning_rate)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    started = time.perf_counter()
    history: list[dict[str, Any]] = []
    response_tokens = 0
    schedule_steps = curriculum_total_steps or (start_step + steps)
    if schedule_steps < start_step + steps:
        raise ValueError("curriculum_total_steps cannot end before this training segment")
    pacing = max(1, schedule_steps // number_of_turns)

    for step in range(start_step + 1, start_step + steps + 1):
        start = ((step - 1) * batch_size) % len(examples)
        prompts = [
            examples[(start + offset) % len(examples)].prompt
            for offset in range(batch_size)
        ]
        if algorithm in {"vopd", "opd2"}:
            trajectories = collect_student_trajectories(
                student,
                prompts,
                tokenizer,
                max_new_tokens=number_of_turns * tokens_per_turn,
                min_new_tokens=number_of_turns * tokens_per_turn,
                temperature=1.0,
                generator=sampling_generator,
            )
            depth = number_of_turns
        else:
            depth = curriculum_depth(
                step - 1,
                start_depth=1,
                pacing_steps=pacing,
                maximum_depth=number_of_turns,
            )
            if algorithm == "tcod_f2b":
                trajectory_turns = depth
                prefix_turns = 0
            elif algorithm == "tcod_b2f":
                trajectory_turns = number_of_turns
                prefix_turns = number_of_turns - depth
            else:
                trajectory_turns = number_of_turns
                prefix_turns = 0
            trajectories = collect_multiturn_trajectories(
                student,
                prompts,
                tokenizer,
                number_of_turns=trajectory_turns,
                tokens_per_turn=tokens_per_turn,
                temperature=1.0,
                generator=sampling_generator,
                teacher_prefix=teacher if prefix_turns else None,
                teacher_prefix_turns=prefix_turns,
            )

        teacher_signals = score_teacher(teacher, trajectories)
        student_logits = student(trajectories.token_ids, trajectories.attention_mask)
        if algorithm == "vopd":
            output = vopd_loss(student_logits, trajectories, teacher_signals)
        elif algorithm == "opd2":
            assert teacher_base is not None
            base_signals = score_teacher(teacher_base, trajectories)
            output = opd2_loss(
                student_logits, trajectories, teacher_signals, base_signals
            )
        elif algorithm.startswith("tcod_"):
            output = tcod_loss(
                student_logits,
                trajectories,
                teacher_signals,
                depth=depth,
                direction=algorithm.removeprefix("tcod_"),
            )
        elif algorithm == "sod":
            output = sod_loss(student_logits, trajectories, teacher_signals)
        else:
            assert teacher_signals.logits is not None
            intervention = proxy_intervention_from_token_agreement(
                student_logits, teacher_signals.logits, trajectories
            )
            output = sage_opd_loss(
                student_logits,
                trajectories,
                TeacherSignals(
                    logits=teacher_signals.logits,
                    intervene=intervention,
                ),
            )

        if not torch.isfinite(output.loss).item():
            raise FloatingPointError(f"non-finite {algorithm} loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        output.loss.backward()
        gradient_norm = clip_gradient_norm(student.parameters(), 1.0)
        optimizer.step()
        step_tokens = int(trajectories.response_mask.sum().item())
        response_tokens += step_tokens
        history.append(
            {
                "step": float(step),
                "loss": float(output.loss.detach().cpu()),
                "gradient_norm": float(gradient_norm.detach().cpu()),
                "response_tokens": float(response_tokens),
                "wall_seconds": time.perf_counter() - started,
                "algorithm": algorithm,
                "curriculum_depth": float(depth),
                "sage_proxy_judge": float(algorithm == "sage_opd"),
            }
        )
    return TrainResult(
        history=tuple(history),
        optimizer_steps=steps,
        response_tokens=response_tokens,
        wall_seconds=time.perf_counter() - started,
        optimizer_state=optimizer.state_dict(),
    )
