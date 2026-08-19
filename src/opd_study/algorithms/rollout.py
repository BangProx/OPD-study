"""Student trajectory collection and gradient-free teacher scoring."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from opd_study.data.tokenizer import CharacterTokenizer
from opd_study.models.tiny_transformer import TinyCausalLM
from opd_study.types import TeacherSignals, TrajectoryBatch


def _module_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration as error:
        raise ValueError("model must have at least one parameter") from error


def collect_student_trajectories(
    student: TinyCausalLM,
    prompts: Sequence[str],
    tokenizer: CharacterTokenizer,
    *,
    max_new_tokens: int = 64,
    min_new_tokens: int = 0,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> TrajectoryBatch:
    """Sample responses and save detached rollout-time selected log-probabilities."""

    if not prompts:
        raise ValueError("prompts must not be empty")
    device = _module_device(student)
    sequences: list[Tensor] = []
    prompt_lengths: list[int] = []
    for prompt in prompts:
        prompt_tensor = torch.tensor(
            tokenizer.encode(prompt, bos=True), dtype=torch.long, device=device
        ).unsqueeze(0)
        if prompt_tensor.shape[1] >= student.config.max_sequence_length:
            raise ValueError("a prompt is too long for the student context window")
        generated = student.generate(
            prompt_tensor,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            min_new_tokens=min_new_tokens,
            temperature=temperature,
            generator=generator,
        ).squeeze(0)
        sequences.append(generated)
        prompt_lengths.append(prompt_tensor.shape[1])

    sequence_length = max(sequence.shape[0] for sequence in sequences)
    token_ids = torch.full(
        (len(sequences), sequence_length),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(token_ids, dtype=torch.bool)
    response_mask = torch.zeros_like(token_ids, dtype=torch.bool)
    for row, (sequence, prompt_length) in enumerate(zip(sequences, prompt_lengths, strict=False)):
        length = sequence.shape[0]
        token_ids[row, :length] = sequence
        attention_mask[row, :length] = True
        response_mask[row, prompt_length:length] = True

    was_training = student.training
    student.eval()
    with torch.no_grad():
        logits = student(token_ids, attention_mask)
        selected = torch.log_softmax(logits[:, :-1].float(), dim=-1).gather(
            -1, token_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        rollout_logprobs = torch.zeros_like(token_ids, dtype=selected.dtype)
        rollout_logprobs[:, 1:] = selected
    student.train(was_training)
    return TrajectoryBatch(
        token_ids=token_ids,
        attention_mask=attention_mask,
        response_mask=response_mask,
        prompt_lengths=torch.tensor(prompt_lengths, dtype=torch.long, device=device),
        student_logprobs=rollout_logprobs.detach(),
    )


def score_teacher(teacher: nn.Module, trajectories: TrajectoryBatch) -> TeacherSignals:
    """Evaluate a teacher without allocating parameter gradients or changing its mode."""

    device = _module_device(teacher)
    if trajectories.token_ids.device != device:
        raise ValueError("teacher and trajectories must be on the same device")
    was_training = teacher.training
    teacher.eval()
    try:
        with torch.no_grad():
            logits = teacher(trajectories.token_ids, trajectories.attention_mask)
    finally:
        teacher.train(was_training)
    return TeacherSignals(logits=logits.detach())


def collect_multiturn_trajectories(
    student: TinyCausalLM,
    prompts: Sequence[str],
    tokenizer: CharacterTokenizer,
    *,
    number_of_turns: int,
    tokens_per_turn: int,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
    teacher_prefix: TinyCausalLM | None = None,
    teacher_prefix_turns: int = 0,
) -> TrajectoryBatch:
    """Collect student turns, optionally after a teacher-generated B2F prefix.

    A newline is inserted as a deterministic environment observation between turns and
    excluded from the loss.  The calculator lesson replaces this separator with real
    tool feedback; this compact collector keeps algorithm smoke tests model-only.
    """

    if not prompts:
        raise ValueError("prompts must not be empty")
    if min(number_of_turns, tokens_per_turn) < 1:
        raise ValueError("number_of_turns and tokens_per_turn must be positive")
    if not 0 <= teacher_prefix_turns < number_of_turns:
        raise ValueError("teacher_prefix_turns must be within [0, number_of_turns)")
    if teacher_prefix_turns and teacher_prefix is None:
        raise ValueError("B2F collection requires a teacher_prefix model")
    device = _module_device(student)
    if teacher_prefix is not None and _module_device(teacher_prefix) != device:
        raise ValueError("student and teacher prefix model must share a device")

    sequences: list[Tensor] = []
    response_rows: list[list[bool]] = []
    turn_rows: list[list[int]] = []
    prompt_lengths: list[int] = []
    separator = tokenizer.encode("\n")
    for prompt in prompts:
        sequence = torch.tensor(
            tokenizer.encode(prompt, bos=True), dtype=torch.long, device=device
        ).unsqueeze(0)
        prompt_length = sequence.shape[1]
        response = [False] * prompt_length
        turn_ids = [-1] * prompt_length
        for turn in range(number_of_turns):
            actor = (
                teacher_prefix
                if teacher_prefix is not None and turn < teacher_prefix_turns
                else student
            )
            assert actor is not None
            before = sequence.shape[1]
            sequence = actor.generate(
                sequence,
                max_new_tokens=tokens_per_turn,
                min_new_tokens=tokens_per_turn,
                eos_token_id=tokenizer.eos_token_id,
                temperature=temperature,
                generator=generator,
            )
            generated_count = sequence.shape[1] - before
            student_turn = turn >= teacher_prefix_turns
            response.extend([student_turn] * generated_count)
            turn_ids.extend([turn if student_turn else -1] * generated_count)
            if turn + 1 < number_of_turns:
                separator_tensor = torch.tensor(
                    separator, dtype=torch.long, device=device
                ).unsqueeze(0)
                if (
                    sequence.shape[1] + separator_tensor.shape[1]
                    >= student.config.max_sequence_length
                ):
                    raise ValueError("multi-turn trajectory exceeded the context window")
                sequence = torch.cat((sequence, separator_tensor), dim=1)
                response.extend([False] * len(separator))
                turn_ids.extend([-1] * len(separator))
        sequences.append(sequence.squeeze(0))
        response_rows.append(response)
        turn_rows.append(turn_ids)
        prompt_lengths.append(prompt_length)

    sequence_length = max(sequence.shape[0] for sequence in sequences)
    shape = (len(sequences), sequence_length)
    token_ids = torch.full(shape, tokenizer.pad_token_id, dtype=torch.long, device=device)
    attention = torch.zeros(shape, dtype=torch.bool, device=device)
    response_mask = torch.zeros(shape, dtype=torch.bool, device=device)
    turn_ids_tensor = torch.full(shape, -1, dtype=torch.long, device=device)
    for index, (sequence, response, turn_ids) in enumerate(
        zip(sequences, response_rows, turn_rows, strict=False)
    ):
        length = sequence.shape[0]
        token_ids[index, :length] = sequence
        attention[index, :length] = True
        response_mask[index, :length] = torch.tensor(
            response, dtype=torch.bool, device=device
        )
        turn_ids_tensor[index, :length] = torch.tensor(
            turn_ids, dtype=torch.long, device=device
        )
    was_training = student.training
    student.eval()
    with torch.no_grad():
        logits = student(token_ids, attention)
        selected = torch.log_softmax(logits[:, :-1].float(), dim=-1).gather(
            -1, token_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        rollout_logprobs = torch.zeros_like(token_ids, dtype=selected.dtype)
        rollout_logprobs[:, 1:] = selected
    student.train(was_training)
    return TrajectoryBatch(
        token_ids=token_ids,
        attention_mask=attention,
        response_mask=response_mask,
        prompt_lengths=torch.tensor(prompt_lengths, dtype=torch.long, device=device),
        student_logprobs=rollout_logprobs.detach(),
        turn_ids=turn_ids_tensor,
        step_ids=turn_ids_tensor.clone(),
        terminal=torch.ones(len(sequences), dtype=torch.bool, device=device),
    )
